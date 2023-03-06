#!/usr/bin/env python
from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import scipy.signal as signal
import os
import pickle
import imageio

import sapp_gym
from ACNet import ACNet

from tensorflow.python.client import device_lib
dev_list = device_lib.list_local_devices()
print(dev_list)
#assert len(dev_list) > 1


def make_gif(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
    imageio.mimwrite(fname,images,subrectangles=True)
    print("wrote gif")

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Worker:
    def __init__(self, game, global_network):
        self.env = game
        self.global_network = global_network
        self.global_network.initial_feed()

        self.nextGIF = episode_count  # For GIFs output

    def train(self, rollout, gamma, bootstrap_value):
        global episode_count
        self.global_network.LSTM.reset_states()
        # [s[0],s[1],a,r,s1,d,v[0,0],train_valid]
        rollout = np.array(rollout, dtype=object)
        observ = np.stack(rollout[:, 0])
        goal_vec = np.stack(rollout[:, 1])
        actions = rollout[:, 2]
        rewards = rollout[:, 3]
        values = rollout[:, 6]
        valids = np.squeeze(np.stack(rollout[:, 7]))

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value], dtype=object)
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]

        self.value_plus = np.asarray(values.tolist() + [bootstrap_value], dtype=object)
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        with tf.GradientTape() as tape:
            discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)
            advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
            valids = tf.convert_to_tensor(valids, dtype=tf.float32)
            goal_vec = tf.convert_to_tensor(goal_vec, dtype=tf.float32)
            observ = tf.convert_to_tensor(observ, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            ls, v_l, p_l, e_l, v_n, valid_l = self.global_network.compute_loss(actions, valids, discounted_rewards, advantages, observ, goal_vec)
            loss = tf.reduce_sum(ls)
            value_loss = v_l.numpy()
            policy_loss = p_l.numpy()
            valid_loss = valid_l.numpy()
            entropy = e_l.numpy()
            var_norms = v_n.numpy()
        gradient = tape.gradient(loss, self.global_network.trainable_weights)
        gradient, g_n = tf.clip_by_global_norm(gradient, GRAD_CLIP)
        grad_norms = g_n.numpy()
        self.global_network.apply_gradients(gradient)
        return value_loss/len(rollout), policy_loss/len(rollout), valid_loss/len(rollout), entropy/len(rollout), grad_norms, var_norms

    def shouldRun(self, coord, episode_count):
        if TRAINING:
            return (not coord.should_stop())
        else:
            return (episode_count < NUM_EXPS)

    def work(self, max_episode_length, gamma, coord, saver):
        global episode_count, episode_rewards, episode_lengths, episode_mean_values, episode_invalid_ops
        total_steps, i_buf = 0, 0

        while self.shouldRun(coord, episode_count):
            episode_buffer, episode_values = [], []
            episode_reward = episode_step_count = episode_inv_count = 0
            d = False

            # Initial state from the environment
            s, validActions = self.env.reset()
            self.global_network.LSTM.reset_states()

            saveGIF = False
            if OUTPUT_GIFS and ((not TRAINING) or (episode_count >= self.nextGIF)):
                saveGIF = True
                self.nextGIF = episode_count + 256
                GIF_episode = int(episode_count)
                episode_frames = [self.env._render(mode='rgb_array', screen_height=900, screen_width=900)]

            while True:  # Give me something!
                # Take an action using probabilities from policy network output.
                a_dist, v = self.global_network(tf.convert_to_tensor([s[0]], dtype=tf.float32),
                                                tf.convert_to_tensor([s[1]], dtype=tf.float32))

                a_dist = a_dist.numpy()
                v = v.numpy()

                if (not (np.argmax(a_dist.flatten()) in validActions)):
                    episode_inv_count += 1

                train_valid = np.zeros(a_size)
                train_valid[validActions] = 1

                valid_dist = np.array([a_dist[0, validActions]])
                valid_dist /= np.sum(valid_dist)

                if TRAINING:
                    #                        if(not (np.argmax(a_dist.flatten()) in validActions)):
                    #                            episode_inv_count += 1
                    #                            a     = validActions[ np.random.choice(range(valid_dist.shape[1])) ]
                    #                        else:
                    a = validActions[np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]
                else:
                    a = np.argmax(a_dist.flatten())
                    if a not in validActions or not GREEDY:
                        a = validActions[np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]

                s1, r, d, validActions = self.env.step(a)

                if saveGIF:
                    episode_frames.append(self.env._render(mode='rgb_array', screen_width=900, screen_height=900))

                episode_buffer.append([s[0], s[1], a, r, s1, d, v[0, 0], train_valid])
                episode_values.append(v[0, 0])
                episode_reward += r
                s = s1
                total_steps += 1
                episode_step_count += 1

                if d == True:
                    print('\n{} Goodbye World. We did it!'.format(episode_step_count), end='\n')

                # If the episode hasn't ended, but the experience buffer is full, then we
                # make an update step using that experience rollout.
                if TRAINING and (len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0 or d):
                    # Since we don't know what the true final return is, we "bootstrap" from our current value estimation.
                    if len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE:
                        episode_buffer_training = episode_buffer[-EXPERIENCE_BUFFER_SIZE:]
                    else:
                        episode_buffer_training = episode_buffer[:]

                    if d:
                        s1Value = 0
                    else:
                        _, s1Value = self.global_network(tf.convert_to_tensor([s[0]], dtype=tf.float32),
                                                         tf.convert_to_tensor([s[1]], dtype=tf.float32))

                    v_l, p_l, valid_l, e_l, g_n, v_n = self.train(episode_buffer_training, gamma, s1Value)

                if episode_step_count >= max_episode_length or d:
                    break

            episode_lengths.append(episode_step_count)
            episode_mean_values.append(np.nanmean(episode_values))
            episode_invalid_ops.append(episode_inv_count)
            episode_rewards.append(episode_reward)

            if not TRAINING:
                if episode_count < NUM_EXPS:
                    plan_durations[episode_count] = episode_step_count
                episode_count += 1
                print('({}) Thread 0: {} steps, {:.2f} reward ({} invalids).'.format(episode_count,
                                                                                     episode_step_count,
                                                                                     episode_reward,
                                                                                     episode_inv_count))
                GIF_episode = int(episode_count)
            else:
                episode_count += 1

                if episode_count % SUMMARY_WINDOW == 0:
                    if episode_count % 100 == 0:
                        print('Saving Model', end='\n')
                        saver.save(curr_episode)
                        print('Saved Model', end='\n')
                    mean_reward = np.nanmean(episode_rewards[-SUMMARY_WINDOW:])
                    mean_length = np.nanmean(episode_lengths[-SUMMARY_WINDOW:])
                    mean_value = np.nanmean(episode_mean_values[-SUMMARY_WINDOW:])
                    mean_invalid = np.nanmean(episode_invalid_ops[-SUMMARY_WINDOW:])

                    with global_summary.as_default():
                        tf.summary.scalar(name='Perf/Reward', data=mean_reward, step=episode_count)
                        tf.summary.scalar(name='Perf/Length', data=mean_length, step=episode_count)
                        tf.summary.scalar(name='Perf/Valid Rate',data=(mean_length - mean_invalid) / mean_length, step=episode_count)

                        tf.summary.scalar(name='Losses/Value Loss', data=v_l, step=episode_count)
                        tf.summary.scalar(name='Losses/Policy Loss', data=p_l, step=episode_count)
                        tf.summary.scalar(name='Losses/Valid Loss', data=valid_l, step=episode_count)
                        tf.summary.scalar(name='Losses/Grad Norm', data=g_n, step=episode_count)
                        tf.summary.scalar(name='Losses/Var Norm', data=v_n, step=episode_count)
                        global_summary.flush()

                    if printQ:
                        print('{} Tensorboard updated'.format(episode_count), end='\r')

            if saveGIF:
                # Dump episode frames for external gif generation (otherwise, makes the jupyter kernel crash)
                time_per_step = 0.1
                images = np.array(episode_frames)
                if TRAINING:
                    make_gif(images,
                             '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(gifs_path, GIF_episode, episode_step_count,
                                                                      episode_reward))
                else:
                    make_gif(images, '{}/episode_{:d}_{:d}.gif'.format(gifs_path, GIF_episode, episode_step_count),
                             duration=len(images) * time_per_step, true_image=True, salience=False)
            if SAVE_EPISODE_BUFFER:
                with open('gifs3D/episode_{}.dat'.format(GIF_episode), 'wb') as file:
                    pickle.dump(episode_buffer, file)


# Learning parameters
max_episode_length     = 128
episode_count          = 0
GRAD_CLIP = 1000.0
EPISODE_START          = episode_count
gamma                  = .95 # discount rate for advantage estimation and reward discounting
#moved network parameters to ACNet.py
EXPERIENCE_BUFFER_SIZE = 128
GRID_SIZE              = 11 #the size of the FOV grid to apply to each agent
ENVIRONMENT_SIZE       = (10,32)#the total size of the environment (length of one side)
OBSTACLE_DENSITY       = (0,.3) #range of densities
DIAG_MVMT              = False # Diagonal movements allowed?
a_size                 = 5 + int(DIAG_MVMT)*4
NUM_CHANNEL = 3
SUMMARY_WINDOW         = 10
LR_Q                   = 8.e-5
load_model             = False
RESET_TRAINER          = False
model_path             = 'model_sapp_vanilla'
gifs_path              = 'gifs_sapp_vanilla'
train_path             = 'train_sapp_vanilla'
GLOBAL_NET_SCOPE       = 'global'

# Simulation options
FULL_HELP              = False
OUTPUT_GIFS            = False
SAVE_EPISODE_BUFFER    = False

# Testing
TRAINING               = True
GREEDY                 = False
NUM_EXPS               = 100
MODEL_NUMBER           = 313000

# Shared arrays for tensorboard
episode_rewards        = []
episode_lengths        = []
episode_mean_values    = []
episode_invalid_ops    = []
rollouts               = None
printQ                 = False # (for headless)


print("Hello World")
if not os.path.exists(model_path):
    os.makedirs(model_path)

if not TRAINING:
    gifs_path += '_tests'
    if SAVE_EPISODE_BUFFER and not os.path.exists('gifs3D'):
        os.makedirs('gifs3D')

#Create a directory to save episode playback gifs to
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

#with tf.device("/gpu:0"): # uncomment to run on GPU, and comment next line
with tf.device("/cpu:0"):
    lr = tf.constant(LR_Q)
    trainer = tf.keras.optimizers.Nadam(learning_rate=lr)

    master_network = ACNet(a_size, trainer, True, NUM_CHANNEL, GRID_SIZE) # Generate global network
    global_summary = tf.summary.create_file_writer(train_path)

    gameEnv = sapp_gym.SAPPEnv(DIAGONAL_MOVEMENT=DIAG_MVMT, SIZE=ENVIRONMENT_SIZE,
                               observation_size=GRID_SIZE, PROB=OBSTACLE_DENSITY)

    worker = Worker(gameEnv, master_network)

    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=trainer, net=master_network)
    saver = tf.train.CheckpointManager(ckpt, model_path, checkpoint_name='model', max_to_keep=3)

    if load_model == True:
        print('Loading Model...')
        ckpt.restore(saver.latest_checkpoint)
        curr_episode = int(ckpt.step.numpy())
        print("curr_episode set to ", curr_episode)
    else:
        curr_episode = 0

    coord = tf.train.Coordinator()

    worker.work(max_episode_length,gamma,coord,saver)

if not TRAINING:
    print([np.mean(episode_lengths), np.sqrt(np.var(episode_lengths)), np.mean(np.asarray(episode_lengths < max_episode_length, dtype=float))])
