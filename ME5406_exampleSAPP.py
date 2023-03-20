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

        self.nextGIF = episode_count # For GIFs output

    def train(self, rollout, sess, gamma, bootstrap_value, rnn_state0, imitation=False):
        global episode_count

        # [s[0],s[1],a,r,s1,d,v[0,0],train_valid]
        rollout     = np.array(rollout)
        observ      = rollout[:,0]
        goal_vec    = rollout[:,1]
        actions     = rollout[:,2]
        rewards     = rollout[:,3]
        values      = rollout[:,6]
        valids      = rollout[:,7]

        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]

        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {
            self.global_network.target_v:np.stack(discounted_rewards),
            self.global_network.inputs:np.stack(observ),
            self.global_network.goal_pos:np.stack(goal_vec),
            self.global_network.actions:actions,
            self.global_network.train_valid:np.stack(valids),
            self.global_network.advantages:advantages,
            self.global_network.state_in[0]:rnn_state0[0],
            self.global_network.state_in[1]:rnn_state0[1]
        }
        
        v_l,p_l,valid_l,e_l,g_n,v_n,_ = sess.run([self.global_network.value_loss,
            self.global_network.policy_loss,
            self.global_network.valid_loss,
            self.global_network.entropy,
            self.global_network.grad_norms,
            self.global_network.var_norms,
            self.global_network.apply_grads],
            feed_dict=feed_dict)

        return v_l, p_l, valid_l, e_l, g_n, v_n

    def shouldRun(self, coord, episode_count):
        if TRAINING:
            return (not coord.should_stop())
        else:
            return (episode_count < NUM_EXPS)

    def work(self, max_episode_length, gamma, sess, coord, saver):
        global episode_count, episode_rewards, episode_lengths, episode_mean_values, episode_invalid_ops
        total_steps, i_buf = 0, 0

        with sess.as_default(), sess.graph.as_default():
            while self.shouldRun(coord, episode_count):
                episode_buffer, episode_values = [], []
                episode_reward = episode_step_count = episode_inv_count = 0
                d = False

                # Initial state from the environment
                s, validActions       = self.env.reset()
                rnn_state             = self.global_network.state_init
                rnn_state0            = rnn_state

                saveGIF = False
                if OUTPUT_GIFS and ((not TRAINING) or (episode_count >= self.nextGIF)):
                    saveGIF = True
                    self.nextGIF = episode_count + 256
                    GIF_episode = int(episode_count)
                    episode_frames = [ self.env._render(mode='rgb_array',screen_height=900,screen_width=900) ]
                    
                while True: # Give me something!
                    #Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state = sess.run([self.global_network.policy,
                                                   self.global_network.value,
                                                   self.global_network.state_out],
                                         feed_dict={self.global_network.inputs:[s[0]],
                                                    self.global_network.goal_pos:[s[1]],
                                                    self.global_network.state_in[0]:rnn_state[0],
                                                    self.global_network.state_in[1]:rnn_state[1]})

                    if(not (np.argmax(a_dist.flatten()) in validActions)):
                        episode_inv_count += 1

                    train_valid = np.zeros(a_size)
                    train_valid[validActions] = 1

                    valid_dist = np.array([a_dist[0,validActions]])
                    valid_dist /= np.sum(valid_dist)

                    if TRAINING:
#                        if(not (np.argmax(a_dist.flatten()) in validActions)):
#                            episode_inv_count += 1
#                            a     = validActions[ np.random.choice(range(valid_dist.shape[1])) ]
#                        else:
                        a     = validActions[ np.random.choice(range(valid_dist.shape[1]),p=valid_dist.ravel()) ]
                    else:
                        a         = np.argmax(a_dist.flatten())
                        if a not in validActions or not GREEDY:
                            a     = validActions[ np.random.choice(range(valid_dist.shape[1]),p=valid_dist.ravel()) ]

                    s1, r, d, validActions = self.env.step(a)

                    if saveGIF:
                        episode_frames.append(self.env._render(mode='rgb_array',screen_width=900,screen_height=900))

                    episode_buffer.append([s[0],s[1],a,r,s1,d,v[0,0],train_valid])
                    episode_values.append(v[0,0])
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
                            s1Value = sess.run(self.global_network.value, 
                                 feed_dict={self.global_network.inputs:np.array([s[0]])
                                            ,self.global_network.goal_pos:[s[1]]
                                            ,self.global_network.state_in[0]:rnn_state[0]
                                            ,self.global_network.state_in[1]:rnn_state[1]})[0,0]

                        v_l, p_l, valid_l, e_l, g_n, v_n = self.train(episode_buffer_training,sess,gamma,s1Value,rnn_state0)
                        rnn_state0                       = rnn_state

                    if episode_step_count >= max_episode_length or d:
                        break

                episode_lengths.append(episode_step_count)
                episode_mean_values.append(np.nanmean(episode_values))
                episode_invalid_ops.append(episode_inv_count)
                episode_rewards.append(episode_reward)

                if not TRAINING:
                    episode_count += 1
                    print('({}) Thread 0: {} steps, {:.2f} reward ({} invalids).'.format(episode_count, episode_step_count, episode_reward, episode_inv_count))
                    GIF_episode = int(episode_count)
                else:
                    episode_count += 1

                    if episode_count % SUMMARY_WINDOW == 0:
                        if episode_count % 100 == 0:
                            print ('Saving Model', end='\n')
                            saver.save(sess, model_path+'/model-'+str(int(episode_count))+'.cptk')
                            print ('Saved Model', end='\n')
                        mean_reward = np.nanmean(episode_rewards[-SUMMARY_WINDOW:])
                        mean_length = np.nanmean(episode_lengths[-SUMMARY_WINDOW:])
                        mean_value = np.nanmean(episode_mean_values[-SUMMARY_WINDOW:])
                        mean_invalid = np.nanmean(episode_invalid_ops[-SUMMARY_WINDOW:])

                        summary = tf.Summary()
                        summary.value.add(tag='Perf/Reward', simple_value=mean_reward)
                        summary.value.add(tag='Perf/Length', simple_value=mean_length)
                        summary.value.add(tag='Perf/Valid Rate', simple_value=(mean_length-mean_invalid)/mean_length)

                        summary.value.add(tag='Losses/Value Loss', simple_value=v_l)
                        summary.value.add(tag='Losses/Policy Loss', simple_value=p_l)
                        summary.value.add(tag='Losses/Valid Loss', simple_value=valid_l)
                        summary.value.add(tag='Losses/Grad Norm', simple_value=g_n)
                        summary.value.add(tag='Losses/Var Norm', simple_value=v_n)
                        global_summary.add_summary(summary, int(episode_count))

                        global_summary.flush()

                        if printQ:
                            print('{} Tensorboard updated'.format(episode_count), end='\r')

                if saveGIF:
                    # Dump episode frames for external gif generation (otherwise, makes the jupyter kernel crash)
                    time_per_step = 0.1
                    images = np.array(episode_frames)
                    if TRAINING:
                        make_gif(images, '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(gifs_path,GIF_episode,episode_step_count,episode_reward))
                    else:
                        make_gif(images, '{}/episode_{:d}_{:d}.gif'.format(gifs_path,GIF_episode,episode_step_count), duration=len(images)*time_per_step,true_image=True,salience=False)
                if SAVE_EPISODE_BUFFER:
                    with open('gifs3D/episode_{}.dat'.format(GIF_episode), 'wb') as file:
                        pickle.dump(episode_buffer, file)


# Learning parameters
max_episode_length     = 128
episode_count          = 0
EPISODE_START          = episode_count
gamma                  = .95 # discount rate for advantage estimation and reward discounting
#moved network parameters to ACNet.py
EXPERIENCE_BUFFER_SIZE = 128
GRID_SIZE              = 11 #the size of the FOV grid to apply to each agent
ENVIRONMENT_SIZE       = (10,32)#the total size of the environment (length of one side)
OBSTACLE_DENSITY       = (0,.3) #range of densities
DIAG_MVMT              = False # Diagonal movements allowed?
a_size                 = 5 + int(DIAG_MVMT)*4
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
OUTPUT_GIFS            = True
SAVE_EPISODE_BUFFER    = False

# Testing
TRAINING               = True
GREEDY                 = False
NUM_EXPS               = 100
MODEL_NUMBER           = 140500

# Shared arrays for tensorboard
episode_rewards        = []
episode_lengths        = []
episode_mean_values    = []
episode_invalid_ops    = []
rollouts               = None
printQ                 = False # (for headless)


tf.reset_default_graph()
print("Hello World")
if not os.path.exists(model_path):
    os.makedirs(model_path)
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True

if not TRAINING:
    gifs_path += '_tests'
    if SAVE_EPISODE_BUFFER and not os.path.exists('gifs3D'):
        os.makedirs('gifs3D')

#Create a directory to save episode playback gifs to
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

#with tf.device("/gpu:0"): # uncomment to run on GPU, and comment next line
with tf.device("/cpu:0"):
    trainer = tf.contrib.opt.NadamOptimizer(learning_rate=LR_Q, use_locking=True)

    master_network = ACNet(GLOBAL_NET_SCOPE, a_size, trainer, True, GRID_SIZE) # Generate global network

    gameEnv = sapp_gym.SAPPEnv(DIAGONAL_MOVEMENT=DIAG_MVMT, SIZE=ENVIRONMENT_SIZE, 
                               observation_size=GRID_SIZE, PROB=OBSTACLE_DENSITY)

    worker  = Worker(gameEnv, master_network)

    global_summary = tf.summary.FileWriter(train_path)
    saver = tf.train.Saver(max_to_keep=2)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()

        if load_model == True:
            print ('Loading Model...')
            if not TRAINING:
                with open(model_path+'/checkpoint', 'w') as file:
                    file.write('model_checkpoint_path: "model-{}.cptk"'.format(MODEL_NUMBER))
                    file.close()
            ckpt = tf.train.get_checkpoint_state(model_path)
            p=ckpt.model_checkpoint_path
            p=p[p.find('-')+1:]
            p=p[:p.find('.')]
            if TRAINING:
                episode_count = int(p)
            else:
                episode_count = 0
            saver.restore(sess,ckpt.model_checkpoint_path)
            print("episode_count set to ",episode_count)
            if RESET_TRAINER:
                trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, use_locking=True)

        worker.work(max_episode_length,gamma,sess,coord,saver)

if not TRAINING:
    print([np.mean(episode_lengths), np.sqrt(np.var(episode_lengths)), np.mean(np.asarray(np.asarray(episode_lengths) < max_episode_length, dtype=float))])
