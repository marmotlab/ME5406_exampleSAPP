#!/usr/bin/env python
from __future__ import division

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy.signal as signal
import os
import pickle
import imageio

import sapp_gym
from ACNet import ACNet

dev_list = torch.cuda.is_available()
print(dev_list)


def make_gif(images, fname, duration=2, true_image=False, salience=False, salIMGS=None):
    imageio.mimwrite(fname,images,subrectangles=True)
    print("wrote gif")


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Worker:
    def __init__(self, game, global_network, trainer, device):
        self.env = game
        self.global_network = global_network
        self.trainer = trainer
        self.device = device

        self.nextGIF = episode_count  # For GIFs output

    def train(self, rollout, gamma, bootstrap_value):
        global episode_count

        rollout_size = len(rollout)
        # obs, goal, a, r, train_valid, rnn_state0[0], rnn_state0[1]]
        # [128, 3, 11, 11]
        observ = torch.stack([rollout[i][0] for i in range(rollout_size)]).squeeze(1)
        # [128, 3]
        goal_vec = torch.stack([rollout[i][1] for i in range(rollout_size)]).squeeze(1)
        # [128, 1]
        actions = torch.stack([rollout[i][2] for i in range(rollout_size)])
        # [128, 1]
        rewards = torch.stack([rollout[i][3] for i in range(rollout_size)])
        # [128, 5]
        valids = torch.stack([rollout[i][5] for i in range(rollout_size)])
        # [128, 1, 128]
        h_n = torch.stack([rollout[i][6] for i in range(rollout_size)]).squeeze(1).permute(1, 0, 2)
        # [128, 1, 128]
        h_c = torch.stack([rollout[i][7] for i in range(rollout_size)]).squeeze(1).permute(1, 0, 2)

        # Forward view
        policies, values, policies_sig, _ = self.global_network(observ, goal_vec, (h_n, h_c))

        # Backward view
        # Calculate target values for value loss
        # [129, 1]
        rewards_plus = torch.cat((rewards, bootstrap_value), 0).detach().cpu().numpy()
        discounted_rewards = discount(rewards_plus, gamma)[:-1]
        discounted_rewards = torch.FloatTensor(discounted_rewards.copy()).to(self.device)

        # Calculate advantage values for policy loss
        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"

        # [129, 1]
        value_plus = torch.cat((values, bootstrap_value), 0).detach().cpu().numpy()
        advantages = rewards.cpu().numpy() + gamma * value_plus[1:] - value_plus[:-1]
        advantages = discount(advantages, gamma)
        advantages = torch.FloatTensor(advantages.copy()).to(self.device)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        responsible_outputs = torch.gather(policies, 1, actions)

        v_l = 0.5 * torch.sum(torch.square(discounted_rewards.detach() - values))
        e_l = - 0.01 * torch.sum(policies * torch.log(torch.clip(policies, 1e-10, 1.0)))
        p_l = - torch.sum(torch.log(torch.clip(responsible_outputs, 1e-15, 1.0)) * advantages.detach())
        valid_l = - 0.5 * torch.sum(torch.log(torch.clip(policies_sig, 1e-10, 1.0)) * valids + torch.log(torch.clip(1 - policies_sig, 1e-10, 1.0)) * (1 - valids))
        loss = v_l + p_l - e_l + valid_l
        self.trainer.zero_grad()
        loss.backward()
        g_n = torch.nn.utils.clip_grad_norm_(self.global_network.parameters(), 1000)
        self.trainer.step()
        return v_l, p_l, valid_l, e_l, g_n

    def work(self, max_episode_length, gamma):
        global episode_count, episode_rewards, episode_lengths, episode_mean_values, episode_invalid_ops
        total_steps, i_buf = 0, 0

        while True:
            episode_buffer, episode_values = [], []
            episode_reward = episode_step_count = episode_inv_count = 0
            d = False
            # Initial state from the environment
            s, validActions = self.env.reset()
            h = torch.zeros(1, 1, 128).to(self.device)
            c = torch.zeros(1, 1, 128).to(self.device)
            rnn_state0 = (h, c)
            rnn_state = rnn_state0

            saveGIF = False
            if OUTPUT_GIFS and ((not TRAINING) or (episode_count >= self.nextGIF)):
                saveGIF = True
                self.nextGIF = episode_count + 256
                GIF_episode = int(episode_count)
                episode_frames = [self.env._render(mode='rgb_array', screen_height=900, screen_width=900)]

            while True:  # Give me something!
                # Take an action using probabilities from policy network output.
                obs = torch.FloatTensor(s[0]).unsqueeze(0).to(self.device)
                goal = torch.FloatTensor(s[1]).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    a_dist, v, _, rnn_state = self.global_network(obs, goal, rnn_state)
                if not (torch.argmax(a_dist.flatten()) in validActions):
                    episode_inv_count += 1

                train_valid = np.zeros(a_size)
                train_valid[validActions] = 1

                valid_dist = a_dist[0, validActions]
                valid_dist /= torch.sum(valid_dist)

                if TRAINING:
                    #                        if(not (np.argmax(a_dist.flatten()) in validActions)):
                    #                            episode_inv_count += 1
                    #                            a     = validActions[ np.random.choice(range(valid_dist.shape[1])) ]
                    #                        else:
                    a = validActions[torch.multinomial(valid_dist, 1)]
                else:
                    a = torch.argmax(valid_dist)
                    if a not in validActions or not GREEDY:
                        a = validActions[torch.multinomial(valid_dist, 1)]

                s1, r, d, validActions = self.env.step(a)

                if saveGIF:
                    episode_frames.append(self.env._render(mode='rgb_array', screen_width=900, screen_height=900))

                r = torch.FloatTensor([r]).to(self.device)
                a = torch.tensor([a], dtype=torch.int64).to(self.device)
                train_valid = torch.FloatTensor(train_valid).to(self.device)
                episode_buffer.append([obs, goal, a, r, v[0, 0], train_valid, rnn_state0[0], rnn_state0[1]])
                episode_values.append(v[0, 0].cpu().numpy())
                episode_reward += r.cpu().numpy()
                s = s1
                total_steps += 1
                episode_step_count += 1
                rnn_state0 = rnn_state

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
                        s1Value = torch.FloatTensor([[0]]).to(self.device)
                    else:
                        obs = torch.FloatTensor(s[0]).unsqueeze(0).to(self.device)
                        goal = torch.FloatTensor(s[1]).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            _, s1Value, _, _ = self.global_network(obs, goal, rnn_state)

                    v_l, p_l, valid_l, e_l, g_n = self.train(episode_buffer_training, gamma, s1Value)
                    rnn_state0 = rnn_state

                if episode_step_count >= max_episode_length or d:
                    break

            episode_lengths.append(episode_step_count)
            episode_mean_values.append(np.nanmean(episode_values))
            episode_invalid_ops.append(episode_inv_count)
            episode_rewards.append(episode_reward)

            if not TRAINING:
                episode_count += 1
                print('({}) Thread 0: {} steps, {:.2f} reward ({} invalids).'.format(episode_count, episode_step_count,episode_reward, episode_inv_count))
                GIF_episode = int(episode_count)
            else:
                episode_count += 1

                if episode_count % SUMMARY_WINDOW == 0:
                    if episode_count % 100 == 0:
                        print('Saving model', end='\n')
                        checkpoint = {"model": self.global_network.state_dict(),
                                      "optimizer": self.trainer.state_dict(),
                                      "episode": episode_count}
                        path_checkpoint = "./" + model_path + "/checkpoint.pth"
                        torch.save(checkpoint, path_checkpoint)
                        print('Saved model', end='\n')
                    mean_reward = np.nanmean(episode_rewards[-SUMMARY_WINDOW:])
                    mean_length = np.nanmean(episode_lengths[-SUMMARY_WINDOW:])
                    mean_value = np.nanmean(episode_mean_values[-SUMMARY_WINDOW:])
                    mean_invalid = np.nanmean(episode_invalid_ops[-SUMMARY_WINDOW:])

                    writer.add_scalar(tag='Perf/Reward', scalar_value=mean_reward, global_step=episode_count)
                    writer.add_scalar(tag='Perf/Length', scalar_value=mean_length, global_step=episode_count)
                    writer.add_scalar(tag='Perf/Value', scalar_value=mean_value, global_step=episode_count)
                    writer.add_scalar(tag='Perf/Valid Rate', scalar_value=(mean_length - mean_invalid) / mean_length, global_step=episode_count)

                    writer.add_scalar(tag='Losses/Value Loss', scalar_value=v_l, global_step=episode_count)
                    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=p_l, global_step=episode_count)
                    writer.add_scalar(tag='Losses/Valid Loss', scalar_value=valid_l, global_step=episode_count)
                    writer.add_scalar(tag='Losses/Grad Norm', scalar_value=g_n, global_step=episode_count)

                    if printQ:
                        print('{} Tensorboard updated'.format(episode_count), end='\r')

            if saveGIF:
                # Dump episode frames for external gif generation (otherwise, makes the jupyter kernel crash)
                time_per_step = 0.1
                images = np.array(episode_frames)
                if TRAINING:
                    make_gif(images,'{}/episode_{:d}_{:d}_{:.1f}.gif'.format(gifs_path, GIF_episode, episode_step_count, float(episode_reward)))
                else:
                    make_gif(images, '{}/episode_{:d}_{:d}.gif'.format(gifs_path, GIF_episode, episode_step_count), duration=len(images) * time_per_step, true_image=True, salience=False)
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

# with torch.device("cpu"): # uncomment to run on GPU, and comment next line
device = torch.device('cuda')

master_network = ACNet(a_size, NUM_CHANNEL=3, GRID_SIZE=GRID_SIZE).to(device) # Generate global network
optimizer = optim.Adam(master_network.parameters(), lr=LR_Q)

gameEnv = sapp_gym.SAPPEnv(DIAGONAL_MOVEMENT=DIAG_MVMT, SIZE=ENVIRONMENT_SIZE,
                           observation_size=GRID_SIZE, PROB=OBSTACLE_DENSITY)
writer = SummaryWriter(train_path)

if load_model:
    print('Loading Model...')
    checkpoint = torch.load(model_path + '/checkpoint.pth')
    master_network.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    curr_episode = checkpoint['episode']
    print("episode_count set to ", curr_episode)
    if RESET_TRAINER:
        optimizer = optim.Adam(master_network.parameters(), lr=LR_Q)

worker = Worker(gameEnv, master_network, optimizer, device)
worker.work(max_episode_length, gamma)


if not TRAINING:
    print([np.mean(episode_lengths), np.sqrt(np.var(episode_lengths)), np.mean(np.asarray(episode_lengths < max_episode_length, dtype=float))])

