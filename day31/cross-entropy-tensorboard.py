#!/usr/bin/env python


# use when 
#   - short episodes
#   - frequent reward system

# it is
#   - model free
#   - policy based
#   - on policy

import gym
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

EPISODE = namedtuple('Episode', ['reward', 'steps'])
EPISODESTEP = namedtuple('EpisodeStep', field_names=['observation', 'action'])

class MyNet(nn.Module):
    
    def __init__(self, obs_dim, actions):
        super(MyNet, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(in_features = obs_dim, out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = actions)
        )

    def forward(self, x):
        return self.pipe(x)

def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EPISODESTEP(observation=obs, action=action))
        if is_done:
            batch.append(EPISODE(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs

def filter_batch(batch, percent):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percent)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))
    
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = gym.wrappers.Monitor(env, directory="cross_entropy", force=True)
    # observation size
    obs_size = env.observation_space.shape[0]
    # possible action types
    n_actions = env.action_space.n
    # initial observation
    observation = env.reset()

    writer = SummaryWriter(comment="-cartpole")

    net = MyNet(obs_size, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)

    for iter_no, batch in enumerate(iterate_batches(env, net, 32)):
        print 'Iterating {} time with a size of {}'.format(iter_no, len(batch))
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, 70) 
        optimizer.zero_grad()
        # training model with just cream of data
        action_scores_v = net(obs_v)
        # calculating loss
        loss_v = objective(action_scores_v, acts_v)
        # back propogating the error to the intermediate nodes
        loss_v.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        
        if reward_m > 199:
            print("Solved!")
            break
    writer.close()