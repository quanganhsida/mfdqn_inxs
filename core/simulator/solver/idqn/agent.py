from .replay_memory import ReplayMemory, Transition
from .model import DQN

import torch.optim as optim
import torch.nn as nn
import random
import torch
import math

class Agent:

    def __init__(self, args):
        # save args
        self.args = args
        # initialize model
        self.policy_net = DQN(args)
        self.target_net = DQN(args)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # initialize other components
        self.optimizer = optim.AdamW(self.policy_net.parameters(),
                                     lr=args.dqn_lr,
                                     amsgrad=True)
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayMemory(args.dqn_memory_size)
        self.t = 0

    def select_action(self, observation):
        # extract args
        args = self.args
        # assert that observation.shape = (n_observation, )
        assert len(observation.shape) == 1
        # calculate exploration prob
        if args.mode == 'train':
            eps = args.dqn_eps_end + (args.dqn_eps_start - args.dqn_eps_end) * \
                    math.exp(-1. * self.t / args.dqn_eps_decay)
        elif args.mode == 'test':
            eps = args.dqn_eps_end
        else:
            raise NotImplementedError
        # inference
        if random.random() < eps: # exploration
            channel = torch.argmax(observation.reshape(args.n_channel, args.n_power_level)[:, 0])
            power_level = torch.tensor(args.n_power_level - 1)
            action = channel * args.n_power_level + power_level
        else: # exploitation
            with torch.no_grad():
                action = self.policy_net(observation).max(0).indices
        # increase step counter
        self.t += 1
        return action

    def store_transition(self, observation, action, next_observation, reward):
        self.memory.push(observation, action, next_observation, reward)

    def optimize(self):
        # extract args
        args       = self.args
        memory     = self.memory
        criterion  = self.criterion
        optimizer  = self.optimizer
        policy_net = self.policy_net
        target_net = self.target_net
        # check if enough data has been stored in memory yet
        if len(self.memory) < args.dqn_batch_size:
            return 0
        # sample transitions
        transitions        = memory.sample(args.dqn_batch_size)
        batch              = Transition(*zip(*transitions))
        b_observation      = torch.tensor(batch.observation, dtype=torch.float)
        b_action           = torch.tensor(batch.action, dtype=torch.int64)
        b_next_observation = torch.tensor(batch.next_observation, dtype=torch.float)
        b_reward           = torch.tensor(batch.reward)
        # compute loss
        q_values           = policy_net(b_observation).gather(1, b_action[..., None]).squeeze()
        next_q_values      = target_net(b_next_observation).max(1).values
        expected_q_values  = b_reward + args.dqn_gamma * next_q_values
        loss               = criterion(q_values, expected_q_values)
        # optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
        return loss.item()

    def soft_target_update(self):
        # extract args
        args       = self.args
        policy_net = self.policy_net
        target_net = self.target_net
        # update
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * args.dqn_tau + \
                                         target_net_state_dict[key] * (1 - args.dqn_tau)
        target_net.load_state_dict(target_net_state_dict)
