from .replay_memory import ReplayMemory, Transition
from .model import DQN

import torch.optim as optim
import torch.nn as nn
import numpy as np
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
            channel = torch.argmax(observation[:args.n_channel * args.n_power_level].reshape(args.n_channel, args.n_power_level)[:, 0])
            power_level = torch.tensor(args.n_power_level - 1)
            action = channel * args.n_power_level + power_level
        else: # exploitation
            with torch.no_grad():
                action = self.policy_net(observation).max(0).indices
        # increase step counter
        self.t += 1
        return action

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
