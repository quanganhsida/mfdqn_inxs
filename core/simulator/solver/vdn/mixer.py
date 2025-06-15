import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import math

from .replay_memory import ReplayMemory, Transition
from .agent import Agent

class Mixer(nn.Module):

    def __init__(self, env, args):
        super().__init__()
        # save args
        self.args = args
        self.env  = env
        # initialize agents
        self.agents = [Agent(args) for _ in range(args.n_subnetwork)]
        for i, agent in enumerate(self.agents):
            setattr(self, f'policy_net_{i}', agent.policy_net)
        # initialize other components
        self.optimizer = optim.AdamW(self.parameters(),
                                     lr=args.dqn_lr,
                                     amsgrad=True)
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayMemory(args.dqn_memory_size)
        self.t = 0
        #
        self.prev_action = np.stack([np.random.randint(low=0, high=args.n_channel, size=[args.n_subnetwork]),
                                     np.full([args.n_subnetwork], args.n_power_level - 1)], axis=-1)

    def select_action(self, observation):
        # extract args
        args = self.args
        env  = self.env
        # select random action
        action = []
        observation = torch.tensor(observation).to(dtype=torch.float)
        # select action
        action = []
        for i in range(args.n_subnetwork):
            a = self.agents[i].select_action(observation[i])
            action.append(a)
        action = torch.stack(action).squeeze().detach().cpu().numpy()
        channel = action // args.n_power_level
        power_level = action % args.n_power_level
        action = np.stack([channel, power_level], axis=-1)
        # fuse elements from action to prev_action
        indices = np.where(np.random.rand(*action.shape) < args.p_new_action)
        self.prev_action[indices] = action[indices]
        return self.prev_action

    def store_transition(self, observation, action, next_observation, reward):
        args = self.args
        int_action = action[:, 0] * args.n_power_level + action[:, 1]
        min_reward = self.env.multi_agent_reward.min()
        multi_agent_reward = self.env.multi_agent_reward
        reward = 0.5 * (min_reward + multi_agent_reward)
        self.memory.push(observation, int_action,
                         next_observation, reward)

    def soft_target_update(self):
        for agent in self.agents:
            agent.soft_target_update()

    def optimize(self):
        # extract args
        args      = self.args
        agents    = self.agents
        memory    = self.memory
        criterion = self.criterion
        optimizer = self.optimizer
        # check if enough data has been stored in memory yet
        if len(self.memory) < args.dqn_batch_size:
            return 0
        # sample transitions
        transitions           = memory.sample(args.dqn_batch_size)
        batch                 = Transition(*zip(*transitions))
        b_observation         = torch.tensor(batch.observation, dtype=torch.float)        # bs, n_agent, n_observation
        b_action              = torch.tensor(batch.action, dtype=torch.int64)             # bs, n_agent
        b_next_observation    = torch.tensor(batch.next_observation, dtype=torch.float)   # bs, n_agent, n_observation
        b_reward              = torch.tensor(batch.reward)                                # bs, n_agent
        # compute loss
        all_q_values      = torch.zeros_like(b_reward)
        all_next_q_values = torch.zeros_like(b_reward)
        for k, agent in enumerate(agents):
            q_values                = agent.policy_net(b_observation[:, k, :]).gather(1, b_action[:, k][..., None]).squeeze()
            next_q_values           = agent.target_net(b_next_observation[:, k, :]).max(1).values
            all_q_values[:, k]      = q_values
            all_next_q_values[:, k] = next_q_values
        q_values = all_q_values.sum(dim=1)
        reward = b_reward.sum(dim=1)
        next_q_values = all_q_values.sum(dim=1)
        expected_q_values = reward + args.dqn_gamma * next_q_values
        loss              = criterion(q_values, expected_q_values)
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
