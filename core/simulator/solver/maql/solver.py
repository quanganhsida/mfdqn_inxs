import numpy as np
import time
import simulator

from ..q_heuristic import get_observation
from ..q_heuristic import Solver as QHeuristicSolver


class Solver(QHeuristicSolver):

    def __init__(self, args):
        #
        super().__init__(args)
        # initialize Q-table
        self.q_table = np.zeros([args.n_subnetwork, 2 ** args.n_channel, args.n_channel, args.n_power_level])
        # initialize exploration rate
        self.eps = args.eps_max
        #
        self.prev_action = np.stack([np.random.randint(low=0, high=args.n_channel, size=[args.n_subnetwork]),
                                     np.full([args.n_subnetwork], args.n_power_level - 1)], axis=-1)

    def binary_observation_to_int(self, observation):
        string = ''.join(map(str, observation.astype(int)))
        index = int(string, 2)
        return index

    def select_action(self, observation):
        # extract args
        args = self.args
        # initialize action at random
        action = np.stack([
                    np.random.randint(low=0, high=args.n_channel, size=[args.n_subnetwork]),
                    np.full([args.n_subnetwork], args.n_power_level - 1)
        ], axis=-1)
        # check which action exploiting q-table
        r = np.random.rand(args.n_subnetwork)
        indices = np.where(r > self.eps)[0]
        for n in indices:
            int_observation = self.binary_observation_to_int(observation[n])
            q_values = self.q_table[n, int_observation]
            top_channels, top_power_levels = np.where(q_values == q_values.max())
            i = np.random.randint(len(top_channels))
            action[n, 0] = top_channels[i]
            action[n, 1] = top_power_levels[i]
        # fuse elements from action to prev_action
        indices = np.where(np.random.rand(*action.shape) < args.p_new_action)
        self.prev_action[indices] = action[indices]
        return self.prev_action

    def update(self, observation, action, next_observation, reward):
        # extract args
        args  = self.args
        alpha = args.learning_rate
        gamma = args.gamma
        env   = self.env
        reward = env.multi_agent_reward
        # decay exploration rate
        self.eps = max(args.eps_min, self.eps - (args.eps_max - args.eps_min) / args.eps_step)
        # update q_tables
        for n in range(args.n_subnetwork):
            int_observation      = self.binary_observation_to_int(observation[n])
            int_next_observation = self.binary_observation_to_int(observation[n])
            channel              = action[n, 0]
            power_level          = action[n, 1]
            self.q_table[n, int_observation, channel, power_level] = \
                    (1 - alpha) * self.q_table[n, int_observation, channel, power_level] + \
                    alpha * (reward[n] + gamma * np.max(self.q_table[n, int_next_observation]))

    def add_info(self, info):
        info['eps'] = self.eps

    def test(self):
        # extract args
        args = self.args
        env = self.env
        monitor = self.monitor
        # reset environment
        observation = env.reset()
        done = False
        step = 0
        # iteratively run until done
        for step in range(args.n_test_step):
            tic = time.time()
            # select random action
            action = self.select_action(observation)
            # send to env
            next_observation, reward, done, info = env.step(action)
            # update solver
            self.update(observation, action, next_observation, reward)
            # update observation
            if step % args.n_step == 0:
                # reset
                observation = env.reset()
            else:
                # update observation
                observation = next_observation
            step += 1
            toc = time.time()
            info['time'] = toc - tic
            # add info
            self.add_info(info)
            # logging to cache
            monitor.step(info)
        # export cache to csv
        monitor.export_csv()
