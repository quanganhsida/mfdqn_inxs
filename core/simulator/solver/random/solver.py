import numpy as np
import time

import simulator

def get_observation(self):
    return None

class Solver:

    def __init__(self, args):
        # save args
        self.args = args
        # create env
        self.env = simulator.Env(args)
        # assign get observation function
        self.env.get_observation = get_observation.__get__(self.env)
        # load monitor
        self.monitor = simulator.Monitor(args)
        #
        self.prev_action = np.stack([np.random.randint(low=0, high=args.n_channel, size=[args.n_subnetwork]),
                                     np.full([args.n_subnetwork], args.n_power_level - 1)], axis=-1)

    def select_action(self, observation):
        # extract args
        args = self.args
        # action (n_agent=n_subnetwork, channel_id, power_level)
        action = np.stack([
                    np.random.randint(low=0, high=args.n_channel, size=[args.n_subnetwork]),
                    np.full([args.n_subnetwork], args.n_power_level - 1)
        ], axis=-1)
        # fuse elements from action to prev_action
        indices = np.where(np.random.rand(*action.shape) < args.p_new_action)
        self.prev_action[indices] = action[indices]
        return self.prev_action

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
            # update observation
            if step % args.n_step == 0:
                # reset
                observation = env.reset()
            else:
                # update observation
                observation = next_observation
            step += 1
            # logging to cache
            toc = time.time()
            info['time'] = toc - tic
            monitor.step(info)
        # export cache to csv
        monitor.export_csv()

    def train(self):
        raise NotImplementedError
