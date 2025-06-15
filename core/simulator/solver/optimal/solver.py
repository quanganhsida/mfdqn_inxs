import numpy as np
import time

import simulator

def get_observation(self):
    # extract args
    args   = self.args
    t      = self.global_step
    csi    = self.csi
    rx_np  = self.rx_noise_power
    Wk     = args.bandwidth / args.n_channel
    if self.action is None:
        observation = np.ones([args.n_subnetwork, args.n_channel])
        return observation
    action = self.action
    # extract action
    channel  = action[:, 0]
    tx_power = action[:, 1]
    # build full tx power
    all_tx_power = np.zeros([args.n_subnetwork, args.n_channel], dtype=np.int32)
    for n in range(args.n_subnetwork):
        c = channel[n]
        p = tx_power[n]
        all_tx_power[n, c] = p
    level2dbm = np.linspace(args.tx_power_min, args.tx_power_max, args.n_power_level)
    all_tx_power_dbm = level2dbm[all_tx_power]
    all_tx_power_w = self.dbm2w(all_tx_power_dbm) # transmit
    all_rx_power_w = np.zeros_like(all_tx_power_w) # receive
    all_int_power_w = np.zeros_like(all_tx_power_w) # interference
    all_sinr = np.zeros_like(all_tx_power_w) # sinr
    for n in range(args.n_subnetwork):
        # receive power at connected subnetwork
        all_rx_power_w[n, :] = all_tx_power_w[n, :] * csi[t % len(csi), :, n, n]
        # inteference from other user using same channel
        for i in range(args.n_subnetwork):
            all_int_power_w[n, :] += all_tx_power_w[i, :] * csi[t % len(csi), :, i, n]
        # convert to sinr
        all_sinr[n, :] = all_rx_power_w[n, :] / (all_int_power_w[n, :] + rx_np)
    # convert to quantized state
    observation = all_sinr
    return observation

class Solver:

    def __init__(self, args):
        # save args
        self.args = args
        # create env
        self.env = simulator.Env(args)
        self.env.get_observation = get_observation.__get__(self.env)
        # load monitor
        self.monitor = simulator.Monitor(args)

    def select_action(self, observation):
        # extract args
        args = self.args
        # action (n_agent=n_subnetwork, channel_id, power_level)
        channel = np.argmax(observation, axis=1)
        power_level = np.full([args.n_subnetwork], args.n_power_level - 1)
        action = np.stack([channel, power_level,], axis=-1)
        return action

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
            toc = time.time()
            info['time'] = toc - tic
            # logging to cache
            monitor.step(info)
        # export cache to csv
        monitor.export_csv()

    def train(self):
        raise NotImplementedError
