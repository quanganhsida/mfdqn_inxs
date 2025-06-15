import numpy as np
import os

class Env:

    def __init__(self, args):
        # save args
        self.args = args
        # load csi matrix
        path = os.path.join(args.data_dir, f'csi_{args.n_subnetwork}_{args.deploy_length}.npy')
        self.csi = np.load(path)
        # self.csi[:, ...] = self.csi[0, ...]
        # initialize global step
        self.global_step = 0

    def dbm2w(self, dbm):
        watts = 10 ** ((dbm - 30) / 10)
        return watts

    @property
    def rx_noise_power(self):
        # extract args
        args = self.args
        return 10 ** (-174 + args.rx_nf + 10 * np.log10(args.bandwidth / args.n_channel))

    def get_reward(self, action):
        # extract args
        args = self.args
        csi = self.csi
        t = self.global_step
        rx_np = self.rx_noise_power
        Wk = args.bandwidth / args.n_channel
        T = len(csi)
        # extract action
        channel = action[:, 0]
        tx_power = action[:, 1]
        level2dbm = np.linspace(args.tx_power_min, args.tx_power_max, args.n_power_level)
        tx_power_dbm = level2dbm[tx_power]
        tx_power_w = self.dbm2w(tx_power_dbm) # transmit
        rx_power_w = np.zeros_like(tx_power_w) # receive
        int_power_w = np.zeros_like(tx_power_w) # interference
        sinr = np.zeros_like(tx_power_w) # sinr
        capacity = np.zeros_like(tx_power_w) # sinr
        for n in range(args.n_subnetwork):
            # receive power at connected subnetwork
            rx_power_w[n] = tx_power_w[n] * csi[t % T, channel[n], n, n]
            # inteference from other user using same channel
            for i in np.where(channel == channel[n])[0]:
                int_power_w[n] += tx_power_w[i] * csi[t % T, channel[n], i, n]
            # convert to sinr
            sinr[n] = rx_power_w[n] / (int_power_w[n] + rx_np)
            # convert to MB/s
            capacity[n] = Wk * np.log2(1 + sinr[n]) / 8 /1e6
        # reward
        reward = np.min(capacity)
        self.multi_agent_reward = capacity
        # info
        info = {
            'step': t,
            'reward': reward,
            'mean_capacity': np.mean(capacity),
            'min_capacity': np.min(capacity),
            'max_capacity': np.max(capacity),
        }
        return reward, info

    def get_observation(self):
        raise NotImplementedError

    def reset(self):
        self.action = None
        observation = self.get_observation()
        return observation

    def step(self, action):
        # extract args
        args = self.args
        # save action for future
        self.action = action
        # get reward and info
        reward, info = self.get_reward(action)
        # increase global step
        self.global_step += 1
        # get next observation
        next_observation = self.get_observation()
        # get done
        done = False
        return next_observation, reward, done, info
