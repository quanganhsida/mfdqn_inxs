import itertools as it
import numpy as np
import pickle
import torch
import math
import os
import time

import simulator
from .mixer import Mixer

def get_observation(self):
    # extract args
    args   = self.args
    t      = self.global_step
    csi    = self.csi
    rx_np  = self.rx_noise_power
    Wk     = args.bandwidth / args.n_channel
    if self.action is None:
        observation = np.ones([args.n_subnetwork, args.n_channel, args.n_power_level])
        observation = observation.reshape(args.n_subnetwork, -1)
        return observation

    # extract action
    action = self.action
    channel  = action[:, 0]
    tx_power = action[:, 1]

    # compute all tx power (n_subnetwork, n_channel)
    all_tx_power = np.zeros([args.n_subnetwork, args.n_channel], dtype=np.int32)
    for n in range(args.n_subnetwork):
        c = channel[n]
        p = tx_power[n]
        all_tx_power[n, c] = p
    level2dbm = np.linspace(args.tx_power_min, args.tx_power_max, args.n_power_level)
    all_tx_power_dbm = level2dbm[all_tx_power]
    all_tx_power_w = self.dbm2w(all_tx_power_dbm) # transmit
    all_possible_tx_power_w = self.dbm2w(level2dbm)

    # compute all sinr (n_subnetwork, n_channel, n_power_level)
    all_rx_power_w  = np.zeros([args.n_subnetwork, args.n_channel, args.n_power_level], dtype=float)
    all_int_power_w = np.zeros_like(all_tx_power_w) # interference
    all_sinr        = np.zeros_like(all_rx_power_w) # sinr
    #
    for n in range(args.n_subnetwork):
        # receive power at connected subnetwork [n_channel, n_power_level] per agent n
        all_rx_power_w[n, :, :] = csi[max(t-1, 0) % len(csi), :, n, n][:, None] @ all_possible_tx_power_w[None, :]
        # inteference from other user using same channel
        for i in range(args.n_subnetwork):
            all_int_power_w[n, :] += all_tx_power_w[i, :] * csi[max(t-1, 0) % len(csi), :, i, n]
        # convert to sinr
        all_sinr[n, :, :] = all_rx_power_w[n, :, :] / (all_int_power_w[n, :][:, None] + rx_np)
    # reshape to [n_subnetwork, n_channel * n_power_level]

    all_sinr = all_sinr.reshape(args.n_subnetwork, -1)
    all_sinr_db = 10 * np.log10(all_sinr)
    observation = all_sinr_db / np.abs(args.tx_power_min)
    # print(observation.min(), observation.max())
    return observation

def get_mf_action(self):
    # extract args
    args   = self.args
    if self.action is None:
        mf_action = np.full([args.n_subnetwork, args.n_channel * args.n_power_level], 1 / (args.n_channel * args.n_power_level))
        return mf_action
    # extract data
    action      = self.action
    channel     = action[:, 0]
    power_level = action[:, 1]
    csi         = self.csi
    t = self.global_step
    T = len(csi)
    # compute peer2peer interference between subnetworks
    rx_power_w = np.zeros([args.n_subnetwork, args.n_subnetwork])
    tx_power = action[:, 1]
    level2dbm = np.linspace(args.tx_power_min, args.tx_power_max, args.n_power_level)
    tx_power_dbm = level2dbm[tx_power]
    tx_power_w = self.dbm2w(tx_power_dbm) # transmit
    for n, i in it.product(range(args.n_subnetwork), range(args.n_subnetwork)):
        # tx is subnetwork i, rx is subnetwork n
        rx_power_w[n, i] = tx_power_w[n] * csi[t % T, channel[n], i, n]
    # print(rx_power_w.min(), rx_power_w.max())
    # print(rx_power_w.mean(), rx_power_w.std())
    # compute mean field action
    mf_action = np.zeros([args.n_subnetwork, args.n_channel, args.n_power_level])
    for n in range(args.n_subnetwork):
        neighbors = np.where(rx_power_w > args.neighbor_rx_power_threshold)[0]
        mf_action[neighbors, channel[n], power_level[n]] += 1
        mf_action[n, channel[n], power_level[n]] -= 1
    # mf_action /= (args.n_subnetwork - 1)
    mf_action = mf_action.reshape(args.n_subnetwork, -1) / 9
    # print(mf_action.min(), mf_action.max())
    return mf_action

class Solver:

    def __init__(self, args):
        # save args
        self.args = args
        # create env
        self.env = simulator.Env(args)
        # assign get observation function
        self.env.get_observation = get_observation.__get__(self.env)
        self.env.get_mf_action = get_mf_action.__get__(self.env)
        # load monitor
        self.monitor = simulator.Monitor(args)
        # load mixer
        self.mixer = Mixer(self.env, args)

    def test(self):
        # extract args
        args = self.args
        env = self.env
        monitor = self.monitor
        # load model
        self.load()
        # reset environment
        observation = env.reset()
        done = False
        step = 0
        # iteratively run until done
        for step in range(args.n_test_step):
            tic = time.time()
            # select random action
            action = self.mixer.select_action(observation)
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

    def get_eps(self, t):
        # extract args
        args = self.args
        # calculate exploration prob
        eps = args.dqn_eps_end + (args.dqn_eps_start - args.dqn_eps_end) * \
                math.exp(-1. * t / args.dqn_eps_decay)
        return eps

    def train(self):
        # extract args
        args    = self.args
        env     = self.env
        monitor = self.monitor
        # reset environment
        observation = env.reset()
        mf_action   = env.get_mf_action()
        done        = False
        step        = 0
        #
        # iteratively run environment until done
        try:
            for step in range(args.n_train_step):
                # select greedy action
                action = self.mixer.select_action(observation)
                # send to env
                next_observation, reward, done, info = env.step(action)
                next_mf_action = env.get_mf_action()
                # store transition in memory
                self.mixer.store_transition(observation, mf_action, action, next_observation, next_mf_action, reward)
                # optimize model
                loss         = self.mixer.optimize()
                info['loss'] = loss
                info['eps']  = self.get_eps(step)
                # soft target update
                self.mixer.soft_target_update()
                # logging
                monitor.step(info)
                # update observation
                observation = next_observation
                mf_action   = next_mf_action
                step       += 1
        except KeyboardInterrupt:
            pass
        finally:
            # export csv
            monitor.export_csv()
            # save model
            self.save()
        # export csv
        monitor.export_csv()
        # save model
        self.save()

    def save(self):
        args = self.args
        state_dict = self.mixer.state_dict()
        path = os.path.join(args.model_dir, f'{self.monitor.label}.pkl')
        with open(path, 'wb') as fp:
            pickle.dump(state_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        args = self.args
        path = os.path.join(args.model_dir, f'{self.monitor.label}.pkl')
        if os.path.exists(path):
            print(f'[+] loading model from {path}')
            with open(path, 'rb') as fp:
                state_dict = pickle.load(fp)
            self.mixer.load_state_dict(state_dict)
