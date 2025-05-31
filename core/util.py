import argparse
import torch
import os

def create_folders(args):
    ls = [args.figure_dir, args.data_dir, args.csv_dir, args.model_dir]
    for folder in ls:
        if not os.path.exists(folder):
            os.makedirs(folder)

def set_default_device(args):
    torch.set_default_device(args.device)

def get_args():
    # create args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='main')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    # simulation scenario
    parser.add_argument('--n_test_step', type=int, default=10000)
    parser.add_argument('--n_train_step', type=int, default=10000)
    parser.add_argument('--n_step', type=int, default=1000)
    # network scenario
    parser.add_argument('--n_subnetwork', type=int, default=20)
    parser.add_argument('--n_channel', type=int, default=4)
    parser.add_argument('--n_power_level', type=int, default=3)
    parser.add_argument('--tx_power_min', type=float, default=-20) # dBm
    parser.add_argument('--tx_power_max', type=float, default=-10) # dBm
    parser.add_argument('--rx_nf', type=float, default=5) # dB
    parser.add_argument('--bandwidth', type=float, default=40e6) # Hz
    parser.add_argument('--n_snapshot', type=int, default=1000)
    parser.add_argument('--deploy_length', type=float, default=20)   # m
    parser.add_argument('--subnetwork_radius', type=float, default=1) # m
    # solver
    parser.add_argument('--solver', type=str, default='random')
    parser.add_argument('--q_heuristic_sinr_threshold', type=float, default=0.7)
    # maql solver
    parser.add_argument('--eps_min', type=float, default=0.01)
    parser.add_argument('--eps_max', type=float, default=0.99)
    parser.add_argument('--eps_step', type=float, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--p_new_action', type=float, default=1.0)
    # idqn solver
    parser.add_argument('--dqn_n_hidden', type=int, default=128)
    parser.add_argument('--dqn_n_layer', type=int, default=2)
    parser.add_argument('--dqn_lr', type=int, default=3e-5)
    parser.add_argument('--dqn_memory_size', type=int, default=100000)
    parser.add_argument('--dqn_gamma', type=float, default=0.9)
    parser.add_argument('--dqn_eps_start', type=float, default=0.9)
    parser.add_argument('--dqn_eps_end', type=float, default=0.00)
    parser.add_argument('--dqn_eps_decay', type=float, default=300)
    parser.add_argument('--dqn_batch_size', type=int, default=500)
    parser.add_argument('--dqn_tau', type=float, default=0.001)
    parser.add_argument('--dqn_boltzman_tau_start', type=float, default=200)
    parser.add_argument('--dqn_boltzman_tau_end', type=float, default=0.01)
    # data directory
    parser.add_argument('--figure_dir', type=str, default='../data/figure')
    parser.add_argument('--model_dir', type=str, default='../data/model')
    parser.add_argument('--data_dir', type=str, default='../data/data')
    parser.add_argument('--csv_dir', type=str, default='../data/csv')
    # cuda
    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, default='cuda:0')
    else:
        parser.add_argument('--device', type=str, default='cpu')
    # plot
    parser.add_argument('--metric', type=str, default='reward')
    parser.add_argument('--n_smooth', type=int, default=150)
    # parse args
    args = parser.parse_args()
    # create folders
    create_folders(args)
    # set default device cuda
    set_default_device(args)
    # additional args
    args.n_action = args.n_channel * args.n_power_level
    return args
