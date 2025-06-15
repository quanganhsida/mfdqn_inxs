import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def smooth_line(x, y, args):
    T = int(np.floor(len(x) // args.n_smooth) * args.n_smooth)
    x = x[:T].reshape(-1, args.n_smooth)
    x = x[:, 0]
    y = y[:T].reshape(-1, args.n_smooth)
    y = np.mean(y, axis=1)
    return x, y

def plot_boxplot(args):
    # define solvers
    if args.mode == 'train':
        # solvers = ['mfdqn']
        solvers = ['idqn', 'MFDQN', 'vdn']
    elif args.mode == 'test':
        # solvers = ['random', 'greedy', 'optimal', 'q_heuristic', 'maql', 'idqn', 'mfdqn']
        solvers = ['random', 'greedy', 'q_heuristic', 'maql', 'idqn', 'MFDQN', 'vdn']
    else:
        raise NotImplementedError
    # load csv
    labels = []
    values = []
    for solver in solvers:
        path = os.path.join(args.csv_dir, args.mode, f'{solver}_{args.n_subnetwork}_{args.deploy_length}.csv')
        df = pd.read_csv(path)
        # x = df['step'].to_numpy()
        y = df[args.metric].to_numpy()
        labels.append(solver)
        values.append(y)
    # decorate
    plt.boxplot(values, labels=labels, showfliers=False)
    plt.xlabel('method')
    plt.ylabel(f'{args.metric}')
    # plt.legend()
    plt.tight_layout()
    path = os.path.join(args.figure_dir, f'{args.scenario}_{args.n_subnetwork}_{args.deploy_length}_{args.mode}_{args.metric}.pdf')
    plt.savefig(path)
