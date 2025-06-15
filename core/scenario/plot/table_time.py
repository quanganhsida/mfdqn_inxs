import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def table_time(args):
    # define solvers
    solvers = ['random', 'greedy', 'q_heuristic', 'maql', 'idqn', 'MFDQN', 'vdn']
    args.mode = 'test'
    # load csv
    for solver in solvers:
        Y = []
        for n in [10, 20, 40, 80]:
            path = os.path.join(args.csv_dir, args.mode, f'{solver}_{n}_{n:0.1f}.csv')
            df = pd.read_csv(path)
            y = df['time'].to_numpy() / n * 1000
            Y.append(y)
        Y = np.array(Y)
        print(f'{solver} {Y.mean():.2e} {Y.std():.2e}')
