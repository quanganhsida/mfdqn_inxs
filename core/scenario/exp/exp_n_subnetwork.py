import itertools as it
import tqdm
import os

def exp_n_subnetwork(args):
    cmds = [
        # Gen data
	    'python3 main.py --scenario=prepare_data --n_snapshot=5000',
        # Plot topo
		'python3 main.py --scenario=plot_network',
        # Train
        'python3 main.py --scenario=main --mode=train --n_train_step=5000 --solver=mfdqn',
        'python3 main.py --scenario=main --mode=train --n_train_step=5000 --solver=idqn',
        'python3 main.py --scenario=main --mode=train --n_train_step=5000 --solver=vdn',
        # Plot train
        'python3 main.py --scenario=plot_metric --mode=train --n_smooth=250 --metric=reward',
        'python3 main.py --scenario=plot_metric --mode=train --n_smooth=250 --metric=loss',
        # 'python3 main.py --scenario=plot_metric --mode=train --n_smooth=250 --metric=eps',
        # Test
		'python3 main.py --scenario=main --n_test_step=5000 --solver=random',
		'python3 main.py --scenario=main --n_test_step=5000 --solver=q_heuristic',
		'python3 main.py --scenario=main --n_test_step=5000 --solver=greedy',
		'python3 main.py --scenario=main --n_test_step=5000 --solver=maql',
		'python3 main.py --scenario=main --n_test_step=5000 --solver=idqn',
		'python3 main.py --scenario=main --n_test_step=5000 --solver=mfdqn',
		'python3 main.py --scenario=main --n_test_step=5000 --solver=vdn',
        # Plot test
        'python3 main.py --scenario=plot_metric --n_smooth=250 --metric=reward',
        # 'python3 main.py --scenario=plot_metric --n_smooth=250 --metric=mean_capacity',
        # 'python3 main.py --scenario=plot_metric --n_smooth=250 --metric=min_capacity',
    ]
    N = [80]
    for n, cmd in tqdm.tqdm(it.product(N, cmds)):
        cmd_ = f'{cmd} --deploy_length={n} --n_subnetwork={n}'
        print(cmd_)
        os.system(cmd_)
