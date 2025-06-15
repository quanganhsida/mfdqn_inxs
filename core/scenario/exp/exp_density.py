import itertools as it
import tqdm
import os

def exp_density(args):
    cmds = [
        # Gen data
	    # 'python3 main.py --scenario=prepare_data --n_snapshot=5000 --n_subnetwork=20',
        # Plot topo
		# 'python3 main.py --scenario=plot_network --n_subnetwork=20',
        # Train
        # 'python3 main.py --scenario=main --mode=train --n_train_step=5000 --solver=mfdqn --n_subnetwork=20',
        # 'python3 main.py --scenario=main --mode=train --n_train_step=5000 --solver=idqn --n_subnetwork=20',
        # 'python3 main.py --scenario=main --mode=train --n_train_step=5000 --solver=vdn --n_subnetwork=20',
        # Plot train
        # 'python3 main.py --scenario=plot_metric --mode=train --n_smooth=250 --n_subnetwork=20 --metric=reward',
        # 'python3 main.py --scenario=plot_metric --mode=train --n_smooth=250 --n_subnetwork=20 --metric=loss',
        # 'python3 main.py --scenario=plot_metric --mode=train --n_smooth=250 --n_subnetwork=20 --metric=eps',
        # Test
		# 'python3 main.py --scenario=main --n_subnetwork=20 --n_test_step=5000 --solver=random',
		# 'python3 main.py --scenario=main --n_subnetwork=20 --n_test_step=5000 --solver=q_heuristic',
		# 'python3 main.py --scenario=main --n_subnetwork=20 --n_test_step=5000 --solver=greedy',
		# 'python3 main.py --scenario=main --n_subnetwork=20 --n_test_step=5000 --solver=maql',
		# 'python3 main.py --scenario=main --n_subnetwork=20 --n_test_step=5000 --solver=idqn',
		# 'python3 main.py --scenario=main --n_subnetwork=20 --n_test_step=5000 --solver=mfdqn',
		# 'python3 main.py --scenario=main --n_subnetwork=20 --n_test_step=5000 --solver=vdn',
        # Plot test
        # 'python3 main.py --scenario=plot_metric --n_smooth=250 --n_subnetwork=20 --metric=reward',
        # 'python3 main.py --scenario=plot_metric --n_smooth=250 --n_subnetwork=20 --metric=mean_capacity',
        # 'python3 main.py --scenario=plot_metric --n_smooth=250 --n_subnetwork=20 --metric=min_capacity',
        'python3 main.py --scenario=plot_boxplot --n_subnetwork=20 --metric=reward',
        'python3 main.py --scenario=plot_boxplot --n_subnetwork=20 --metric=mean_capacity',
        'python3 main.py --scenario=plot_boxplot --n_subnetwork=20 --metric=min_capacity',
    ]
    deploy_lengths = [10, 20, 40]
    for deploy_length, cmd in tqdm.tqdm(it.product(deploy_lengths, cmds)):
        cmd_ = f'{cmd} --{deploy_length=}'
        print(cmd_)
        os.system(cmd_)
