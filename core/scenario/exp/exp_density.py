import itertools as it
import tqdm
import os

def exp_density(args):
    cmds = [
		'python3 main.py --scenario=plot_network --n_subnetwork=50',
		# 'python3 main.py --scenario=main --n_subnetwork=50 --solver=mfdqn',
		# 'python3 main.py --scenario=main --n_subnetwork=50 --solver=idqn',
		# 'python3 main.py --scenario=main --n_subnetwork=50 --solver=greedy',
		# 'python3 main.py --scenario=main --n_subnetwork=50 --solver=maql',
		# 'python3 main.py --scenario=main --n_subnetwork=50 --solver=q_heuristic',
		# 'python3 main.py --scenario=main --n_subnetwork=50 --solver=random',
    ]
    deploy_lengths = [10, 20, 50, 100]
    for deploy_length, cmd in tqdm.tqdm(it.product(deploy_lengths, cmds)):
        cmd_ = f'{cmd} --{deploy_length=}'
        print(cmd_)
        os.system(cmd_)
