import itertools as it
import tqdm
import os

def exp_time(args):
    cmds = [
        # Test
		'python3 main.py --scenario=main --n_test_step=5000 --solver=q_heuristic',
		'python3 main.py --scenario=main --n_test_step=5000 --solver=random',
		# 'python3 main.py --scenario=main --n_test_step=5000 --solver=greedy',
		# 'python3 main.py --scenario=main --n_test_step=5000 --solver=maql',
		# 'python3 main.py --scenario=main --n_test_step=5000 --solver=idqn',
		# 'python3 main.py --scenario=main --n_test_step=5000 --solver=mfdqn',
		# 'python3 main.py --scenario=main --n_test_step=5000 --solver=vdn',
        # Plot test
        # 'python3 main.py --scenario=plot_line_time --n_subnetwork=20 --metric=time',
    ]
    N = [10, 20, 40, 80]
    for n, cmd in tqdm.tqdm(it.product(N, cmds)):
        cmd_ = f'{cmd} --n_subnetwork={n} --deploy_length={n}'
        print(cmd_)
        os.system(cmd_)
