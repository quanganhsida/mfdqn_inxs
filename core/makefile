all: install_cuda prepare train test plot

install_cuda:
	pip3 install -r requirements.txt
	pip3 install torch torchvision torchaudio

install_cpu:
	pip3 install -r requirements.txt
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

prepare:
	python3 main.py --scenario=prepare_data

prepare_density:
	python3 main.py --scenario=prepare_data --n_snapshot=1000 --n_subnetwork=50 --deploy_length=10
	python3 main.py --scenario=prepare_data --n_snapshot=1000 --n_subnetwork=50 --deploy_length=20
	python3 main.py --scenario=prepare_data --n_snapshot=1000 --n_subnetwork=50 --deploy_length=50
	python3 main.py --scenario=prepare_data --n_snapshot=1000 --n_subnetwork=50 --deploy_length=100

prepare_n_subnetwork:
	python3 main.py --scenario=prepare_data --n_snapshot=1000 --n_subnetwork=10 --deploy_length=10
	python3 main.py --scenario=prepare_data --n_snapshot=1000 --n_subnetwork=20 --deploy_length=20
	python3 main.py --scenario=prepare_data --n_snapshot=1000 --n_subnetwork=50 --deploy_length=50
	python3 main.py --scenario=prepare_data --n_snapshot=1000 --n_subnetwork=100 --deploy_length=100

plot:
	python3 main.py --scenario=plot_metric --n_smooth=500 --metric=reward & \
	python3 main.py --scenario=plot_metric --n_smooth=500 --metric=mean_capacity & \
	python3 main.py --scenario=plot_metric --n_smooth=500 --metric=min_capacity & \
	python3 main.py --scenario=plot_metric --mode=train --n_smooth=500 --metric=reward & \
	python3 main.py --scenario=plot_metric --mode=train --n_smooth=500 --metric=loss & \
	python3 main.py --scenario=plot_metric --mode=train --n_smooth=500 --metric=eps & \
	wait

test:
	python3 main.py --scenario=main --solver=mfdqn
	python3 main.py --scenario=main --solver=idqn
	python3 main.py --scenario=main --solver=greedy
	python3 main.py --scenario=main --solver=maql
	python3 main.py --scenario=main --solver=q_heuristic
	python3 main.py --scenario=main --solver=random

train:
	python3 main.py --scenario=main --mode=train --solver=mfdqn
	python3 main.py --scenario=main --mode=train --solver=idqn
