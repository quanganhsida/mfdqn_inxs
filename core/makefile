all: prepare plot

install:
	pip3 install -r requirements.txt
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

install_cuda:
	pip3 install -r requirements.txt
	pip3 install torch torchvision torchaudio

prepare:
	python3 main.py --scenario=prepare_data

plot:
	python3 main.py --scenario=plot_metric --n_smooth=500 --metric=reward & \
	python3 main.py --scenario=plot_metric --n_smooth=500 --metric=mean_capacity & \
	python3 main.py --scenario=plot_metric --n_smooth=500 --metric=min_capacity & \
	python3 main.py --scenario=plot_metric --mode=train --n_smooth=500 --metric=reward & \
	python3 main.py --scenario=plot_metric --mode=train --n_smooth=500 --metric=loss & \
	python3 main.py --scenario=plot_metric --mode=train --n_smooth=500 --metric=eps & \
	wait

run:
	python3 main.py --scenario=main --solver=greedy
	python3 main.py --scenario=main --solver=maql
	python3 main.py --scenario=main --solver=optimal
	python3 main.py --scenario=main --solver=q_heuristic
	python3 main.py --scenario=main --solver=random
	python3 main.py --scenario=main --solver=mfdqn
	python3 main.py --scenario=main --solver=idqn

train:
	python3 main.py --scenario=main --mode=train --solver=mfdqn
	python3 main.py --scenario=main --mode=train --solver=idqn
