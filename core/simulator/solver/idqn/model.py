import torch.nn.functional as F
import torch.nn as nn
import torch

class DQN(nn.Module):

    def __init__(self, args):
        super().__init__()
        n_observation = n_action = args.n_channel * args.n_power_level
        self.fc1 = nn.Linear(n_observation, args.dqn_n_hidden)
        self.fc2 = nn.Linear(args.dqn_n_hidden, args.dqn_n_hidden)
        self.fc3 = nn.Linear(args.dqn_n_hidden, n_action)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
