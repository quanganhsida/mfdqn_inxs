import torch.nn.functional as F
import torch.nn as nn
import torch

class DQN(nn.Module):

    def __init__(self, args):
        super().__init__()
        n_observation = args.n_channel * args.n_power_level * 2
        n_action = args.n_channel * args.n_power_level
        self.fc_in = nn.Linear(n_observation, args.dqn_n_hidden)
        self.fc_hiddens = nn.ModuleList()
        for _ in range(args.dqn_n_layer - 1):
            self.fc_hiddens.append(nn.Linear(args.dqn_n_hidden,
                                             args.dqn_n_hidden))
        self.fc_out = nn.Linear(args.dqn_n_hidden, n_action)

    def forward(self, x):
        x = self.fc_in(x)
        x = F.relu(x)
        for layer in self.fc_hiddens:
            x = layer(x)
            x = F.relu(x)
        x = self.fc_out(x)
        return x
