import torch
import torch.nn as nn
from torch.autograd import Variable

# Initialize weights according to https://arxiv.org/abs/1502.01852
# Normal Distribution:
def initialize_weights_kn(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.kaiming_normal_(m.weight,a = 0.2, mode='fan_in', nonlinearity='leaky_relu')#, gain=nn.init.calculate_gain('leaky_relu',0.2))

# Uniform distribution:
def initialize_weights_ku(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.kaiming_uniform_(m.weight,a = 0.2, mode='fan_in', nonlinearity='leaky_relu')#, gain=nn.init.calculate_gain('leaky_relu',0.2))

# Initialize weights according to https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
# Normal Distribution:
def initialize_weights_xn(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu',0.2))

# Uniform distribution
def initialize_weights_xu(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu',0.2))

# Initialize weights according to https://openreview.net/forum?id=_wzZwKpTDF_9C
def initialize_weights_orth_(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.orthogonal(m.weight)

class FeedForward(torch.nn.Module):
    def __init__(self, n_inputs: int, hidden_size: int, num_layers: int, n_outputs: int, drop: float):
        super(FeedForward, self).__init__()
        self.n_inputs = n_inputs
        self.hiddensize = hidden_size
        sl = [torch.nn.Linear(self.n_inputs, self.hiddensize),
              torch.nn.Dropout(drop)]
        for i in range(num_layers-1):
            sl.append(torch.nn.PReLU(num_parameters=self.hiddensize))
            sl.append(torch.nn.Linear(self.hiddensize, self.hiddensize))
            sl.append(torch.nn.Dropout(drop))
        sl.append(torch.nn.PReLU(num_parameters=self.hiddensize))
        sl.append(torch.nn.Linear(self.hiddensize, n_outputs))
        self.network = torch.nn.Sequential(*sl)
        self.apply(initialize_weights_kn)

    def forward(self, x):
        return self.network(x)
