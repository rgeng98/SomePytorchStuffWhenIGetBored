import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

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

class CNN(torch.nn.Module):
    def __init__(self, input_dim: int, kernel: int, num_layers: int, n_outputs: int):
        super(CNN, self).__init__()
        if torch.cuda.is_available():
            print("Using CUDA")
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        # This code is going to calculate the number of output features from the convolutional layers and create the appropriate number of inputs for the last fully connected layer
        self.input_dim = input_dim
        n_channels_in = 3
        n_channels_out = 64
        padding = 3
        stride = 3
        sl = [torch.nn.Conv2d(n_channels_in, n_channels_out, kernel_size=kernel, stride=stride, padding=padding), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        # Set the padding to 1 for all layers after the first one
        padding = 1
        stride = 1
        for i in range(num_layers-2):
            n_channels_in = n_channels_out
            if n_channels_out <=256:
                n_channels_out = n_channels_out * 2
            if kernel > 3:
                kernel = kernel - 2
            sl.append(torch.nn.Conv2d(n_channels_in, n_channels_out, kernel_size=kernel, padding=padding))
            sl.append(torch.nn.ReLU())
            sl.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

        n_channels_in = n_channels_out
        if n_channels_out <=256:
            n_channels_out = n_channels_out * 2
        sl.append(torch.nn.Conv2d(n_channels_in, n_channels_out, kernel_size=1))
        sl.append(torch.nn.ReLU())
        sl.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))  
        sl.append(torch.nn.Flatten())     
        self.network = torch.nn.Sequential(*sl)
        self.network = self.network.to(self.device)
        sample_in = torch.randn(3, input_dim[0], input_dim[1]).unsqueeze(0).to(self.device)
        sample_out = self.network(sample_in)
        print(n_outputs)
        self.outlayer = torch.nn.Linear(sample_out.shape[1], n_outputs)
        self.outlayer = self.outlayer.to(self.device)
        self.apply(initialize_weights_kn)
        

    def forward(self, x):
        x = x.to(self.device)
        x = self.network(x)
        return self.outlayer(x)
