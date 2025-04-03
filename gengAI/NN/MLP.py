import gengAI.NN.FeedForward
import torch
import torch.nn as nn
from torch.autograd import Variable
import json

# Multilayer perceptron
class MLP(object):

    def __init__(self, config_file,
                 device=torch.device("cuda:0" if torch.cuda.is_available()
                                     else "cpu")):
        with open(config_file, 'r') as f:
            config = json.load(f)
        n_inputs = config["n_inputs"]
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        n_outputs = config["n_outputs"]
        drop = config["drop"]
        self.Network = gengAI.NN.FeedForward.FeedForward(n_inputs, hidden_size, num_layers, n_outputs, drop)
        self.device = device
        self.Network.to(self.device)
        if config["optimizer"] == "Adam":
            self.Optim =  torch.optim.Adam(self.Network.parameters(), lr=config["learning_rate"]) 
        if config["optimizer"] == "RAdam":
            self.Optim =  torch.optim.RAdam(self.Network.parameters(), lr=config["learning_rate"]) 
        if config["criterion"] == "L1Loss":
            self.Loss = torch.nn.L1Loss()
        if torch.cuda.is_available():
            print("Using cuda")


    def train(self, inputs, targets):
        inputs, t = inputs.to(self.device), targets.to(self.device)
        self.Optim.zero_grad()
        outputs = self.Network(Variable(inputs))
        error = self.Loss(outputs, Variable(t))
        error.backward()
        self.Optim.step()
        return pos, error.item()


    def test(self, inputs, targets):
        inputs, t = inputs.to(self.device), targets.to(self.device)
        with torch.no_grad():
            pos = self.Network(inputs)
            error1 = self.Loss(pos, t)
            error = error1
            return pos, error.item()

    def deploy(self, inputs, a1):
        inputs = inputs.to(self.device)
        with torch.no_grad():
            pos = self.Network(inputs)
            self.x = float(pos[0][0]) + float(a1[0][0])
            self.y = float(pos[0][1]) + float(a1[0][1])
            self.z = float(pos[0][2]) + float(a1[0][2])
        return pos

