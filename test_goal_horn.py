import gengAI
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import ImageGrab
import os
import time

if __name__ == "__main__":
    input_dim = [1024, 1024]
    # nn = gengAI.NN.Convolutional.CNN(input_dim, 7, 2, 2).to(torch.device("cuda:0"))
    print("Loading model")
    nn = torch.load("StarsGoalHornAI.pt", weights_only=False)
    nn.eval()
    print(nn)
    print("Creating transform")
    transform = transforms.Compose([
        transforms.Resize((input_dim[0], input_dim[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    print("Create function to transform screenshot to tensor")
    ToTensor = transforms.ToTensor()
    while True:
        SS = ImageGrab.grab()
        tensor = ToTensor(SS).to(torch.device("cuda:0"))
        outputs = nn(transform(SS).unsqueeze(0))
        print(outputs.cpu().detach().numpy())
        if torch.sigmoid(outputs.cpu()).detach().numpy()[0][0] > 0.5:
            print("Goal?")
        else:
            print("No goal?")
        time.sleep(0.1)


    