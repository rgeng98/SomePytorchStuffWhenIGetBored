import gengAI
import torch
import RPi.GPIO as GPIO
from torchvision import transforms
from PIL import ImageGrab
import time
import pygame
import random
import zmq

outpin = 18

if __name__ == "__main__":
    input_dim = [1024, 1024]
    pygame.init()
    sounds = pygame.mixer.Sound("GoalHornAudio/GoalHorn4.wav"),
    #     pygame.mixer.Sound("GoalHornAudio/GoalHorn2.wav"),
    #     pygame.mixer.Sound("GoalHornAudio/GoalHorn3.wav"),
    #     pygame.mixer.Sound("GoalHornAudio/GoalHorn4.wav")
    # ]
    GPIO.setmode(GPIO.BCM)  
    GPIO.setup(outpin, GPIO.OUT, initial=GPIO.LOW)

    print("Loading model")
    nn = torch.load("StarsGoalHornAI.pth", weights_only=False)
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

    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://localhost:5555")

    message = "GOAL"
    
    running = True
    while True:
        SS = ImageGrab.grab()
        tensor = ToTensor(SS).to(torch.device("cuda:0"))
        outputs = nn(transform(SS).unsqueeze(0))
        if torch.sigmoid(outputs.cpu()).detach().numpy()[0][0] > 0.5:
            time.sleep(0.1)
        else:
            print("Goal")
            socket.send_string(message)
            GPIO.output(outpin, GPIO.HIGH)
            sound = random.choice(sounds)
            sound.play()
            time.sleep(40)
            GPIO.output(outpin, GPIO.LOW)
        
    socket.close()
    context.term()
        


    