import datetime

from pynput import keyboard
from PIL import ImageGrab
import sys

def on_press(key):
    if key.char == "q":
        SS = ImageGrab.grab()
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = f"starsGoalDatabase/train/Goal/{time}.jpg"
        SS.save(save_path)

def on_release(key):
    if key == keyboard.Key.esc:
        return False


# Collect events until released
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()       