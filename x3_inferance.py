import pickle
import time

import numpy as np
import tensorflow as tf

from direct_keys import PressKey, ReleaseKey, W, A, S, D
from grab_keys import grab_keys
from grab_screen import grab_screen
from model import Model
from x1_collect_data import fps_stuff2

simulate = True
index_to_label = None
model: Model = None
last_print_time = time.time()


def release_all():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


def press_label(label):
    global last_print_time
    if time.time() - last_print_time > 1:
        last_print_time = time.time()
        print(label)
    if simulate:
        return
    if label == "NO":
        release_all()
    elif label == "WA":
        PressKey(W)
        PressKey(A)
        ReleaseKey(S)
        ReleaseKey(D)
    elif label == "WD":
        PressKey(W)
        ReleaseKey(A)
        ReleaseKey(S)
        PressKey(D)
    elif label == "AS":
        ReleaseKey(W)
        PressKey(A)
        PressKey(S)
        ReleaseKey(D)
    elif label == "SD":
        ReleaseKey(W)
        ReleaseKey(A)
        PressKey(S)
        PressKey(D)
    elif label == "W":
        PressKey(W)
        ReleaseKey(A)
        ReleaseKey(S)
        ReleaseKey(D)
    elif label == "A":
        ReleaseKey(W)
        PressKey(A)
        ReleaseKey(S)
        ReleaseKey(D)
    elif label == "D":
        ReleaseKey(W)
        ReleaseKey(A)
        ReleaseKey(S)
        PressKey(D)
    elif label == "S":
        ReleaseKey(W)
        ReleaseKey(A)
        PressKey(S)
        ReleaseKey(D)

def main(_):
    model_path = "./models/VHAQJMFYIV"
    global index_to_label, model
    with open('./{}/label_to_index.pkl'.format(model_path), 'rb') as f:
        label_to_index = pickle.load(f)
    index_to_label = {v: k for (k, v) in label_to_index.items()}
    model = Model(index_to_label, model_path=model_path)
    print("Number of labels: {}".format(len(label_to_index.keys())))

    start_time = time.time()
    x = 1
    counter = 0

    while True:
        label = model.predict(get_screen_dict())
        press_label(label)
        counter, start_time = fps_stuff2(counter, start_time, x)


def get_screen_dict():
    return {"x": np.expand_dims(grab_screen(), axis=0)}


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
