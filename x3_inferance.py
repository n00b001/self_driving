import pickle
import time

import numpy as np
import tensorflow as tf

from direct_keys import PressKey, ReleaseKey, W, A, S, D
from grab_screen import grab_screen
from model import Model
from x1_collect_data import fps_stuff2

simulate = False
index_to_label = None
model: Model = None
last_print_time = time.time()


def on_click(x, y, button, pressed):
    global simulate
    if pressed:
        print('Mouse clicked with {2}'.format(x, y, button))
        if button.name == "left":
            print("Turning on simulation...")
            simulate = True
        elif button.name == "right":
            print("Turning off simulation...")
            simulate = False
        print("Simulate: {}".format(simulate))


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
    # release_all()
    if label == "WA":
        PressKey(W)
        PressKey(A)
        ReleaseKey(S)
        ReleaseKey(D)
    elif label == "WD":
        PressKey(W)
        PressKey(D)
        ReleaseKey(A)
        ReleaseKey(S)
    elif label == "AS":
        PressKey(A)
        PressKey(S)
        ReleaseKey(W)
        ReleaseKey(D)
    elif label == "SD":
        PressKey(S)
        PressKey(D)
        ReleaseKey(W)
        ReleaseKey(A)
    elif label == "A":
        PressKey(A)
        ReleaseKey(W)
        ReleaseKey(S)
        ReleaseKey(D)
    elif label == "D":
        PressKey(D)
        ReleaseKey(W)
        ReleaseKey(A)
        ReleaseKey(S)
    elif label == "S":
        PressKey(S)
        ReleaseKey(W)
        ReleaseKey(A)
        ReleaseKey(D)
    elif label == "W":
        PressKey(W)
        ReleaseKey(A)
        ReleaseKey(S)
        ReleaseKey(D)


def main(_):
    model_path = "./models/MHMHYFRXLM"
    global index_to_label, model
    with open('./{}/label_to_index.pkl'.format(model_path), 'rb') as f:
        label_to_index = pickle.load(f)
    index_to_label = {v: k for (k, v) in label_to_index.items()}
    model = Model(index_to_label, model_path=model_path, dropout=0.0)
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
