import time

import tensorflow as tf

from direct_keys import PressKey, ReleaseKey, W, A, S, D
from model import Model
from x1_collect_data import fps_stuff2

simulate = True
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
        pass
    # release_all()
    elif label == "WA":
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
        # time.sleep(0.05)
        # release_all()


def main(_):
    model_path = "C:\\Users\\xfant\\PycharmProjects\\self_driving2\\models\\TSZIATHCIA"
    global model
    model = Model(
        model_path=model_path,
        predict=True,
    )

    start_time = time.time()
    x = 1
    counter = 0

    while True:
        label = model.predict()
        press_label(label)
        counter, start_time = fps_stuff2(counter, start_time, x)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
