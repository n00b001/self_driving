import pickle
import time

import numpy as np
import tensorflow as tf

from direct_keys import PressKey, ReleaseKey, W, A, S, D
from grab_screen import grab_screen
from model import Model
from x1_collect_data import fps_stuff2
from x2_train_net import preprocess_image, to_dict

simulate = True
index_to_label = None
model = None


def release_all():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


def press_label(label):
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
    model_path = "./models/WRKHZOWXNC"
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
        label = normalised_screen_to_label()
        press_label(label)
        counter, start_time = fps_stuff2(counter, start_time, x)


def normalised_screen_to_label():
    predictions = model.predict(get_screen_dict())
    output = predictions["class_ids"][0]
    label = index_to_label[int(output)]
    return label


def get_screen_dict():
    if False:
        image = tf.convert_to_tensor(grab_screen())
        pre_processed = preprocess_image(image, training=False)
        pre_processed = tf.expand_dims(pre_processed, axis=0)
        return to_dict(pre_processed)
    return {"x": np.expand_dims(np.divide(grab_screen(), 255), axis=0)}
    # return {"x": np.zeros((1, 96, 96, 3), dtype=np.float32)}


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
