import pickle

from direct_keys import PressKey, ReleaseKey, W, A, S, D
from grab_screen import grab_screen

# def get_model(model_path):
#     return predictor.from_saved_model(model_path)
from x1_collect_data import resize
from x2_train_net import get_model
import numpy as np
simulate = True

def release_all():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


def press_label(label):
    if simulate:
        print(label)
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


def main(model_path):
    with open('label_to_index.pkl', 'rb') as f:
        label_to_index = pickle.load(f)
    index_to_label = {v: k for (k, v) in label_to_index.items()}
    with open('image_count.pkl', 'rb') as f:
        image_count = pickle.load(f)
    model = get_model(num_classes=image_count, model_path=model_path)
    while True:
        screen = grab_screen()
        resized_screen = resize(screen)
        resized_screen = resized_screen.astype("float32")
        resized_screen /= 255.0
        label = normalised_screen_to_label(index_to_label, model, resized_screen)
        press_label(label)


def normalised_screen_to_label(index_to_label, model, resized_screen):
    resized_screen = np.expand_dims(resized_screen, axis=0)
    output = model.predict({"x": resized_screen})
    output_list = list(output)
    label = index_to_label[int(output_list[0])]
    return label


if __name__ == '__main__':
    main("./models/LBAUNNNGJX")
