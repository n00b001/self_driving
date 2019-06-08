import os
import re

import cv2
import numpy as np
import tensorflow as tf

from consts import IMAGE_SIZE
from dataset_keras import process_image_np
from grab_screen import grab_screen
from model import Model


def get_layer_by_name(model, layer_name):
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
    else:
        print("\n".join([x.name for x in model.layers]))
        raise Exception(f"Can't find: {layer_name}")


def show_all_layers(activations, img, shape):
    average_layer_image = np.mean([
        cv2.resize(np.mean(a, axis=-1)[0], dsize=(shape[1], shape[0])) for a in activations
    ], axis=0)
    normalized = (average_layer_image - average_layer_image.min()) \
                 / (average_layer_image.max() - average_layer_image.min())

    heatmap = np.uint8(255 * normalized)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imshow("Original", img)
    cv2.imshow("Main", superimposed_img)
    cv2.waitKey(1)


def heatmap_3(model: Model, images_list=None):
    base_model = model.model.layers[0]
    intersting_layers = []

    # incepotionresnet
    # reg = "block_.*relu"
    # for l in model.layers[0].layers[100:135]:
    #     if re.match(reg, l.name):
    #         intersting_layers.append(l)

    # mobilenetv2
    # reg = "conv2d_.*"
    # reg = "activation_.*"
    # for l in model.layers[0].layers:
    #     if re.match(reg, l.name):
    #         intersting_layers.append(l)

    # mobilenetv1
    reg = ".*relu"
    for l in base_model.layers:
        if re.match(reg, l.name):
            intersting_layers.append(l)
    if len(intersting_layers) == 0:
        get_layer_by_name(base_model, "asdf")

    layer_outputs = [layer.output for layer in intersting_layers]

    # Creates a model that will return these outputs, given the model input
    activation_model = tf.keras.models.Model(
        inputs=base_model.input, outputs=layer_outputs
    )

    cv2.namedWindow("Main", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    if images_list is not None:
        for x in images_list:
            show_activations(activation_model, x)
        cv2.waitKey(1) & 0xFF
        exit(0)
    while True:
        show_activations(activation_model, images_list)


def show_activations(activation_model, img_path):
    if img_path is None:
        scr = grab_screen()
    else:
        scr = cv2.imread(img_path)
    scr = cv2.resize(scr, (IMAGE_SIZE, IMAGE_SIZE))

    img = np.expand_dims(process_image_np(scr.astype(np.float32)), axis=0)
    activations = activation_model.predict(img)
    show_all_layers(activations, scr, scr.shape[:-1])
    if img_path is not None:
        cv2.waitKey(0)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    images_list = [
        "C:\\Users\\xfant\\PycharmProjects\\self_driving2\\data\\W\\AAAOYAZVNA_2.jpg",
        "C:\\Users\\xfant\\PycharmProjects\\self_driving2\\data\\W\\AAACBGIXKK.jpg",
        "C:\\Users\\xfant\\PycharmProjects\\self_driving2\\data\\A\\AAAOYAZVNA_55.jpg",
        "C:\\Users\\xfant\\PycharmProjects\\self_driving2\\data\\A\\AAAHWYBLNQ.jpg",
        "C:\\Users\\xfant\\PycharmProjects\\self_driving2\\data\\D\\AAAOYAZVNA_44.jpg",
        "C:\\Users\\xfant\\PycharmProjects\\self_driving2\\data\\D\\AAIRLVGYTB.jpg",
        "C:\\Users\\xfant\\PycharmProjects\\self_driving2\\data\\S\\ABRFGAESDR_159.jpg",
        "C:\\Users\\xfant\\PycharmProjects\\self_driving2\\data\\S\\ABOVZDNNBB.jpg",
    ]

    model_path = "C:\\Users\\xfant\\PycharmProjects\\self_driving2\\models\\JCJYWJIYEI\\fine_weights_epoch_0_13\\1559831114"
    model_path = "C:\\Users\\xfant\\PycharmProjects\\self_driving2\\models\\TSZIATHCIA"
    # model_path = "C:\\Users\\xfant\\PycharmProjects\\self_driving2\\models\\TSZIATHCIA\\fine_weights_epoch_0_13\\1559916282"
    model = Model(
        model_path=model_path,
        predict=True
    )

    heatmap_3(model, images_list=None)
    # heatmap_3(model, images_list=images_list)


if __name__ == '__main__':
    main()
