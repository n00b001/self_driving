import re

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from dataset_keras import process_image_np
from grab_screen import grab_screen
from x2_2_train_net import get_base_model, get_model


def load_model(path):
    num_classes = 9
    base_model = get_base_model()
    model = get_model(base_model=base_model, num_classes=num_classes)
    model.load_weights(path)
    return model


def get_layer_by_name(model, layer_name):
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
    else:
        print("\n".join([x.name for x in model.layers]))
        raise Exception(f"Can't find: {layer_name}")


def show_all_layers(intersting_layers, activations, img):
    layer_names = [x.name for x in intersting_layers]
    average_layer_images = []
    shape = img.shape[:-1]

    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        dimentions = len(layer_activation.shape)
        all_layer_images = []
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                if dimentions == 4:
                    channel_image = layer_activation[0, :, :, col * images_per_row + row]
                elif dimentions == 3:
                    channel_image = layer_activation[:, :, col * images_per_row + row]
                else:
                    raise Exception(f"Unexpected dimentionality: {dimentions}")

                channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                all_layer_images.append(channel_image)
        all_layer_images = np.array(all_layer_images)
        average_layer_image = np.average(all_layer_images, axis=0)
        resized_average = cv2.resize(average_layer_image, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        average_layer_images.append(resized_average)
    average_layer_images_np = np.array(average_layer_images)
    average_feature_map = np.average(average_layer_images_np, axis=0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img, aspect='auto', interpolation='none')
    plt.imshow(average_feature_map, aspect='auto', interpolation='none', cmap='plasma', alpha=0.3)
    plt.draw()
    plt.pause(0.1)


def heatmap_3(model):
    # base_model = model.layers[0]
    # intersting_layers = []
    # reg = "block_.*relu"
    # for l in model.layers[0].layers[100:135]:
    #     if re.match(reg, l.name):
    #         intersting_layers.append(l)

    base_model = model.layers[0]
    intersting_layers = []
    # reg = "conv2d_.*"
    reg = "activation_.*"
    for l in model.layers[0].layers:
        if re.match(reg, l.name):
            intersting_layers.append(l)
    if len(intersting_layers) == 0:
        get_layer_by_name(base_model, "asdf")

    layer_outputs = [layer.output for layer in intersting_layers]

    # Creates a model that will return these outputs, given the model input
    activation_model = tf.keras.models.Model(
        inputs=base_model.input, outputs=layer_outputs
    )

    while True:
        scr = grab_screen()
        img = process_image_np(scr.astype(np.float32))
        expanded_img = np.expand_dims(img, axis=0)

        activations = activation_model.predict(expanded_img)
        # Returns a list of five Numpy arrays: one array per layer activation

        show_all_layers(intersting_layers, activations, scr)


def main():
    img_path = "C:\\Users\\xfant\\PycharmProjects\\self_driving2\\data\\W\\AAAOYAZVNA_2.jpg"
    # img_path = "C:\\Users\\xfant\\PycharmProjects\\self_driving2\\data\\W\\AAACBGIXKK.jpg"
    # model_path = "C:\\Users\\xfant\\PycharmProjects\\self_driving2\\models\\IDBULYIUOJ\\fine_weights_epoch_4\\1558129135" # mobilenetv2
    model_path = "C:\\Users\\xfant\\PycharmProjects\\self_driving2\\models\\BQPEMNVUJT\\fine_weights_epoch_11\\1558707601" # inceptionRestnmet
    output_path = "."
    model = tf.contrib.saved_model.load_keras_model(model_path)

    heatmap_3(model)


if __name__ == '__main__':
    main()
