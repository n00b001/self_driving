import os

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.data.experimental import AUTOTUNE
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from consts import IMAGE_SIZE, BATCH_SIZE, SHUFFLE_BUFFER
from file_stuff import split_paths


def load_image(path, label):
    image = tf.image.decode_jpeg(tf.read_file(path), channels=3)
    return image, label


def get_raw_ds(all_image_labels, all_image_paths, is_training):
    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
    ds = ds.map(
        map_func=load_image,
        num_parallel_calls=os.cpu_count()
    )
    if is_training:
        ds = ds.map(
            map_func=img_augmentation,
            num_parallel_calls=os.cpu_count()
        )
        ds = ds.shuffle(SHUFFLE_BUFFER)
    ds = ds.apply(tf.data.experimental.map_and_batch(
        map_func=preprocess_from_path_label,
        batch_size=BATCH_SIZE,
        num_parallel_batches=os.cpu_count(),
    ))
    ds = ds.apply(tf.data.experimental.ignore_errors())
    # todo: this cache is a real pain in the ASS, it helps with speed,
    #   but it sometimes just fills up GPU RAM and freezes the PC
    #   this might be caused by something else, I don't know
    # todo: another error: There appears to be a concurrent caching iterator running
    # ds = ds.cache(filename=os.path.join(CACHE_LOCATION, "cache.tf"))
    return ds.prefetch(buffer_size=AUTOTUNE)


def process_image_np(image):
    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    image_mean = resized.mean()
    variance = np.subtract(np.mean(np.square(resized), axis=(-1, -2, -3), keepdims=True),
                           np.square(image_mean))
    variance = variance * (variance > 0)
    stddev = np.sqrt(variance)

    min_stddev = np.reciprocal(np.sqrt(resized.size))
    pixel_value_scale = max(stddev, min_stddev)

    new_image = np.subtract(resized, image_mean)
    new_image = np.divide(new_image, pixel_value_scale)

    return new_image


def get_datasets(x, y):
    xtr, xte = split_paths(x)
    ytr, yte = split_paths(y)
    train_ds = get_raw_ds(
        is_training=True,
        all_image_paths=xtr,
        all_image_labels=ytr
    )
    test_ds = get_raw_ds(
        is_training=False,
        all_image_paths=xte,
        all_image_labels=yte
    )

    steps_per_epoch = round(len(xtr)) // BATCH_SIZE
    steps_per_epoch = 10

    val_steps = round(len(xte)) // BATCH_SIZE

    return train_ds, test_ds, steps_per_epoch, val_steps


def img_augmentation(x, label):
    # todo; DO NOT RANDOM LEFT RIGHT FLIP - WE ARE TRYING TO PREDICT DRIVING LEFT OR RIGHT
    # if np.random.uniform(0, 1) < thresh:
    #     # todo: always seems to produce same quality
    #     x = tf.image.random_jpeg_quality(x, 30, 100)
    x_new = tf.cast(x, tf.dtypes.float32)
    x_new = tf.image.random_brightness(x_new, 0.5)
    x_new = tf.image.random_contrast(x_new, 0.4, 1.4)
    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, 0.5)

    def set_lower_part_of_image_to_black():
        # shape of images is (IMG_SIZE, IMG_SIZE, IMG_DEPTH)
        shape_f32 = tf.shape(x_new)
        # get Y size (96 pixels for example)
        y_shape = shape_f32[0]
        # get X size (96 pixels for example)
        x_shape = shape_f32[1]
        # multiply y size by 0.62 to get boundry we want to keep (keep between 0 and 96*0.62 (59.52))
        y_pos = tf.cast(tf.multiply(tf.cast(y_shape, tf.dtypes.float32), tf.constant(0.62)), y_shape.dtype)
        # create 2d array of shape 96 (width) x 59.52 (y_pos)
        ones = tf.ones(shape=(y_pos, x_shape))
        # how much padding to add in the y dimension so it matches image
        padding = tf.subtract(y_shape, y_pos)
        # padding must be integer
        padding = tf.convert_to_tensor([[0, padding], [0, 0]], dtype=tf.dtypes.int32)
        # pad with zeros so it matches image
        mask = tf.pad(ones, padding)
        # cast mask to input dtype so we can multiply
        cast = tf.cast(mask, x_new.dtype)
        # multiply original image with mask to set y>59.52 to 0 (black)
        return tf.multiply(x_new, cast[:, :, tf.newaxis])

    def just_return():
        return x_new

    x_new = tf.cond(pred, set_lower_part_of_image_to_black, just_return)
    x_new = tf.image.random_hue(x_new, 0.06)
    x_new = tf.image.random_saturation(x_new, 0.1, 1.9)
    x_new += tf.random_normal(shape=tf.shape(x_new), mean=0, stddev=10, dtype=x_new.dtype)
    x = tf.cast(tf.clip_by_value(x_new, 0, 255), x.dtype)
    return x, label


def preprocess_from_path_label(image, label):
    image = tf.image.per_image_standardization(image)
    return tf.image.resize_images(image, (IMAGE_SIZE, IMAGE_SIZE), method=ResizeMethod.AREA), label


if __name__ == '__main__':
    img_path = "C:\\Users\\xfant\\PycharmProjects\\self_driving2\\data\\W\\AAAJFIGKSO.jpg"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # config = tf.ConfigProto(
    #     device_count={'GPU': 0}
    # )
    # sess = tf.Session(config=config)
    sess = tf.Session()
    val, _ = load_image(img_path, None)
    val, _ = img_augmentation(val, _)
    # val = tf.dtypes.cast(val, tf.dtypes.uint8)

    while True:
        runs = []
        for i in range(10):
            runs.append(sess.run(val))
            # time.sleep(1)
            # ran = cv2.cvtColor(ran, cv2.COLOR_BGR2RGB)
        for ran in runs:
            plt.imshow(ran, aspect='auto', interpolation='none')
            plt.draw()
            plt.pause(0.4)
