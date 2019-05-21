import os
import random

import cv2
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE
import numpy as np
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from consts import IMAGE_SIZE, BATCH_SIZE, SHUFFLE_BUFFER


def get_raw_ds(all_image_labels, all_image_paths, is_training):
    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
    ds = ds.apply(tf.data.experimental.map_and_batch(
        map_func=load_and_preprocess_from_path_label,
        batch_size=BATCH_SIZE,
        num_parallel_batches=os.cpu_count(),
    ))
    ds = ds.apply(tf.data.experimental.ignore_errors())
    # todo: this cache is a real pain in the ASS, it helps with speed,
    #   but it sometimes just fills up GPU RAM and freezes the PC
    #   this might be caused by something else, I don't know
    # todo: another error: There appears to be a concurrent caching iterator running
    # ds = ds.cache(filename=os.path.join(CACHE_LOCATION, "cache.tf"))
    if is_training:
        ds = ds.map(
            map_func=img_augmentation,
            num_parallel_calls=os.cpu_count()
        )
        ds = ds.shuffle(SHUFFLE_BUFFER)
    return ds.prefetch(buffer_size=AUTOTUNE)


def process_image(image):
    resized = tf.image.resize_images(image, (IMAGE_SIZE, IMAGE_SIZE), method=ResizeMethod.AREA)
    norm = tf.image.per_image_standardization(resized)
    return norm


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


def img_augmentation(x, label):
    if random.random() < 0.5:
        x = tf.image.random_flip_left_right(x)
    if random.random() < 0.01:
        x = tf.image.random_brightness(x, 50)
    if random.random() < 0.01:
        x = tf.image.random_contrast(x, 50, 150)
    if random.random() < 0.9:
        x += tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)
    if random.random() < 0.1:
        hue = lambda h_func: tf.image.random_hue(h_func, 0.1)
        x = tf.map_fn(hue, x)
    if random.random() < 0.1:
        jpeg = lambda j_func: tf.image.random_jpeg_quality(j_func, 50, 100)
        x = tf.map_fn(jpeg, x)
    if random.random() < 0.1:
        sat = lambda s_func: tf.image.random_saturation(s_func, 50, 150)
        x = tf.map_fn(sat, x)
    return x, label


def load_and_preprocess_from_path_label(path, label):
    image = tf.image.decode_jpeg(tf.read_file(path), channels=3)
    return process_image(image), label
