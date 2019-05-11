import os
import random

import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

from consts import SHUFFLE_BUFFER, IMAGE_SIZE, BATCH_SIZE


def get_initialized_dataset(is_training, all_image_paths, all_image_labels, label_weight_list=None):
    ds = get_raw_ds(all_image_labels, all_image_paths, is_training, label_weight_list)
    ds = ds.make_one_shot_iterator()
    return ds.get_next()


def get_raw_ds(all_image_labels, all_image_paths, is_training, label_weight_list=None):
    if label_weight_list is None:
        label_weight_list = tf.convert_to_tensor([1 for _ in range(len(all_image_labels))])
    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels, label_weight_list))
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


def load_and_preprocess_image(path):
    image = tf.image.decode_jpeg(tf.read_file(path), channels=3)
    return tf.image.resize_images(image, (IMAGE_SIZE, IMAGE_SIZE))


def img_augmentation(image, label):
    x = image["x"]
    w = image["w"]
    random_num = random.random()
    if random_num < 0.1:
        x = tf.image.random_flip_left_right(x)
    if random_num < 0.03:
        x = tf.image.random_brightness(x, 100)
    if random_num < 0.03:
        x = tf.image.random_contrast(x, 0, 100)
    # todo: below needs changing before it'll work for batches
    # if random_num < 0.005:
    #     image = tf.image.random_hue(image, 0.5)
    # if random_num < 0.1:
    #     image = tf.image.random_jpeg_quality(image, 0, 100)
    # if random_num < 0.01:
    #     image = tf.image.random_saturation(image, 0, 100)
    return to_dict(x, w), label


def load_and_preprocess_from_path_label(path, label, weight):
    x = load_and_preprocess_image(path)
    return to_dict(x, weight), label


def to_dict(x, w):
    return {
        "x": x,
        "w": w
    }
