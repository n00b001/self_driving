import os
import pathlib
import pickle
import random

import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

from consts import DATA_DIR, BATCH_SIZE, EPOCHS, SHUFFLE_BUFFER, MAX_TRAINING_STEPS
from model import Model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
random.seed = 1337


def load_and_preprocess_image(path, training):
    image = tf.image.decode_jpeg(tf.read_file(path), channels=3)
    if training:
        return preprocess_image(image, training)
    return image


def preprocess_image(image, training=True):
    if training:
        if random.random() < 0.1:
            image = tf.image.random_flip_left_right(image)
        if random.random() < 0.03:
            image = tf.image.random_brightness(image, 100)
        if random.random() < 0.03:
            image = tf.image.random_contrast(image, 0, 100)
        if random.random() < 0.005:
            image = tf.image.random_hue(image, 0.5)
        if random.random() < 0.1:
            image = tf.image.random_jpeg_quality(image, 0, 100)
        if random.random() < 0.01:
            image = tf.image.random_saturation(image, 0, 100)
    return image


def load_and_preprocess_from_path_label_train(path, label):
    return to_dict(load_and_preprocess_image(path, True)), label


def to_dict(x):
    return {"x": x}


def load_and_preprocess_from_path_label_test(path, label):
    return to_dict(load_and_preprocess_image(path, False)), label


def get_dataset(is_training, model_path):
    all_image_paths, number_of_labels = get_paths_and_count()

    train_ds, test_ds = split_paths(all_image_paths)
    if is_training:
        all_image_paths = train_ds
    else:
        all_image_paths = test_ds

    label_to_index = dict((name, index) for index, name in enumerate(number_of_labels.keys()))
    with open('./{}/label_to_index.pkl'.format(model_path), 'wb') as f:
        pickle.dump(label_to_index, f, pickle.HIGHEST_PROTOCOL)
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

    if is_training:
        ds = ds.shuffle(SHUFFLE_BUFFER)
        ds = ds.apply(tf.data.experimental.map_and_batch(
            map_func=load_and_preprocess_from_path_label_train,
            batch_size=BATCH_SIZE,
            num_parallel_batches=os.cpu_count(),
        ))
    else:
        ds = ds.apply(tf.data.experimental.map_and_batch(
            map_func=load_and_preprocess_from_path_label_test,
            batch_size=BATCH_SIZE,
            num_parallel_batches=os.cpu_count(),
        ))

    ds = ds.prefetch(buffer_size=AUTOTUNE)
    ds = ds.make_one_shot_iterator()
    ds = ds.get_next()
    return ds


def split_paths(all_image_paths):
    train_split_index = int(0.8 * len(all_image_paths))
    train_ds = all_image_paths[:train_split_index]
    test_ds = all_image_paths[train_split_index:]
    return train_ds, test_ds


def get_paths_and_count():
    data_root = pathlib.Path(DATA_DIR)
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    max_class_examples = max([len(os.listdir(os.path.join(DATA_DIR, k))) for k in label_names])

    number_of_labels = {}
    for k in label_names:
        num_examples = len(os.listdir(os.path.join(DATA_DIR, k)))
        threshold = max_class_examples * 0.01
        if num_examples > threshold:
            number_of_labels[k] = num_examples

    all_image_paths = []
    for k, _ in number_of_labels.items():
        files = [os.path.join(DATA_DIR, k, x) for x in os.listdir(os.path.join(DATA_DIR, k))]
        all_image_paths.extend(files)
    random.shuffle(all_image_paths)
    return all_image_paths, number_of_labels


def train(model: Model):
    for i in range(EPOCHS):
        print("Starting epoch: {}".format(i + 1))
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: get_dataset(is_training=True, model_path=model.model_path),
            max_steps=MAX_TRAINING_STEPS,
        )

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: get_dataset(is_training=False, model_path=model.model_path),
            steps=None
        )
        stats = tf.estimator.train_and_evaluate(model.model, train_spec, eval_spec)
        print("Stats: {}".format(stats))


def main(_):
    all_images, number_of_labels = get_paths_and_count()
    train_ds, test_ds = split_paths(all_images)

    print("Test size: {}".format(len(test_ds)))
    print("Train size: {}".format(len(train_ds)))
    num_images = len(all_images)
    print("Number of images: {}".format(num_images))
    print("Labels: {}".format(number_of_labels))

    model = Model(number_of_labels)
    train(model)
    print("")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
