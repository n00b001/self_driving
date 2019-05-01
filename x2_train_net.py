import os
import pathlib
import pickle
import random

import tensorflow as tf
from tensorflow.contrib.learn import DNNClassifier
from tensorflow.python.data.experimental import AUTOTUNE

from consts import DATA_DIR, IMAGE_SIZE, BATCH_SIZE, MAX_TRAINING_STEPS, EPOCHS, MODEL_DIR, IMAGE_DEPTH, SHUFFLE_BUFFER
from file_stuff import get_random_str

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
random.seed = 1337


def get_model(num_classes, model_path=None):
    if model_path is None:
        model_path = "./{}/{}".format(MODEL_DIR, get_random_str())
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if not os.path.exists(model_path):
        os.mkdir(model_path)
        architecture = [256]
        with open('./{}/architecture.pkl'.format(model_path), 'wb') as f:
            pickle.dump(architecture, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open('./{}/architecture.pkl'.format(model_path), 'rb') as f:
            architecture = pickle.load(f)

    print("Model path: {}".format(model_path))
    feature_columns = [tf.feature_column.numeric_column("x", shape=[IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH])]
    classifier = DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=architecture,
        optimizer=tf.train.AdamOptimizer(1e-4),
        n_classes=num_classes,
        activation_fn=tf.nn.leaky_relu,
        dropout=0.5,
        model_dir=model_path
    )
    return classifier, model_path


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range
    return image


# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label(path, label):
    # return load_and_preprocess_image(path), label
    return dict({"x": load_and_preprocess_image(path)}), label


def get_dataset(is_training, model_path):
    data_root = pathlib.Path(DATA_DIR)
    all_image_paths, image_count = get_paths_and_count(data_root)
    with open('./{}/image_count.pkl'.format(model_path), 'wb') as f:
        pickle.dump(image_count, f, pickle.HIGHEST_PROTOCOL)

    train_size = int(0.8 * image_count)
    if is_training:
        all_image_paths = all_image_paths[:train_size]
    else:
        all_image_paths = all_image_paths[train_size:]

    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    with open('./{}/label_to_index.pkl'.format(model_path), 'wb') as f:
        pickle.dump(label_to_index, f, pickle.HIGHEST_PROTOCOL)
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

    ds = ds.map(load_and_preprocess_from_path_label)

    if is_training:
        ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=SHUFFLE_BUFFER))
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        ds = ds.make_one_shot_iterator()
        ds = ds.get_next()
        return ds
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    ds = ds.make_one_shot_iterator()
    ds = ds.get_next()
    return ds


def get_paths_and_count(data_root):
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    image_count = len(all_image_paths)
    print("Number of items: {}".format(image_count))
    return all_image_paths, image_count


def train(model, model_path):
    for i in range(EPOCHS):
        print("Starting epoch: {}".format(i))
        model.fit(
            input_fn=lambda: get_dataset(is_training=True, model_path=model_path),
            steps=MAX_TRAINING_STEPS
        )
        stats = model.evaluate(
            input_fn=lambda: get_dataset(is_training=False, model_path=model_path)
        )
        print("Stats: {}".format(stats))


def main():
    data_root = pathlib.Path(DATA_DIR)
    _, num_classes = get_paths_and_count(data_root)
    model, model_path = get_model(num_classes)
    train(model, model_path)
    print("")


if __name__ == '__main__':
    main()
