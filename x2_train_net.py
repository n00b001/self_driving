import os
import pathlib
import pickle
import random

import tensorflow as tf

from consts import DATA_DIR, IMAGE_SIZE, BATCH_SIZE, EPOCHS, MODEL_DIR, IMAGE_DEPTH, SHUFFLE_BUFFER
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
        architecture = [256, 32]
        with open('./{}/architecture.pkl'.format(model_path), 'wb') as f:
            pickle.dump(architecture, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open('./{}/architecture.pkl'.format(model_path), 'rb') as f:
            architecture = pickle.load(f)

    print("Model path: {}".format(model_path))
    feature_columns = [tf.feature_column.numeric_column("x", shape=[IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH])]
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=architecture,
        optimizer=tf.train.AdamOptimizer(1e-3),
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
    image = tf.math.divide(image, 255)  # normalize to [0,1] range

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
    # if bool(random.getrandbits(1)):
    #     image = tf.image.random_flip_up_down(image)
    # if bool(random.getrandbits(1)):
    #     image = tf.image.random_brightness(image)
    # if bool(random.getrandbits(1)):
    #     image = tf.image.random_contrast(image)
    # if bool(random.getrandbits(1)):
    #     image = tf.image.random_crop(image)
    # if bool(random.getrandbits(1)):
    #     image = tf.image.random_hue(image)
    # if bool(random.getrandbits(1)):
    #     image = tf.image.random_jpeg_quality(image)
    # if bool(random.getrandbits(1)):
    #     image = tf.image.random_saturation(image)
    return image


# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label(path, label):
    # return load_and_preprocess_image(path), label
    return dict({"x": load_and_preprocess_image(path)}), label


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
        map_func=load_and_preprocess_from_path_label,
        batch_size=BATCH_SIZE,
        num_parallel_batches=8,
    ))

    ds = ds.prefetch(buffer_size=BATCH_SIZE)
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


def train(model, model_path):
    for i in range(EPOCHS):
        print("Starting epoch: {}".format(i + 1))
        model.train(
            input_fn=lambda: get_dataset(is_training=True, model_path=model_path),
            # steps=MAX_TRAINING_STEPS
        )
        print("Evaluating...")
        stats = model.evaluate(
            input_fn=lambda: get_dataset(is_training=False, model_path=model_path)
        )
        print("Stats: {}".format(stats))


def main():
    all_images, number_of_labels = get_paths_and_count()
    train_ds, test_ds = split_paths(all_images)
    print("Test size: {}".format(len(test_ds)))
    print("Train size: {}".format(len(train_ds)))
    num_images = len(all_images)
    print("Number of images: {}".format(num_images))
    print("Labels: {}".format(number_of_labels))
    model, model_path = get_model(len(number_of_labels.keys()))
    train(model, model_path)
    print("")


if __name__ == '__main__':
    main()
