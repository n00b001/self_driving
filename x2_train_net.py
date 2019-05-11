import os
import pathlib
import random
import traceback

import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

from consts import BATCH_SIZE, EPOCHS, SHUFFLE_BUFFER, MAX_TRAINING_STEPS, IMAGE_SIZE, DATA_DIR, CACHE_LOCATION
from model import Model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
random.seed = 1337


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
    # w = weight_table.lookup(label)
    return to_dict(x, weight), label


def to_dict(x, w):
    return {
        "x": x,
        "w": w
    }


def get_dataset(is_training, class_examples):
    # global number_of_labels
    all_image_paths, _ = get_paths_and_count()

    train_ds, test_ds = split_paths(all_image_paths)
    if is_training:
        all_image_paths = train_ds
    else:
        all_image_paths = test_ds

    # label_to_index = dict((name, index) for index, name in enumerate(number_of_labels.keys()))
    # with open('./{}/label_to_index.pkl'.format(model_path), 'wb') as f:
    #     pickle.dump(label_to_index, f, pickle.HIGHEST_PROTOCOL)

    all_image_labels = [pathlib.Path(path).parent.name for path in all_image_paths]
    # all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

    max_val = max(class_examples.values())
    label_weight = {k: 2.0 - (v / max_val) for k, v in class_examples.items()}
    # keys = tf.convert_to_tensor(list(label_weight.keys()))
    # values = tf.convert_to_tensor(list(label_weight.values()))
    # weight_table = tf.contrib.lookup.HashTable(
    #     tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1
    # )
    label_weight_list = [label_weight[l] for l in all_image_labels]

    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels, label_weight_list))

    ds = ds.apply(tf.data.experimental.map_and_batch(
        map_func=load_and_preprocess_from_path_label,
        batch_size=BATCH_SIZE,
        num_parallel_batches=os.cpu_count(),
    ))
    ds = ds.apply(tf.data.experimental.ignore_errors())
    # todo: this cache is a real pain in the ASS
    # ds = ds.cache(filename=os.path.join(CACHE_LOCATION, "cache.tf"))
    if is_training:
        ds = ds.map(
            map_func=img_augmentation,
            num_parallel_calls=os.cpu_count()
        )
        ds = ds.shuffle(SHUFFLE_BUFFER)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    # todo: ValueError: Failed to create a one-shot iterator for a dataset.
    # `Dataset.make_one_shot_iterator()` does not support datasets that capture stateful objects,
    #   such as a `Variable` or `LookupTable`. In these cases, use `Dataset.make_initializable_iterator()`.
    ds = ds.make_one_shot_iterator()
    # ds = ds.make_initializable_iterator()
    # tf.get_default_session().run(ds.initializer)
    ds = ds.get_next()
    return ds


def split_paths(all_image_paths):
    train_split_index = int(0.8 * len(all_image_paths))
    train_ds = all_image_paths[:train_split_index]
    test_ds = all_image_paths[train_split_index:]
    return train_ds, test_ds


def is_good_data(item):
    if not item.is_dir() \
            and item.suffix == ".jpg" \
            and os.stat(str(item)).st_size:
        return True
    return False


def get_paths_and_count():
    print("Getting paths and count...")
    data_root = pathlib.Path(DATA_DIR)
    label_names = [item for item in data_root.glob('*/') if item.is_dir()]
    all_file_names = {
        d.name: [f for f in d.glob("*") if is_good_data(f)]
        for d in label_names
    }
    class_examples = {k: len(v) for k, v in all_file_names.items()}
    threshold = max(class_examples.values()) * 0.01
    for k, v in class_examples.items():
        if v < threshold:
            del all_file_names[k]
    class_examples = {k: len(v) for k, v in all_file_names.items()}

    all_image_paths = []
    for x in all_file_names.values():
        all_image_paths.extend([str(y) for y in x])
    random.shuffle(all_image_paths)
    return all_image_paths, class_examples


def train(model: Model):
    for i in range(EPOCHS):
        print("Starting epoch: {}".format(i + 1))
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: get_dataset(is_training=True, class_examples=model.class_examples),
            max_steps=MAX_TRAINING_STEPS,
        )

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: get_dataset(is_training=False, class_examples=model.class_examples),
            steps=None,
        )
        stats = tf.estimator.train_and_evaluate(model.model, train_spec, eval_spec)
        print("Stats: {}".format(stats))


def main(_):
    os.makedirs(CACHE_LOCATION, exist_ok=True)
    all_images, class_examples = get_paths_and_count()
    train_ds, test_ds = split_paths(all_images)

    print("Test size: {}".format(len(test_ds)))
    print("Train size: {}".format(len(train_ds)))
    num_images = len(all_images)
    print("Number of images: {}".format(num_images))
    print("Labels: {}".format(class_examples))

    label_to_index = dict((name, index) for index, name in enumerate(class_examples.keys()))
    print("label_to_index: {}".format(label_to_index))

    model = Model(class_examples=class_examples,
                  label_to_index=label_to_index)
    train(model)
    print("")


if __name__ == '__main__':
    try:
        print("Starting up training...")
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.app.run(main)
    except Exception as e:
        [os.remove(os.path.join(CACHE_LOCATION, x)) for x in os.listdir(CACHE_LOCATION)]
        traceback.print_exc()
        print(e)
