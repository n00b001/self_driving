import os
import pathlib
import pickle
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

from consts import DATA_DIR, BATCH_SIZE, EPOCHS, SHUFFLE_BUFFER, MAX_TRAINING_STEPS
from grab_screen import grab_screen
from model import Model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
random.seed = 1337


def load_and_preprocess_image(path, training):
    image = tf.image.decode_jpeg(tf.read_file(path), channels=3)
    if training:
        return preprocess_image(image, training)
    return image


def preprocess_image(image, training=True):
    # image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE])
    # image = tf.math.divide(image, 255)  # normalize to [0,1] range

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


# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label_train(path, label):
    # return load_and_preprocess_image(path), label
    return to_dict(load_and_preprocess_image(path, True)), label


def to_dict(x):
    return {"x": x}


# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label_test(path, label):
    # return load_and_preprocess_image(path), label
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
        # ds = ds.repeat()
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
        # model.train(
        #     input_fn=lambda: get_dataset(is_training=True, model_path=model_path),
        #     steps=MAX_TRAINING_STEPS
        # )
        # print("Evaluating...")
        # stats = model.evaluate(
        #     input_fn=lambda: get_dataset(is_training=False, model_path=model_path)
        # )
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

        # feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)

        # x = {"x": tf.placeholder(shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH], dtype=tf.float32, name="x")}
        # # serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(x)
        # serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features=x)
        # receiver_tensor = {
        #     "receiver": x
        # }
        # features = {
        #     'feature': tf.image.resize_images(receiver_tensor['x'], [IMAGE_SIZE, IMAGE_SIZE])
        # }
        # print("Saving as SavedModel...")
        # model.export_savedmodel(
        #     model_path,
        #     lambda: tf.estimator.export.ServingInputReceiver(features, receiver_tensor),
        #     # strip_default_attrs=True,
        #     as_text=True
        # )
        # tf.get_variable_scope().reuse_variables()
        # predictions = model.predict(input_fn=get_screen_dict)  # , yield_single_examples=True)
        # output = list(predictions)[0]
        # label = index_to_label[int(output)]
        # return label

#
# def get_screen_dict():
#     if False:
#         image = tf.convert_to_tensor(grab_screen())
#         pre_processed = preprocess_image(image, training=False)
#         pre_processed = tf.expand_dims(pre_processed, axis=0)
#         return to_dict(pre_processed)
#     # return {"x": np.expand_dims(np.divide(grab_screen(), 255), axis=0)}
#     return {"x": np.expand_dims(np.zeros((96, 96, 3), dtype=np.float32), axis=0)}, None


def main(_):
    all_images, number_of_labels = get_paths_and_count()
    train_ds, test_ds = split_paths(all_images)

    print("Test size: {}".format(len(test_ds)))
    print("Train size: {}".format(len(train_ds)))
    num_images = len(all_images)
    print("Number of images: {}".format(num_images))
    print("Labels: {}".format(number_of_labels))

    model = Model(number_of_labels)
    # model, model_path = get_model(len(number_of_labels.keys()))
    train(model)
    print("")


if __name__ == '__main__':
    # main()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
