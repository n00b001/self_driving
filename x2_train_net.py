import os
import random
import traceback

import tensorflow as tf

from consts import EPOCHS, MAX_TRAINING_STEPS, CACHE_LOCATION
from dataset import get_initialized_dataset
from file_stuff import get_paths_and_count, split_paths
from model import Model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
random.seed = 1337


def train(model: Model):
    path_train, path_test = split_paths(model.all_image_paths)
    label_train, label_test = split_paths(model.all_image_labels)
    # weight_train, weight_test = split_paths(model.label_weight_list)

    for i in range(EPOCHS):
        print("Starting epoch: {}".format(i + 1))
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: get_initialized_dataset(
                is_training=True,
                all_image_paths=path_train,
                all_image_labels=label_train,
                label_weight_list=weight_train,
            ),
            max_steps=MAX_TRAINING_STEPS,
        )

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: get_initialized_dataset(
                is_training=False,
                all_image_paths=path_test,
                all_image_labels=label_test,
            ),
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

    model = Model(
        all_image_paths=all_images,
        num_classes=len(class_examples.keys())
    )
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
