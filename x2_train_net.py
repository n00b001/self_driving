import os
import random
import traceback

import tensorflow as tf

from consts import EPOCHS, CACHE_LOCATION, LEARNING_RATE
from file_stuff import get_paths_and_count, split_paths
from model import Model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
random.seed = 1337


def main(_):
    model_path = None
    while True:
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
            num_classes=len(class_examples.keys()),
            model_path=model_path
        )
        model.train(EPOCHS)
        model_path = model.model_path
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
