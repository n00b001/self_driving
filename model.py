import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

from consts import MODEL_DIR, IMAGE_SIZE, IMAGE_DEPTH
from dataset_keras import process_image_np
from file_stuff import get_random_str, get_labels, get_latest_dir
# from x2_2_train_net import get_base_model, get_model
from grab_screen import grab_screen

QUEUE_SIZE = 1


def predict_tflite(input_data, input_details, model, output_details):
    model.set_tensor(input_details[0]['index'], input_data)
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])
    return output_data


def get_model(base_model, num_classes):
    # Trainable classification head
    maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
    # Layer classification head with feature detector
    model = tf.keras.Sequential([
        base_model,
        maxpool_layer,
        prediction_layer
    ])
    return model


def get_base_model():
    print("Getting model...")
    # # Pre-trained model with MobileNetV2
    # base_model = tf.keras.applications.MobileNetV2(
    #     input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH),
    #     include_top=False,
    #     alpha=1.0,
    #     weights='imagenet'

    # Pre-trained model with MobileNet
    base_model = tf.keras.applications.MobileNet(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH),
        include_top=False,
        alpha=1.0,
        weights='imagenet'
    )
    # # Pre-trained model with MobileNetV2
    # base_model = tf.keras.applications.InceptionResNetV2(
    #     input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH),
    #     include_top=False,
    #     weights='imagenet'
    # )
    # Freeze the pre-trained model weights
    base_model.trainable = True
    return base_model


def get_empty_model(num_classes):
    base_model = get_base_model()
    model = get_model(base_model=base_model, num_classes=num_classes)
    return model


def load_model(path, num_classes):
    model = get_empty_model(num_classes)
    model.load_weights(path)
    return model


class Model:
    def __init__(
            self,
            all_image_paths=None,
            num_classes=None,
            encoder=None,
            model_path=None,
            verbose=False,
            predict=False,
    ):
        if not predict and all_image_paths is None:
            raise Exception("Must give all image paths to model for training")
        self.verbose = verbose
        self.all_image_paths = all_image_paths
        self.model_path = model_path
        self.encoder = encoder

        if predict:
            self.encoder = pickle.load(open(model_path + "/encoder_file.p", "rb"))
            num_classes = pickle.load(open(model_path + "/num_classes.p", "rb"))
            self.model = self.get_model(
                model_path=model_path, num_classes=num_classes
            )

            # self.input_queue = Queue(maxsize=QUEUE_SIZE)
            # self.output_queue = Queue(maxsize=QUEUE_SIZE)

            # We set the generator thread as daemon
            # (see https://docs.python.org/3/library/threading.html#threading.Thread.daemon)
            # This means that when all other threads are dead,
            # this thread will not prevent the Python program from exiting
            # self.prediction_thread = Thread(target=self.predict_from_queue, daemon=True)
            # self.prediction_thread.start()
        else:
            encoder = LabelBinarizer()
            self.model = self.get_model(
                model_path=model_path, num_classes=num_classes
            )
            self.all_image_labels = encoder.fit_transform(get_labels(all_image_paths))
            # self.label_weight_list = get_label_weights(self.all_image_labels, self.class_examples)

            pickle.dump(encoder, open(model_path + "encoder_file.p", "wb"))
            pickle.dump(num_classes, open(model_path + "num_classes.p", "wb"))

    def generate_from_queue(self):
        """ Generator which yields items from the input queue.
        This lives within our 'prediction thread'.
        """

        while True:
            yield process_image_np(self.input_queue.get())

    def predict_from_queue(self):
        """ Adds a prediction from the model to the output_queue.
        This lives within our 'prediction thread'.
        Note: estimators accept generators as inputs and return generators as output.
        Here, we are iterating through the output generator, which will be
        populated in lock-step with the input generator.
        """

        for i in self.model.predict(input_fn=self.queued_predict_input_fn):
            if self.verbose:
                print('Putting in output queue')
            self.output_queue.put(i)

    def predict(self, features=None):
        # self.input_queue.put(features)
        # predictions = self.output_queue.get()
        #
        # output = predictions["class_ids"][0]
        if features is None:
            features = grab_screen()
        output = self.model.predict(
            np.expand_dims(process_image_np(features.astype(np.float32)), axis=0),
            workers=8,
            use_multiprocessing=True
        )
        label = self.encoder.inverse_transform(output)
        return str(label[0])

    def get_model(self, num_classes, model_path=None):

        if model_path is None:
            model_path = "./{}/{}".format(MODEL_DIR, get_random_str())
        if not os.path.exists(MODEL_DIR):
            os.mkdir(MODEL_DIR)

        if os.path.exists(model_path):
            latest = get_latest_dir(model_path)
            saved_model = get_latest_dir(latest)
            paths = [
                model_path,
                latest,
                saved_model,
            ]
            model = None
            for p in paths:
                try:
                    model = tf.contrib.saved_model.load_keras_model(p)
                    break
                except:
                    try:
                        model = load_model(p, num_classes)
                        break
                    except:
                        pass
            if model is None:
                raise Exception("Model couldn't be loaded!")
        else:
            model = get_empty_model(num_classes)

        print("Model path: {}".format(model_path))
        self.model_path = model_path
        return model

    def queued_predict_input_fn(self):
        """
        Queued version of the `predict_input_fn` in FlowerClassifier.
        Instead of yielding a dataset from data as a parameter,
        we construct a Dataset from a generator,
        which yields from the input queue.
        """

        # Fetch the inputs from the input queue
        output_types = {
            'x': tf.float32
        }

        return tf.data.Dataset.from_generator(self.generate_from_queue,
                                              output_types=output_types)
