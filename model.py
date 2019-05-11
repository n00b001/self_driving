import os
import pickle
from queue import Queue
from threading import Thread

import tensorflow as tf
from tensorflow_estimator.python.estimator.run_config import RunConfig

from consts import MODEL_DIR, DEFAULT_ARCHITECTURE, IMAGE_SIZE, IMAGE_DEPTH
from file_stuff import get_random_str

QUEUE_SIZE = 1


class Model:
    def __init__(self, class_examples: dict, label_to_index: dict, model_path=None, verbose=False, dropout=0.5):
        self.verbose = verbose
        self.class_examples = class_examples
        self.label_to_index = label_to_index
        index_to_label = {v: k for (k, v) in self.label_to_index.items()}
        self.index_to_label = index_to_label
        self.model_path = model_path
        self.model = self.get_model(
            model_path=model_path, dropout=dropout, label_to_index=label_to_index
        )
        self.input_queue = Queue(maxsize=QUEUE_SIZE)
        self.output_queue = Queue(maxsize=QUEUE_SIZE)

        # We set the generator thread as daemon
        # (see https://docs.python.org/3/library/threading.html#threading.Thread.daemon)
        # This means that when all other threads are dead,
        # this thread will not prevent the Python program from exiting
        self.prediction_thread = Thread(target=self.predict_from_queue, daemon=True)
        self.prediction_thread.start()

    def generate_from_queue(self):
        """ Generator which yields items from the input queue.
        This lives within our 'prediction thread'.
        """

        while True:
            yield self.input_queue.get()

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

    def predict(self, features):
        features = dict(features)
        self.input_queue.put(features)
        predictions = self.output_queue.get()

        output = predictions["class_ids"][0]
        # todo: this will break if we use string labels
        # label = self.index_to_label[int(output)]
        return str(output)

    def get_model(self, label_to_index, model_path=None, dropout=0.5):
        if model_path is None:
            model_path = "./{}/{}".format(MODEL_DIR, get_random_str())
        if not os.path.exists(MODEL_DIR):
            os.mkdir(MODEL_DIR)

        if not os.path.exists(model_path):
            os.mkdir(model_path)
            architecture = DEFAULT_ARCHITECTURE
            with open('./{}/architecture.pkl'.format(model_path), 'wb') as f:
                pickle.dump(architecture, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open('./{}/architecture.pkl'.format(model_path), 'rb') as f:
                architecture = pickle.load(f)

        print("Model path: {}".format(model_path))
        feature_columns = [tf.feature_column.numeric_column(
            "x", shape=[IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH],
            normalizer_fn=lambda x: tf.math.divide(x, 255)
        )]

        r = RunConfig(
            # log_step_count_steps=50,
            log_step_count_steps=500,
            save_summary_steps=1_000,
            keep_checkpoint_max=2,
        )

        # classifier = mobilenetv2.get_estimator(
        #     model_dir=model_path, num_classes=len(self.index_to_label.keys())
        # )
        classifier = self.get_estimator(architecture, dropout, feature_columns, model_path, label_to_index, r)
        self.model_path = model_path
        return classifier

    def get_estimator(self, architecture, dropout, feature_columns, model_path, label_to_index, r):
        weight = tf.feature_column.numeric_column('w')
        label_list = list(label_to_index.keys())
        classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=architecture,
            optimizer=tf.train.AdamOptimizer(1e-3),
            n_classes=len(label_list),
            activation_fn=tf.nn.leaky_relu,
            dropout=dropout,
            model_dir=model_path,
            config=r,
            weight_column=weight,
            label_vocabulary=label_list
        )
        return classifier

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
