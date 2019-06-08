import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras.optimizer_v2.adam import Adam

from consts import MODEL_DIR, IMAGE_SIZE, IMAGE_DEPTH, BATCH_SIZE, LEARNING_RATE
from dataset_keras import process_image_np, get_datasets
from file_stuff import get_random_str, get_labels, get_latest_dir
# from x2_2_train_net import get_base_model, get_model
from grab_screen import grab_screen


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
        self.new_model = False
        self.num_classes = num_classes
        self.all_image_labels = None

        self.model = self.load_or_create_model(
            num_classes=self.num_classes
        )

        self.model.summary()
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE, decay=1e-10),
            # optimizer=tf.train.AdamOptimizer(learning_rate=lr_finetune),
            metrics=['accuracy']
        )

    def predict(self, features=None):
        if features is None:
            features = grab_screen()
        output = self.model.predict(
            np.expand_dims(process_image_np(features.astype(np.float32)), axis=0),
            workers=8,
            use_multiprocessing=True
        )
        label = self.encoder.inverse_transform(output)
        return str(label[0])

    def train(self, epochs):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.model_path,
            write_graph=False,
            histogram_freq=0,
            write_images=False,
            batch_size=BATCH_SIZE,
            update_freq=1_000,
            embeddings_freq=0,

        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.model_path, "weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
            monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=False, mode='auto', period=1
        )

        train_ds, test_ds, steps_per_epoch, val_steps = get_datasets(self.all_image_paths, self.all_image_labels)
        history = self.model.fit(
            train_ds.repeat(),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=test_ds.repeat(),
            callbacks=[tensorboard_callback, checkpoint],
            # sample_weight=weights,
            # class_weight=class_weights,
            verbose=1,
            validation_steps=val_steps,
            workers=8,
            shuffle=False,
            use_multiprocessing=True
        )

        tf.contrib.saved_model.save_keras_model(self.model, self.model_path)
        return history

    def load_or_create_model(self, num_classes):
        if self.model_path is None:
            self.model_path = "./{}/{}".format(MODEL_DIR, get_random_str())

        if os.path.exists(self.model_path):
            latest = get_latest_dir(self.model_path)
            saved_model = get_latest_dir(latest)
            paths = [
                self.model_path,
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

            self.encoder = pickle.load(open(self.model_path + "/encoder_file.p", "rb"))
            self.num_classes = pickle.load(open(self.model_path + "/num_classes.p", "rb"))
        else:
            os.mkdir(self.model_path)
            model = get_empty_model(num_classes)
            self.new_model = True
            self.encoder = LabelBinarizer()
            self.all_image_labels = self.encoder.fit_transform(get_labels(self.all_image_paths))
            # self.label_weight_list = get_label_weights(self.all_image_labels, self.class_examples)

            pickle.dump(self.encoder, open(self.model_path + "/encoder_file.p", "wb"))
            pickle.dump(self.num_classes, open(self.model_path + "/num_classes.p", "wb"))

        print("Model path: {}".format(self.model_path))
        return model
