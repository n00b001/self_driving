import os
import pickle
import time
import traceback
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import class_weight
from tensorflow.python.keras.optimizer_v2.adam import Adam

from consts import IMAGE_SIZE, EPOCHS, IMAGE_DEPTH, BATCH_SIZE, MODEL_DIR, FINE_TUNE_EPOCHS, LEARNING_RATE
from dataset_keras import get_raw_ds, process_image_np
from file_stuff import get_paths_and_count, get_labels, split_paths, get_label_weights, get_random_str, get_latest_dir
from grab_screen import grab_screen
from x1_collect_data import fps_stuff2
from x3_inferance import press_label

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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


def train(
        train_ds, test_ds, model, steps_per_epoch, learning_rate, model_path,
        tensorboard_callback, weights, class_weights, val_steps, checkpoint
):
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        # optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=get_metrics()
    )

    print("Training...")
    model.summary()
    history = model.fit(train_ds.repeat(),
                        epochs=EPOCHS,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=test_ds.repeat(),
                        validation_steps=val_steps,
                        callbacks=[tensorboard_callback, checkpoint],
                        # class_weight=class_weights,
                        # sample_weight=weights,
                        # verbose=2,
                        verbose=1,
                        workers=1,
                        use_multiprocessing=False
                        # class_weight=class_weights
                        )

    tf.contrib.saved_model.save_keras_model(model, model_path)
    # model.save(model_path + ".full_model.h5")
    return history


def show_graph(history, name="Course"):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label=f'{name} Training Accuracy')
    plt.plot(val_acc, label=f'{name} Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel(f'{name} Accuracy')
    plt.title(f'{name} Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label=f'{name} Training Loss')
    plt.plot(val_loss, label=f'{name} Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title(f'{name} Training and Validation Loss')
    plt.xlabel('epoch')


def fine_tune(
        model, train_ds, test_ds, steps_per_epoch, total_epochs, fine_model_path,
        tensorboard_callback, weights, class_weights, val_steps, checkpoint
):
    model.summary()
    history = model.fit(train_ds.repeat(),
                        steps_per_epoch=steps_per_epoch,
                        epochs=total_epochs,
                        # initial_epoch=EPOCHS,
                        validation_data=test_ds.repeat(),
                        callbacks=[tensorboard_callback, checkpoint],
                        # sample_weight=weights,
                        # class_weight=class_weights,
                        # verbose=2,
                        verbose=1,
                        validation_steps=val_steps,
                        workers=8,
                        use_multiprocessing=True)

    tf.contrib.saved_model.save_keras_model(model, fine_model_path)
    # model.save(fine_model_path + ".full_model.h5")
    return history


def setup_for_fine_tune(base_model, model, learning_rate=0.001):
    # Unfreeze all layers of MobileNetV2
    base_model.trainable = True
    # Refreeze layers until the layers we want to fine-tune
    # for layer in base_model.layers[:25]:
    # for layer in base_model.layers[:25]:
    # layer.trainable = False
    # Use a lower learning rate
    # lr_finetune = learning_rate / 10.0
    # Recompile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate),
                  # optimizer=tf.train.AdamOptimizer(learning_rate=lr_finetune),
                  metrics=get_metrics()
                  )
    return model


def get_metrics():
    return ['accuracy']


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model, None


def predict(fine_model_path, num_classes, encoder):
    latest = get_latest_dir(fine_model_path)
    tf_lite_path = os.path.join(fine_model_path, "converted_model.tflite")
    if os.path.exists(tf_lite_path):
        model = tf.lite.Interpreter(model_path=tf_lite_path)
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(latest)
        model = converter.convert()
        open(tf_lite_path, "wb").write(model)
        model = tf.lite.Interpreter(model_path=tf_lite_path)
    model.allocate_tensors()

    start_time = time.time()
    x = 1
    counter = 0
    # Get input and output tensors.
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    print(f"Input: {input_details}")
    print(f"Output: {output_details}")
    while True:
        scr = grab_screen()
        scr_typed = scr.astype(np.float32)
        img = process_image_np(scr_typed)
        output = predict_tflite([img], input_details, model, output_details)
        label = encoder.inverse_transform(output)[0]
        press_label(label)
        counter, start_time = fps_stuff2(counter, start_time, x)


def predict_tflite(input_data, input_details, model, output_details):
    model.set_tensor(input_details[0]['index'], input_data)
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])
    return output_data


def main():
    # Increase training epochs for fine-tuning
    total_epochs = EPOCHS + FINE_TUNE_EPOCHS

    random_str = get_random_str()
    random_str = "TSZIATHCIA"
    model_base_dir = os.path.join(MODEL_DIR, random_str)
    os.makedirs(model_base_dir, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, random_str, f'weights_epoch_{EPOCHS}')
    fine_model_path = os.path.join(MODEL_DIR, random_str, f'fine_weights_epoch_0_{total_epochs}')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=model_base_dir,
        write_graph=True,
        histogram_freq=0,
        write_images=True,
        batch_size=BATCH_SIZE,
        update_freq=1_000,
        embeddings_freq=0,

    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_base_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
        monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=False, mode='auto', period=1
    )

    encoder_file = os.path.join(MODEL_DIR, random_str, "encoder_file.p")
    num_classes_file = os.path.join(MODEL_DIR, random_str, "num_classes.p")

    # if True:
    if False and not os.path.exists(fine_model_path):
        print(f"{fine_model_path} doesn't exist...")
        all_images, class_examples = get_paths_and_count()

        num_classes = len(class_examples.keys())
        labels = get_labels(all_images)
        weights = get_label_weights(labels, class_examples)
        class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)

        encoder = LabelBinarizer()
        transfomed_label = encoder.fit_transform(labels)

        pickle.dump(encoder, open(encoder_file, "wb"))
        pickle.dump(num_classes, open(num_classes_file, "wb"))

        train_ds, test_ds = get_datasets(all_images, transfomed_label)

        xtr, xte = split_paths(all_images)
        steps_per_epoch = round(len(xtr)) // BATCH_SIZE
        val_steps = round(len(xte)) // BATCH_SIZE
        print(f"Steps per epoch: {steps_per_epoch}")

        if not os.path.exists(model_path):
            print(f"Tuning {model_path}...")
            base_model = get_base_model()
            model = get_model(base_model=base_model, num_classes=num_classes)
            model = setup_for_fine_tune(base_model, learning_rate=LEARNING_RATE, model=model)
            for i in range(10):
                fine_model_path = os.path.join(MODEL_DIR, random_str, f'fine_weights_epoch_{i}_{total_epochs}')
                history2 = fine_tune(
                    model, train_ds, test_ds, steps_per_epoch, total_epochs,
                    fine_model_path, tensorboard_callback, weights, class_weights,
                    val_steps, checkpoint
                )

    encoder = pickle.load(open(encoder_file, "rb"))
    num_classes = pickle.load(open(num_classes_file, "rb"))
    print(f"Predicting {fine_model_path}...")
    predict(fine_model_path, num_classes=num_classes, encoder=encoder)


def get_class_weights(y, smooth_factor=0):
    """
    We can't use this yet because:
        our one-hot encoded labels aren't hashable (they are a list)
        We can't use the raw string label because:
            keras wants a float label
            :(

    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {cls: float(majority) / count for cls, count in counter.items()}


def get_datasets(x, y):
    xtr, xte = split_paths(x)
    ytr, yte = split_paths(y)
    train_ds = get_raw_ds(
        is_training=True,
        all_image_paths=xtr,
        all_image_labels=ytr
    )
    test_ds = get_raw_ds(
        is_training=False,
        all_image_paths=xte,
        all_image_labels=yte
    )
    return train_ds, test_ds


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()
