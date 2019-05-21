import os
import pickle
import time
import traceback
from collections import Counter

import keras_metrics
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import class_weight

from consts import IMAGE_SIZE, EPOCHS, VAL_STEPS, IMAGE_DEPTH, BATCH_SIZE, MODEL_DIR, FINE_TUNE_EPOCHS, LEARNING_RATE
from dataset_keras import get_raw_ds, process_image_np
from file_stuff import get_paths_and_count, get_labels, split_paths, get_random_str, get_label_weights
from grab_screen import grab_screen
from x1_collect_data import fps_stuff2

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
    # Pre-trained model with MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH),
        include_top=False,
        weights='imagenet'
    )
    # Freeze the pre-trained model weights
    base_model.trainable = False
    return base_model


def train(
        train_ds, test_ds, model, steps_per_epoch, learning_rate, model_path,
        tensorboard_callback, weights, class_weights
):
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=get_metrics()
                  )

    print("Training...")
    model.summary()
    history = model.fit(train_ds.repeat(),
                        epochs=EPOCHS,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=test_ds.repeat(),
                        validation_steps=VAL_STEPS,
                        callbacks=[tensorboard_callback],
                        class_weight=class_weights,
                        # sample_weight=weights,
                        verbose=2,
                        # class_weight=class_weights
                        )

    tf.contrib.saved_model.save_keras_model(model, model_path)
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
        tensorboard_callback, weights, class_weights
):
    # Fine-tune model
    # Note: Set initial_epoch to begin training after epoch 30 since we
    # previously trained for 30 epochs.
    model.summary()
    history = model.fit(train_ds.repeat(),
                        steps_per_epoch=steps_per_epoch,
                        epochs=total_epochs,
                        initial_epoch=EPOCHS,
                        validation_data=test_ds.repeat(),
                        callbacks=[tensorboard_callback],
                        # sample_weight=weights,
                        class_weight=class_weights,
                        verbose=2,
                        validation_steps=VAL_STEPS)

    tf.contrib.saved_model.save_keras_model(model, fine_model_path)
    return history


def setup_for_fine_tune(base_model, model, learning_rate=0.001):
    # Unfreeze all layers of MobileNetV2
    base_model.trainable = True
    # Refreeze layers until the layers we want to fine-tune
    for layer in base_model.layers[:100]:
        layer.trainable = False
    # Use a lower learning rate
    lr_finetune = learning_rate / 10.0
    # Recompile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=lr_finetune),
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

        # img_tensor = tf.convert_to_tensor(scr_typed)
        # img_tensor = process_image(img_tensor)
        # with tf.Session() as sess:
        #     img2 = sess.run(img_tensor)
        img = process_image_np(scr_typed)

        # scr_resiz = np.expand_dims(
        #     np.divide(
        #         cv2.resize(
        #             grab_screen(), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR),
        #         255.0),
        #     axis=0).astype(np.float32)
        output = predict_tflite([img], input_details, model, output_details)
        print(f"Output: {output[0]}")
        label = encoder.inverse_transform(output)[0]
        print(f"label: {label}")
        counter, start_time = fps_stuff2(counter, start_time, x)


def predict_tflite(input_data, input_details, model, output_details):
    model.set_tensor(input_details[0]['index'], input_data)
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])
    return output_data


def get_latest_dir(direct):
    all_subdirs = [os.path.join(direct, d) for d in os.listdir(direct) if os.path.isdir(os.path.join(direct, d))]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    return latest_subdir


def main():
    # Increase training epochs for fine-tuning
    total_epochs = EPOCHS + FINE_TUNE_EPOCHS

    random_str = get_random_str()
    # random_str = "IDBULYIUOJ"
    model_base_dir = os.path.join(MODEL_DIR, random_str)
    os.makedirs(model_base_dir, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, random_str, f'weights_epoch_{EPOCHS}')
    fine_model_path = os.path.join(MODEL_DIR, random_str, f'fine_weights_epoch_{total_epochs}')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=model_base_dir,
        write_graph=True,
        histogram_freq=0,
        write_images=True,
        batch_size=BATCH_SIZE,
        update_freq='batch'
    )

    encoder_file = os.path.join(MODEL_DIR, random_str, "encoder_file.p")
    num_classes_file = os.path.join(MODEL_DIR, random_str, "num_classes.p")

    if not os.path.exists(fine_model_path):
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
        print(f"Steps per epoch: {steps_per_epoch}")

        if not os.path.exists(model_path):
            print(f"Tuning {model_path}...")
            base_model = get_base_model()
            model = get_model(base_model=base_model, num_classes=num_classes)
            history1 = train(
                train_ds, test_ds, model, steps_per_epoch, LEARNING_RATE,
                model_path, tensorboard_callback, weights, class_weights
            )
            model = setup_for_fine_tune(base_model, learning_rate=LEARNING_RATE, model=model)
            history2 = fine_tune(
                model, train_ds, test_ds, steps_per_epoch, total_epochs,
                fine_model_path, tensorboard_callback, weights, class_weights
            )
            show_graph(history1)
            show_graph(history2, name="Fine")
            plt.show()
        else:
            print(f"Fine tuning {fine_model_path}...")
            base_model = get_base_model()
            model = get_model(base_model=base_model, num_classes=num_classes)
            model.load_weights(model_path)
            model = setup_for_fine_tune(base_model, learning_rate=LEARNING_RATE, model=model)
            history2 = fine_tune(
                model, train_ds, test_ds, steps_per_epoch, total_epochs,
                fine_model_path, tensorboard_callback, weights, class_weights
            )
            show_graph(history2, name="Fine")
            plt.show()

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
