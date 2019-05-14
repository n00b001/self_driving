import os
from collections import Counter

import cv2
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

from consts import IMAGE_SIZE, EPOCHS, VAL_STEPS, IMAGE_DEPTH, BATCH_SIZE, MODEL_DIR
from dataset_keras import get_raw_ds
from file_stuff import get_paths_and_count, get_labels, split_paths
from grab_screen import grab_screen
import numpy as np
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_model(base_model, num_classes):
    # Trainable classification head
    maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='sigmoid')
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


def train(train_ds, test_ds, model, steps_per_epoch, learning_rate, model_path):
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy']
                  )

    print("Training...")
    model.summary()
    history = model.fit(train_ds.repeat(),
                        epochs=EPOCHS,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=test_ds.repeat(),
                        validation_steps=VAL_STEPS,
                        # verbose=1,
                        # class_weight=class_weights
                        )

    model.save_weights(model_path)
    return history


def show_graph(history):
    import matplotlib.pyplot as plt
    print(history)
    print(history.history)
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def fine_tune(model, train_ds, test_ds, steps_per_epoch, total_epochs, fine_model_path):
    # Fine-tune model
    # Note: Set initial_epoch to begin training after epoch 30 since we
    # previously trained for 30 epochs.
    model.summary()
    history = model.fit(train_ds.repeat(),
                        steps_per_epoch=steps_per_epoch,
                        epochs=total_epochs,
                        initial_epoch=EPOCHS,
                        validation_data=test_ds.repeat(),
                        validation_steps=VAL_STEPS)

    model.save_weights(fine_model_path)
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
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=lr_finetune),
                  metrics=['accuracy'])
    return model


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model, None


def predict(fine_model_path, num_classes, encoder):
    base_model = get_base_model()
    model = get_model(base_model=base_model, num_classes=num_classes)
    model = setup_for_fine_tune(base_model=base_model, model=model)
    model.load_weights(fine_model_path)
    model.summary()
    while True:
        scr = grab_screen()
        resized = cv2.resize(scr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        scr_resiz = np.expand_dims(resized, axis=0)
        output = model.predict(scr_resiz)
        print(f"Output: {output}")
        label = encoder.inverse_transform(output)
        print(f"label: {label}")


def main():
    # Increase training epochs for fine-tuning
    fine_tune_epochs = 2
    total_epochs = EPOCHS + fine_tune_epochs

    model_path = os.path.join(MODEL_DIR, f'weights_epoch_{EPOCHS}.h5')
    fine_model_path = os.path.join(MODEL_DIR, f'fine_weights_epoch_{total_epochs}.h5')
    learning_rate = 0.0001

    encoder_file = os.path.join(MODEL_DIR, "encoder_file.p")
    num_classes_file = os.path.join(MODEL_DIR, "num_classes.p")

    if not os.path.exists(fine_model_path):
        all_images, class_examples = get_paths_and_count()
        num_classes = len(class_examples.keys())
        labels = get_labels(all_images)
        encoder = LabelBinarizer()
        transfomed_label = encoder.fit_transform(labels)

        pickle.dump(encoder, open(encoder_file, "wb"))
        pickle.dump(num_classes, open(num_classes_file, "wb"))

        train_ds, test_ds = get_datasets(all_images, transfomed_label)

        xtr, xte = split_paths(all_images)
        steps_per_epoch = round(len(xtr)) // BATCH_SIZE

        if not os.path.exists(model_path):
            print("Tuning...")
            base_model = get_base_model()
            model = get_model(base_model=base_model, num_classes=num_classes)
            history = train(train_ds, test_ds, model, steps_per_epoch, learning_rate, model_path)
            show_graph(history)
        else:
            print("Fine tuning...")
            base_model = get_base_model()
            model = get_model(base_model=base_model, num_classes=num_classes)
            model.load_weights(model_path)
            model = setup_for_fine_tune(base_model, learning_rate=learning_rate, model=model)
            history = fine_tune(model, train_ds, test_ds, steps_per_epoch, total_epochs, fine_model_path)
            show_graph(history)

    encoder = pickle.load(open(encoder_file, "rb"))
    num_classes = pickle.load(open(num_classes_file, "rb"))
    print("Predicting...")
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
    main()
