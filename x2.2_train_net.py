import os
from collections import Counter

import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

from consts import IMAGE_SIZE, EPOCHS, VAL_STEPS, IMAGE_DEPTH, BATCH_SIZE, MODEL_DIR
from dataset_keras import get_raw_ds
from file_stuff import get_paths_and_count, get_labels, split_paths

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_model(num_classes):
    print("Getting model...")
    # Pre-trained model with MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH),
        include_top=False,
        weights='imagenet'
    )
    # Freeze the pre-trained model weights
    base_model.trainable = False

    # Trainable classification head
    maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    # Layer classification head with feature detector
    model = tf.keras.Sequential([
        base_model,
        maxpool_layer,
        prediction_layer
    ])
    learning_rate = 0.0001
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy']
                  )
    return model


def train(train_ds, test_ds, model, steps_per_epoch):
    print("Training...")
    history = model.fit(train_ds.repeat(),
                        epochs=EPOCHS,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=test_ds.repeat(),
                        validation_steps=VAL_STEPS,
                        verbose=2,
                        # class_weight=class_weights
                        )

    model.save_weights(os.path.join(MODEL_DIR, f'weights_epoch_{EPOCHS}.h5'))
    return history


def show_graph(history):
    import matplotlib.pyplot as plt
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

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


def main():
    all_images, class_examples = get_paths_and_count()
    num_classes = len(class_examples.keys())
    labels = get_labels(all_images)
    encoder = LabelBinarizer()
    transfomed_label = encoder.fit_transform(labels)

    train_ds, test_ds = get_datasets(all_images, transfomed_label)
    model = get_model(num_classes=num_classes)

    xtr, xte = split_paths(all_images)
    steps_per_epoch = round(len(xtr)) // BATCH_SIZE
    history = train(train_ds, test_ds, model, steps_per_epoch)
    show_graph(history)


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
