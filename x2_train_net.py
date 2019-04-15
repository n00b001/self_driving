import os
import pathlib
import pickle
import random

import tensorflow as tf
from tensorflow.contrib.learn import DNNClassifier
from tensorflow.python.data.experimental import AUTOTUNE

from consts import DATA_DIR, IMAGE_SIZE, BATCH_SIZE, MAX_TRAINING_STEPS, EPOCHS, MODEL_DIR, IMAGE_DEPTH, SHUFFLE_BUFFER
from file_stuff import get_random_str

random.seed = 1337


# def get_dataset():
#     data_root = pathlib.Path(DATA_DIR)
#     all_image_paths = list(data_root.glob('*/*'))
#     all_image_paths = [str(path) for path in all_image_paths]
#     random.shuffle(all_image_paths)
#     all_image_paths_train = all_image_paths[:, int(len(all_image_paths) * 0.8)]
#     all_image_paths_test = all_image_paths[int(len(all_image_paths) * 0.8):]
#     train = tf.data.Dataset.from_tensor_slices(all_image_paths_train)
#     test = tf.data.Dataset.from_tensor_slices(all_image_paths_test)
#     return train, test, len(os.listdir(DATA_DIR))
#
# def get_dataset_keras():
#     datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#         rescale=1. / 255,
#         validation_split=0.2)
#
#     train_generator = datagen.flow_from_directory(
#         DATA_DIR,
#         target_size=(IMAGE_SIZE, IMAGE_SIZE),
#         batch_size=BATCH_SIZE,
#         subset='training')
#
#     val_generator = datagen.flow_from_directory(
#         DATA_DIR,
#         target_size=(IMAGE_SIZE, IMAGE_SIZE),
#         batch_size=BATCH_SIZE,
#         subset='validation')
#     return train_generator, val_generator
#
#
# def get_model(num_classes):
#     # Create the base model from the pre-trained model MobileNet V2
#     base_model = tf.keras.applications.MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH),
#                                                    include_top=False,
#                                                    weights='imagenet')
#     base_model.trainable = False
#     model = tf.keras.Sequential([
#         base_model,
#         tf.keras.layers.Conv2D(32, 3, activation='relu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.Dense(num_classes, activation='softmax')
#     ])
#     model.compile(optimizer=tf.keras.optimizers.Adam(),
#                   loss='categorical_crossentropy',
#                   # metrics=['accuracy']
#                   )
#     model = tf.keras.estimator.model_to_estimator(model)
#     return model

# def train_model_keras(model, train_generator, val_generator, callback):
#     model.summary()
#     len(model.trainable_variables)
#     history = model.fit(
#         train_generator,
#         epochs=EPOCHS,
#         validation_data=val_generator,
#         callbacks=[callback]
#     )
#     return history
# def display_history_keras(history):
#     acc = history.history['accuracy']
#     val_acc = history.history['val_accuracy']
#
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#
#     plt.figure(figsize=(8, 8))
#     plt.subplot(2, 1, 1)
#     plt.plot(acc, label='Training Accuracy')
#     plt.plot(val_acc, label='Validation Accuracy')
#     plt.legend(loc='lower right')
#     plt.ylabel('Accuracy')
#     plt.ylim([min(plt.ylim()), 1])
#     plt.title('Training and Validation Accuracy')
#
#     plt.subplot(2, 1, 2)
#     plt.plot(loss, label='Training Loss')
#     plt.plot(val_loss, label='Validation Loss')
#     plt.legend(loc='upper right')
#     plt.ylabel('Cross Entropy')
#     plt.ylim([0, 1.0])
#     plt.title('Training and Validation Loss')
#     plt.xlabel('epoch')
#     plt.show()
# def setup_tensorboard_keras():
#     return tf.keras.callbacks.TensorBoard(
#         log_dir=get_log_dir(), histogram_freq=0, batch_size=BATCH_SIZE, write_graph=True, write_grads=False,
#         write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
#         embeddings_data=None, update_freq='epoch'
#     )

def get_model(num_classes, model_path=None):
    if model_path is None:
        model_path = "./{}/{}".format(MODEL_DIR, get_random_str())
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    feature_columns = [tf.feature_column.numeric_column("x", shape=[IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH])]
    classifier = DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[256, 32],
        optimizer=tf.train.AdamOptimizer(1e-4),
        n_classes=num_classes,
        activation_fn=tf.nn.leaky_relu,
        dropout=0.5,
        model_dir=model_path
    )
    return classifier


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range
    return image


# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label(path, label):
    # return load_and_preprocess_image(path), label
    return dict({"x": load_and_preprocess_image(path)}), label


def get_dataset(is_training):
    data_root = pathlib.Path(DATA_DIR)
    all_image_paths, image_count = get_paths_and_count(data_root)
    with open('image_count.pkl', 'wb') as f:
        pickle.dump(image_count, f, pickle.HIGHEST_PROTOCOL)

    train_size = int(0.8 * image_count)
    if is_training:
        all_image_paths = all_image_paths[:train_size]
    else:
        all_image_paths = all_image_paths[train_size:]

    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print("Label to index:\n{}".format(label_to_index))
    with open('label_to_index.pkl', 'wb') as f:
        pickle.dump(label_to_index, f, pickle.HIGHEST_PROTOCOL)
    # all_image_labels = [pathlib.Path(path).parent.name for path in all_image_paths]
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

    ds = ds.map(load_and_preprocess_from_path_label)

    if is_training:
        # ds = ds.take(train_size)

        ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=SHUFFLE_BUFFER))
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        ds = ds.make_one_shot_iterator()
        ds = ds.get_next()
        return ds
    # ds = ds.skip(train_size)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    ds = ds.make_one_shot_iterator()
    ds = ds.get_next()
    return ds


def get_paths_and_count(data_root):
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    image_count = len(all_image_paths)
    print("Number of items: {}".format(image_count))
    return all_image_paths, image_count


def train(model):
    for i in range(EPOCHS):
        print("Starting epoch: {}".format(i))
        # model.fit(input_fn=train_generator, steps=MAX_TRAINING_STEPS)
        model.fit(
            input_fn=lambda: get_dataset(is_training=True),
            steps=MAX_TRAINING_STEPS
        )
        stats = model.evaluate(
            input_fn=lambda: get_dataset(is_training=False)
        )
        print("Stats: {}".format(stats))


def main():
    data_root = pathlib.Path(DATA_DIR)
    _, num_classes = get_paths_and_count(data_root)
    model = get_model(num_classes)
    train(model)
    print("")


if __name__ == '__main__':
    main()
