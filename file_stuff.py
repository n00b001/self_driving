import os
import pathlib
import random
import string
from multiprocessing import Process

import cv2

from consts import DATA_DIR, LOG_DIR

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)


def save_to_files(resized_screen, keys):
    p = Process(target=save_func, args=(resized_screen, keys))
    p.start()


def get_log_dir():
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    return os.path.join(LOG_DIR, get_random_str())


def save_func(resized_screen, keys):
    for i in range(len(keys)):
        frame_screen = resized_screen[i]
        frame_keys = keys[i]
        frame_keys = [k for k in frame_keys if k != "B"]
        if len(frame_keys) == 0:
            frame_keys_str = "NO"
        else:
            frame_keys_str = "".join(frame_keys)
        output_path = os.path.join(DATA_DIR, frame_keys_str)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        random_name = get_random_str() + ".jpg"
        file_name = os.path.join(output_path, random_name)
        cv2.imwrite(file_name, frame_screen, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print("Saved!")


def get_random_str():
    return "".join([random.choice(string.ascii_uppercase) for _ in range(10)])


def split_paths(some_list):
    if some_list is None:
        return None, None
    train_split_index = int(0.8 * len(some_list))
    train_ds = some_list[:train_split_index]
    test_ds = some_list[train_split_index:]
    return train_ds, test_ds


def is_good_data(item):
    if not item.is_dir() \
            and item.suffix == ".jpg" \
            and os.stat(str(item)).st_size:
        return True
    return False


def get_paths_and_count():
    print("Getting paths and count...")
    data_root = pathlib.Path(DATA_DIR)
    label_names = [item for item in data_root.glob('*/') if item.is_dir()]
    all_file_names = {
        d.name: [f for f in d.glob("*") if is_good_data(f)]
        for d in label_names
    }
    class_examples = {k: len(v) for k, v in all_file_names.items()}
    threshold = max(class_examples.values()) * 0.01
    for k, v in class_examples.items():
        if v < threshold:
            del all_file_names[k]
    class_examples = {k: len(v) for k, v in all_file_names.items()}

    all_image_paths = []
    for x in all_file_names.values():
        all_image_paths.extend([str(y) for y in x])
    random.shuffle(all_image_paths)
    return all_image_paths, class_examples


def get_labels(all_image_paths):
    return [pathlib.Path(path).parent.name for path in all_image_paths]


def get_label_weights(all_image_labels, class_examples):
    """
    this will generate weights from 2 ~ 1, where 1 is the most common label and 2 is the least common label
    """
    label_weight = get_weight_lookup(class_examples)
    label_weight_list = [label_weight[l] for l in all_image_labels]
    return label_weight_list


def get_weight_lookup(class_examples):
    max_val = max(class_examples.values())
    label_weight = {k: 2.0 - (v / max_val) for k, v in class_examples.items()}
    return label_weight
