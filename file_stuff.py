import os
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
