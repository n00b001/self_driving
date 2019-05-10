import os
import time

from consts import SAVE_LIMIT
from file_stuff import save_to_files
from grab_keys import grab_keys
from grab_screen import grab_screen

# import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
recording = False


def fps_stuff2(counter, start_time, x):
    counter += 1
    if (time.time() - start_time) > x:
        print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()
    return counter, start_time


def handle_keys(keys):
    global recording

    if len(keys) > 0:
        if len(keys) == 1:
            if keys == ["B"]:
                recording = True
                print(f"Recoding: {recording}")
            elif keys == ["N"]:
                recording = False
                print(f"Recoding: {recording}")


def display_keys(keys):
    if len(keys) > 0:
        print(keys)


def main():
    start_time = time.time()
    x = 1
    counter = 0

    feature_data = []
    target_data = []

    while True:
        screen = grab_screen()
        keys = grab_keys()
        handle_keys(keys)
        if recording:
            feature_data.append(screen)
            target_data.append(keys)
            if len(feature_data) > SAVE_LIMIT:
                save_to_files(feature_data, target_data)
                feature_data = []
                target_data = []
        counter, start_time = fps_stuff2(counter, start_time, x)


if __name__ == '__main__':
    main()
