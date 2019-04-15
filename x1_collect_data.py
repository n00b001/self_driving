import time

import cv2

from consts import IMAGE_SIZE, SAVE_LIMIT
from file_stuff import save_to_files
from grab_keys import grab_keys
from grab_screen import grab_screen, display_screen

recording = False


def resize(screen):
    colourized = cv2.cvtColor(screen, cv2.COLOR_RGBA2RGB)
    img = cv2.resize(colourized, (IMAGE_SIZE, IMAGE_SIZE))
    return img


def fps_stuff(fps_num, fps_total, frames, start):
    frames += 1.0
    if time.time() > start + 1.0:
        start = time.time()
        fps_total += frames
        frames = 0
        fps_num += 1.0
    if fps_total > 0:
        print("FPS: {}".format(int(fps_total / fps_num)))
    return frames, start


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
    frames = 0
    start = time.time()
    fps_num = 0
    fps_total = 0
    feature_data = []
    target_data = []

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    while True:
        screen = grab_screen()
        resized_screen = resize(screen)
        keys = grab_keys()
        handle_keys(keys)
        if not recording:
            # display_screen(screen)
            display_screen(resized_screen)
            # display_keys(keys)
        else:
            feature_data.append(resized_screen)
            target_data.append(keys)
            if len(feature_data) > SAVE_LIMIT:
                save_to_files(feature_data, target_data)
                feature_data = []
                target_data = []
        frames, start = fps_stuff(fps_num, fps_total, frames, start)


if __name__ == '__main__':
    main()
