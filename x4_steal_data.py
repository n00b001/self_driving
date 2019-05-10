import os
import random
import traceback
from multiprocessing.pool import ThreadPool

import h5py
import numpy as np

from consts import SAVE_LIMIT
from file_stuff import save_to_files


def main():
    folder = 'D:\data'
    list_files = [os.path.join(folder, file) for file in os.listdir(folder)]
    with ThreadPool(processes=16) as pool:
        pool.map(func=parse_file, iterable=list_files)
        # for file in os.listdir(folder):
        #     filename = os.path.join(folder, file)
        #     f = h5py.File(filename, 'r')
        #     Process(target=parse_file, args=(f)).start()
        # parse_file(f, feature_data, target_data)
    print("")


def parse_file(f):
    f = h5py.File(f, 'r')
    target_data = []
    feature_data = []
    # S, W, A/D
    for x, y in zip(f["features"], f["targets"]):
        try:
            ad_val = y[2]
            w_val = y[1]
            a_val = 0 if ad_val > 0.0 else ad_val * -2.0
            s_val = y[0]
            d_val = 0 if ad_val < 0.0 else ad_val * 2.0
            is_w = random.random() < w_val
            is_a = random.random() < a_val
            is_s = random.random() < s_val
            is_d = random.random() < d_val

            output = ""
            if is_w:
                output += "W"
            if is_a:
                output += "A"
            if is_s:
                output += "S"
            if is_d:
                output += "D"
            output = "".join(sorted(output))
            if output == "":
                output = "NO"

            x = (x * 255.0).astype(np.uint8)
            feature_data.append(x)
            target_data.append(output)
            if len(feature_data) > SAVE_LIMIT:
                save_to_files(feature_data, target_data)
                feature_data = []
                target_data = []
        except Exception as e:
            traceback.print_exc()
            print(e)


if __name__ == '__main__':
    main()
