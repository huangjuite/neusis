import os
import cv2
import pickle
import json
import math
import numpy as np
from scipy.io import savemat
from tqdm import tqdm

def get_radar_fft(filename):
    radar_fft = np.fromfile(filename, dtype=np.float32)
    byte_len = radar_fft.shape[0]
    radar_fft = radar_fft[: byte_len // 2] + 1j * radar_fft[byte_len // 2 :]
    radar_fft = radar_fft.reshape((257, 64, 232))
    radar_fft = np.abs(radar_fft) ** 0.4
    radar_fft = np.linalg.norm(radar_fft, axis=1)  # find magnitude
    return radar_fft.T


def load_data_radar(target):
    dirpath = "./data/{}".format(target)
    infos = "{}/radar_pose_pair.pkl".format(dirpath)
    output_loc = "{}/UnzipData".format(dirpath)

    if not os.path.exists(output_loc):
        os.makedirs(output_loc)
    images = []
    sensor_poses = []

    with open(infos, "rb") as f:
        data = pickle.load(f)
        files = data["files"]
        sensor_poses = data["poses"]

        for f in tqdm(files):
            d = get_radar_fft(f)/5000
            images.append(d)

        # heuristic filtering
        # s = image.shape
        # image[image < 0.2] = 0
        # image[s[0]- 200:, :] = 0

    data = {
        "images": images,
        "images_no_noise": [],
        "sensor_poses": sensor_poses,
        "min_range": 0.5856,
        "max_range": 27.6397,
        "hfov": 180.0,
        "vfov": 40.0,
    }

    savemat("{}/{}.mat".format(dirpath, target), data, oned_as="row")
    return data


if __name__ == "__main__":
    load_data_radar("nsh_atrium")
