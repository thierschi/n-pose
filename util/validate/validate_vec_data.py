from pathlib import Path

import cv2
import numpy as np
from progressbar.bar import ProgressBar

from import_data.unity_data import UnityData

# create colors array with 20 distinguable colors

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (0, 64, 0),
    (0, 0, 64),
    (64, 64, 0),
    (64, 0, 64),
    (0, 64, 64),
]


def draw_keypoints(image, keypoints, color=(0, 255, 0), width=1920, height=1080, radius=5):
    for i in range(len(keypoints) // 2):
        x = int(keypoints[i * 2] * width)
        y = int(keypoints[i * 2 + 1] * height)
        cv2.circle(image, (x, y), radius, color, -1)

    return image


def draw_segmentation(image, segmentation, color=(0, 255, 0), width=1920, height=1080, line_width=3):
    boundary = []
    for i in range(len(segmentation) // 2):
        x = int(segmentation[i * 2] * width)
        y = int(segmentation[i * 2 + 1] * height)
        boundary.append(np.array([x, y]).astype(np.int32))
    pts = np.array(boundary, np.int32)

    cv2.polylines(image, [pts], True, color, line_width)

    return image


def get_newest_file(folder_path: str, file_extension: str):
    folder_path = Path(folder_path)
    file_list = list(folder_path.glob(f'*{file_extension}'))
    if len(file_list) == 0:
        return None

    return max(file_list, key=lambda x: x.stat().st_ctime)


file_path = get_newest_file("../../data/solo_95_vec", ".csv")

data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

unity_data = UnityData('../../data/solo_95')

sequences = np.unique(data[:, 0].astype(int))

with ProgressBar(max_value=len(sequences)) as bar:
    for sequence in sequences:
        bar.next()
        rows = data[data[:, 0] == sequence]

        right_img = cv2.imread(
            [c for c in unity_data.captures if c.sequence == sequence and c.id == "RightCam"][0].file_path)
        left_img = cv2.imread(
            [c for c in unity_data.captures if c.sequence == sequence and c.id == "LeftCam"][0].file_path)

        for i, row in enumerate(rows):
            pos = row[1:4]
            rot = row[4:8]
            kp_l = row[8:18]
            kp_r = row[118:128]
            seg_l = row[18:118]
            seg_r = row[128:228]

            # remove zero values from each array
            pos = pos[pos != 0]
            rot = rot[rot != 0]
            kp_l = kp_l[kp_l != 0]
            kp_r = kp_r[kp_r != 0]
            seg_l = seg_l[seg_l != 0]
            seg_r = seg_r[seg_r != 0]

            color = colors[i % len(colors)]
            right_img = draw_keypoints(right_img, kp_r, color=color)
            # right_img = draw_segmentation(right_img, seg_r, color=color)

            left_img = draw_keypoints(left_img, kp_l, color=color)
            # left_img = draw_segmentation(left_img, seg_l, color=color)

        cv2.imwrite(f"../../data/vec_tests/{sequence}_right.png", right_img)
        cv2.imwrite(f"../../data/vec_tests/{sequence}_left.png", left_img)
