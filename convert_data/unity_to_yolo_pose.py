import os
import shutil
from pathlib import Path

import numpy as np

from import_data.unity_data import UnityData, UnityCapture


# TODO Make code more readable and refactor
# TODO Check outputs like -1e-05

def get_split(n: int, val: float, test: float):
    n_val = int(n * val)
    n_test = int(n * test)
    n_train = n - n_val - n_test

    split = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    np.random.shuffle(split)

    return split


def get_yolo_pose_annotations(capture: UnityCapture, precision: int = 6):
    annotations = []
    img_width = capture.dimension[0]
    img_height = capture.dimension[1]

    for bounding_box in capture.bounding_boxes_2d.values:
        center_x = round((bounding_box.origin[0] + bounding_box.dimension[0] / 2) / img_width, precision)
        center_y = round((bounding_box.origin[1] + bounding_box.dimension[1] / 2) / img_height, precision)
        width = round(bounding_box.dimension[0] / img_width, precision)
        height = round(bounding_box.dimension[1] / img_height, precision)

        keypoint_annotation = next(
            (kp for kp in capture.keypoints.values if kp.instance_id == bounding_box.instance_id), None)
        keypoints = []

        if keypoint_annotation is None:
            continue

        for keypoint in keypoint_annotation.keypoints:
            keypoint_x = round(keypoint.camera_cartesian_location[0] / img_width, precision)
            keypoint_y = round(keypoint.camera_cartesian_location[1] / img_height, precision)
            keypoint_visible = keypoint.state
            keypoints.append(f"{keypoint_x} {keypoint_y} {keypoint_visible}")

        annotations.append(f"{bounding_box.label.id} {center_x} {center_y} {width} {height} {' '.join(keypoints)}")

    return annotations


def unity_to_yolo_pose(unity_data: UnityData, _path: str, include_test: bool = False, precision: int = 6):
    if _path[-1] == "/":
        _path = _path[:-1]

    name = _path.split("/")[-1]

    if not name.isalnum():
        raise Exception("Name contains illegal characters")

    if os.path.exists(_path) and not os.path.isdir(_path):
        raise Exception("Path is not a directory")

    if os.path.exists(_path):
        i = 1
        while os.path.exists(f"{_path}_{i}"):
            i += 1
        _path = f"{_path}_{i}"

    path = Path(_path)
    path.mkdir(parents=True)

    os.mkdir(f"{_path}/train")
    os.mkdir(f"{_path}/val")
    if include_test:
        os.mkdir(f"{_path}/test")

    yaml_path = f"{_path}/{name}.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {_path}\n")
        f.write(f"train: train\n")
        f.write(f"val: val\n")
        if include_test:
            f.write(f"test: test\n")

        f.write("\n")

        f.write("names:")
        for label in unity_data.labels:
            f.write(f"    {label.id}: {label.name}\n")

    valid_captures = [capture for capture in unity_data.captures if
                      capture.bounding_boxes_2d is not None and capture.keypoints is not None]

    split = get_split(len(valid_captures), 0.1, 0.1 if include_test else 0.0)

    for i, s in enumerate(split):
        capture = valid_captures[i]

        shutil.copyfile(capture.file_path, os.path.join(path.as_posix(), s, f'{i}.{capture.file_path.split(".")[-1]}'))

        annotations = get_yolo_pose_annotations(capture)
        with open(os.path.join(path.as_posix(), s, f'{i}.txt'), "w") as f:
            for annotation in annotations:
                f.write(f"{annotation}\n")
