import os
import shutil
from typing import List

from progressbar import ProgressBar

from data.unity_data import Capture, UnityData
from util import generate_random_split
from .data_directory import create_yolo_data_dir


def get_pose_annotations_for_capture(capture: Capture, precision: int = 6):
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
            keypoint_x = round(keypoint.location[0] / img_width, precision)
            keypoint_y = round(keypoint.location[1] / img_height, precision)
            keypoint_visible = 1 if keypoint.state == 2 else 0
            keypoints.append(f"{keypoint_x} {keypoint_y} {keypoint_visible}")

        annotations.append(f"{bounding_box.label.id} {center_x} {center_y} {width} {height} {' '.join(keypoints)}")

    return annotations


def unity_to_yolo_pose(unity_data: UnityData, _path: str, kpts: int, flip_idx: List[int], include_test: bool = False,
                       precision: int = 6):
    path, yaml_path = create_yolo_data_dir(_path, include_test)

    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(_path)}\n")
        f.write(f"train: train\n")
        f.write(f"val: val\n")
        if include_test:
            f.write(f"test: test\n")

        f.write('\n')
        f.write(f'kpt_shape: [{kpts},3]\n')
        f.write(f'flip_idx: {flip_idx}\n')

        f.write("names:\n")
        for label in unity_data.labels:
            f.write(f"    {label.id}: {label.name}\n")

    valid_captures = [capture for capture in unity_data.captures if
                      capture.bounding_boxes_2d is not None and capture.keypoints is not None]

    split = generate_random_split(len(valid_captures), 0.1, 0.1 if include_test else 0.0)

    with ProgressBar(max_value=len(split)) as bar:
        for i, s in enumerate(split):
            bar.update(i)

            capture = valid_captures[i]

            shutil.copyfile(capture.file_path,
                            os.path.join(path.as_posix(), s, f'{i}.{capture.file_path.split(".")[-1]}'))

            annotations = get_pose_annotations_for_capture(capture)
            with open(os.path.join(path.as_posix(), s, f'{i}.txt'), "w") as f:
                for annotation in annotations:
                    f.write(f"{annotation}\n")

    return yaml_path
