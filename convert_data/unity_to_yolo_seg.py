import os
import shutil
from pathlib import Path

import cv2

from import_data.unity_data import UnityData
from util.polygons import get_colored_polygons_from_mask
from util.split import get_split


# TODO Make code more readable and refactor
# TODO Check outputs like -1e-05 -> Test if YOLO cares


def unity_to_yolo_seg(unity_data: UnityData, _path: str, include_test: bool = False,
                      precision: int = 6):
    if _path[-1] == "/":
        _path = _path[:-1]

    name = _path.split("/")[-1]

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
        f.write(f"path: {os.path.abspath(_path)}\n")
        f.write(f"train: train\n")
        f.write(f"val: val\n")
        if include_test:
            f.write(f"test: test\n")

        f.write("names:\n")
        for label in unity_data.labels:
            f.write(f"    {label.id}: {label.name}\n")

    valid_captures = [capture for capture in unity_data.captures if capture.instance_segmentation is not None]

    split = get_split(len(valid_captures), 0.1, 0.1 if include_test else 0.0)

    for i, s in enumerate(split):
        capture = valid_captures[i]

        shutil.copyfile(capture.file_path, os.path.join(path.as_posix(), s, f'{i}.{capture.file_path.split(".")[-1]}'))

        with open(os.path.join(path.as_posix(), s, f'{i}.txt'), "w") as f:
            segmentation_map_path = valid_captures[i].instance_segmentation.file_path
            instances = valid_captures[i].instance_segmentation.instances

            for instance in instances:
                img = cv2.imread(segmentation_map_path)
                _, norm_polygons = get_colored_polygons_from_mask(img, instance.color)

                annotation = f'{instance.label.id}'
                for polygon in norm_polygons:
                    boundary_x = polygon.boundary.coords.xy[0]
                    boundary_y = polygon.boundary.coords.xy[1]

                    coord_pairs = []
                    for n, x in enumerate(boundary_x):
                        y = boundary_y[n]
                        coord_pairs.append(f'{x} {y}')

                    annotation += f' {" ".join(coord_pairs)}'

                f.write(f"{annotation}\n")

    return yaml_path
