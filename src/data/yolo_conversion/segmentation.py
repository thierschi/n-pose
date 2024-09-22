import os
import shutil

import cv2
from progressbar import ProgressBar

from .data_directory import create_yolo_data_dir
from ..unity_data import UnityData
from ...util import generate_random_split, detect_colored_polygons


def unity_to_yolo_seg(unity_data: UnityData, _path: str, include_test: bool = False,
                      precision: int = 6) -> str:
    """
    Convert Unity data to YOLO segmentation format.
    :param unity_data: The data to convert
    :param _path: Path to the directory to save the data to
    :param include_test: Whether to include a test set
    :param precision: Floating point precision
    :return: Path to the yaml file
    """
    path, yaml_path = create_yolo_data_dir(_path, include_test)

    with open(yaml_path, "w") as f:
        # Write metadata
        f.write(f"path: {os.path.abspath(_path)}\n")
        f.write(f"train: train\n")
        f.write(f"val: val\n")
        if include_test:
            f.write(f"test: test\n")

        f.write("names:\n")
        for label in unity_data.labels:
            f.write(f"    {label.id}: {label.name}\n")

    # Collect captures with valid annotations
    valid_captures = [capture for capture in unity_data.captures if capture.instance_segmentation is not None]

    split = generate_random_split(len(valid_captures), 0.1, 0.1 if include_test else 0.0)

    with ProgressBar(max_value=len(split)) as bar:
        for i, s in enumerate(split):
            bar.update(i)

            capture = valid_captures[i]

            # Copy the image file
            shutil.copyfile(capture.file_path, os.path.join(path, s, f'{i}.{capture.file_path.split(".")[-1]}'))

            with open(os.path.join(path, s, f'{i}.txt'), "w") as f:
                segmentation_map_path = valid_captures[i].instance_segmentation.file_path
                img = cv2.imread(segmentation_map_path)

                instances = valid_captures[i].instance_segmentation.instances

                # Detect each instance on map and convert the polygon to string
                for instance in instances:
                    _, norm_polygons = detect_colored_polygons(img, instance.color, precision=precision)

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
