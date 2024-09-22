import os
from pathlib import Path


def create_yolo_data_dir(path: str, include_test: bool = False) -> tuple[str, str]:
    """
    Creates the directory structure for YOLO data.
    :param path: folder path
    :param include_test: Are test images included?
    :return: Tuple: The path to the created directory and the path to the yaml file
    """
    if path[-1] == "/":
        path = path[:-1]

    name = path.split("/")[-1]

    if os.path.exists(path) and not os.path.isdir(path):
        raise Exception("Path is not a directory")

    # Create new directory if path already exists
    if os.path.exists(path):
        i = 1
        while os.path.exists(f"{path}_{i}"):
            i += 1
        path = f"{path}_{i}"

    # Create directory and parents
    path = Path(path)
    path.mkdir(parents=True)

    # Create data directories
    os.mkdir(f"{path}/train")
    os.mkdir(f"{path}/val")
    if include_test:
        os.mkdir(f"{path}/test")

    yaml_path = f"{path}/{name}.yaml"

    return path.as_posix(), yaml_path
