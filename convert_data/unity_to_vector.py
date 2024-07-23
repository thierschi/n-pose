import cv2
import numpy as np

from import_data.unity_data import UnityData, UnityKeypointValue, UnityInstanceSegmentationInstance
from util.polygons import get_colored_polygons_from_mask, get_polygon_boundary, simplify_polygon_group


def convert_kps_to_vector(kp_value: UnityKeypointValue, v_size: int):
    vector = np.zeros(v_size)

    for i, kp in enumerate(kp_value.keypoints):
        if i * 2 >= v_size:
            break

        vector[i * 2] = kp.camera_cartesian_location[0]
        vector[i * 2 + 1] = kp.camera_cartesian_location[1]

    return vector


def convert_iseg_to_vector(instance: UnityInstanceSegmentationInstance, map: cv2.typing.MatLike, v_size: int):
    assert v_size % 2 == 0
    target = v_size / 2

    try:
        _, polygons = get_colored_polygons_from_mask(map, instance.color)
    except NotImplementedError:
        return None

    simple_polygon = simplify_polygon_group(polygons, target=50, tolerance=1e-10)
    x, y = get_polygon_boundary(simple_polygon)

    if len(x) > target:
        return None

    vector = np.zeros(v_size)
    for i, x in enumerate(x):
        if i * 2 >= v_size:
            break

        vector[i * 2] = x
        vector[i * 2 + 1] = y[i]

    return vector


def convert_position_to_vector(data: UnityData):
    return None


def get_vectors_from_unity_capture(data: UnityData, folder_path: str, file_name: str):
    return None
