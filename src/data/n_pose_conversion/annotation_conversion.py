from typing import List

import cv2
import numpy as np

from ..unity_data import Keypoint, KeypointValue, InstanceSegmentationInstance
from ...util import get_keypoint_usability, get_object_direction
from ...util import get_polygon_boundary, detect_colored_polygons, simplify_polygon_group
from ...util import normalize_vector
from ...util import point_to_world


def keypoints_to_vector(kp_value: KeypointValue, v_size: int, image_size=(1920, 1080)) -> np.ndarray:
    """
    Convert keypoint annotation to vector

    :param kp_value: An instance's keypoints
    :param v_size: Target vector size
    :param image_size: Dimension of the image to norm keypoint locations
    :return: KeypointValue as a vector
    """
    vector = np.zeros(v_size)

    for i, kp in enumerate(kp_value.keypoints):
        if i * 2 >= v_size:
            break

        vector[i * 2] = kp.location[0] / image_size[0]
        vector[i * 2 + 1] = kp.location[1] / image_size[1]

    return vector


def i_segmentation_to_vector(instance: InstanceSegmentationInstance, seg_map: cv2.typing.MatLike,
                             v_size: int) -> np.ndarray | None:
    """
    Convert instance segmentation annotation to vector

    :param instance: Instance of instance seg annotation
    :param seg_map:  map on which the instance is present
    :param v_size: Target vector size
    :return: InstanceSegmentationInstance as a vector, None if failed
    """
    assert v_size % 2 == 0
    target = v_size // 2

    # Detect normed polygons on map that match the instance's color
    _, polygons = detect_colored_polygons(seg_map, instance.color)

    try:
        simple_polygon = simplify_polygon_group(polygons, target=v_size // 2, tolerance=1e-10)
        x, y = get_polygon_boundary(simple_polygon)
    except NotImplementedError:
        return None

    if len(x) > target:
        return None

    vector = np.zeros(v_size)
    for i, x in enumerate(x):
        if i * 2 >= v_size:
            break

        vector[i * 2] = x
        vector[i * 2 + 1] = y[i]

    return vector


def position_and_rotation_to_vector(obj_kps: List[Keypoint], camera_pos, camera_rot) -> np.ndarray | None:
    """
    Get position and orientation of object using instance's keypoints

    :param obj_kps: The instance's keypoints
    :param camera_pos: The camera's position in the capture
    :param camera_rot: The camera's rotation in the capture
    :return: A vector representing the object's position and orientation, None if failed
    """
    obj_kps.sort(key=lambda x: x.index)
    kp_usability = get_keypoint_usability(obj_kps)

    points = [kp.camera_cartesian_location for kp in obj_kps]
    m, *_ = np.array(points)
    m_usable, *_ = kp_usability

    direction = get_object_direction(obj_kps)

    if not m_usable or direction is None:
        return None
    position = m
    direction_ep = position + direction

    position = point_to_world(position, camera_pos, camera_rot)
    direction_ep = point_to_world(direction_ep, camera_pos, camera_rot)

    world_direction = direction_ep - position
    world_direction = normalize_vector(world_direction)

    return np.concatenate([position, world_direction])
