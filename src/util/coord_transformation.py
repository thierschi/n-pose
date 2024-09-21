import numpy as np
from scipy.spatial.transform import Rotation as R


def point_to_world(point, camera_pos, camera_rot):
    """
    Transform a point from camera to world coordinates
    :param point: Point to transform
    :param camera_pos: Position of the camera
    :param camera_rot: Rotation of the camera
    :return: Transformed point
    """
    point = np.array(point)
    camera_pos = np.array(camera_pos)
    camera_rot = np.array(camera_rot)

    rotation = R.from_quat(camera_rot)
    rotated_point = rotation.apply(point)

    rotated_point = rotated_point + camera_pos

    return rotated_point


def point_to_cam(point, camera_pos, camera_rot):
    """
    Transform a point from world to camera coordinates
    :param point: Point to transform
    :param camera_pos: Position of the camera
    :param camera_rot: Rotation of the camera
    :return:
    """
    point = np.array(point)
    camera_pos = np.array(camera_pos)
    camera_rot = np.array(camera_rot)

    point = point - camera_pos

    rotation = R.from_quat(camera_rot)
    rotated_point = rotation.apply(point, inverse=True)

    return rotated_point
