import os

import numpy as np

from data.unity_data import UnityData
from visualize import Annotator, CoordType


def split_row(row: np.ndarray, kp_v_size: int, seg_v_size: int):
    seq = row[0].astype(np.int32)
    instance = row[1].astype(np.int32)
    pos = row[2:5]
    direction = row[5:8]
    kp_left = row[8:8 + kp_v_size]
    seg_left = row[8 + kp_v_size:8 + kp_v_size + seg_v_size]
    kp_right = row[8 + kp_v_size + seg_v_size:8 + 2 * kp_v_size + seg_v_size]
    seg_right = row[8 + 2 * kp_v_size + seg_v_size:8 + 2 * kp_v_size + 2 * seg_v_size]

    return seq, instance, pos, direction, kp_left, seg_left, kp_right, seg_right


def convert_vector_coordinates(points: np.ndarray, img_width: int, img_height: int):
    points = points.reshape(-1, 2)
    points[:, 0] = points[:, 0] * img_width
    points[:, 1] = points[:, 1] * img_height
    points = points.astype(np.int32)
    points = [p for p in points if p[0] > 0 and p[1] > 0]

    return points


def validate_vector_data(data: UnityData, v_data_path: str, kp_v_size: int, seg_v_size: int):
    assert os.path.exists(v_data_path)

    vec_data = np.loadtxt(v_data_path, delimiter=',', skiprows=1).astype(np.float32)

    for row in vec_data:
        seq, instance, pos, direction, kp_left, seg_left, kp_right, seg_right = split_row(row, kp_v_size, seg_v_size)

        seq = data.get_sequence(seq)

        for c in seq:
            is_left = c.id == "LeftCam"
            kps = kp_left if is_left else kp_right
            seg = seg_left if is_left else seg_right

            kp_points = convert_vector_coordinates(kps, c.dimension[0], c.dimension[1])
            seg_points = convert_vector_coordinates(seg, c.dimension[0], c.dimension[1])

            a = (Annotator(c)
                 .capture_info()
                 .additional_info(f"instance_id: {instance}")
                 .bb2d(instance, color=(0, 255, 0)))

            a.points(kp_points, CoordType.IMAGE, color=(0, 0, 255))
            a.points(seg_points, CoordType.IMAGE, connect=True, color=(255, 0, 0))

            a.arrow(pos, pos + direction, CoordType.WORLD, color=(0, 0, 255))

            a.save(f'../../data/vec_valid/{c.sequence}_{c.id}_{instance}.png')


def main():
    data = UnityData('../../data/solo_149')
    validate_vector_data(data, '../../data/n_pose/vec_solo_149.csv', 10, 100)


main()
