from pathlib import Path
from typing import List

import cv2
import numpy as np
from progressbar.bar import ProgressBar
from scipy.spatial.transform import Rotation as R

from import_data.unity_data import UnityData, UnityKeypointValue, UnityKeypoint
from import_data.unity_data import UnityInstanceSegmentationInstance
from util.polygons import get_colored_polygons_from_mask, get_polygon_boundary, simplify_polygon_group


def point_to_world(point, camera_pos, camera_rot):
    point = np.array(point)
    camera_pos = np.array(camera_pos)
    camera_rot = np.array(camera_rot)

    rotation = R.from_quat(camera_rot)
    rotated_point = rotation.apply(point)

    rotated_point = rotated_point + camera_pos

    return rotated_point


def point_to_cam(point, camera_pos, camera_rot):
    point = np.array(point)
    camera_pos = np.array(camera_pos)
    camera_rot = np.array(camera_rot)

    point = point - camera_pos

    rotation = R.from_quat(camera_rot)
    rotated_point = rotation.apply(point, inverse=True)

    return rotated_point


def get_keypoint_usability(kps: List[UnityKeypoint]):
    kp_usability = []
    for kp in kps:
        usable = False
        if kp.state != 0:
            for coord in kp.camera_cartesian_location:
                if coord != 0:
                    usable = True
                    break
        kp_usability.append(usable)
    return kp_usability


def double_cross_product(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = b - a
    ac = c - a

    return np.cross(ab, np.cross(ab, ac))


def normalize_vector(v):
    v = np.array(v)
    return v / np.linalg.norm(v)


def get_object_direction_vector(obj_kps: List[UnityKeypoint]):
    obj_kps.sort(key=lambda x: x.index)
    kp_usability = get_keypoint_usability(obj_kps)

    points = [kp.camera_cartesian_location for kp in obj_kps]
    m, fl, fr, bl, br = np.array(points)
    m_usable, fl_usable, fr_usable, bl_usable, br_usable = kp_usability

    if len([x for x in kp_usability if x]) < 3:
        return None

    a, b, c = None, None, None
    inversion_needed = False

    if fl_usable and fr_usable:
        a = fl
        b = fr
        c = m if m_usable else bl if bl_usable else br if br_usable else None
        inversion_needed = False
    elif bl_usable and br_usable:
        a = bl
        b = br
        c = m if m_usable else fl if fl_usable else fr if fr_usable else None
        inversion_needed = True

    if a is None or b is None or c is None:
        return None

    direction = double_cross_product(a, b, c)
    direction = normalize_vector(direction) / 10

    if inversion_needed:
        direction = -direction

    return direction


def convert_kps_to_vector(kp_value: UnityKeypointValue, v_size: int, image_size=(1920, 1080)):
    vector = np.zeros(v_size)

    for i, kp in enumerate(kp_value.keypoints):
        if i * 2 >= v_size:
            break

        vector[i * 2] = kp.location[0] / image_size[0]
        vector[i * 2 + 1] = kp.location[1] / image_size[1]

    return vector


def convert_iseg_to_vector(instance: UnityInstanceSegmentationInstance, seg_map: cv2.typing.MatLike, v_size: int):
    assert v_size % 2 == 0
    target = v_size // 2

    _, polygons = get_colored_polygons_from_mask(seg_map, instance.color)

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


def get_position_vector(obj_kps: List[UnityKeypoint], camera_pos, camera_rot):
    obj_kps.sort(key=lambda x: x.index)
    kp_usability = get_keypoint_usability(obj_kps)

    points = [kp.camera_cartesian_location for kp in obj_kps]
    m, *_ = np.array(points)
    m_usable, *_ = kp_usability

    direction = get_object_direction_vector(obj_kps)

    if not m_usable or direction is None:
        return None
    position = m
    direction_ep = position + direction

    position = point_to_world(position, camera_pos, camera_rot)
    direction_ep = point_to_world(direction_ep, camera_pos, camera_rot)

    world_direction = direction_ep - position
    world_direction = normalize_vector(world_direction)

    return np.concatenate([position, world_direction])


def save_unity_data_as_vector_csv(data: UnityData, folder_path: str, file_name: str, kp_vector_size: int,
                                  seg_vector_size: int, precision=6):
    assert kp_vector_size % 2 == 0
    assert seg_vector_size % 2 == 0

    folder_path = folder_path.rstrip("/")

    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

    file_path = folder_path / f"{file_name}.csv"
    index = 0
    while file_path.exists():
        file_path = folder_path / f"{file_name}_{index}.csv"
        index += 1

    max_sequence = 0
    for capture in data.captures:
        if capture.sequence > max_sequence:
            max_sequence = capture.sequence

    skipping_msgs = []
    total_instances = 0

    with ProgressBar(max_value=max_sequence) as bar:
        with open(file_path, "w") as file:
            header = ['seq', 'pos_x', 'pos_y', 'pos_z', 'dir_x', 'dir_y', 'dir_z']
            header_l = []
            header_r = []
            for i in range(kp_vector_size // 2):
                header_l.extend([f"kp_l_{i}_x", f"kp_l_{i}_y"])
                header_r.extend([f"kp_r_{i}_x", f"kp_r_{i}_y"])
            for i in range(seg_vector_size // 2):
                header_l.extend([f"seg_l_{i}_x", f"seg_l_{i}_y"])
                header_r.extend([f"seg_r_{i}_x", f"seg_r_{i}_y"])
            header.extend(header_l)
            header.extend(header_r)
            file.write(f"{', '.join(header)}\n")

            for i in range(max_sequence):
                bar.next()

                left_capture = None
                right_capture = None

                for capture in data.captures:
                    if capture.sequence == i and capture.id == "RightCam":
                        right_capture = capture
                    elif capture.sequence == i and capture.id == "LeftCam":
                        left_capture = capture
                    if left_capture is not None and right_capture is not None:
                        break

                if left_capture is None or right_capture is None:
                    continue

                instances = []
                for capture in [left_capture, right_capture]:
                    for instance in capture.instance_segmentation.instances:
                        if instance.instance_id not in instances:
                            instances.append(instance.instance_id)

                for instance_id in instances:
                    total_instances += 1

                    # bounding_box_3d_value = None
                    #
                    # for capture in [left_capture, right_capture]:
                    #     if capture.bounding_boxes_3d is None:
                    #         continue
                    #     b = [b for b in capture.bounding_boxes_3d.values if
                    #          b.instance_id == instance_id]
                    #     if len(b) > 0:
                    #         bounding_box_3d_value = b[0]
                    #         break
                    # if bounding_box_3d_value is None:
                    #     skipping_msgs.append(
                    #         f"Skipping {capture.sequence}/{instance_id} because of missing bounding box")
                    #     continue

                    try:
                        kpv_l = [kpv for kpv in left_capture.keypoints.values if kpv.instance_id == instance_id][0]
                        kpv_r = [kpv for kpv in right_capture.keypoints.values if kpv.instance_id == instance_id][0]
                        seg_inst_l = \
                            [seg for seg in left_capture.instance_segmentation.instances if
                             seg.instance_id == instance_id][
                                0]
                        seg_inst_r = [seg for seg in right_capture.instance_segmentation.instances if
                                      seg.instance_id == instance_id][0]
                    except IndexError:
                        skipping_msgs.append(f"Skipping {capture.sequence}/{instance_id} because of missing annotation")
                        continue

                    vectors = [
                        get_position_vector(kpv_l.keypoints, left_capture.position, left_capture.rotation),
                        convert_kps_to_vector(kpv_l, kp_vector_size,
                                              (left_capture.dimension[0], left_capture.dimension[1])),
                        convert_iseg_to_vector(seg_inst_l,
                                               cv2.imread(left_capture.instance_segmentation.file_path),
                                               seg_vector_size),
                        convert_kps_to_vector(kpv_r, kp_vector_size,
                                              (right_capture.dimension[0], right_capture.dimension[1])),
                        convert_iseg_to_vector(seg_inst_r,
                                               cv2.imread(right_capture.instance_segmentation.file_path),
                                               seg_vector_size)
                    ]

                    if len([v for v in vectors if v is None]) > 0:
                        skipping_msgs.append(
                            f"Skipping {capture.sequence}/{instance_id} because of failed conversion ({','.join(['O' if v is None else 'X' for v in vectors])}).")
                        continue

                    vector = [capture.sequence]
                    for v in vectors:
                        vector.extend(v)

                    # write to file
                    file.write(f"{', '.join([str(round(x, precision)) for x in vector])}\n")

    for msg in skipping_msgs:
        print(msg)
    print(f"Skipped {len(skipping_msgs)}/{total_instances} instances in total")
    print(f"Saved to {file_path}")

    return file_path.as_posix()
