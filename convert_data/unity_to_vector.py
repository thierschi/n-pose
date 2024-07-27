from pathlib import Path

import cv2
import numpy as np
from progressbar.bar import ProgressBar

from import_data.unity_data import UnityData, UnityKeypointValue, UnityInstanceSegmentationInstance, \
    Unity3DBoundingBoxValue
from util.polygons import get_colored_polygons_from_mask, get_polygon_boundary, simplify_polygon_group


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
        simple_polygon = simplify_polygon_group(polygons, target=50, tolerance=1e-10)
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


def convert_position_to_vector(bounding_box_3d: Unity3DBoundingBoxValue):
    pos_vector = np.zeros(3)
    pos_vector[0] = bounding_box_3d.translation[0]
    pos_vector[1] = bounding_box_3d.translation[1]
    pos_vector[2] = bounding_box_3d.translation[2]

    rot_vector = np.zeros(4)
    rot_vector[0] = bounding_box_3d.rotation[0]
    rot_vector[1] = bounding_box_3d.rotation[1]
    rot_vector[2] = bounding_box_3d.rotation[2]
    rot_vector[3] = bounding_box_3d.rotation[3]

    return np.concatenate((pos_vector, rot_vector))


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
            header = ['seq', 'pos_x', 'pos_y', 'pos_z', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
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
                    if left_capture is not None and right_capture is not None:
                        break
                    if capture.sequence == i and capture.id == "RightCam":
                        right_capture = capture
                    elif capture.sequence == i and capture.id == "LeftCam":
                        left_capture = capture

                if left_capture is None or right_capture is None:
                    continue

                instances = []
                for capture in [left_capture, right_capture]:
                    for instance in capture.instance_segmentation.instances:
                        if instance.instance_id not in instances:
                            instances.append(instance.instance_id)

                for instance_id in instances:
                    total_instances += 1

                    bounding_box_3d_value = None

                    for capture in [left_capture, right_capture]:
                        if capture.bounding_boxes_3d is None:
                            continue
                        b = [b for b in capture.bounding_boxes_3d.values if
                             b.instance_id == instance_id]
                        if len(b) > 0:
                            bounding_box_3d_value = b[0]
                            break
                    if bounding_box_3d_value is None:
                        skipping_msgs.append(
                            f"Skipping {capture.sequence}/{instance_id} because of missing bounding box")
                        continue

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

                    pos_vector = convert_position_to_vector(bounding_box_3d_value)
                    kpv_l_vector = convert_kps_to_vector(kpv_l, kp_vector_size,
                                                         (left_capture.dimension[0], left_capture.dimension[1]))
                    kpv_r_vector = convert_kps_to_vector(kpv_r, kp_vector_size,
                                                         (right_capture.dimension[0], right_capture.dimension[1]))
                    seg_inst_l_vector = convert_iseg_to_vector(seg_inst_l,
                                                               cv2.imread(left_capture.instance_segmentation.file_path),
                                                               seg_vector_size)
                    seg_inst_r_vector = convert_iseg_to_vector(seg_inst_r,
                                                               cv2.imread(
                                                                   right_capture.instance_segmentation.file_path),
                                                               seg_vector_size)

                    if pos_vector is None or kpv_l_vector is None or kpv_r_vector is None or seg_inst_l_vector is None \
                            or seg_inst_r_vector is None:
                        skipping_msgs.append(f"Skipping {capture.sequence}/{instance_id} because of failed conversion")
                        continue

                    vector = []
                    vector.extend(pos_vector)
                    vector.extend(kpv_l_vector)
                    vector.extend(seg_inst_l_vector)
                    vector.extend(kpv_r_vector)
                    vector.extend(seg_inst_r_vector)

                    # write to file
                    file.write(f"{capture.sequence}, {', '.join([str(round(x, precision)) for x in vector])}\n")

    for msg in skipping_msgs:
        print(msg)
    print(f"Skipped {len(skipping_msgs)}/{total_instances} instances in total")
    print(f"Saved to {file_path}")

    return file_path.as_posix()
