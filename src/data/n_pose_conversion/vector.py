from pathlib import Path

from progressbar import ProgressBar

from .annotation_conversion import *
from ..unity_data import UnityData


def DEPRECATED_unity_to_n_pose(data: UnityData, folder_path: str, file_name: str, kp_vector_size: int,
                               seg_vector_size: int, precision=6) -> str:
    """
    DEPRECATED: Use unity_to_n_pose instead
    Converts UnityData to a CSV file with n-pose vectors

    :param data: UnityData
    :param folder_path: folder path where .csv will be saved
    :param file_name: file_name of the .csv
    :param kp_vector_size: Size of the keypoint vectors
    :param seg_vector_size: Size of the segmentation vectors
    :param precision: presicion of the float values
    :return: full file path of the saved .csv
    """
    assert kp_vector_size % 2 == 0
    assert seg_vector_size % 2 == 0

    folder_path = folder_path.rstrip("/")

    # Create folder path and all parents
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

    # Append index to filename of file with same name exists
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
            # Write csv header
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

            # Write sequence data
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
                        position_and_rotation_to_vector(kpv_l.keypoints, left_capture.position, left_capture.rotation),
                        keypoints_to_vector(kpv_l, kp_vector_size,
                                            (left_capture.dimension[0], left_capture.dimension[1])),
                        i_segmentation_to_vector(seg_inst_l,
                                                 cv2.imread(left_capture.instance_segmentation.file_path),
                                                 seg_vector_size),
                        keypoints_to_vector(kpv_r, kp_vector_size,
                                            (right_capture.dimension[0], right_capture.dimension[1])),
                        i_segmentation_to_vector(seg_inst_r,
                                                 cv2.imread(right_capture.instance_segmentation.file_path),
                                                 seg_vector_size)
                    ]

                    if len([v for v in vectors if v is None]) > 0:
                        skipping_msgs.append(
                            f"Skipping {capture.sequence}/{instance_id} because"
                            f"of failed conversion ({','.join(['O' if v is None else 'X' for v in vectors])}).")
                        continue

                    vector = [capture.sequence]
                    for v in vectors:
                        vector.extend(v)

                    file.write(f"{', '.join([str(round(x, precision)) for x in vector])}\n")

    # Print sequences that were skipped and why
    for msg in skipping_msgs:
        print(msg)
    print(f"Skipped {len(skipping_msgs)}/{total_instances} instances in total")
    print(f"Saved to {file_path}")

    return file_path.as_posix()
