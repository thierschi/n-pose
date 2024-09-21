from pathlib import Path
from typing import Type

from tqdm import tqdm

from .vector_converter import VectorConverter
from ..unity_data import UnityData


def unity_to_n_pose(data: UnityData, file_path: str, converter_class: Type[VectorConverter], kp_v_size: int = 10,
                    seg_v_size: int = 100,
                    precision=6) -> str:
    """
    Converts UnityData to n-pose format and saves it to a .csv file.
    :param data: Data to convert
    :param file_path: Path where to save the .csv file. (Must include extension)
    :param converter_class: Uninitialised class of the converter to use
    :param kp_v_size: Vector size for keypoints
    :param seg_v_size: Vector size for instance segmentation
    :param precision: Floating point precision
    :return: Path to the saved file (different from file_path if file already existed)
    """
    assert kp_v_size % 2 == 0
    assert seg_v_size % 2 == 0

    # Create parent directories if they don't exist
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # check if file exists and rename with index if it does
    index = 0
    path_to_check = file_path
    while path_to_check.exists():
        path_to_check = file_path.parent / f"{file_path.stem}_{index}{file_path.suffix}"
        index += 1
    file_path = path_to_check

    skipping_msgs = []
    total_instances = 0

    bar = tqdm(total=data.len_sequences, desc="Converting")

    with open(file_path, "w") as file:
        # Write csv header
        header = ['seq', 'id', 'pos_x', 'pos_y', 'pos_z', 'dir_x', 'dir_y', 'dir_z']
        header_l = []
        header_r = []
        for seq in range(kp_v_size // 2):
            header_l.extend([f"kp_l_{seq}_x", f"kp_l_{seq}_y"])
            header_r.extend([f"kp_r_{seq}_x", f"kp_r_{seq}_y"])
        for seq in range(seg_v_size // 2):
            header_l.extend([f"seg_l_{seq}_x", f"seg_l_{seq}_y"])
            header_r.extend([f"seg_r_{seq}_x", f"seg_r_{seq}_y"])
        header.extend(header_l)
        header.extend(header_r)
        file.write(f"{', '.join(header)}\n")

        for seq in data.sequence_ids:
            bar.update()

            left_capture = None
            right_capture = None

            # Find the left and right captures for the current sequence
            for capture in data.get_sequence(seq):
                if capture.sequence == seq and capture.id == "RightCam":
                    right_capture = capture
                elif capture.sequence == seq and capture.id == "LeftCam":
                    left_capture = capture
                if left_capture is not None and right_capture is not None:
                    break

            if left_capture is None or right_capture is None:
                continue

            for instance_id in left_capture.instance_segmentation.ids():
                total_instances += 1

                converter = converter_class()

                try:  # Obligatory try-except block to catch any errors during conversion
                    converter.use(left_capture, instance_id)
                    converter.keypoints(kp_v_size)
                    converter.instance_segmentation(seg_v_size)

                    converter.position()
                    converter.direction()

                    converter.use(right_capture, instance_id)
                    converter.keypoints(kp_v_size)
                    converter.instance_segmentation(seg_v_size)

                    if not converter.was_successful:
                        skipping_msgs.append(f"Skipping {seq} {instance_id} because of unsuccessful conversion")
                        continue
                except Exception as e:
                    skipping_msgs.append(f"Skipping {seq} {instance_id} because of error: {e}")
                    continue

                # Combine and write the vectors
                vector = [seq, instance_id]
                vector.extend(converter.output_vector)
                vector.extend(converter.input_vector)

                file.write(f"{', '.join([str(round(x, precision)) for x in vector])}\n")

    for msg in skipping_msgs:
        print(msg)
    print(f"Skipped {len(skipping_msgs)}/{total_instances} instances in total")
    print(f"Saved to {file_path}")

    return file_path.as_posix()
