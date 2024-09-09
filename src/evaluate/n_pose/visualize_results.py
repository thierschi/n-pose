import cv2
import numpy as np
import torch
from torch import nn

from ...data.n_pose_conversion.annotation_conversion import keypoints_to_vector, i_segmentation_to_vector, \
    position_and_rotation_to_vector
from ...data.unity_data import UnityData
from ...visualize import Annotator, CoordType


def visualize_n_pose(model: nn.Module, data: UnityData, sequence: int, instance_id: int, device="cpu"):
    captures = data.get_sequence(sequence)
    left_capture = None
    right_capture = None

    for capture in captures:
        if capture.id == "RightCam":
            right_capture = capture
        elif capture.id == "LeftCam":
            left_capture = capture
        if left_capture is not None and right_capture is not None:
            break

    assert left_capture is not None and right_capture is not None

    input_vectors = [
        keypoints_to_vector(left_capture.keypoints[instance_id], 10),
        keypoints_to_vector(right_capture.keypoints[instance_id], 10),
        i_segmentation_to_vector(left_capture.instance_segmentation[instance_id],
                                 cv2.imread(left_capture.instance_segmentation.file_path), 100),
        i_segmentation_to_vector(right_capture.instance_segmentation[instance_id],
                                 cv2.imread(right_capture.instance_segmentation.file_path), 100)
    ]

    input = []
    for v in input_vectors:
        input.extend(v)
    input = torch.from_numpy(np.array([input]).astype(np.float32)).to(device)

    output = None
    model.eval()
    with torch.no_grad():
        output = model(input)

    output = output.cpu().numpy()

    true = position_and_rotation_to_vector(left_capture.keypoints[instance_id].keypoints, left_capture.position,
                                           left_capture.rotation)
    pos, dir = np.split(true, 2)
    # dir = pos + dir

    print(true, output)

    (Annotator(left_capture)
     .capture_info()
     .keypoints(instance_id)
     .bb2d(instance_id)
     .arrow(pos, dir, CoordType.WORLD, color=(0, 0, 255))
     .arrow(output[0, :3], output[0, 3:], CoordType.WORLD, color=(0, 255, 0))
     .line(pos, output[0, :3], CoordType.WORLD, color=(255, 0, 0))
     .additional_info(f"Distance: {np.linalg.norm(pos - output[0, :3]):.2f}")
     .show())
