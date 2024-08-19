from dataclasses import dataclass
from typing import List

from .bounding_box_2d import BoundingBox2DAnnotation
from .bounding_box_3d import BoundingBox3DAnnotation
from .depth import DepthAnnotation
from .instance_segmentation import InstanceSegmentationAnnotation
from .keypoint import KeypointAnnotation
from .label import Label
from .semantic_segmentation import SemanticSegmentationAnnotation


@dataclass(frozen=True)
class Capture:
    id: str
    sequence: int
    description: str
    position: List[float]
    rotation: List[float]
    velocity: List[float]
    acceleration: List[float]
    file_path: str
    image_format: str
    dimension: List[float]
    projection: str
    matrix: List[float]
    instance_segmentation: InstanceSegmentationAnnotation
    semantic_segmentation: SemanticSegmentationAnnotation
    bounding_boxes_2d: BoundingBox2DAnnotation
    bounding_boxes_3d: BoundingBox3DAnnotation
    depth: DepthAnnotation
    keypoints: KeypointAnnotation

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]):
        path = _dict['path']
        annotations = {}

        for annotation in _dict['annotations']:
            annotations[annotation['@type']] = annotation

        instance_segmentation = None if InstanceSegmentationAnnotation.type_name not in annotations else \
            InstanceSegmentationAnnotation.from_dict(annotations[InstanceSegmentationAnnotation.type_name],
                                                     labels, path)
        semantic_segmentation = None if SemanticSegmentationAnnotation.type_name not in annotations else \
            SemanticSegmentationAnnotation.from_dict(annotations[SemanticSegmentationAnnotation.type_name],
                                                     labels, path)
        bounding_boxes_2d = None if BoundingBox2DAnnotation.type_name not in annotations else \
            BoundingBox2DAnnotation.from_dict(annotations[BoundingBox2DAnnotation.type_name], labels)
        bounding_boxes_3d = None if BoundingBox3DAnnotation.type_name not in annotations else \
            BoundingBox3DAnnotation.from_dict(annotations[BoundingBox3DAnnotation.type_name], labels)
        depth = None if DepthAnnotation.type_name not in annotations else \
            DepthAnnotation.from_dict(annotations[DepthAnnotation.type_name], path)
        keypoints = None if KeypointAnnotation.type_name not in annotations else \
            KeypointAnnotation.from_dict(annotations[KeypointAnnotation.type_name], labels)

        return Capture(_dict['id'],
                       int(path.split('/')[-1].split('.')[-1]),
                       _dict['description'],
                       _dict['position'],
                       _dict['rotation'],
                       _dict['velocity'],
                       _dict['acceleration'],
                       path + '/' + _dict['filename'],
                       _dict['imageFormat'],
                       _dict['dimension'],
                       _dict['projection'],
                       _dict['matrix'],
                       instance_segmentation,
                       semantic_segmentation,
                       bounding_boxes_2d,
                       bounding_boxes_3d,
                       depth,
                       keypoints)
