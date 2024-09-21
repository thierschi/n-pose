from dataclasses import dataclass
from typing import List

from .label import Label
from ...util import RGBAColor


@dataclass(frozen=True)
class SemanticSegmentationInstance:
    """
    SemanticSegmentationInstance represents a single instance in a Unity semantic segmentation annotation.
    """
    label: Label
    color: RGBAColor

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]) -> 'SemanticSegmentationInstance':
        """
        Create a SemanticSegmentationInstance from a dictionary.
        :param _dict:
        :param labels: List of all labels in UnityData
        :return: A SemanticSegmentationInstance object
        """
        label = None
        color = None

        for label in labels:
            if label.name == _dict['labelName']:
                label = label
                break

        if 'color' in _dict:
            color = RGBAColor(_dict['color'][0],
                              _dict['color'][1],
                              _dict['color'][2],
                              _dict['color'][3])

        return SemanticSegmentationInstance(label, color)


@dataclass(frozen=True)
class SemanticSegmentationAnnotation:
    """
    SemanticSegmentationAnnotation represents a Unity semantic segmentation annotation.
    """
    type_name = 'type.unity.com/unity.solo.SemanticSegmentationAnnotation'

    id: str
    sensor_id: str
    description: str
    image_format: str
    dimension: List[float]
    file_path: str
    instances: List[SemanticSegmentationInstance]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label], path: str) -> 'SemanticSegmentationAnnotation':
        """
        Create a SemanticSegmentationAnnotation from a dictionary.
        :param _dict:
        :param labels: List of all labels in UnityData
        :param path: Base path of sequence
        :return: A SemanticSegmentationAnnotation object
        """
        instances = [SemanticSegmentationInstance.from_dict(instance, labels) for instance in
                     _dict['instances']] if 'instances' in _dict else []

        return SemanticSegmentationAnnotation(_dict['id'],
                                              _dict['sensorId'],
                                              _dict['description'],
                                              _dict['imageFormat'],
                                              _dict['dimension'],
                                              path + '/' + _dict['filename'],
                                              instances)
