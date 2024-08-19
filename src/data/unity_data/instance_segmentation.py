from dataclasses import dataclass
from typing import List

from util.color import RGBAColor
from .label import Label


@dataclass(frozen=True)
class InstanceSegmentationInstance:
    instance_id: int
    label: Label
    color: RGBAColor

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]):
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

        return InstanceSegmentationInstance(_dict['instanceId'], label, color)


@dataclass(frozen=True)
class InstanceSegmentationAnnotation:
    type_name = 'type.unity.com/unity.solo.InstanceSegmentationAnnotation'

    id: str
    sensor_id: str
    description: str
    image_format: str
    dimension: List[float]
    file_path: str
    instances: List[InstanceSegmentationInstance]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label], path: str):
        instances = [InstanceSegmentationInstance.from_dict(instance, labels) for instance in
                     _dict['instances']] if 'instances' in _dict else []

        return InstanceSegmentationAnnotation(_dict['id'],
                                              _dict['sensorId'],
                                              _dict['description'],
                                              _dict['imageFormat'],
                                              _dict['dimension'],
                                              path + '/' + _dict['filename'],
                                              instances)
