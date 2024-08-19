from dataclasses import dataclass
from typing import List

from .label import Label


@dataclass(frozen=True)
class BoundingBox3DValue:
    instance_id: int
    label: Label
    translation: List[float]
    size: List[float]
    rotation: List[float]
    velocity: List[float]
    acceleration: List[float]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]):
        label = None

        for label in labels:
            if label.name == _dict['labelName']:
                label = label
                break

        return BoundingBox3DValue(_dict['instanceId'],
                                  label,
                                  _dict['translation'],
                                  _dict['size'],
                                  _dict['rotation'],
                                  _dict['velocity'],
                                  _dict['acceleration'])


@dataclass(frozen=True)
class BoundingBox3DAnnotation:
    type_name = 'type.unity.com/unity.solo.BoundingBox3DAnnotation'

    id: str
    sensor_id: str
    description: str
    values: List[BoundingBox3DValue]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]):
        values = [BoundingBox3DValue.from_dict(value, labels) for value in
                  _dict['values']] if 'values' in _dict else []

        return BoundingBox3DAnnotation(_dict['id'],
                                       _dict['sensorId'],
                                       _dict['description'],
                                       values)
