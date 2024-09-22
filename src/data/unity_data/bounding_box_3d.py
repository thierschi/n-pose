from dataclasses import dataclass
from typing import List

from .label import Label


@dataclass(frozen=True)
class BoundingBox3DValue:
    """
    BoundingBox3DValue represents a single 3D bounding box in Unity.
    """
    instance_id: int
    label: Label
    translation: List[float]
    size: List[float]
    rotation: List[float]
    velocity: List[float]
    acceleration: List[float]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]) -> 'BoundingBox3DValue':
        """
        Create a BoundingBox3DValue from a dictionary.
        :param _dict:
        :param labels: List of all labels in UnityData
        :return: A BoundingBox3DValue object
        """
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
    """
    BoundingBox3DAnnotation represents a Unity 3D bounding box annotation.
    """
    type_name = 'type.unity.com/unity.solo.BoundingBox3DAnnotation'

    id: str
    sensor_id: str
    description: str
    values: List[BoundingBox3DValue]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]) -> 'BoundingBox3DAnnotation':
        """
        Create a BoundingBox3DAnnotation from a dictionary.
        :param _dict:
        :param labels: List of all labels in UnityData
        :return: A BoundingBox3DAnnotation object
        """
        values = [BoundingBox3DValue.from_dict(value, labels) for value in
                  _dict['values']] if 'values' in _dict else []

        return BoundingBox3DAnnotation(_dict['id'],
                                       _dict['sensorId'],
                                       _dict['description'],
                                       values)

    def __getitem__(self, _id):
        item = [v for v in self.values if v.instance_id == _id]

        if len(item) == 0:
            return None
        return item[0]

    def ids(self) -> List[int]:
        """
        Get all instance ids in the annotation.
        :return: List of instance ids
        """
        return [v.instance_id for v in self.values]
