from dataclasses import dataclass
from typing import List

from .label import Label


@dataclass(frozen=True)
class BoundingBox2DValue:
    """
    BoundingBox2DValue represents a single 2D bounding box in Unity.
    """
    instance_id: int
    label: Label
    origin: List[float]
    dimension: List[float]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]) -> 'BoundingBox2DValue':
        """
        Create a BoundingBox2DValue from a dictionary.
        :param _dict:
        :param labels: List of all labels in UnityData
        :return: A BoundingBox2DValue object
        """
        label = None

        for label in labels:
            if label.name == _dict['labelName']:
                label = label
                break

        return BoundingBox2DValue(_dict['instanceId'],
                                  label,
                                  _dict['origin'],
                                  _dict['dimension'])


@dataclass(frozen=True)
class BoundingBox2DAnnotation:
    """
    BoundingBox2DAnnotation represents a Unity 2D bounding box annotation.
    """
    type_name = 'type.unity.com/unity.solo.BoundingBox2DAnnotation'

    id: str
    sensor_id: str
    description: str
    values: List[BoundingBox2DValue]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]) -> 'BoundingBox2DAnnotation':
        """
        Create a BoundingBox2DAnnotation from a dictionary.
        :param _dict:
        :param labels: List of all labels in UnityData
        :return: A BoundingBox2DAnnotation object
        """
        values = [BoundingBox2DValue.from_dict(value, labels) for value in
                  _dict['values']] if 'values' in _dict else []

        return BoundingBox2DAnnotation(_dict['id'],
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
