from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TransformValue:
    """
    TransformValue represents a single transformation in Unity
    """
    instance_id: int
    position: List[float]
    rotation: List[float]

    @staticmethod
    def from_dict(_dict: dict) -> 'TransformValue':
        """
        Create a TransformValue from a dictionary.
        :param _dict:
        :return: A TransformValue object
        """
        return TransformValue(_dict['instanceId'],
                              _dict['position'],
                              _dict['rotation'])


@dataclass(frozen=True)
class TransformAnnotation:
    """
    TransformAnnotation represents a KIARA transform annotation.
    """
    type_name = 'type.experimental-surgery.com/KIARA.Transform'

    id: str
    sensor_id: str
    description: str
    values: List[TransformValue]

    @staticmethod
    def from_dict(_dict: dict) -> 'TransformAnnotation':
        """
        Create a TransformAnnotation from a dictionary.
        :param _dict:
        :return: A TransformAnnotation object
        """
        values = [TransformValue.from_dict(value) for value in
                  _dict['values']] if 'values' in _dict else []

        return TransformAnnotation(_dict['id'],
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
        Get all instance ids in the annotation
        :return: List of instance IDs
        """
        return [v.instance_id for v in self.values]
