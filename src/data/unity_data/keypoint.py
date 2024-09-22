from dataclasses import dataclass
from typing import List

from .label import Label


@dataclass
class Keypoint:
    """
    Keypoint represents a single keypoint in Unity
    """
    index: int
    location: List[float]
    camera_cartesian_location: List[float]
    state: int

    @staticmethod
    def from_dict(_dict: dict) -> 'Keypoint':
        """
        Create a Keypoint from a dictionary.
        :param _dict:
        :return: A Keypoint object
        """
        return Keypoint(_dict['index'],
                        _dict['location'],
                        _dict['cameraCartesianLocation'],
                        _dict['state'])


@dataclass
class KeypointValue:
    """
    KeypointValue represents all keypoints for a single instance in Unity
    """
    instance_id: int
    label: Label
    pose: str
    keypoints: List[Keypoint]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]) -> 'KeypointValue':
        """
        Create a KeypointValue from a dictionary.
        :param _dict:
        :param labels: List of all labels in UnityData
        :return: A  KeypointValue object
        """
        label = None

        for label in labels:
            if label.unity_id == _dict['labelId']:
                label = label
                break

        keypoints = [Keypoint.from_dict(keypoint) for keypoint in _dict['keypoints']]

        return KeypointValue(_dict['instanceId'],
                             label,
                             _dict['pose'],
                             keypoints)


@dataclass
class KeypointAnnotation:
    """
    KeypointAnnotation represents a Unity keypoint annotation
    """
    type_name = 'type.unity.com/unity.solo.KeypointAnnotation'

    id: str
    sensor_id: str
    description: str
    template_id: str
    values: List[KeypointValue]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]) -> 'KeypointAnnotation':
        """
        Create a KeypointAnnotation from a dictionary.
        :param _dict:
        :param labels: List of all labels in UnityData
        :return: A KeypointAnnotation object
        """
        values = [KeypointValue.from_dict(value, labels) for value in
                  _dict['values']] if 'values' in _dict else []

        return KeypointAnnotation(_dict['id'],
                                  _dict['sensorId'],
                                  _dict['description'],
                                  _dict['templateId'],
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
