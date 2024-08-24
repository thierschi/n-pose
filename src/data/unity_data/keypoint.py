from dataclasses import dataclass
from typing import List

from .label import Label


@dataclass
class Keypoint:
    index: int
    location: List[float]
    camera_cartesian_location: List[float]
    state: int

    @staticmethod
    def from_dict(_dict: dict):
        return Keypoint(_dict['index'],
                        _dict['location'],
                        _dict['cameraCartesianLocation'],
                        _dict['state'])


@dataclass
class KeypointValue:
    instance_id: int
    label: Label
    pose: str
    keypoints: List[Keypoint]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]):
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
    type_name = 'type.unity.com/unity.solo.KeypointAnnotation'

    id: str
    sensor_id: str
    description: str
    template_id: str
    values: List[KeypointValue]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]):
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
