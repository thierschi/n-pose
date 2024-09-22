from dataclasses import dataclass
from typing import List

from .label import Label
from ...util import RGBAColor


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
    """
    InstanceSegmentationAnnotation represents a Unity instance segmentation annotation.
    """
    type_name = 'type.unity.com/unity.solo.InstanceSegmentationAnnotation'

    id: str
    sensor_id: str
    description: str
    image_format: str
    dimension: List[float]
    file_path: str
    instances: List[InstanceSegmentationInstance]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label], path: str) -> 'InstanceSegmentationAnnotation':
        """
        Create a InstanceSegmentationAnnotation from a dictionary.
        :param _dict:
        :param labels: List of all labels in UnityData
        :param path: Base path for capture, i.e. the path of the sequence folder
        :return: A InstanceSegmentationAnnotation object
        """
        instances = [InstanceSegmentationInstance.from_dict(instance, labels) for instance in
                     _dict['instances']] if 'instances' in _dict else []

        return InstanceSegmentationAnnotation(_dict['id'],
                                              _dict['sensorId'],
                                              _dict['description'],
                                              _dict['imageFormat'],
                                              _dict['dimension'],
                                              path + '/' + _dict['filename'],
                                              instances)

    def __getitem__(self, _id):
        item = [v for v in self.instances if v.instance_id == _id]

        if len(item) == 0:
            return None
        return item[0]

    def ids(self) -> List[int]:
        """
        Get a list of all instance IDs in the annotation.
        :return: List of instance IDs
        """
        return [v.instance_id for v in self.instances]
