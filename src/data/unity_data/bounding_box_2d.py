from dataclasses import dataclass
from typing import List

from .label import Label


@dataclass(frozen=True)
class BoundingBox2DValue:
    instance_id: int
    label: Label
    origin: List[float]
    dimension: List[float]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]):
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
    type_name = 'type.unity.com/unity.solo.BoundingBox2DAnnotation'

    id: str
    sensor_id: str
    description: str
    values: List[BoundingBox2DValue]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]):
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

    def ids(self):
        return [v.instance_id for v in self.values]
