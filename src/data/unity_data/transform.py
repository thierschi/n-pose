from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TransformValue:
    instance_id: int
    position: List[float]
    rotation: List[float]

    @staticmethod
    def from_dict(_dict: dict):
        return TransformValue(_dict['instanceId'],
                              _dict['position'],
                              _dict['rotation'])


@dataclass(frozen=True)
class TransformAnnotation:
    type_name = 'type.experimental-surgery.com/KIARA.Transform'

    id: str
    sensor_id: str
    description: str
    values: List[TransformValue]

    @staticmethod
    def from_dict(_dict: dict):
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

    def ids(self):
        return [v.instance_id for v in self.values]
