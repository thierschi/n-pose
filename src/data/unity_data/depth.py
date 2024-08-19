from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DepthAnnotation:
    type_name = 'type.unity.com/unity.solo.DepthAnnotation'

    sensor_id: str
    description: str
    measurement_strategy: str
    image_format: str
    dimension: List[float]
    file_path: str

    @staticmethod
    def from_dict(_dict: dict, path: str):
        return DepthAnnotation(_dict['sensorId'],
                               _dict['description'],
                               _dict['measurementStrategy'],
                               _dict['imageFormat'],
                               _dict['dimension'],
                               path + '/' + _dict['filename'])
