from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RGBColor:
    r: int
    g: int
    b: int


@dataclass(frozen=True)
class RGBAColor:
    r: int
    g: int
    b: int
    a: int


@dataclass(frozen=True)
class Label:
    id: int
    unity_id: int
    name: str


@dataclass(frozen=True)
class UnityInstanceSegmentationInstance:
    instance_id: int
    label: Label
    color: RGBAColor


@dataclass(frozen=True)
class UnityInstanceSegmentation:
    id: str
    sensor_id: str
    description: str
    image_format: str
    dimension: List[float]
    file_path: str
    instances: List[UnityInstanceSegmentationInstance]


@dataclass(frozen=True)
class UnitySemanticSegmentationInstance:
    label: Label
    color: RGBAColor


@dataclass(frozen=True)
class UnitySemanticSegmentation:
    id: str
    sensor_id: str
    description: str
    image_format: str
    dimension: List[float]
    file_path: str
    instances: List[UnitySemanticSegmentationInstance]


@dataclass(frozen=True)
class Unity2DBoundingBoxValue:
    instance_id: int
    label: Label
    origin: List[float]
    dimension: List[float]


@dataclass(frozen=True)
class Unity2DBoundingBox:
    id: str
    sensor_id: str
    description: str
    values: List[Unity2DBoundingBoxValue]


@dataclass(frozen=True)
class UnityDepth:
    sensor_id: str
    description: str
    measurement_strategy: str
    image_format: str
    dimension: List[float]
    file_path: str


@dataclass(frozen=True)
class Unity3DBoundingBoxValue:
    instance_id: int
    label: Label
    translation: List[float]
    size: List[float]
    rotation: List[float]
    velocity: List[float]
    acceleration: List[float]


@dataclass(frozen=True)
class Unity3DBoundingBox:
    id: str
    sensor_id: str
    description: str
    values: List[Unity3DBoundingBoxValue]


@dataclass(frozen=True)
class UnityCapture:
    id: str
    description: str
    position: List[float]
    rotation: List[float]
    velocity: List[float]
    acceleration: List[float]
    file_path: str
    image_format: str
    dimension: List[float]
    projection: str
    matrix: List[float]
