import glob
import json
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

        return UnityInstanceSegmentationInstance(_dict['instanceId'], label, color)


@dataclass(frozen=True)
class UnityInstanceSegmentation:
    id: str
    sensor_id: str
    description: str
    image_format: str
    dimension: List[float]
    file_path: str
    instances: List[UnityInstanceSegmentationInstance]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label], path: str):
        instances = [UnityInstanceSegmentationInstance.from_dict(instance, labels) for instance in
                     _dict['instances']] if 'instances' in _dict else []

        return UnityInstanceSegmentation(_dict['id'],
                                         _dict['sensorId'],
                                         _dict['description'],
                                         _dict['imageFormat'],
                                         _dict['dimension'],
                                         path + '/' + _dict['filename'],
                                         instances)


@dataclass(frozen=True)
class UnitySemanticSegmentationInstance:
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

        return UnitySemanticSegmentationInstance(label, color)


@dataclass(frozen=True)
class UnitySemanticSegmentation:
    id: str
    sensor_id: str
    description: str
    image_format: str
    dimension: List[float]
    file_path: str
    instances: List[UnitySemanticSegmentationInstance]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label], path: str):
        instances = [UnitySemanticSegmentationInstance.from_dict(instance, labels) for instance in
                     _dict['instances']] if 'instances' in _dict else []

        return UnitySemanticSegmentation(_dict['id'],
                                         _dict['sensorId'],
                                         _dict['description'],
                                         _dict['imageFormat'],
                                         _dict['dimension'],
                                         path + '/' + _dict['filename'],
                                         instances)


@dataclass(frozen=True)
class Unity2DBoundingBoxValue:
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

        return Unity2DBoundingBoxValue(_dict['instanceId'],
                                       label,
                                       _dict['origin'],
                                       _dict['dimension'])


@dataclass(frozen=True)
class Unity2DBoundingBox:
    id: str
    sensor_id: str
    description: str
    values: List[Unity2DBoundingBoxValue]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]):
        values = [Unity2DBoundingBoxValue.from_dict(value, labels) for value in
                  _dict['values']] if 'values' in _dict else []

        return Unity2DBoundingBox(_dict['id'],
                                  _dict['sensorId'],
                                  _dict['description'],
                                  values)


@dataclass(frozen=True)
class UnityDepth:
    sensor_id: str
    description: str
    measurement_strategy: str
    image_format: str
    dimension: List[float]
    file_path: str

    @staticmethod
    def from_dict(_dict: dict, path: str):
        return UnityDepth(_dict['sensorId'],
                          _dict['description'],
                          _dict['measurementStrategy'],
                          _dict['imageFormat'],
                          _dict['dimension'],
                          path + '/' + _dict['filename'])


@dataclass(frozen=True)
class Unity3DBoundingBoxValue:
    instance_id: int
    label: Label
    translation: List[float]
    size: List[float]
    rotation: List[float]
    velocity: List[float]
    acceleration: List[float]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]):
        label = None

        for label in labels:
            if label.name == _dict['labelName']:
                label = label
                break

        return Unity3DBoundingBoxValue(_dict['instanceId'],
                                       label,
                                       _dict['translation'],
                                       _dict['size'],
                                       _dict['rotation'],
                                       _dict['velocity'],
                                       _dict['acceleration'])


@dataclass(frozen=True)
class Unity3DBoundingBox:
    id: str
    sensor_id: str
    description: str
    values: List[Unity3DBoundingBoxValue]

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]):
        values = [Unity3DBoundingBoxValue.from_dict(value, labels) for value in
                  _dict['values']] if 'values' in _dict else []

        return Unity3DBoundingBox(_dict['id'],
                                  _dict['sensorId'],
                                  _dict['description'],
                                  values)


@dataclass(frozen=True)
class UnityCapture:
    id: str
    sequence: int
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
    instance_segmentation: UnityInstanceSegmentation
    semantic_segmentation: UnitySemanticSegmentation
    bounding_boxes_2d: Unity2DBoundingBox
    bounding_boxes_3d: Unity3DBoundingBox
    depth: UnityDepth

    @staticmethod
    def from_dict(_dict: dict, labels: List[Label]):
        path = _dict['path']
        annotations = {}

        for annotation in _dict['annotations']:
            annotations[annotation['@type'].split('/')[-1]] = annotation

        instance_segmentation = None if 'unity.solo.InstanceSegmentationAnnotation' not in annotations else \
            UnityInstanceSegmentation.from_dict(annotations['unity.solo.InstanceSegmentationAnnotation'], labels, path)
        semantic_segmentation = None if 'unity.solo.SemanticSegmentationAnnotation' not in annotations else \
            UnitySemanticSegmentation.from_dict(annotations['unity.solo.SemanticSegmentationAnnotation'], labels, path)
        bounding_boxes_2d = None if 'unity.solo.BoundingBox2DAnnotation' not in annotations else \
            Unity2DBoundingBox.from_dict(annotations['unity.solo.BoundingBox2DAnnotation'], labels)
        bounding_boxes_3d = None if 'unity.solo.BoundingBox3DAnnotation' not in annotations else \
            Unity3DBoundingBox.from_dict(annotations['unity.solo.BoundingBox3DAnnotation'], labels)
        depth = None if 'unity.solo.DepthAnnotation' not in annotations else \
            UnityDepth.from_dict(annotations['unity.solo.DepthAnnotation'], path)

        return UnityCapture(_dict['id'],
                            int(path.split('/')[-1].split('.')[-1]),
                            _dict['description'],
                            _dict['position'],
                            _dict['rotation'],
                            _dict['velocity'],
                            _dict['acceleration'],
                            path + '/' + _dict['filename'],
                            _dict['imageFormat'],
                            _dict['dimension'],
                            _dict['projection'],
                            _dict['matrix'],
                            instance_segmentation,
                            semantic_segmentation,
                            bounding_boxes_2d,
                            bounding_boxes_3d,
                            depth)


class UnityData:
    data_path: str
    labels: List[Label]
    captures: List[UnityCapture]

    def read_labels(self):
        with open(f"{self.data_path}/annotation_definitions.json") as f:
            data = json.load(f)
            annotation_definitions = data['annotationDefinitions']

        labels = []
        counter = 0

        for annotation_definition in annotation_definitions:
            if 'spec' not in annotation_definition or not isinstance(annotation_definition['spec'], list) or len(
                    annotation_definition['spec']) == 0:
                continue

            for spec in annotation_definition['spec']:
                if 'label_id' not in spec or 'label_name' not in spec:
                    continue

                # If no label with name exists, create a label
                if not any(label.name == spec['label_name'] for label in labels):
                    labels.append(Label(counter, spec['label_id'], spec['label_name']))
                    counter += 1

        self.labels = labels

    def read_captures(self):
        frame_data_files = glob.glob(self.data_path + '/**/*frame_data.json', recursive=True)
        frame_data_files.sort(key=lambda x: int(x.split('/')[-2].split('.')[-1]))

        captures = []
        for file in frame_data_files:
            with open(file) as f:
                data_dict = json.load(f)

                # Continue if data_dict has no key captures or captures is empty or not an array
                if 'captures' not in data_dict or not isinstance(data_dict['captures'], list) or len(
                        data_dict['captures']) == 0:
                    continue

                for capture in data_dict['captures']:
                    capture['path'] = '/'.join(file.split('/')[:-1])
                    captures.append(capture)

        self.captures = [UnityCapture.from_dict(capture, self.labels) for capture in captures]

    def __init__(self, data_path: str):
        self.data_path = data_path[:-1] if data_path.endswith('/') else data_path
        self.read_labels()
        self.read_captures()
