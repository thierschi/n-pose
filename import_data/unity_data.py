import glob
import json
from dataclasses import dataclass
from typing import List

from util.color import RGBAColor


@dataclass(frozen=True)
class UnityLabel:
    id: int
    unity_id: int
    name: str


@dataclass(frozen=True)
class UnityInstanceSegmentationInstance:
    instance_id: int
    label: UnityLabel
    color: RGBAColor

    @staticmethod
    def from_dict(_dict: dict, labels: List[UnityLabel]):
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
class UnityInstanceSegmentationAnnotation:
    type_name = 'type.unity.com/unity.solo.InstanceSegmentationAnnotation'

    id: str
    sensor_id: str
    description: str
    image_format: str
    dimension: List[float]
    file_path: str
    instances: List[UnityInstanceSegmentationInstance]

    @staticmethod
    def from_dict(_dict: dict, labels: List[UnityLabel], path: str):
        instances = [UnityInstanceSegmentationInstance.from_dict(instance, labels) for instance in
                     _dict['instances']] if 'instances' in _dict else []

        return UnityInstanceSegmentationAnnotation(_dict['id'],
                                                   _dict['sensorId'],
                                                   _dict['description'],
                                                   _dict['imageFormat'],
                                                   _dict['dimension'],
                                                   path + '/' + _dict['filename'],
                                                   instances)


@dataclass(frozen=True)
class UnitySemanticSegmentationInstance:
    label: UnityLabel
    color: RGBAColor

    @staticmethod
    def from_dict(_dict: dict, labels: List[UnityLabel]):
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
class UnitySemanticSegmentationAnnotation:
    type_name = 'type.unity.com/unity.solo.SemanticSegmentationAnnotation'

    id: str
    sensor_id: str
    description: str
    image_format: str
    dimension: List[float]
    file_path: str
    instances: List[UnitySemanticSegmentationInstance]

    @staticmethod
    def from_dict(_dict: dict, labels: List[UnityLabel], path: str):
        instances = [UnitySemanticSegmentationInstance.from_dict(instance, labels) for instance in
                     _dict['instances']] if 'instances' in _dict else []

        return UnitySemanticSegmentationAnnotation(_dict['id'],
                                                   _dict['sensorId'],
                                                   _dict['description'],
                                                   _dict['imageFormat'],
                                                   _dict['dimension'],
                                                   path + '/' + _dict['filename'],
                                                   instances)


@dataclass(frozen=True)
class Unity2DBoundingBoxValue:
    instance_id: int
    label: UnityLabel
    origin: List[float]
    dimension: List[float]

    @staticmethod
    def from_dict(_dict: dict, labels: List[UnityLabel]):
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
class Unity2DBoundingBoxAnnotation:
    type_name = 'type.unity.com/unity.solo.BoundingBox2DAnnotation'

    id: str
    sensor_id: str
    description: str
    values: List[Unity2DBoundingBoxValue]

    @staticmethod
    def from_dict(_dict: dict, labels: List[UnityLabel]):
        values = [Unity2DBoundingBoxValue.from_dict(value, labels) for value in
                  _dict['values']] if 'values' in _dict else []

        return Unity2DBoundingBoxAnnotation(_dict['id'],
                                            _dict['sensorId'],
                                            _dict['description'],
                                            values)


@dataclass(frozen=True)
class UnityDepthAnnotation:
    type_name = 'type.unity.com/unity.solo.DepthAnnotation'

    sensor_id: str
    description: str
    measurement_strategy: str
    image_format: str
    dimension: List[float]
    file_path: str

    @staticmethod
    def from_dict(_dict: dict, path: str):
        return UnityDepthAnnotation(_dict['sensorId'],
                                    _dict['description'],
                                    _dict['measurementStrategy'],
                                    _dict['imageFormat'],
                                    _dict['dimension'],
                                    path + '/' + _dict['filename'])


@dataclass(frozen=True)
class Unity3DBoundingBoxValue:
    instance_id: int
    label: UnityLabel
    translation: List[float]
    size: List[float]
    rotation: List[float]
    velocity: List[float]
    acceleration: List[float]

    @staticmethod
    def from_dict(_dict: dict, labels: List[UnityLabel]):
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
class Unity3DBoundingBoxAnnotation:
    type_name = 'type.unity.com/unity.solo.BoundingBox3DAnnotation'

    id: str
    sensor_id: str
    description: str
    values: List[Unity3DBoundingBoxValue]

    @staticmethod
    def from_dict(_dict: dict, labels: List[UnityLabel]):
        values = [Unity3DBoundingBoxValue.from_dict(value, labels) for value in
                  _dict['values']] if 'values' in _dict else []

        return Unity3DBoundingBoxAnnotation(_dict['id'],
                                            _dict['sensorId'],
                                            _dict['description'],
                                            values)


@dataclass
class UnityKeypoint:
    index: int
    location: List[float]
    camera_cartesian_location: List[float]
    state: int

    @staticmethod
    def from_dict(_dict: dict):
        return UnityKeypoint(_dict['index'],
                             _dict['location'],
                             _dict['cameraCartesianLocation'],
                             _dict['state'])


@dataclass
class UnityKeypointValue:
    instance_id: int
    label: UnityLabel
    pose: str
    keypoints: List[UnityKeypoint]

    @staticmethod
    def from_dict(_dict: dict, labels: List[UnityLabel]):
        label = None

        for label in labels:
            if label.unity_id == _dict['labelId']:
                label = label
                break

        keypoints = [UnityKeypoint.from_dict(keypoint) for keypoint in _dict['keypoints']]

        return UnityKeypointValue(_dict['instanceId'],
                                  label,
                                  _dict['pose'],
                                  keypoints)


@dataclass
class UnityKeypointAnnotation:
    type_name = 'type.unity.com/unity.solo.KeypointAnnotation'

    id: str
    sensor_id: str
    description: str
    template_id: str
    values: List[UnityKeypointValue]

    @staticmethod
    def from_dict(_dict: dict, labels: List[UnityLabel]):
        values = [UnityKeypointValue.from_dict(value, labels) for value in
                  _dict['values']] if 'values' in _dict else []

        return UnityKeypointAnnotation(_dict['id'],
                                       _dict['sensorId'],
                                       _dict['description'],
                                       _dict['templateId'],
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
    instance_segmentation: UnityInstanceSegmentationAnnotation
    semantic_segmentation: UnitySemanticSegmentationAnnotation
    bounding_boxes_2d: Unity2DBoundingBoxAnnotation
    bounding_boxes_3d: Unity3DBoundingBoxAnnotation
    depth: UnityDepthAnnotation
    keypoints: UnityKeypointAnnotation

    @staticmethod
    def from_dict(_dict: dict, labels: List[UnityLabel]):
        path = _dict['path']
        annotations = {}

        for annotation in _dict['annotations']:
            annotations[annotation['@type']] = annotation

        instance_segmentation = None if UnityInstanceSegmentationAnnotation.type_name not in annotations else \
            UnityInstanceSegmentationAnnotation.from_dict(annotations[UnityInstanceSegmentationAnnotation.type_name],
                                                          labels, path)
        semantic_segmentation = None if UnitySemanticSegmentationAnnotation.type_name not in annotations else \
            UnitySemanticSegmentationAnnotation.from_dict(annotations[UnitySemanticSegmentationAnnotation.type_name],
                                                          labels, path)
        bounding_boxes_2d = None if Unity2DBoundingBoxAnnotation.type_name not in annotations else \
            Unity2DBoundingBoxAnnotation.from_dict(annotations[Unity2DBoundingBoxAnnotation.type_name], labels)
        bounding_boxes_3d = None if Unity3DBoundingBoxAnnotation.type_name not in annotations else \
            Unity3DBoundingBoxAnnotation.from_dict(annotations[Unity3DBoundingBoxAnnotation.type_name], labels)
        depth = None if UnityDepthAnnotation.type_name not in annotations else \
            UnityDepthAnnotation.from_dict(annotations[UnityDepthAnnotation.type_name], path)
        keypoints = None if UnityKeypointAnnotation.type_name not in annotations else \
            UnityKeypointAnnotation.from_dict(annotations[UnityKeypointAnnotation.type_name], labels)

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
                            depth,
                            keypoints)


class UnityData:
    data_path: str
    labels: List[UnityLabel]
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
                    labels.append(UnityLabel(counter, spec['label_id'], spec['label_name']))
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
