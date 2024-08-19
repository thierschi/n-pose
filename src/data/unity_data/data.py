import glob
import json
from typing import List

from .capture import Capture
from .label import Label


class UnityData:
    data_path: str
    labels: List[Label]
    captures: List[Capture]

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

        self.captures = [Capture.from_dict(capture, self.labels) for capture in captures]

    def __init__(self, data_path: str):
        self.data_path = data_path[:-1] if data_path.endswith('/') else data_path
        self.read_labels()
        self.read_captures()
