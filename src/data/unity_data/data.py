import glob
import json
from typing import List, Dict

from .capture import Capture
from .label import Label


class UnityData:
    """
    UnityData represents a collection of captures from Unity.
    """
    data_path: str
    labels: List[Label]
    sequences: Dict[int, List[Capture]]

    def read_labels(self):
        """
        Read labels from annotation_definitions.json.
        """
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
        """
        Read captures from frame_data.json files.
        """
        # Find all frame_data.json files in the sequences
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

        # Collect sequences for efficient access
        self.sequences = {}
        for capture in captures:
            c = Capture.from_dict(capture, self.labels)

            if c.sequence not in self.sequences:
                self.sequences[c.sequence] = []
            self.sequences[c.sequence].append(c)

    def __init__(self, data_path: str):
        self.data_path = data_path[:-1] if data_path.endswith('/') else data_path
        self.read_labels()
        self.read_captures()

    @property
    def len_sequences(self):
        return len(self.sequences)

    @property
    def sequence_ids(self):
        return list(self.sequences.keys())

    def get_sequence(self, sequence: int) -> List[Capture]:
        """
        Get all captures in a sequence.
        :param sequence:
        :return: List of Captures
        """
        return self.sequences[sequence]

    @property
    def captures(self):
        return [capture for sequence in self.sequences.values() for capture in sequence]
