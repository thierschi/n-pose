import os
import glob
import json
import shutil

import numpy as np

from create_polygons import mask_to_polygons

def get_captures_from_data(data_path):
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    folders.sort(key=lambda x: int(x.split('.')[1]))

    metadata_files = glob.glob(f'{data_path}/**/*.frame_data.json', recursive=True)

    captures = []
    for metadata_file in metadata_files:
        metadata = json.load(open(metadata_file, 'r'))
        folder_path = os.path.dirname(metadata_file)

        if 'captures' not in metadata:
            continue

        local_captures = []
        for capture_data in metadata['captures']:
            instance_segmentation = next(
                (a for a in capture_data["annotations"]
                 if a['@type'] == "type.unity.com/unity.solo.InstanceSegmentationAnnotation"),
                None
            )

            if instance_segmentation is None:
                continue

            instances = []
            for instance in instance_segmentation['instances']:
                instances.append({
                    'label_id': instance["labelId"],
                    'color': instance["color"]
                })

            capture = {
                'image': f'{folder_path}/{capture_data["filename"]}',
                'instance_segmentation': f'{folder_path}/{instance_segmentation["filename"]}',
                'instances': instances,
            }
            local_captures.append(capture)
        captures.extend(local_captures)

    return captures


def get_labels(data_path):
    data = json.load(open(os.path.join(data_path, 'annotation_definitions.json'), 'r'))

    instance_segmentation_definitions = next(
                (a for a in data['annotationDefinitions']
                 if a['@type'] == "type.unity.com/unity.solo.InstanceSegmentationAnnotation"),
                None
            )

    if instance_segmentation_definitions is None:
        return None

    specs = instance_segmentation_definitions['spec']

    labels = {}
    for spec in specs:
        labels[spec['label_id']] = spec['label_name']

    return labels


def create_training_data(data_path, captures, labels, tests=True):
    if os.path.exists(data_path):
        shutil.rmtree(data_path)

    os.mkdir(data_path)

    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    test_path = os.path.join(data_path, 'test')

    os.mkdir(train_path)
    os.mkdir(val_path)
    if tests:
        os.mkdir(test_path)

    yolo_captures = []
    for capture in captures:
        polygons, normalized_polygons = mask_to_polygons(capture['instance_segmentation'])

        polygons_as_str = []
        for polygon in normalized_polygons:
            boundary_x = polygon.boundary.coords.xy[0]
            boundary_y = polygon.boundary.coords.xy[1]

            coord_pairs = []
            for i, x in enumerate(boundary_x):
                y = boundary_y[i]
                coord_pairs.append(f'{x} {y}')

            polygons_as_str.append(f'{capture["instances"][0]["label_id"] - 1} {" ".join(coord_pairs)}')

            yolo_captures.append({
                'image': capture['image'],
                'mask': '\n'.join(polygons_as_str),
            })

    shuffled_yolo_captures = np.array(yolo_captures)
    np.random.shuffle(shuffled_yolo_captures)

    training = shuffled_yolo_captures[:int(len(shuffled_yolo_captures) * (0.8 if tests else 0.9))]
    validation = shuffled_yolo_captures[-int(len(shuffled_yolo_captures) * 0.1):]
    testing = shuffled_yolo_captures[-int(len(shuffled_yolo_captures) * 0.1):] if tests else []

    i = 0

    for train in training:
        shutil.copyfile(train['image'], os.path.join(train_path, str(i) + '.png'))
        with open(os.path.join(train_path, str(i) + '.txt'), 'w') as f:
            f.write(train['mask'])
        i += 1

    for val in validation:
        shutil.copyfile(val['image'], os.path.join(val_path, str(i) + '.png'))
        with open(os.path.join(val_path, str(i) + '.txt'), 'w') as f:
            f.write(val['mask'])
        i += 1

    if tests:
        for test in testing:
            shutil.copyfile(test['image'], os.path.join(test_path, str(i) + '.png'))
            with open(os.path.join(test_path, str(i) + '.txt'), 'w') as f:
                f.write(test['mask'])
            i += 1

    yaml_path = os.path.join(data_path, 'training.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f'path: {os.path.abspath(data_path)}\n')
        f.write('train: train\n')
        f.write('val: val\n')
        if tests:
            f.write('test: test\n')
        f.write('\n')
        f.write('names:\n')
        for i, (k, v) in enumerate(labels.items()):
            f.write(f'    {k - 1}: {v}\n')

    return yaml_path

