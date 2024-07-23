import cv2
from ultralytics import YOLO

from import_data.unity_data import UnityData
from util.polygons import get_colored_polygons_from_mask

data = UnityData('../data/solo_93')
model = YOLO("../solo_93_seg_x.pt")

boundary_points = []
yolo_boundary_points = []

for capture in data.captures:
    print('-------------------------------------------')
    print(f'Processing {capture.sequence}-{capture.id}')

    if capture.instance_segmentation is not None:
        print(f'Found {len(capture.instance_segmentation.instances)} instances in {capture.sequence}-{capture.id}')
        for instance in capture.instance_segmentation.instances:
            img = cv2.imread(capture.instance_segmentation.file_path)
            _, norm_polygons = get_colored_polygons_from_mask(img, instance.color)

            points = 0
            for polygon in norm_polygons:
                boundary_x = polygon.boundary.coords.xy[0]
                points += len(boundary_x)

            boundary_points.append(points)
            print(
                f'Found {points} boundary points for {instance.label.name} ({instance.instance_id}) in {capture.sequence}-{capture.id}')

    prediction = model.predict(capture.file_path)
    masks = prediction[0].masks

    if masks is None:
        print(f'YOLO: No masks found for {capture.sequence}-{capture.id}')
        continue

    boundaries = masks.xy
    print(f'YOLO found {len(masks)} instances for {capture.sequence}-{capture.id}')

    i = 0
    for b in boundaries:
        yolo_boundary_points.append(len(b))
        print(f'Yolo found {len(b)} boundary points for instance {i} in {capture.sequence}-{capture.id}')
        i += 1

# Save the boundary_points to a text file. First line should be the array separated by spaces.
# After that write the average, mean, deviation, and min and max values
# Do the same for yolo_boundary_points
# The Path for the file is ../data/boundary_points.txt and ../data/yolo_boundary_points.txt
# Use the following format:
# boundary_points: 1 2 3 4 5
# average: 3
# mean: 3
# deviation: 1.4142135623730951
# min: 1
# max: 5
# yolo_boundary_points: 1 2 3 4 5
# average: 3
# mean: 3
# deviation: 1.4142135623730951
# min: 1
# max: 5
import numpy as np

boundary_points = np.array(boundary_points)
yolo_boundary_points = np.array(yolo_boundary_points)

with open('../data/boundary_points.txt', 'w') as f:
    f.write(f'boundary_points: {" ".join(map(str, boundary_points))}\n')
    f.write(f'average: {np.average(boundary_points)}\n')
    f.write(f'mean: {np.mean(boundary_points)}\n')
    f.write(f'deviation: {np.std(boundary_points)}\n')
    f.write(f'min: {np.min(boundary_points)}\n')
    f.write(f'max: {np.max(boundary_points)}\n')
    
with open('../data/yolo_boundary_points.txt', 'w') as f:
    f.write(f'yolo_boundary_points: {" ".join(map(str, yolo_boundary_points))}\n')
    f.write(f'average: {np.average(yolo_boundary_points)}\n')
    f.write(f'mean: {np.mean(yolo_boundary_points)}\n')
    f.write(f'deviation: {np.std(yolo_boundary_points)}\n')
    f.write(f'min: {np.min(yolo_boundary_points)}\n')
    f.write(f'max: {np.max(yolo_boundary_points)}\n')

print('Done')
