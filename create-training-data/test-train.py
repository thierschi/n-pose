import sys

from ultralytics import YOLO

from convert_data import *

print("Python version")
print(sys.version)
print("Version info.")
print(sys.version_info)
print(os.getcwd())

captures = get_captures_from_data('../data/data')
labels = get_labels('../data/data')
yaml_path = create_training_data('../data/training_data', captures, labels)
# yaml_path = 'coco8-seg.yaml'

# Load a model
model = YOLO('yolov8m-seg.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data=yaml_path, epochs=100, imgsz=640, device='cpu', project='../logs')
