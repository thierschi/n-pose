import os

from ultralytics import YOLO

from src.data.unity_data import UnityData
from src.data.yolo_conversion import unity_to_yolo_pose

"""
This file can be used to train the YOLO pose model on the Unity dataset.
A Script for training all models at once can be found in the scripts/yolo directory.
"""

# Change directory to the root of the project
path = os.getcwd()
while not os.path.exists(os.path.join(path, 'src')):
    path = os.path.join(path, "..")
os.chdir(path)

# Change these paths to the correct paths for your data
DATA_PATH = 'data/solo_161'
LOG_PATH = 'logs/solo_161'

# Device: cpu for CPU, cuda for NVIDIA GPU, mps for Apple Silicon
DEVICE = 'cuda'

# Load and convert data
data = UnityData(DATA_PATH)
yaml_path = unity_to_yolo_pose(data, f'{DATA_PATH}_yolo_pose', 5, [0, 2, 1, 4, 3], include_test=True)

# Train the pose model
model = YOLO("yolov8x-pose.pt")
results = model.train(data=yaml_path, epochs=100, imgsz=640, device=DEVICE, project=f'{LOG_PATH}_pose')
