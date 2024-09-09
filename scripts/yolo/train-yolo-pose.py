import os

from ultralytics import YOLO

from src.data.unity_data import UnityData
from src.data.yolo_conversion import unity_to_yolo_pose

path = os.getcwd()
while not os.path.exists(os.path.join(path, 'src')):
    path = os.path.join(path, "..")
os.chdir(path)

data = UnityData('data/solo_161')
yaml_path = unity_to_yolo_pose(data, 'data/solo_161_yolo_pose', 5, [0, 2, 1, 4, 3], include_test=True)

model = YOLO("yolov8x-pose.pt")

results = model.train(data=yaml_path, epochs=100, imgsz=640, device='cuda', project='logs/solo_161_pose')
