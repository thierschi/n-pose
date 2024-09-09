import os

from ultralytics import YOLO

from src.data.unity_data import UnityData
from src.data.yolo_conversion import unity_to_yolo_pose, unity_to_yolo_seg

path = os.getcwd()
while not os.path.exists(os.path.join(path, 'src')):
    path = os.path.join(path, "..")
os.chdir(path)

data = UnityData('data/solo_161')
pose_yaml_path = unity_to_yolo_pose(data, 'data/solo_161_yolo_pose', 5, [0, 2, 1, 4, 3], include_test=True)
yaml_path = unity_to_yolo_seg(data, 'data/solo_161_yolo_seg', include_test=True)

model = YOLO("yolov8x-pose.pt")
results = model.train(data=pose_yaml_path, epochs=100, imgsz=640, device='cuda', project='logs/solo_161_pose')

model = YOLO("yolov8x-seg.pt")
results = model.train(data=yaml_path, epochs=100, imgsz=640, device='cuda', project='logs/solo_161_seg')
