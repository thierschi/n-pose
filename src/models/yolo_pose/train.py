from ultralytics import YOLO

from ...data.unity_data import UnityData
from ...data.yolo_conversion import unity_to_yolo_pose

data = UnityData('data/solo_93')
yaml_path = unity_to_yolo_pose(data, 'data/solo_93_yolo', 5, [0, 2, 1, 4, 3], include_test=True)

model = YOLO("logs/solo_90/train/weights/solo_93_seg_x.pt")

results = model.train(data=yaml_path, epochs=100, imgsz=640, device='cuda', project='logs/solo_93')
