from ultralytics import YOLO

from convert_data.unity_to_yolo_pose import unity_to_yolo_pose
from import_data.unity_data import UnityData

data = UnityData('data/solo_93')
yaml_path = unity_to_yolo_pose(data, 'data/solo_93_yolo', 5, [0, 2, 1, 4, 3], include_test=True)

model = YOLO("logs/solo_90/train/weights/solo_93_seg_x.pt")

results = model.train(data=yaml_path, epochs=100, imgsz=640, device='cuda', project='logs/solo_93')
