from ultralytics import YOLO

from src.data.unity_data import UnityData
from src.data.yolo_conversion import unity_to_yolo_seg

data = UnityData('data/solo_161')
yaml_path = unity_to_yolo_seg(data, 'data/solo_161_yolo_seg', include_test=True)

model = YOLO("yolov8x-seg.pt")

results = model.train(data=yaml_path, epochs=100, imgsz=640, device='cuda', project='logs/solo_161_seg')
