from ultralytics import YOLO

from ...data.unity_data import UnityData
from ...data.yolo_conversion import unity_to_yolo_seg

data = UnityData('data/solo_93')
yaml_path = unity_to_yolo_seg(data, 'data/solo_93_yolo_seg', include_test=True)

model = YOLO("yolov8x-seg.pt")

results = model.train(data=yaml_path, epochs=100, imgsz=640, device='cuda', project='logs/solo_93_seg')
