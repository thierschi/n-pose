from ultralytics import YOLO

from convert_data.unity_to_yolo_seg import unity_to_yolo_seg
from import_data.unity_data import UnityData

data = UnityData('data/solo_93')
yaml_path = unity_to_yolo_seg(data, 'data/solo_93_yolo_seg', include_test=True)

model = YOLO("yolov8x-seg.pt")

results = model.train(data=yaml_path, epochs=100, imgsz=640, device='cuda', project='logs/solo_93_seg')
