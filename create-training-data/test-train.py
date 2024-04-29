from convert_data import *
from ultralytics import YOLO

captures = get_captures_from_data('../first-training/data')
labels = get_labels('../first-training/data')
yaml_path = create_training_data('./training_data', captures, labels)



# Load a model
model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data=yaml_path, epochs=1, imgsz=640)

print(results)