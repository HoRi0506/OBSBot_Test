from ultralytics import YOLO

model = YOLO(model='yolo11n-pose.pt')
model.export(format="onnx")