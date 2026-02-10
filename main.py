# https://docs.ultralytics.com/models/yolov8
from ultralytics import YOLO

path = "images/sampleImage.jpg"
model = YOLO("model/yolov10x.pt")
 5 persons, 16 cars, 409.4ms - Speed: 2.2ms

model = YOLO("model/yolov8x.pt")
 5 persons, 22 cars, 1 truck, 1 traffic light, 519.7ms Speed: 1.7ms
