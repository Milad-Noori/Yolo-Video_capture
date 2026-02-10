# https://docs.ultralytics.com/models/yolov8
from ultralytics import YOLO

path = "images/sampleImage.jpg"
model = YOLO("model/yolov10x.pt")
 5 persons, 16 cars, 409.4ms - Speed: 2.2ms

model = YOLO("model/yolov8x.pt")
 5 persons, 22 cars, 1 truck, 1 traffic light, 519.7ms Speed: 1.7ms


results = model.predict(source=path)
results[0].show()
print(results[0].names)

endregion

region Step 2 ...

path = "images/sampleImage.jpg"
model = YOLO("yolov8n.pt")
results = model.predict(source=path, classes=[2])
results[0].show()
