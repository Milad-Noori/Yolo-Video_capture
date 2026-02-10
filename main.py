# https://docs.ultralytics.com/models/yolov8
from ultralytics import YOLO
#
# path = "images/sampleImage.jpg"
# model = YOLO("model/yolov10x.pt")
#  5 persons, 16 cars, 409.4ms - Speed: 2.2ms
#
# model = YOLO("model/yolov8x.pt")
#  5 persons, 22 cars, 1 truck, 1 traffic light, 519.7ms Speed: 1.7ms
#
#
# results = model.predict(source=path)
# results[0].show()
# print(results[0].names)
#
# endregion
#
# region Step 2 ...
#
# path = "images/sampleImage.jpg"
# model = YOLO("yolov8n.pt")
# results = model.predict(source=path, classes=[2])
# results[0].show()

path = "images/sampleImage.jpg"
model = YOLO("model/yolov8x.pt")
selected_class = [0,2]
results = model.predict(source=path, classes=selected_class, save=True, conf = 0.7)
results[0].show()
path = "images/sampleImage.jpg"
model = YOLO("model/yolov8x.pt")
selected_class = [0,2]
results = model.predict(source=path, classes=selected_class, save=True, conf = 0.7)
results[0].show()
results[0].save(filename="result.jpg")
########################################################

import cv2

def process_video(path: str):
    vs = cv2.VideoCapture(path)
    model = YOLO("model/yolov8x.pt")

    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        results = model.predict(frame, stream=False)
        detection_classes = results[0].names
        # results[0].show()
        for result in results:
            for data in result.boxes.data.tolist():
                # print(data)
                code = data[5]
                draw_box(data=data, image=frame, name=detection_classes[code])
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)