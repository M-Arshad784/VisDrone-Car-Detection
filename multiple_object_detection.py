# import cv2
# import cvzone
# from ultralytics import YOLO, YOLOWorld
#
# img = cv2.imread('2j.jpg')
# img = cv2.resize(img, (900, 700))
#
# # Load all models
# model = YOLO('best.pt')
#
# # Run detection
# results = model(img, conf=0.25)
#
# for result in results:
#     for box in result.boxes:
#
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         conf = float(box.conf[0])
#         cls = int(box.cls[0])
#         name = model.names[cls]
#
#         cvzone.cornerRect(img, (x1, y1, x2, y2))
#         cvzone.putTextRect(img,f'{name} {conf:.2f}',(x1, y1 - 10),colorR=(255, 0, 0),scale=0.8, thickness=1)
#
#
# cv2.imshow('Object Detection', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import cvzone
from fsspec.config import conf
from ultralytics import YOLO, YOLOWorld

cap = cv2.VideoCapture('video.mp4')
model = YOLO('best.pt')

while True:
    success, img = cap.read()
    img = cv2.resize(img, (500, 500))
    results = model(img, conf=0.7)
    if not success:
        break

    for result in results:
        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = model.names[cls]
            cvzone.cornerRect(img, (x1, y1, x2, y2))
            cvzone.putTextRect(img, f'{name}, {conf:.2f}', (x1, y1-10), colorR=(255, 0, 0), scale=0.8, thickness=1)

    cv2.imshow('Car Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



