import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO("best.pt")

# Image
image_path = "test.jpg"
results = model(image_path)
results[0].show()

# Webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("Fruit Freshness Detector", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
