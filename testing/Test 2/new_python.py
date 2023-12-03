import cv2
import numpy as np
from ultralytics import YOLO

capCamera = cv2.VideoCapture(0)
capVideo = cv2.VideoCapture(1)

model = YOLO("D:/OneDrive/OneDrive - Student Ambassadors/College notes/Sem 7/github/Sentinel-tracking-system/testing/Test 2/best_human5s.pt")

model.predict(source="1", show=True, conf=0.5)
# Fix: Add display=True to show the prediction on the screen
