import cv2
import numpy as np
from ultralytics import YOLO

def predict_objects1():
    model = YOLO("D:/OneDrive/OneDrive - Student Ambassadors/College notes/Sem 7/github/Sentinel-tracking-system/testing/Test 2/best_human5s.pt")
    model.predict(source="1", show=True, conf=0.5)

predict_objects1()
