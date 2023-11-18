import cv2
from ultralytics import YOLO
#from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from ultralytics import YOLO
import torch
import cv2 
import numpy as np
import pathlib
import matplotlib.pyplot as plt



model = YOLO("D:/Downloads/best.pt")
model.predict(source="0", show=True, conf=0.5)
