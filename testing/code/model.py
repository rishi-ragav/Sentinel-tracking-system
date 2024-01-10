import cv2
from ultralytics import YOLO
#from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import torch
import cv2 
import numpy as np
import pathlib
import matplotlib.pyplot as plt



model = YOLO("Rishi Weights")
model.predict(source="0", show=True, conf=0.5)