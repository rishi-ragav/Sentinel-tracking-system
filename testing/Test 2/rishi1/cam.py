from ultralytics import YOLO
import cv2

model = YOLO('rishi1n_50e_seed14.pt')
model.predict(source="1", show=True, conf=0.5)