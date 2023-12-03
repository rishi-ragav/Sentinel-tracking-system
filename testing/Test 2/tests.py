import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera 0")
else:
    print("Camera 0 is opened successfully")

cap.release()
