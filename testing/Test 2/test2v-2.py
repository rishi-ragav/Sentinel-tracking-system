import cv2
from ultralytics import YOLO
from ultralytics.models.yolo import DetectionPredictor

model = YOLO("D:/OneDrive/OneDrive - Student Ambassadors/College notes/Sem 7/github/Sentinel-tracking-system/testing/Test 2/model.pt")
model.predict(source="0", show=True, conf=0.75)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Check if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
