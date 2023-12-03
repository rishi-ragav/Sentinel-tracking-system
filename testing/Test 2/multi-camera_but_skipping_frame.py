import cv2
from ultralytics import YOLO
import cv2
from ultralytics import YOLO



# Initialize webcams
cap1 = cv2.VideoCapture(0)  # Use the appropriate index for the first webcam
cap2 = cv2.VideoCapture(1)  # Use the appropriate index for the second webcam

# Load YOLO model
model = YOLO("D:/OneDrive/OneDrive - Student Ambassadors/College notes/Sem 7/github/Sentinel-tracking-system/testing/Test 2/best_human5s.pt")

while True:
    # Read frames from both webcams
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # Predict using YOLO model
    predictions1 = model.predict(source=frame1, show=True, conf=0.5)
    predictions2 = model.predict(source=frame2, show=True, conf=0.5)

    # Display predictions in separate windows

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcams and close OpenCV windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
