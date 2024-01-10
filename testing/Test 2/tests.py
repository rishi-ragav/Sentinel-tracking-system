import cv2
import torch
import numpy as np
from torchvision import transforms
from deep_sort import build_tracker
from deep_sort import feature_extractor

# Load face recognition model
model = torch.load('best_human5n.pt')
model.eval()

# Define a transformation for input images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Function to perform face recognition and return features
def extract_features(frame, faces):
    features = []

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_tensor = transform(face).unsqueeze(0)

        # Forward pass through the face recognition model
        with torch.no_grad():
            feature = model(face_tensor)

        features.append(feature.numpy())

    return features

# Initialize DeepSORT tracker
tracker = build_tracker("deep_sort_model/mars-small128.pb", use_cuda=True)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Perform face detection (replace with your own detection logic)
    faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Extract features for DeepSORT input
    features = extract_features(frame, faces)

    # Update DeepSORT tracker
    trackers = tracker.update(features)

    # Draw bounding boxes on the frame
    for bbox in trackers:
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Tracking', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
