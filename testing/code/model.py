import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Step 1: Load the images from the my-images folder
image_dir = 'my-images'
images = []
for filename in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir, filename))
    if img is not None:
        images.append(img)

# Step 2: Preprocess the images
img_size = (100, 100)
gray_images = []
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, img_size)
    gray_images.append(gray_resized)

# Step 3: Use a pre-trained face detection model to detect faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_regions = []
for gray_img in gray_images:
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face_regions.append(gray_img[y:y+h, x:x+w])

# Step 4: Extract the face regions from the images
face_size = (50, 50)
face_images = []
for face in face_regions:
    face_resized = cv2.resize(face, face_size)
    face_images.append(face_resized)

# Step 5: Preprocess the face regions
X = np.array(face_images) / 255.0
y = np.ones(len(X))

# Step 6: Train a new model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X[0].shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, validation_split=0.2)

# Step 7: Evaluate the model
test_dir = 'test-images'
test_images = []
for filename in os.listdir(test_dir):
    img = cv2.imread(os.path.join(test_dir, filename))
    if img is not None:
        test_images.append(img)
test_gray_images = []
for img in test_images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, img_size)
    test_gray_images.append(gray_resized)
test_face_regions = []
for gray_img in test_gray_images:
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        test_face_regions.append(gray_img[y:y+h, x:x+w])
test_face_images = []
for face in test_face_regions:
    face_resized = cv2.resize(face, face_size)
    test_face_images.append(face_resized)
X_test = np.array(test_face_images) / 255.0
y_test = np.ones(len(X_test))
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')
# Step 8: Export the result accuracy in a text file which appends with each execution
with open('accuracy.txt', 'a') as f:
    f.write(f'Test accuracy: {accuracy}\n')
