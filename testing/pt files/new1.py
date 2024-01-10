from ultralytics import YOLO

# Load a model
  # load an official model
model = YOLO('D:/OneDrive/OneDrive - Student Ambassadors/College notes/Sem 7/github/Sentinel-tracking-system/testing/pt files/detecting-rishi-v1-n.pt')  # load a custom trained model

# Export the model
model.export(format='tfjs')