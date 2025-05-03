from ultralytics import YOLO
import torch

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load a YOLOv8 model (nano, small, or medium)
model = YOLO("yolov8s.pt")  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.

# Train the model
results = model.train(
    data=r"datasets\screws\screw.yaml",         # Path to your dataset config file
    epochs=50,                 # Number of training epochs
    imgsz=640,                 # Input image size
    batch=16,                  # Adjust depending on GPU RAM
    device=0,                  # 0 = first GPU (or 'cpu' if you want to force CPU)
    name="screw_detector_v1",  # Experiment name / output folder
    workers=4,                 # Number of data loading workers
    verbose=True               # Print training progress
)

print("✅ Training complete. Weights saved to:")
print("runs/detect/screw_detector_v1/weights/best.pt")
