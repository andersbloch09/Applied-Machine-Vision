import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

# Define the same model architecture
class CombinedModel(nn.Module):
    def __init__(self, base_model):
        super(CombinedModel, self).__init__()
        self.base_model = base_model
        self.bbox_head = nn.Sequential(
            nn.Linear(128, 4),  # 4 outputs for bounding box (x_min, y_min, x_max, y_max)
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.base_model(x)
        classification = features
        bbox = self.bbox_head(features)
        return classification, bbox

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
base_model.classifier = nn.Sequential(
    nn.Linear(base_model.last_channel, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 4)  # 4 classes for classification
)
model = CombinedModel(base_model)
model.load_state_dict(torch.load("screw_cnn_fold1.pth", map_location=device))
model.to(device)
model.eval()

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names (ensure these match your training dataset)
class_names = ['Screwtype1', 'Screwtype2', 'Screwtype3', 'Screwtype4']

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Preprocess the frame
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    input_tensor = transform(input_frame).unsqueeze(0).to(device)  # Add batch dimension

    # Make predictions
    with torch.no_grad():
        classifications, bboxes = model(input_tensor)
        _, predicted_class = torch.max(classifications, 1)
        predicted_label = class_names[predicted_class.item()]
        bbox = bboxes[0].cpu().numpy()

    # Scale bounding box coordinates back to the original frame size
    h, w, _ = frame.shape
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)

    # Draw the bounding box and label on the frame
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(frame, predicted_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Live Screw Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()