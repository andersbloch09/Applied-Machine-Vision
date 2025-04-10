import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np  

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture (must match training script)
model = models.mobilenet_v2(pretrained=False)
model.classifier = nn.Sequential(
    nn.Linear(model.last_channel, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 4)  # 4 screw classes
)

# Load trained weights
model.load_state_dict(torch.load("screw_cnn_fold1.pth", map_location=device))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load and preprocess image
image_path = r"C:\Users\ander\OneDrive\UNI\VT2\Applied Machine Vision\Applied-Machine-Vision\dataset_path\Screwtype3\Screwtype3_266.jpg"  # <- replace with your test image
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dim

# Inference
with torch.no_grad():
    output = model(img_tensor)
    predicted_class = output.argmax(1).item()

print(f"Predicted class index: {predicted_class}")

# Enable cudnn benchmark for faster performance if input size is fixed
torch.backends.cudnn.benchmark = True

# Image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Class labels
class_names = ['Screwtype1', 'Screwtype2', 'Screwtype3', 'Screwtype4']

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert to RGB and preprocess
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(input_frame).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        predicted_label = class_names[predicted_class.item()]

    # Draw prediction on frame
    cv2.putText(frame, predicted_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Live Screw Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
