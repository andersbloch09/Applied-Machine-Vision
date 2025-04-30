import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from collections import Counter

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = "final_model.pth"

# Update label names
CLASS_NAMES = ['No screw', 'Bolt', 'Wood screw', 'Machine screw', 'Hook screw']
NUM_CLASSES = len(CLASS_NAMES)

# --- Load Model ---
model = models.mobilenet_v2()
model.classifier = nn.Sequential(
    nn.Linear(model.last_channel, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, NUM_CLASSES)
)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- Transforms ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- Prediction State ---
frame_buffer = []
buffer_size = 10
last_prediction = None

print("Running classification... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
        pred_label = CLASS_NAMES[pred.item()]
        frame_buffer.append(pred_label)

    # Keep buffer size
    if len(frame_buffer) > buffer_size:
        frame_buffer.pop(0)

    # Classify only if stable
    if len(frame_buffer) == buffer_size:
        most_common, count = Counter(frame_buffer).most_common(1)[0]
        if count > buffer_size // 2 and most_common != last_prediction:
            last_prediction = most_common
            print(f"Detected: {most_common} ({confidence.item() * 100:.1f}%)")

    # Visual display
    text = f"{pred_label} ({confidence.item() * 100:.1f}%)"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Screw Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
