import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np  

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = "screw_cnn_fold1.pth"  # or your final checkpoint
CLASS_NAMES = ['Noscrew1', 'Screwtype1', 'Screwtype2', 'Screwtype3', 'Screwtype4']
NUM_CLASSES = len(CLASS_NAMES)

# --- Model Definition & Load ---
model = models.mobilenet_v2(pretrained=False)
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
# static image
static_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Grayscale conversion
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Lighting adjustments
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# webcam frames
webcam_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),  # Grayscale conversion
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Lighting adjustments
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Static Image Inference ---
image_path = r".\dataset_path\Screwtype3\Screwtype3_266.jpg"
img = Image.open(image_path).convert("RGB")
img_tensor = static_transform(img).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    output = model(img_tensor)
    probs = torch.softmax(output, dim=1)
    confidence, pred = torch.max(probs, 1)
    label = CLASS_NAMES[pred.item()]
print(f"Static Image → {label} ({confidence.item()*100:.1f}%)")

# --- Live Webcam Inference ---
torch.backends.cudnn.benchmark = True
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # preprocess
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = webcam_transform(rgb).unsqueeze(0).to(DEVICE)

    # predict
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
        label = CLASS_NAMES[pred.item()]

    # overlay
    text = f"{label} ({confidence.item()*100:.1f}%)"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Screw Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
