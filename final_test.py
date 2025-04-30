import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from collections import Counter

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise fallback to CPU
WEIGHTS_PATH = "final_model.pth"  # Path to the trained model weights

CLASS_NAMES = ['No screw', 'Bolt', 'Wood screw', 'Machine screw', 'Hook screw']  # Class labels for predictions
NUM_CLASSES = len(CLASS_NAMES)  # Number of classes

# --- Load Model ---
# Initialize the MobileNetV2 model
model = models.mobilenet_v2()
# Replace the classifier with a custom one for the specific number of classes
model.classifier = nn.Sequential(
    nn.Linear(model.last_channel, 128),  # Fully connected layer with 128 units
    nn.ReLU(),  # Activation function
    nn.Dropout(0.4),  # Dropout for regularization
    nn.Linear(128, NUM_CLASSES)  # Output layer with NUM_CLASSES units
)
# Load the trained weights into the model
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)  # Move the model to the appropriate device (CPU/GPU)
model.eval()  # Set the model to evaluation mode (disables dropout, etc.)

# --- Transforms ---
# Define preprocessing transformations for input images
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array to PIL image
    transforms.Grayscale(num_output_channels=3),  # Convert to grayscale with 3 channels
    transforms.Resize((224, 224)),  # Resize to 224x224 (required by MobileNetV2)
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean
                         std=[0.229, 0.224, 0.225])  # Normalize using ImageNet std
])

# --- Webcam Setup ---
# Initialize the webcam
cap = cv2.VideoCapture(0)  # Open the default webcam (index 0)
if not cap.isOpened():
    print("Error: Could not open webcam.")  # Exit if the webcam cannot be accessed
    exit()

# --- Prediction State ---
frame_buffer = []  # Buffer to store recent predictions for stability
buffer_size = 10  # Number of frames to consider for stable predictions
last_prediction = None  # Store the last stable prediction

print("Running classification... Press 'q' to exit.")

# --- Main Loop ---
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if the frame cannot be read

    # Convert the frame from BGR (OpenCV format) to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Apply the preprocessing transformations
    tensor = transform(rgb).unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device

    # Perform inference
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(tensor)  # Get model predictions
        probs = torch.softmax(output, dim=1)  # Apply softmax to get probabilities
        confidence, pred = torch.max(probs, 1)  # Get the highest probability and its index
        pred_label = CLASS_NAMES[pred.item()]  # Map the index to the corresponding class label
        frame_buffer.append(pred_label)  # Add the prediction to the buffer

    # Maintain the buffer size
    if len(frame_buffer) > buffer_size:
        frame_buffer.pop(0)  # Remove the oldest prediction if the buffer exceeds its size

    # Determine stable predictions
    if len(frame_buffer) == buffer_size:
        # Find the most common prediction in the buffer
        most_common, count = Counter(frame_buffer).most_common(1)[0]
        # Update the last prediction if it is stable (appears more than half the buffer size)
        if count > buffer_size // 2 and most_common != last_prediction:
            last_prediction = most_common
            print(f"Detected: {most_common} ({confidence.item() * 100:.1f}%)")  # Print the stable prediction

    # Display the prediction on the video feed
    text = f"{pred_label} ({confidence.item() * 100:.1f}%)"  # Format the prediction text
    cv2.putText(frame, text, (10, 30),  # Add text to the frame
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Font, size, color, thickness
    cv2.imshow("Screw Classifier", frame)  # Show the frame with predictions

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows