import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from collections import Counter

# Configure logging
logging.basicConfig(
    filename="training_log_final.txt",  # Log file name
    level=logging.INFO,           # Log level
    format="%(asctime)s — %(message)s",  # Log format with timestamp
    datefmt="%Y-%m-%d %H:%M:%S"   # Date format
)

# Path to dataset (must contain subdirectories for each class: Noscrew1 + Screwtype1–4)
dataset_path = "./dataset_path"

# Hyperparameters
IMG_HEIGHT, IMG_WIDTH = 224, 224  # Image dimensions for resizing
BATCH_SIZE = 32                  # Number of samples per batch
LEARNING_RATE = 0.001            # Learning rate for the optimizer
EPOCHS = 5                       # Number of training epochs
NUM_CLASSES = 5                  # Total number of classes (Noscrew1 + 4 screw types)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),  # Resize images to 224x224
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels (for compatibility with MobileNetV2)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random lighting adjustments
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406],  # Normalize using ImageNet mean and std
                         [0.229, 0.224, 0.225]),
])

# Load the entire dataset using ImageFolder
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
class_names = full_dataset.classes  # Get class names from dataset
print("Classes:", class_names)

# Create DataLoader for the entire dataset
train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Compute class distribution in the dataset
labels_in_dataset = full_dataset.targets
counts = Counter(labels_in_dataset)
max_count = max(counts.values())
weights = [max_count / counts[i] for i in range(NUM_CLASSES)]
weight_tensor = torch.tensor(weights, dtype=torch.float, device="cuda" if torch.cuda.is_available() else "cpu")
print("Class counts:", counts)
print("Loss weights:", weights)

# Initialize MobileNetV2 model with pre-trained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.classifier = nn.Sequential(
    nn.Linear(model.last_channel, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, NUM_CLASSES),
)
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = out.argmax(dim=1)
        running_total += lbls.size(0)
        running_correct += (preds == lbls).sum().item()

    train_acc = 100 * running_correct / running_total
    print(f"Epoch {epoch}/{EPOCHS} — Train Loss: {running_loss:.4f}, Acc: {train_acc:.2f}%")
    logging.info(f"Epoch {epoch}/{EPOCHS} — Train Loss: {running_loss:.4f}, Acc: {train_acc:.2f}%")

# Save the final model
torch.save(model.state_dict(), "final_model.pth")
print("\nFinal model saved as final_model.pth")
logging.info("Final model saved as final_model.pth")