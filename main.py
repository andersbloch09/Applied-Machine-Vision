import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import os

# Path to dataset
dataset_path = "./dataset_path"

# Hyperparameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
K_FOLDS = 5

# Data transformation
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset from folders: Screwtype1, Screwtype2, Screwtype3, Screwtype4
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
class_names = dataset.classes
print(f"Classes found: {class_names}")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# K-Fold setup
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n--- Fold {fold + 1}/{K_FOLDS} ---")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    # Load pre-trained MobileNetV2 and replace classifier
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 4)  # 4 screw classes
    )
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

        # Save best model for this fold
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = f"screw_cnn_fold{fold+1}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model for fold {fold+1} saved to {best_model_path}")

# Save final model
torch.save(model.state_dict(), "screw_classification_final.pth")
print("\nFinal model saved as screw_classification_final.pth")
