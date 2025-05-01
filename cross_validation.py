import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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
EPOCHS = 5                     # Number of training epochs
K_FOLDS = 5                      # Number of folds for K-Fold cross-validation
NUM_CLASSES = 5                  # Total number of classes (Noscrew1 + 4 screw types)

# Define image transformations
# Includes resizing, grayscale conversion, lighting adjustments, and normalization
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),  # Resize images to 224x224
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels (for compatibility with MobileNetV2)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random lighting adjustments
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406],  # Normalize using ImageNet mean and std
                         [0.229, 0.224, 0.225]),
])

# Load the entire dataset using ImageFolder
# Assumes dataset_path contains subdirectories for each class
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
class_names = full_dataset.classes  # Get class names from dataset
print("Classes:", class_names)

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize K-Fold cross-validation
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

# Loop through each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset), 1):
    print(f"\n--- Fold {fold}/{K_FOLDS} ---")

    # Split dataset into training and validation subsets for this fold
    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Compute class distribution in the training set
    labels_in_fold = [full_dataset.targets[i] for i in train_idx]
    counts = Counter(labels_in_fold)
    max_count = max(counts.values())
    weights = [max_count / counts[i] for i in range(NUM_CLASSES)]
    weight_tensor = torch.tensor(weights, dtype=torch.float, device=device)
    print("Class counts:", counts)
    print("Loss weights:", weights)

    # Initialize MobileNetV2 model with pre-trained weights
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

    best_val_loss = float("inf")  # Initialize the best validation loss to infinity

    # Training and validation loop for the current fold
    for epoch in range(1, EPOCHS + 1):
        # — Train —
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

        # — Validate —
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss = criterion(out, lbls)
                val_loss += loss.item()
                preds = out.argmax(dim=1)
                val_total += lbls.size(0)
                val_correct += (preds == lbls).sum().item()

                # Collect predictions and true labels for evaluation
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbls.cpu().numpy())

        val_acc = 100 * val_correct / val_total
        print(f"Validation Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # Save the model if it achieves the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"screw_cnn_fold{fold}.pth")
            print(f"→ Saved best model for fold {fold}")
        
            # Evaluate the best model
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='weighted')
            recall = recall_score(all_labels, all_preds, average='weighted')
        
            # Log and print metrics
            log_message = (
                f"Best Model Evaluation (Fold {fold}):\n"
                f"Accuracy: {accuracy:.4f}\n"
                f"Precision: {precision:.4f}\n"
                f"Recall: {recall:.4f}\n"
            )
        
            print(log_message)
            logging.info(log_message)
        
            # Generate classification report
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds, target_names=class_names))
        
            # Plot confusion matrix and save it to a file
            conf_matrix = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title(f'Confusion Matrix (Fold {fold})')

            # Save the confusion matrix plot to a file
            plt.savefig(f"confusion_matrix_fold_{fold}.png")
            plt.close()  # Close the plot to avoid blocking execution

