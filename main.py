import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from collections import Counter

# Path to dataset (must contain Noscrew1 + Screwtype1–4)
dataset_path = "./dataset_path"

# Hyperparameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
K_FOLDS = 5
NUM_CLASSES = 5  # Noscrew1 + 4 screw types

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Load full dataset once
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
class_names = full_dataset.classes
print("Classes:", class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset), 1):
    print(f"\n--- Fold {fold}/{K_FOLDS} ---")

    train_ds = Subset(full_dataset, train_idx)
    val_ds   = Subset(full_dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # Compute per-class counts in this training split
    labels_in_fold = [full_dataset.targets[i] for i in train_idx]
    counts = Counter(labels_in_fold)
    max_count = max(counts.values())
    # weight[i] = max_count / counts[i]
    weights = [max_count / counts[i] for i in range(NUM_CLASSES)]
    weight_tensor = torch.tensor(weights, dtype=torch.float, device=device)
    print("Class counts:", counts)
    print("Loss weights:", weights)

    # Build MobileNetV2 with explicit weights API
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, NUM_CLASSES),
    )
    model.to(device)

    # Weighted cross‑entropy loss
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
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
            running_total   += lbls.size(0)
            running_correct += (preds == lbls).sum().item()

        train_acc = 100 * running_correct / running_total
        print(f"Epoch {epoch}/{EPOCHS} — Train Loss: {running_loss:.4f}, Acc: {train_acc:.2f}%")

        # — Validate —
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss = criterion(out, lbls)
                val_loss += loss.item()
                preds = out.argmax(dim=1)
                val_total   += lbls.size(0)
                val_correct += (preds == lbls).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"Validation Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # Save best model for this fold
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"screw_cnn_fold{fold}.pth")
            print(f"→ Saved best model for fold {fold}")

# Save final model
torch.save(model.state_dict(), "screw_classification_final.pth")
print("\nFinal model saved as screw_classification_final.pth")
