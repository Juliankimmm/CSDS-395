import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import shutil
import random
from pathlib import Path

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Paths ---
BASE_MODEL_PATH = "RanKing-app/ml/models/moderation_model.pth"  # previously trained weights
NEW_IMAGES_DIR = "RanKing-app/ml/datasets/new_images"  # folder where you drop new images
TRAIN_DIR = "RanKing-app/ml/datasets/train"
VAL_DIR = "RanKing-app/ml/datasets/val"
VAL_SPLIT = 0.2

CATEGORY_LABEL = {
    "neutral": "safe",
    "porn": "unsafe",
    "sexy": "unsafe",
    "hentai": "unsafe",
    "drawings": "unsafe",
}

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Function to process new images ---
def process_new_images():
    if not os.path.exists(NEW_IMAGES_DIR):
        return

    for category in os.listdir(NEW_IMAGES_DIR):
        cat_path = Path(NEW_IMAGES_DIR) / category
        if not cat_path.is_dir():
            continue

        images = [p for p in cat_path.iterdir() if p.is_file()]
        if not images:
            continue

        random.shuffle(images)
        split_idx = int(len(images) * (1 - VAL_SPLIT))
        train_imgs, val_imgs = images[:split_idx], images[split_idx:]

        # Determine target dirs
        label = CATEGORY_LABEL.get(category, "unsafe")
        train_target = Path(TRAIN_DIR) / label
        val_target = Path(VAL_DIR) / label
        train_target.mkdir(parents=True, exist_ok=True)
        val_target.mkdir(parents=True, exist_ok=True)

        # Copy files to main dataset
        for src in train_imgs:
            shutil.copy(src, train_target / src.name)
        for src in val_imgs:
            shutil.copy(src, val_target / src.name)

        # Delete processed images from new_images
        for img in images:
            img.unlink()

        print(f"Processed {len(images)} new images for category '{category}'.")

# --- Process new images ---
process_new_images()

# --- Datasets ---
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- Model ---
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model.to(device)

# --- Load previous weights if they exist ---
if os.path.exists(BASE_MODEL_PATH):
    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
    print("Loaded pretrained weights from moderation_model.pth")
else:
    print("No existing weights found. Training from scratch.")

# --- Loss & Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- Fine-tuning loop ---
num_epochs = 5

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    train_loss, correct_train, total_train = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()

    train_loss /= total_train
    train_acc = correct_train / total_train

    # --- Validation ---
    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            total_val += labels.size(0)
            correct_val += predicted.eq(labels).sum().item()

    val_loss /= total_val
    val_acc = correct_val / total_val

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# --- Save updated weights ---
torch.save(model.state_dict(), BASE_MODEL_PATH)
print(f"Updated weights saved to {BASE_MODEL_PATH}")


# test commit