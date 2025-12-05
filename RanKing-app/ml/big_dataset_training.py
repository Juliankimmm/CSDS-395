import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil
import time

# --- Config ---
SAFE_IMAGES_DIR = Path("RanKing-app/ml/datasets/new_images")
TRAIN_SAFE_DIR = Path("RanKing-app/ml/datasets/train/safe")
MODEL_PATH = "RanKing-app/ml/models/moderation_model.pth"

BATCH_SIZE = 64
MAX_EPOCHS = 200
LEARNING_RATE = 1e-4
PATIENCE = 10
SAMPLE_BATCHES_FOR_ESTIMATE = 2  # How many batches to sample for time estimate

# --- Custom Dataset ---
class SafeImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = list(Path(img_dir).glob("*"))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = 0
        return image, label


def estimate_training_time(model, dataloader, device):
    """Run a few batches to estimate total training time."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    it = iter(dataloader)
    start_time = time.time()
    for _ in range(SAMPLE_BATCHES_FOR_ESTIMATE):
        try:
            images, labels = next(it)
        except StopIteration:
            break
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    elapsed = time.time() - start_time
    total_batches = len(dataloader) * MAX_EPOCHS
    estimated_total_time = (elapsed / SAMPLE_BATCHES_FOR_ESTIMATE) * total_batches
    return estimated_total_time


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    TRAIN_SAFE_DIR.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = SafeImageDataset(SAFE_IMAGES_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"Loaded {len(dataset)} safe images from {SAFE_IMAGES_DIR}")

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.to(device)

    if Path(MODEL_PATH).exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Loaded existing moderation_model.pth")

    # --- Pre-training estimate ---
    est_seconds = estimate_training_time(model, dataloader, device)
    print(f"Estimated total training time: {est_seconds/60:.1f} minutes (~{est_seconds/3600:.2f} hours)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float("inf")
    epochs_no_improve = 0
    epoch_times = []

    for epoch in range(MAX_EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(dataloader)
        epoch_duration = time.time() - start_time
        epoch_times.append(epoch_duration)

        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_time_sec = avg_epoch_time * (MAX_EPOCHS - (epoch + 1))
        remaining_time_min = remaining_time_sec / 60

        print(f"Epoch {epoch+1}/{MAX_EPOCHS}, Loss: {avg_loss:.4f}, "
              f"Time: {epoch_duration:.1f}s, Est. remaining: {remaining_time_min:.1f} min")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Updated moderation_model.pth (epoch {epoch+1}, loss {avg_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs. Stopping early.")
            break

    # --- Move images ---
    print(f"Moving {len(dataset)} images from {SAFE_IMAGES_DIR} -> {TRAIN_SAFE_DIR}")
    for img_path in tqdm(list(SAFE_IMAGES_DIR.glob("*")), desc="Moving images"):
        shutil.move(str(img_path), str(TRAIN_SAFE_DIR / img_path.name))

    print("All images moved. Training finished.")


if __name__ == "__main__":
    train()
