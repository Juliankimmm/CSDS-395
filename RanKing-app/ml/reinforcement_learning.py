import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import json
import os

from your_model_file import ConvNextWithCLIP, DEVICE, IMG_SIZE, NUM_CLASSES  # import your trained model

# --- Paths ---
BASE_MODEL_PATH = Path("RanKing-app/ml/models/moderation_model.pth")
TEMP_JSON_PATH = Path("RanKing-app/ml/models/temp.json")
NEW_IMAGES_DIR = Path("RanKing-app/ml/datasets/new_images")
TRAIN_DIR = Path("RanKing-app/ml/datasets/train")

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
NEW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# --- Image transforms ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# --- Load temperature scaling ---
if TEMP_JSON_PATH.exists():
    with open(TEMP_JSON_PATH, "r") as f:
        temp_data = json.load(f)
    TEMPERATURE = temp_data.get("temperature", 1.0)
else:
    TEMPERATURE = 1.0

softmax = nn.Softmax(dim=1)

# --- Load model ---
def load_model():
    model = ConvNextWithCLIP(clip_weight=0.3)
    if BASE_MODEL_PATH.exists():
        print(f"Loading model weights from {BASE_MODEL_PATH} ...")
        model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE))
    else:
        print("No trained model found. Initializing from scratch.")
    return model.to(DEVICE)

model = load_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- Predict function with temperature scaling ---
def predict(img_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        logits = logits / TEMPERATURE
        probs = softmax(logits)
        conf, pred_idx = torch.max(probs, dim=1)
        label = "safe" if pred_idx.item() == 0 else "unsafe"
    return label, conf.item()

# --- Human feedback training loop ---
def human_feedback_train():
    images = [p for p in NEW_IMAGES_DIR.iterdir() if p.is_file()]
    if not images:
        print("No new images found.")
        return

    for img_path in images:
        try:
            img = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            print(f"Skipping invalid file: {img_path}")
            continue

        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        pred_label, confidence = predict(img_tensor)
        print(f"\nPredicted: {pred_label} ({confidence*100:.2f}% confidence) for {img_path.name}")

        # --- Human feedback ---
        while True:
            feedback = input("Is this image SAFE? (y/n): ").lower()
            if feedback in ["y", "n"]:
                break
            print("Enter 'y' for safe or 'n' for unsafe.")

        label = 0 if feedback == "y" else 1
        label_name = "safe" if label == 0 else "unsafe"

        # --- Train model on feedback ---
        model.train()
        optimizer.zero_grad()
        target = torch.tensor([label], dtype=torch.long).to(DEVICE)
        output = model(img_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        print(f" Trained on {img_path.name} as '{label_name}' (loss={loss.item():.6f})")

        # --- Move image to correct train folder ---
        dest_folder = TRAIN_DIR / label_name
        dest_folder.mkdir(parents=True, exist_ok=True)
        img_path.rename(dest_folder / img_path.name)

    # --- Save updated model ---
    torch.save(model.state_dict(), BASE_MODEL_PATH)
    print(f"\nUpdated model saved to {BASE_MODEL_PATH}")

if __name__ == "__main__":
    human_feedback_train()
