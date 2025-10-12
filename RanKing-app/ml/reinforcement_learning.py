import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import os

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Paths ---
BASE_MODEL_PATH = Path("RanKing-app/ml/models/moderation_model_full.pth")
NEW_IMAGES_DIR = Path("RanKing-app/ml/datasets/new_images")
TRAIN_DIR = Path("RanKing-app/ml/datasets/train")

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
NEW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# --- Image transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Model loader with safe checkpoint handling ---
def load_model():
    print("Loading model...")
    try:
        # Try loading pretrained EfficientNet_B0
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 2)  # 2 classes: safe/unsafe

        if BASE_MODEL_PATH.exists():
            print(f"Loading saved weights from {BASE_MODEL_PATH} ...")
            state_dict = torch.load(BASE_MODEL_PATH, map_location=device)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(" Some keys were missing/unexpected but model was loaded safely.")
            else:
                print(" Model weights loaded successfully.")

        else:
            print("No saved model found. Using pretrained ImageNet EfficientNet_B0.")

        return model.to(device)

    except Exception as e:
        print(" Error loading EfficientNet:", e)
        print("Falling back to MobileNetV3 Small...")
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
        return model.to(device)


# --- Initialize model ---
model = load_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
softmax = nn.Softmax(dim=1)


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

        # --- Predict ---
        model.eval()
        with torch.no_grad():
            img_tensor = transform(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            probs = softmax(output)
            confidence, pred_idx = torch.max(probs, dim=1)
            pred_label = "safe" if pred_idx.item() == 0 else "unsafe"

        print(f"\nPredicted: {pred_label} ({confidence.item()*100:.2f}% confidence)")
        print(f"Image: {img_path.name}")

        # --- Get human feedback ---
        while True:
            feedback = input("Is this image SAFE? (y/n): ").lower()
            if feedback in ["y", "n"]:
                break
            print("Please enter 'y' for safe or 'n' for unsafe.")

        label = 0 if feedback == "y" else 1
        label_name = "safe" if label == 0 else "unsafe"

        # --- Train model on feedback ---
        model.train()
        optimizer.zero_grad()
        target = torch.tensor([label], dtype=torch.long).to(device)
        output = model(img_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        print(f" Trained on {img_path.name} as '{label_name}' (loss={loss.item():.6f})")

        # --- Move to train dataset folder ---
        dest_folder = TRAIN_DIR / label_name
        dest_folder.mkdir(parents=True, exist_ok=True)
        img_path.rename(dest_folder / img_path.name)

    # --- Save updated model ---
    torch.save(model.state_dict(), BASE_MODEL_PATH)
    print(f"\n Updated model saved to {BASE_MODEL_PATH}")


# --- Main entry point ---
if __name__ == "__main__":
    human_feedback_train()
