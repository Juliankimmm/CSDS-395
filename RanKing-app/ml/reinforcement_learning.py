import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import os

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Paths ---
BASE_MODEL_PATH = "RanKing-app/ml/models/moderation_model.pth"
NEW_IMAGES_DIR = "RanKing-app/ml/datasets/new_images"
TRAIN_DIR = "RanKing-app/ml/datasets/train"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(NEW_IMAGES_DIR, exist_ok=True)

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Load Model ---
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model.to(device)

if os.path.exists(BASE_MODEL_PATH):
    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
    print("Loaded pretrained weights.")
else:
    print("Training from scratch.")

# --- Loss & Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- Human-in-the-loop function with confidence ---
def human_feedback_train():
    new_path = Path(NEW_IMAGES_DIR)
    images = [p for p in new_path.iterdir() if p.is_file()]
    if not images:
        print("No new images found.")
        return

    softmax = nn.Softmax(dim=1)

    for img_path in images:
        # Show image
        img = Image.open(img_path).convert("RGB")
        img.show()

        # Model prediction
        model.eval()
        with torch.no_grad():
            img_tensor = transform(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            probs = softmax(output)
            confidence, pred_idx = torch.max(probs, dim=1)
            pred_label = "safe" if pred_idx.item() == 0 else "unsafe"

        print(f"Model predicts: {pred_label} "
              f"with confidence: {confidence.item()*100:.2f}%")

        # Get human label
        while True:
            feedback = input(f"Is this image SAFE? (y/n) for {img_path.name}: ").lower()
            if feedback in ["y", "n"]:
                break
            print("Invalid input. Type 'y' for safe or 'n' for unsafe.")

        label = 0 if feedback == 'y' else 1
        label_name = "safe" if label == 0 else "unsafe"

        # Train for 1 step
        model.train()
        optimizer.zero_grad()
        target = torch.tensor([label], dtype=torch.long).to(device)
        output = model(img_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        print(f"Trained on {img_path.name} as {label_name}.\n")

        # Move image to train folder
        train_folder = Path(TRAIN_DIR) / label_name
        train_folder.mkdir(parents=True, exist_ok=True)
        img_path.rename(train_folder / img_path.name)

    # Save updated model
    torch.save(model.state_dict(), BASE_MODEL_PATH)
    print(f"Updated model saved to {BASE_MODEL_PATH}")

if __name__ == "__main__":
    human_feedback_train()
