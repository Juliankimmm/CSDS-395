import os
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

# Base dataset directory (same as your moderation dataset)
BASE_DIR = "RanKing-app/ml/datasets"
TRAIN_DIR = os.path.join(BASE_DIR, "train/safe")
VAL_DIR = os.path.join(BASE_DIR, "val/safe")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Download Fashion-MNIST
transform = transforms.Compose([transforms.ToTensor()])
full_dataset = datasets.FashionMNIST(
    root=os.path.join(BASE_DIR, "fashion_mnist"),  # raw download cache
    train=True,
    download=True,
    transform=transform
)

# Split into train / val (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Save helper
def save_images(dataset, target_dir, prefix="img"):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for idx, (img_tensor, _) in enumerate(loader):
        img = transforms.ToPILImage()(img_tensor.squeeze())
        img_path = os.path.join(target_dir, f"{prefix}_{idx}.png")
        img.save(img_path)

print("Saving train images...")
save_images(train_dataset, TRAIN_DIR, prefix="train")

print("Saving validation images...")
save_images(val_dataset, VAL_DIR, prefix="val")

print(f"Fashion-MNIST saved under {TRAIN_DIR} and {VAL_DIR}")
