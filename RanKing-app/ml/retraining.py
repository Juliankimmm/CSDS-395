import os, argparse, json
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import models, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.optim.swa_utils import AveragedModel, SWALR
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384
NUM_CLASSES = 2
CLIP_WEIGHT_DEFAULT = 0.3

# -------------------------- Dataset ---------------------------------
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, clip_embeddings=None):
        self.samples = []
        self.transform = transform
        self.clip_embeddings = clip_embeddings  # dictionary {image_path: embedding}
        label_map = {"safe": 0, "unsafe": 1}
        for label_name, label_val in label_map.items():
            class_dir = os.path.join(root_dir, label_name)
            if os.path.exists(class_dir):
                for file in os.listdir(class_dir):
                    if file.lower().endswith(("png", "jpg", "jpeg")):
                        path = os.path.join(class_dir, file)
                        self.samples.append((path, label_val))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Always return a tensor for clip_feat, never None
        if self.clip_embeddings is not None and path in self.clip_embeddings:
            clip_feat = torch.tensor(self.clip_embeddings[path], dtype=torch.float32)
        else:
            clip_feat = torch.zeros(512, dtype=torch.float32)

        return image, torch.tensor(label, dtype=torch.long), path, clip_feat


# -------------------------- Model ---------------------------------
class ConvNextWithCLIP(nn.Module):
    def __init__(self, clip_weight=CLIP_WEIGHT_DEFAULT, clip_dim=512):
        super().__init__()
        self.clip_weight = clip_weight
        self.clip_dim = clip_dim

        # Use pretrained ConvNeXt backbone for stability
        self.backbone = models.convnext_base(weights="IMAGENET1K_V1")
        in_features = self.backbone.classifier[2].in_features

        # Remove default classifier head
        self.backbone.classifier = nn.Identity()

        # Always assume CLIP can exist (even if zero embeddings, we'll pass 512 zeros)
        combined_features = in_features + 512
        self.classifier = nn.Linear(combined_features, NUM_CLASSES)

    def forward(self, x, clip_feats=None):
        conv_feat = self.backbone(x)
        if conv_feat.ndim == 4:
            conv_feat = conv_feat.mean(dim=[2, 3])
        
        if clip_feats is None or clip_feats.ndim != 2:
            clip_feats = torch.zeros(conv_feat.size(0), 512, device=conv_feat.device)

        clip_feats = F.normalize(clip_feats, dim=-1)
        features = torch.cat([conv_feat, clip_feats * self.clip_weight], dim=1)
        return self.classifier(features)

# -------------------------- Loss & Metrics ---------------------------------
def criterion_fn(outputs, targets, class_weights=None):
    if class_weights is not None:
        class_weights = class_weights.to(DEVICE)
        return F.cross_entropy(outputs, targets, weight=class_weights)
    return F.cross_entropy(outputs, targets)

def accuracy_fn(outputs, targets):
    preds = torch.argmax(outputs, dim=1)
    return (preds == targets).float().mean().item()

# -------------------------- Mixup / CutMix ---------------------------------
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, class_weights=None):
    return lam * criterion(pred, y_a, class_weights) + (1 - lam) * criterion(pred, y_b, class_weights)

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

# -------------------------- Training / Validation ---------------------------------
def train_epoch(model, dataloader, optimizer, epoch, class_weights=None, mixup_alpha=0.4, cutmix_alpha=1.0):
    model.train()
    running_loss = 0
    for imgs, labels, paths, clip_feats in tqdm(dataloader, desc=f"Train {epoch}", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        if np.random.rand() < 0.5:
            imgs, targets_a, targets_b, lam = mixup_data(imgs, labels, mixup_alpha)
        else:
            imgs, targets_a, targets_b, lam = cutmix_data(imgs, labels, cutmix_alpha)
        optimizer.zero_grad()
        outputs = model(imgs, clip_feats=clip_feats)
        loss = mixup_criterion(criterion_fn, outputs, targets_a, targets_b, lam, class_weights)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(dataloader.dataset)

@torch.no_grad()
def validate_epoch(model, dataloader):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels, paths, clip_feats in tqdm(dataloader, desc="Validate", leave=False):
        imgs = imgs.to(DEVICE)
        outputs = model(imgs, clip_feats=clip_feats)
        probs = F.softmax(outputs, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    preds = (np.array(all_probs) > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    return acc, f1, auc, np.array(all_probs), np.array(all_labels)

# -------------------------- Temperature Scaling ---------------------------------
def temperature_scale(logits, labels):
    logits = torch.tensor(logits, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    T = torch.tensor([1.0], requires_grad=True)
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    def eval():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss
    optimizer.step(eval)
    return T.item()

# -------------------------- Main Training Loop ---------------------------------
def main(args):
    # Load precomputed embeddings if provided
    clip_embeddings_train = None
    clip_embeddings_val = None

    if args.precomputed_clip_embeddings_train and os.path.exists(args.precomputed_clip_embeddings_train):
        clip_embeddings_train = np.load(args.precomputed_clip_embeddings_train, allow_pickle=True).item()
        print(f"Loaded precomputed train CLIP embeddings from {args.precomputed_clip_embeddings_train}")

    if args.precomputed_clip_embeddings_val and os.path.exists(args.precomputed_clip_embeddings_val):
        clip_embeddings_val = np.load(args.precomputed_clip_embeddings_val, allow_pickle=True).item()
        print(f"Loaded precomputed val CLIP embeddings from {args.precomputed_clip_embeddings_val}")

    transform_train = transforms.Compose([
        AutoAugment(AutoAugmentPolicy.IMAGENET),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor()
    ])
    transform_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    train_ds = ImageDataset(args.train_dir, transform_train, clip_embeddings_train)
    val_ds = ImageDataset(args.val_dir, transform_val, clip_embeddings_val)

    labels_list = [lbl for _, lbl in train_ds.samples]
    class_counts = np.bincount(labels_list)
    print(f"Dataset imbalance: safe={class_counts[0]}, unsafe={class_counts[1]}")
    weights = 1. / class_counts
    sample_weights = [weights[lbl] for lbl in labels_list]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    class_weights = torch.tensor([class_counts[1]/class_counts[0], 1.0], dtype=torch.float32)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    clip_dim = 512 if (clip_embeddings_train is not None or clip_embeddings_val is not None) else 0

    model = ConvNextWithCLIP(clip_weight=args.clip_weight, clip_dim=clip_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.lr * 0.5)

    best_f1 = 0.0
    best_probs, best_labels, best_T = None, None, 1.0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_dl, optimizer, epoch, class_weights)
        acc, f1, auc, probs, labels = validate_epoch(model, val_dl)
        swa_model.update_parameters(model)
        swa_scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs} | TrainLoss: {train_loss:.4f} | ValAcc: {acc:.3f} | ValF1: {f1:.3f} | ValAUC: {auc:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_probs, best_labels = probs, labels
            torch.save(model.state_dict(), args.save_model)
            np.save(args.save_probs, probs)
            np.save(args.save_labels, labels)
            print("Saved new best model.")

    best_T = temperature_scale(torch.tensor(best_probs).unsqueeze(1).repeat(1, NUM_CLASSES), torch.tensor(best_labels))
    print("Optimal temperature:", best_T)
    with open(args.save_temp, "w") as f:
        json.dump({"temperature": best_T}, f)

# -------------------------- CLI ---------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="RanKing-app/ml/datasets/train")
    parser.add_argument("--val_dir", default="RanKing-app/ml/datasets/val")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--save_model", default="RanKing-app/ml/models/moderation_model.pth")
    parser.add_argument("--save_probs", default="RanKing-app/ml/models/probs.npy")
    parser.add_argument("--save_labels", default="RanKing-app/ml/models/labels.npy")
    parser.add_argument("--save_temp", default="RanKing-app/ml/models/temp.json")
    parser.add_argument("--clip_weight", type=float, default=CLIP_WEIGHT_DEFAULT)
    parser.add_argument("--precomputed_clip_embeddings", default=None, help="Path to .npy file with precomputed CLIP embeddings")
    parser.add_argument("--precomputed_clip_embeddings_train", default=None, help="Path to .npy file with precomputed CLIP embeddings for training set")
    parser.add_argument("--precomputed_clip_embeddings_val", default=None, help="Path to .npy file with precomputed CLIP embeddings for validation set")

    args = parser.parse_args()
    main(args)
