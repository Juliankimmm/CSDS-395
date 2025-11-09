#!/usr/bin/env python3
"""
Enhanced ConvNeXt + CLIP hybrid training script for extreme class imbalance.
Improvements included:
- Extreme minority oversampling with configurable factor
- Class-specific strong augmentation for minority class
- Class-balanced reweighting (effective number)
- Curriculum sampling (balanced early epochs -> real distribution later)
- Per-class label smoothing inside FocalSmoothLoss
- Per-class metric reporting
- Progressive resizing
- Gradient accumulation
- ROC-based threshold calibration
- TTA (flip + center crop)
- Robust corrupted image handling
- Fixed early stopping with proper validation loss
"""

import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score, roc_curve, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import models, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.optim.swa_utils import AveragedModel, SWALR
import torch.optim as optim
import random
import math
from collections import Counter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -------------------------- Dataset ---------------------------------
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, clip_embeddings=None, strong_aug_transform=None):
        self.samples = []
        self.transform = transform
        self.strong_aug_transform = strong_aug_transform
        self.clip_embeddings = clip_embeddings or {}
        label_map = {"safe": 0, "unsafe": 1}
        for label_name, label_val in label_map.items():
            class_dir = os.path.join(root_dir, label_name)
            if os.path.exists(class_dir):
                for file in os.listdir(class_dir):
                    if file.lower().endswith(("png", "jpg", "jpeg")):
                        path = os.path.join(class_dir, file)
                        self.samples.append((path, label_val))
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root_dir}. Check dataset structure.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            # If corrupted, pick a random valid sample to avoid crashing
            rand_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(rand_idx)
        # Use stronger augmentation for minority class (label=1, unsafe)
        if self.strong_aug_transform and label == 1 and np.random.rand() < 0.7:
            image = self.strong_aug_transform(image)
        elif self.transform:
            image = self.transform(image)
        clip_feat = torch.tensor(self.clip_embeddings.get(path, np.zeros(512)), dtype=torch.float32)
        return image, torch.tensor(label, dtype=torch.long), path, clip_feat


# -------------------------- Model ---------------------------------
class ConvNextWithCLIP(nn.Module):
    def __init__(self, img_size=384, clip_weight=0.3, clip_dim=512, dropout=0.3, freeze_layers=0):
        super().__init__()
        self.clip_weight = nn.Parameter(torch.tensor(clip_weight, dtype=torch.float32))
        self.clip_dim = clip_dim
        self.backbone = models.convnext_base(weights="IMAGENET1K_V1")
        # replace classifier with identity to extract features
        try:
            in_features = self.backbone.classifier[2].in_features
        except Exception:
            with torch.no_grad():
                dummy = torch.randn(1, 3, img_size, img_size)
                feat = self.backbone(dummy)
                if feat.ndim == 4:
                    feat = feat.mean(dim=[2, 3])
                in_features = feat.shape[1]
        self.backbone.classifier = nn.Identity()
        if freeze_layers > 0:
            params = list(self.backbone.features.parameters())
            for p in params[:freeze_layers]:
                p.requires_grad = False
        combined_features = in_features + clip_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(combined_features, NUM_CLASSES)
        # auxiliary small head on CLIP features (helps semantic signal)
        self.clip_head = nn.Sequential(
            nn.Linear(clip_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x, clip_feats=None):
        conv_feat = self.backbone(x)
        if conv_feat.ndim == 4:
            conv_feat = conv_feat.mean(dim=[2, 3])
        if clip_feats is None or clip_feats.ndim != 2:
            clip_feats = torch.zeros(conv_feat.size(0), self.clip_dim, device=conv_feat.device)
        clip_feats = F.normalize(clip_feats, dim=-1)
        features = torch.cat([conv_feat, clip_feats * self.clip_weight], dim=1)
        logits_main = self.classifier(self.dropout(features))
        logits_clip = self.clip_head(clip_feats)  # auxiliary logits
        # fuse logits: weighted average (learnable clip_weight affects contribution)
        fused_logits = (logits_main + logits_clip * self.clip_weight) / (1.0 + torch.abs(self.clip_weight))
        return fused_logits


# -------------------------- Loss ---------------------------------
class FocalSmoothLoss(nn.Module):
    """
    Combines focal loss with per-class label smoothing and optional alpha weighting.
    smoothing can be scalar (applied to all classes) or vector (per-class).
    alpha is a 1D tensor with per-class weights.
    """
    def __init__(self, gamma=2.0, smoothing=0.1, alpha=None):
        super().__init__()
        self.gamma = gamma
        if isinstance(smoothing, (list, tuple, np.ndarray)):
            self.smoothing = torch.tensor(smoothing, dtype=torch.float32)
        else:
            self.smoothing = smoothing
        self.alpha = alpha

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        device = logits.device
        with torch.no_grad():
            if isinstance(self.smoothing, torch.Tensor):
                # per-class smoothing: create true dist per sample
                smooth_vec = self.smoothing.to(device)
                true_dist = torch.zeros_like(logits)
                # fill with class-wise smoothing fraction over other classes
                for c in range(num_classes):
                    filler = smooth_vec[c] / (num_classes - 1)
                    true_dist[:, :] = filler
                # then set true label prob
                for i, t in enumerate(targets):
                    prob = 1.0 - smooth_vec[t]
                    true_dist[i, t] = prob
            else:
                smoothing = float(self.smoothing)
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(smoothing / (num_classes - 1))
                true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
        logprobs = F.log_softmax(logits, dim=-1)
        ce = -(true_dist * logprobs).sum(dim=1)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            alpha = self.alpha.to(device)
            loss = loss * alpha[targets]
        return loss.mean()


# -------------------------- Utilities ---------------------------------
def class_balanced_weights(num_samples_per_cls, beta=0.9999):
    """
    Compute class-balanced weights (Cui et al.) for re-weighting.
    num_samples_per_cls: array-like where index == class id
    returns: normalized weights per class as torch tensor
    """
    num_samples_per_cls = np.array(num_samples_per_cls, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, num_samples_per_cls)
    effective_num[effective_num == 0] = 1e-8
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * len(num_samples_per_cls)
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(samples):
    """
    Given list of (path,label) samples, return per-sample weights inverse to class frequency.
    """
    labels = [lbl for _, lbl in samples]
    counts = Counter(labels)
    weights_per_class = {cls: 1.0 / (counts[cls] + 1e-9) for cls in counts}
    sample_weights = [weights_per_class[lbl] for lbl in labels]
    return sample_weights, counts


def find_best_threshold(y_true, y_probs):
    """
    Find threshold that maximizes (TPR - FPR) or maximizes F1 on validation set.
    Returns both thresholds (roc_optimal, f1_optimal).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    # Youden's J statistic (maximize tpr - fpr)
    j_scores = tpr - fpr
    j_idx = np.argmax(j_scores)
    roc_thresh = thresholds[j_idx]

    # F1-based
    best_f1 = -1.0
    best_thresh_f1 = 0.5
    for th in np.linspace(0.01, 0.99, 99):
        preds = (y_probs >= th).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh_f1 = th
    return roc_thresh, best_thresh_f1


def compute_per_class_metrics(y_true, y_pred, y_probs):
    """Compute precision, recall, F1 for each class"""
    metrics = {}
    for cls in range(NUM_CLASSES):
        cls_name = "safe" if cls == 0 else "unsafe"
        mask = y_true == cls
        if mask.sum() > 0:
            cls_pred = y_pred[mask]
            cls_true = y_true[mask]
            metrics[f"{cls_name}_precision"] = precision_score(cls_true, cls_pred, pos_label=cls, zero_division=0)
            metrics[f"{cls_name}_recall"] = recall_score(cls_true, cls_pred, pos_label=cls, zero_division=0)
            metrics[f"{cls_name}_f1"] = f1_score(cls_true, cls_pred, pos_label=cls, zero_division=0)
            metrics[f"{cls_name}_count"] = mask.sum()
    return metrics


# -------------------------- Mixup / CutMix ---------------------------------
def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y, y[index], lam


# -------------------------- Validation ---------------------------------
@torch.no_grad()
def validate_epoch(model, dataloader, tta=True):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels, paths, clip_feats in tqdm(dataloader, desc="Validate", leave=False):
        imgs, clip_feats = imgs.to(DEVICE), clip_feats.to(DEVICE)
        if tta:
            # outputs: original, horizontally flipped, center crop
            outputs = model(imgs, clip_feats)
            outputs_flipped = model(torch.flip(imgs, dims=[3]), clip_feats)
            # simple center crop TTA
            _, _, H, W = imgs.shape
            crop_sz = int(0.9 * min(H, W))
            y1 = (H - crop_sz) // 2
            x1 = (W - crop_sz) // 2
            imgs_cropped = imgs[:, :, y1:y1 + crop_sz, x1:x1 + crop_sz]
            imgs_cropped = F.interpolate(imgs_cropped, size=(H, W), mode='bilinear', align_corners=False)
            outputs_crop = model(imgs_cropped, clip_feats)
            probs = (F.softmax(outputs, dim=1)[:, 1] + F.softmax(outputs_flipped, dim=1)[:, 1] + F.softmax(outputs_crop, dim=1)[:, 1]) / 3
        else:
            outputs = model(imgs, clip_feats)
            probs = F.softmax(outputs, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    all_probs, all_labels = np.array(all_probs), np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    bal_acc = balanced_accuracy_score(all_labels, preds)
    
    # Per-class metrics
    per_class = compute_per_class_metrics(all_labels, preds, all_probs)
    
    return acc, f1, auc, bal_acc, all_probs, all_labels, per_class


# -------------------------- Main ---------------------------------
def main(args):
    # progressive resizing schedule (list of sizes to use at different stages)
    resize_schedule = sorted([int(s) for s in args.resize_schedule.split(',')]) if args.resize_schedule else [args.img_size]
    
    # transform factories
    def train_transform(img_size):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            AutoAugment(AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.4),
        ])
    
    def strong_aug_transform(img_size):
        """Stronger augmentation for minority class"""
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
            AutoAugment(AutoAugmentPolicy.IMAGENET),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.5),
        ])

    def val_transform(img_size):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    # initial dataset load (we will recreate dataloaders when resize or sampler changes)
    train_ds = ImageDataset(args.train_dir, transform=train_transform(resize_schedule[0]), 
                           strong_aug_transform=strong_aug_transform(resize_schedule[0]))
    val_ds = ImageDataset(args.val_dir, transform=val_transform(resize_schedule[-1]))

    # compute sample weights and class counts
    sample_weights, class_counts = compute_sample_weights(train_ds.samples)
    class_counts_list = [class_counts.get(i, 0) for i in range(NUM_CLASSES)]
    print(f"\n{'='*60}")
    print(f"Class Distribution (train):")
    print(f"  Safe (0):   {class_counts_list[0]:,} samples")
    print(f"  Unsafe (1): {class_counts_list[1]:,} samples")
    print(f"  Imbalance ratio: {class_counts_list[0]/max(class_counts_list[1], 1):.1f}:1")
    print(f"{'='*60}\n")

    # class-balanced alpha using effective number
    cb_weights = class_balanced_weights(class_counts_list, beta=args.cb_beta)
    print(f"Class-balanced weights (alpha): {cb_weights.numpy()}")

    # per-class label smoothing: smaller smoothing for minority (avoid over-regularizing minority)
    smoothing_vec = np.array([args.smoothing_majority, args.smoothing_minority], dtype=np.float32)
    print(f"Label smoothing: [safe={args.smoothing_majority}, unsafe={args.smoothing_minority}]\n")

    # criterion uses class-balanced alpha
    alpha_tensor = cb_weights
    criterion = FocalSmoothLoss(gamma=args.focal_gamma, smoothing=smoothing_vec, alpha=alpha_tensor)

    # two samplers: balanced_sampler for curriculum stage, weighted_sampler for realistic stage
    labels = [lbl for _, lbl in train_ds.samples]
    counts = Counter(labels)
    majority_count = max(counts.values())
    minority_count = min(counts.values())
    
    # EXTREME oversampling for minority: each minority sample weighted much higher
    oversample_factor = args.minority_oversample_factor
    print(f"Minority oversample factor: {oversample_factor}x")
    print(f"Expected minority samples per epoch (curriculum): ~{int(minority_count * oversample_factor):,}\n")
    
    balanced_sample_weights = []
    for lbl in labels:
        if counts[lbl] == minority_count:  # minority class
            weight = majority_count * oversample_factor / (counts[lbl] + 1e-9)
        else:  # majority class
            weight = 1.0
        balanced_sample_weights.append(weight)
    balanced_sampler = WeightedRandomSampler(balanced_sample_weights, num_samples=len(train_ds), replacement=True)

    # Weighted (real distribution) sampler - still oversample minority but less aggressively
    weighted_sample_weights = [majority_count / (counts[lbl] + 1e-9) if counts[lbl] == minority_count else 1.0 for lbl in labels]
    weighted_sampler = WeightedRandomSampler(weighted_sample_weights, num_samples=len(train_ds), replacement=True)

    # dataloaders initial (use balanced_sampler if curriculum_epochs > 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=balanced_sampler if args.curriculum_epochs > 0 else weighted_sampler, num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model and optimizer
    model = ConvNextWithCLIP(img_size=resize_schedule[-1], clip_weight=args.clip_weight, dropout=args.dropout, freeze_layers=args.freeze_backbone).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_t0)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.lr * 0.5)
    scaler = torch.amp.GradScaler('cuda')

    early_stopper = EarlyStopping(patience=args.early_stopping_patience, min_delta=args.early_stopping_min_delta)
    best_f1 = 0.0
    best_epoch = -1
    global_step = 0
    best_threshold = 0.5

    print(f"Starting training for {args.epochs} epochs")
    print(f"Curriculum phase: first {args.curriculum_epochs} epochs with balanced sampling\n")

    for epoch in range(args.epochs):
        # progressive resizing: pick transform size based on epoch
        schedule_idx = min(len(resize_schedule) - 1, math.floor(epoch / max(1, args.epochs // len(resize_schedule))))
        current_img_size = resize_schedule[schedule_idx]
        
        # update dataset transforms
        train_ds.transform = train_transform(current_img_size)
        train_ds.strong_aug_transform = strong_aug_transform(current_img_size)
        
        # sampler selection: curriculum -> balanced for first curriculum_epochs
        use_balanced = epoch < args.curriculum_epochs
        sampler_to_use = balanced_sampler if use_balanced else weighted_sampler
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler_to_use, num_workers=args.num_workers, pin_memory=True)
        
        sampling_mode = "BALANCED (Curriculum)" if use_balanced else "WEIGHTED (Realistic)"
        print(f"\nEpoch {epoch+1}/{args.epochs} - {sampling_mode} - ImgSize: {current_img_size}")

        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for step, (imgs, labels, paths, clip_feats) in enumerate(tqdm(train_dl, desc=f"Train", leave=False)):
            imgs, labels, clip_feats = imgs.to(DEVICE), labels.to(DEVICE), clip_feats.to(DEVICE)
            
            # augmentation mixing
            if args.use_mixup and np.random.rand() < 0.5:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=args.mixup_alpha)
            else:
                imgs, y_a, y_b, lam = cutmix_data(imgs, labels, alpha=args.cutmix_alpha)
            
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(imgs, clip_feats)
                loss = lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)
                loss = loss / args.accum_steps
            
            scaler.scale(loss).backward()
            
            if (step + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
            
            running_loss += loss.item() * imgs.size(0) * args.accum_steps

        # validation
        acc, f1, auc, bal_acc, all_probs, all_labels, per_class = validate_epoch(model, val_dl, tta=args.tta)
        
        # compute validation loss for early stopping (use negative F1)
        val_loss = -f1
        
        # compute best threshold on validation
        if len(np.unique(all_labels)) > 1:
            roc_thresh, f1_thresh = find_best_threshold(all_labels, all_probs)
        else:
            roc_thresh, f1_thresh = 0.5, 0.5
        best_threshold = f1_thresh
        
        scheduler.step(epoch)
        swa_model.update_parameters(model)
        swa_scheduler.step()
        
        # Print results
        print(f"Loss: {running_loss/len(train_dl.dataset):.6f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | BalAcc: {bal_acc:.4f}")
        print(f"  Per-class metrics:")
        print(f"    Safe   - P: {per_class.get('safe_precision', 0):.4f} | R: {per_class.get('safe_recall', 0):.4f} | F1: {per_class.get('safe_f1', 0):.4f} | N: {per_class.get('safe_count', 0)}")
        print(f"    Unsafe - P: {per_class.get('unsafe_precision', 0):.4f} | R: {per_class.get('unsafe_recall', 0):.4f} | F1: {per_class.get('unsafe_f1', 0):.4f} | N: {per_class.get('unsafe_count', 0)}")
        print(f"  Best Threshold (F1): {best_threshold:.3f}")
        
        # checkpoint best by F1
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': f1,
                'threshold': best_threshold,
                'per_class_metrics': per_class
            }, args.save_model)
            print(f"  ✓ New best F1! Model saved.")
        
        if early_stopper.step(val_loss):
            print("\nEarly stopping triggered.")
            break

    # After training: apply SWA update & save final model
    print("\nFinalizing SWA model...")
    torch.optim.swa_utils.update_bn(train_dl, swa_model)
    final_path = args.save_model.replace(".pth", "_swa.pth")
    torch.save({
        'model_state_dict': swa_model.module.state_dict() if hasattr(swa_model, 'module') else swa_model.state_dict(),
        'threshold': best_threshold
    }, final_path)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best F1: {best_f1:.4f} at epoch {best_epoch+1}")
    print(f"  Best model: {args.save_model}")
    print(f"  SWA model: {final_path}")
    print(f"  Recommended threshold: {best_threshold:.3f}")
    print(f"{'='*60}\n")


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def step(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# -------------------------- CLI ---------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvNeXt + CLIP hybrid for extreme imbalance")
    parser.add_argument("--train_dir", default="./datasets/train")
    parser.add_argument("--val_dir", default="./datasets/val")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--resize_schedule", type=str, default="256,320,384", help="comma-separated sizes for progressive resizing")
    parser.add_argument("--clip_weight", type=float, default=0.3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--save_model", default="./models/moderation_model_best.pth")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cb_beta", type=float, default=0.9999, help="beta for class-balanced weights")
    parser.add_argument("--smoothing_majority", type=float, default=0.10, help="label smoothing for majority class")
    parser.add_argument("--smoothing_minority", type=float, default=0.05, help="label smoothing for minority class")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--use_mixup", action="store_true", help="enable mixup (otherwise default to cutmix)")
    parser.add_argument("--accum_steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--curriculum_epochs", type=int, default=5, help="number of early epochs to use balanced sampling")
    parser.add_argument("--freeze_backbone", type=int, default=0, help="freeze first N layers of backbone features")
    parser.add_argument("--scheduler_t0", type=int, default=5)
    parser.add_argument("--tta", action="store_true", help="use test-time augmentation during validation")
    parser.add_argument("--early_stopping_patience", type=int, default=4)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001)
    args = parser.parse_args()
    main(args)