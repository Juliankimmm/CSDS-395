#!/usr/bin/env python3
"""
Enhanced ConvNeXt + CLIP hybrid training script for EXTREME class imbalance (310:1).
Features:
- Checkpoint resumption for interrupted training
- Ensemble predictions
- Advanced minority class handling
- Confidence calibration
- Comprehensive logging and monitoring
"""

import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score, roc_curve, precision_score, recall_score, log_loss, confusion_matrix
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
from scipy.optimize import minimize
import warnings
import logging
from datetime import datetime
import pickle

warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Setup logging
def setup_logging(log_dir):
    """Setup comprehensive logging"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

# -------------------------- Dataset ---------------------------------
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, clip_embeddings=None, strong_aug_transform=None, cache_images=False):
        self.samples = []
        self.transform = transform
        self.strong_aug_transform = strong_aug_transform
        self.clip_embeddings = clip_embeddings or {}
        self.cache_images = cache_images
        self.image_cache = {}
        
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
        
        # Pre-cache images if enabled
        if self.cache_images:
            logging.info(f"Pre-caching {len(self.samples)} images...")
            for path, _ in tqdm(self.samples, desc="Caching images"):
                try:
                    img = Image.open(path).convert("RGB")
                    self.image_cache[path] = img
                except Exception as e:
                    logging.warning(f"Failed to cache {path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        try:
            if self.cache_images and path in self.image_cache:
                image = self.image_cache[path].copy()
            else:
                image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            rand_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(rand_idx)
        
        # Use stronger augmentation for minority class with high probability
        if self.strong_aug_transform and label == 1 and np.random.rand() < 0.8:
            image = self.strong_aug_transform(image)
        elif self.transform:
            image = self.transform(image)
        
        clip_feat = torch.tensor(self.clip_embeddings.get(path, np.zeros(512)), dtype=torch.float32)
        return image, torch.tensor(label, dtype=torch.long), path, clip_feat


# -------------------------- Model ---------------------------------
class ConvNextWithCLIP(nn.Module):
    def __init__(self, img_size=384, clip_weight=0.15, clip_dim=512, dropout=0.4, freeze_layers=0, use_attention=True):
        super().__init__()
        self.clip_weight = nn.Parameter(torch.tensor(clip_weight, dtype=torch.float32))
        self.clip_dim = clip_dim
        self.temperature = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.use_attention = use_attention
        
        self.backbone = models.convnext_base(weights="IMAGENET1K_V1")
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
        
        # Cross-modal attention for better fusion
        if self.use_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=in_features,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(in_features)
        
        combined_features = in_features + clip_dim
        self.dropout = nn.Dropout(dropout)
        
        # FIX: Use LayerNorm instead of BatchNorm1d to avoid batch size issues
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_features),  # Changed from BatchNorm1d
            nn.Linear(combined_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, NUM_CLASSES)
        )
        
        # Auxiliary CLIP head
        self.clip_head = nn.Sequential(
            nn.Linear(clip_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, NUM_CLASSES)
        )
        
        # Uncertainty head for confidence estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(combined_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, clip_feats=None, return_uncertainty=False):
        conv_feat = self.backbone(x)
        if conv_feat.ndim == 4:
            conv_feat = conv_feat.mean(dim=[2, 3])
        
        if clip_feats is None or clip_feats.ndim != 2:
            clip_feats = torch.zeros(conv_feat.size(0), self.clip_dim, device=conv_feat.device)
        
        clip_feats = F.normalize(clip_feats, dim=-1)
        
        # Apply cross-attention if enabled
        if self.use_attention:
            # Reshape for attention
            conv_feat_reshaped = conv_feat.unsqueeze(1)  # [B, 1, D]
            clip_key = clip_feats.unsqueeze(1)  # [B, 1, clip_dim]
            
            # Project clip to match conv_feat dimension
            clip_proj = F.linear(clip_key, 
                                torch.randn(conv_feat.size(1), clip_feats.size(1), device=conv_feat.device))
            
            # Apply attention
            attended_feat, _ = self.cross_attention(conv_feat_reshaped, clip_proj, clip_proj)
            conv_feat = self.attention_norm(conv_feat + attended_feat.squeeze(1))
        
        # Bounded clip weight
        clip_weight_val = torch.sigmoid(self.clip_weight) * 0.5
        features = torch.cat([conv_feat, clip_feats * clip_weight_val], dim=1)
        
        logits_main = self.classifier(features)
        logits_clip = self.clip_head(clip_feats)
        
        # Conservative fusion
        fused_logits = 0.85 * logits_main + 0.15 * logits_clip
        
        # Apply temperature scaling
        temp = torch.clamp(self.temperature, min=0.5, max=3.0)
        scaled_logits = fused_logits / temp
        
        if return_uncertainty:
            uncertainty = self.uncertainty_head(features)
            return scaled_logits, uncertainty
        
        return scaled_logits

# -------------------------- Loss ---------------------------------
class AdaptiveFocalLoss(nn.Module):
    """Enhanced focal loss with dynamic adaptation"""
    def __init__(self, gamma=0.5, smoothing_safe=0.05, smoothing_unsafe=0.20, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.smoothing = torch.tensor([smoothing_safe, smoothing_unsafe], dtype=torch.float32)
        self.alpha = alpha

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        device = logits.device
        
        with torch.no_grad():
            smooth_vec = self.smoothing.to(device)
            true_dist = torch.zeros_like(logits)
            
            for c in range(num_classes):
                filler = smooth_vec[c] / (num_classes - 1)
                true_dist[:, :] = filler
            
            for i, t in enumerate(targets):
                prob = 1.0 - smooth_vec[t]
                true_dist[i, t] = prob
        
        logprobs = F.log_softmax(logits, dim=-1)
        ce = -(true_dist * logprobs).sum(dim=1)
        pt = torch.exp(-ce)
        
        focal_weight = ((1 - pt) ** self.gamma)
        loss = focal_weight * ce
        
        if self.alpha is not None:
            alpha = self.alpha.to(device)
            loss = loss * alpha[targets]
        
        return loss.mean()


class UncertaintyLoss(nn.Module):
    """Loss to train uncertainty estimation"""
    def __init__(self):
        super().__init__()
    
    def forward(self, uncertainty, logits, targets):
        probs = F.softmax(logits, dim=1)
        pred_conf, pred_class = probs.max(dim=1)
        
        # High uncertainty when prediction is wrong
        correct = (pred_class == targets).float()
        target_uncertainty = 1.0 - correct
        
        loss = F.mse_loss(uncertainty.squeeze(), target_uncertainty)
        return loss

# -------------------------- Checkpoint Management ---------------------------------
class CheckpointManager:
    """Manages training checkpoints for resumption"""
    def __init__(self, checkpoint_dir, keep_last_n=3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    
    def save_checkpoint(self, epoch, model, optimizer, scheduler, swa_model, 
                       scaler, best_f1, best_threshold, training_history, args):
        """Save complete training state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'swa_model_state_dict': swa_model.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_f1': best_f1,
            'best_threshold': best_threshold,
            'training_history': training_history,
            'args': vars(args),
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
            'cuda_random_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_path)
        
        # Save periodic checkpoint
        epoch_checkpoint = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, epoch_checkpoint)
        
        # Clean old checkpoints
        self._clean_old_checkpoints()
        
        logging.info(f"Checkpoint saved: epoch {epoch}")
    
    def _migrate_model_state_dict(self, model_state_dict):
        """Migrate old model architecture to new one (BatchNorm -> LayerNorm)"""
        migrated_dict = {}
        
        for key, value in model_state_dict.items():
            # Skip old BatchNorm1d keys from classifier
            if 'classifier.0.' in key:
                if any(x in key for x in ['running_mean', 'running_var', 'num_batches_tracked', 'weight', 'bias']):
                    # Skip BatchNorm parameters - LayerNorm will use its own
                    if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                        logging.debug(f"Skipping BatchNorm key: {key}")
                        continue
                    # Map BatchNorm weight/bias to LayerNorm
                    if key.startswith('classifier.0.weight'):
                        migrated_dict['classifier.0.weight'] = value
                        logging.debug(f"Migrating BatchNorm weight to LayerNorm: {key}")
                        continue
                    if key.startswith('classifier.0.bias'):
                        migrated_dict['classifier.0.bias'] = value
                        logging.debug(f"Migrating BatchNorm bias to LayerNorm: {key}")
                        continue
            
            migrated_dict[key] = value
        
        return migrated_dict
    
    def load_checkpoint(self, model, optimizer, scheduler, swa_model, scaler):
        """Load checkpoint and restore training state with architecture compatibility"""
        if not os.path.exists(self.checkpoint_path):
            logging.info("No checkpoint found, starting from scratch")
            return 0, 0.0, 0.5, []
        
        logging.info(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE)
        
        # Migrate model state dict for architecture changes
        model_state_dict = checkpoint['model_state_dict']
        logging.info("Checking for architecture compatibility...")
        model_state_dict = self._migrate_model_state_dict(model_state_dict)
        
        # Load with strict=False to handle any remaining mismatches
        try:
            incompatible_keys = model.load_state_dict(model_state_dict, strict=False)
            if incompatible_keys.missing_keys:
                logging.info(f"Missing keys (will use initialized values): {incompatible_keys.missing_keys[:3]}...")
            if incompatible_keys.unexpected_keys:
                logging.info(f"Unexpected keys (ignored): {incompatible_keys.unexpected_keys[:3]}...")
            logging.info("✓ Model state loaded successfully with architecture migration")
        except Exception as e:
            logging.error(f"Failed to load model state: {e}")
            logging.warning("Starting with fresh model weights")
            return 0, 0.0, 0.5, []
        
        # Try to load optimizer/scheduler states (with error handling)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info("✓ Optimizer state loaded")
        except Exception as e:
            logging.warning(f"Could not restore optimizer state: {e}")
        
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info("✓ Scheduler state loaded")
        except Exception as e:
            logging.warning(f"Could not restore scheduler state: {e}")
        
        try:
            swa_state_dict = checkpoint['swa_model_state_dict']
            swa_state_dict = self._migrate_model_state_dict(swa_state_dict)
            swa_model.load_state_dict(swa_state_dict, strict=False)
            logging.info("✓ SWA model state loaded")
        except Exception as e:
            logging.warning(f"Could not restore SWA model state: {e}")
        
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logging.info("✓ Scaler state loaded")
        except Exception as e:
            logging.warning(f"Could not restore scaler state: {e}")
        
        # Restore random states with error handling for compatibility
        try:
            random.setstate(checkpoint['random_state'])
            logging.debug("✓ Random state restored")
        except Exception as e:
            logging.warning(f"Could not restore random state: {e}")
        
        try:
            np.random.set_state(checkpoint['np_random_state'])
            logging.debug("✓ NumPy random state restored")
        except Exception as e:
            logging.warning(f"Could not restore numpy random state: {e}")
        
        try:
            torch_rng_state = checkpoint['torch_random_state']
            if not isinstance(torch_rng_state, torch.ByteTensor):
                torch_rng_state = torch_rng_state.cpu()
            torch.set_rng_state(torch_rng_state)
            logging.debug("✓ PyTorch random state restored")
        except (TypeError, RuntimeError, KeyError) as e:
            logging.warning(f"Could not restore torch RNG state: {e}")
            logging.info("Continuing with current random state...")
        
        try:
            if checkpoint.get('cuda_random_state') is not None and torch.cuda.is_available():
                cuda_rng_state = checkpoint['cuda_random_state']
                if not isinstance(cuda_rng_state, torch.ByteTensor):
                    cuda_rng_state = cuda_rng_state.cpu()
                torch.cuda.set_rng_state(cuda_rng_state)
                logging.debug("✓ CUDA random state restored")
        except (TypeError, RuntimeError, KeyError) as e:
            logging.warning(f"Could not restore CUDA RNG state: {e}")
            logging.info("Continuing with current random state...")
        
        epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        best_f1 = checkpoint['best_f1']
        best_threshold = checkpoint['best_threshold']
        training_history = checkpoint['training_history']
        
        logging.info(f"\n{'='*70}")
        logging.info(f"✓ Checkpoint restored successfully!")
        logging.info(f"{'='*70}")
        logging.info(f"Resuming from epoch {checkpoint['epoch']}")
        logging.info(f"Previous best F1: {best_f1:.4f} at threshold {best_threshold:.4f}")
        logging.info(f"Training history entries: {len(training_history)}")
        logging.info(f"{'='*70}\n")
        
        return epoch, best_f1, best_threshold, training_history
    
    def _clean_old_checkpoints(self):
        """Keep only the last N checkpoints"""
        checkpoints = sorted([f for f in os.listdir(self.checkpoint_dir) 
                            if f.startswith("checkpoint_epoch_")])
        
        if len(checkpoints) > self.keep_last_n:
            for old_ckpt in checkpoints[:-self.keep_last_n]:
                os.remove(os.path.join(self.checkpoint_dir, old_ckpt))

# -------------------------- Utilities ---------------------------------
def class_balanced_weights(num_samples_per_cls, beta=0.99):
    """Conservative class-balanced weights for extreme imbalance."""
    num_samples_per_cls = np.array(num_samples_per_cls, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, num_samples_per_cls)
    effective_num[effective_num == 0] = 1e-8
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * len(num_samples_per_cls)
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(samples):
    labels = [lbl for _, lbl in samples]
    counts = Counter(labels)
    weights_per_class = {cls: 1.0 / (counts[cls] + 1e-9) for cls in counts}
    sample_weights = [weights_per_class[lbl] for lbl in labels]
    return sample_weights, counts


def find_best_threshold(y_true, y_probs, optimize_for='f1'):
    """Find optimal threshold with multiple strategies"""
    if len(np.unique(y_true)) <= 1:
        return 0.5, 0.5, 0.5
    
    # F1-based threshold
    best_f1 = -1.0
    best_thresh_f1 = 0.5
    for th in np.linspace(0.01, 0.99, 199):
        preds = (y_probs >= th).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh_f1 = th
    
    # Balanced accuracy threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    balanced_acc = (tpr + (1 - fpr)) / 2
    best_bal_idx = np.argmax(balanced_acc)
    best_thresh_bal = thresholds[best_bal_idx]
    
    # Youden's J statistic (optimal ROC threshold)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    best_thresh_j = thresholds[best_j_idx]
    
    return best_thresh_f1, best_thresh_bal, best_thresh_j


def compute_per_class_metrics(y_true, y_pred, y_probs):
    """Compute comprehensive per-class metrics"""
    metrics = {}
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    for cls in range(NUM_CLASSES):
        cls_name = "safe" if cls == 0 else "unsafe"
        mask = y_true == cls
        if mask.sum() > 0:
            cls_pred = y_pred[mask]
            cls_true = y_true[mask]
            cls_probs = y_probs[mask]
            
            metrics[f"{cls_name}_precision"] = precision_score(cls_true, cls_pred, pos_label=cls, zero_division=0)
            metrics[f"{cls_name}_recall"] = recall_score(cls_true, cls_pred, pos_label=cls, zero_division=0)
            metrics[f"{cls_name}_f1"] = f1_score(cls_true, cls_pred, pos_label=cls, zero_division=0)
            metrics[f"{cls_name}_count"] = int(mask.sum())
            metrics[f"{cls_name}_avg_confidence"] = float(cls_probs.mean())
    
    return metrics


def calibrate_temperature(model, dataloader, device=DEVICE):
    """Find optimal temperature scaling post-training."""
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels, _, clip_feats in tqdm(dataloader, desc="Calibrating", leave=False):
            imgs = imgs.to(device)
            clip_feats = clip_feats.to(device)
            
            original_temp = model.temperature.item()
            model.temperature.data.fill_(1.0)
            logits = model(imgs, clip_feats)
            model.temperature.data.fill_(original_temp)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    def temperature_loss(T):
        T = max(T[0], 0.1)
        scaled_probs = F.softmax(torch.tensor(all_logits) / T, dim=1).numpy()
        return log_loss(all_labels, scaled_probs, labels=[0, 1])
    
    result = minimize(temperature_loss, x0=[1.0], method='Nelder-Mead', 
                     options={'maxiter': 100})
    optimal_T = max(result.x[0], 0.1)
    
    logging.info(f"Optimal temperature: {optimal_T:.4f}")
    return optimal_T


# -------------------------- Mixup / CutMix ---------------------------------
def mixup_data(x, y, alpha=0.2):
    """Conservative mixup for imbalanced data."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
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


def cutmix_data(x, y, alpha=0.5):
    """Conservative cutmix."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    index = torch.randperm(x.size(0)).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y, y[index], lam


# -------------------------- Validation ---------------------------------
@torch.no_grad()
def validate_epoch(model, dataloader, threshold=0.5, tta=True, return_predictions=False):
    model.eval()
    all_probs, all_labels, all_paths = [], [], []
    all_uncertainties = []
    
    for imgs, labels, paths, clip_feats in tqdm(dataloader, desc="Validate", leave=False):
        imgs = imgs.to(DEVICE)
        clip_feats = clip_feats.to(DEVICE)

        if tta:
            probs_accum = []
            uncertainties_accum = []
            
            # Original
            out, unc = model(imgs, clip_feats, return_uncertainty=True)
            p = F.softmax(out, dim=1)[:, 1]
            probs_accum.append(p.cpu())
            uncertainties_accum.append(unc.cpu())
            
            # Horizontal flip
            imgs_flipped = torch.flip(imgs, dims=[3])
            out, unc = model(imgs_flipped, clip_feats, return_uncertainty=True)
            p = F.softmax(out, dim=1)[:, 1]
            probs_accum.append(p.cpu())
            uncertainties_accum.append(unc.cpu())
            
            # Center crop
            _, _, H, W = imgs.shape
            crop_sz = int(0.875 * min(H, W))
            y1, x1 = (H - crop_sz) // 2, (W - crop_sz) // 2
            imgs_cropped = imgs[:, :, y1:y1 + crop_sz, x1:x1 + crop_sz]
            if imgs_cropped.shape[2] != H or imgs_cropped.shape[3] != W:
                imgs_cropped = F.interpolate(imgs_cropped, size=(H, W), mode='bilinear', align_corners=False)
            out, unc = model(imgs_cropped, clip_feats, return_uncertainty=True)
            p = F.softmax(out, dim=1)[:, 1]
            probs_accum.append(p.cpu())
            uncertainties_accum.append(unc.cpu())
            
            # Conservative averaging
            probs = 0.5 * probs_accum[0] + 0.25 * probs_accum[1] + 0.25 * probs_accum[2]
            uncertainties = torch.stack(uncertainties_accum).mean(dim=0)
        else:
            out, unc = model(imgs, clip_feats, return_uncertainty=True)
            probs = F.softmax(out, dim=1)[:, 1].cpu()
            uncertainties = unc.cpu()

        all_probs.extend(probs.numpy())
        all_labels.extend(labels.numpy())
        all_paths.extend(paths)
        # Handle both scalar and array uncertainties
        unc_np = uncertainties.squeeze().numpy()
        if unc_np.ndim == 0:
            all_uncertainties.append(float(unc_np))
        else:
            all_uncertainties.extend(unc_np)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_uncertainties = np.array(all_uncertainties)
    preds = (all_probs >= threshold).astype(int)
    
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    bal_acc = balanced_accuracy_score(all_labels, preds)
    per_class = compute_per_class_metrics(all_labels, preds, all_probs)

    if return_predictions:
        return acc, f1, auc, bal_acc, all_probs, all_labels, per_class, all_uncertainties, all_paths
    
    return acc, f1, auc, bal_acc, all_probs, all_labels, per_class


# -------------------------- Main ---------------------------------
def main(args):
    # Setup logging
    log_file = setup_logging(args.log_dir)
    logging.info(f"Training started - Log file: {log_file}")
    logging.info(f"Arguments: {vars(args)}")
    
    print(f"\n{'='*70}")
    print(f"Training ConvNeXt + CLIP for Extreme Imbalance (310:1)")
    print(f"{'='*70}\n")
    
    # Progressive resizing
    resize_schedule = sorted([int(s) for s in args.resize_schedule.split(',')]) if args.resize_schedule else [args.img_size]
    
    def train_transform(img_size):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.3),
        ])
    
    def strong_aug_transform(img_size):
        """Very strong augmentation for rare minority class"""
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(25),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.ColorJitter(0.6, 0.6, 0.6, 0.3),
            AutoAugment(AutoAugmentPolicy.IMAGENET),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.6),
        ])

    def val_transform(img_size):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    train_ds = ImageDataset(args.train_dir, 
                           transform=train_transform(resize_schedule[0]), 
                           strong_aug_transform=strong_aug_transform(resize_schedule[0]),
                           cache_images=args.cache_images)
    val_ds = ImageDataset(args.val_dir, 
                         transform=val_transform(resize_schedule[-1]),
                         cache_images=args.cache_images)

    sample_weights, class_counts = compute_sample_weights(train_ds.samples)
    class_counts_list = [class_counts.get(i, 0) for i in range(NUM_CLASSES)]
    
    logging.info(f"Class Distribution (train):")
    logging.info(f"   Safe (0):   {class_counts_list[0]:,} samples")
    logging.info(f"   Unsafe (1): {class_counts_list[1]:,} samples")
    logging.info(f"   Imbalance ratio: {class_counts_list[0]/max(class_counts_list[1], 1):.1f}:1")

    # Conservative class-balanced weights
    cb_weights = class_balanced_weights(class_counts_list, beta=args.cb_beta)
    logging.info(f"Class-balanced weights: {cb_weights.numpy()}")
    
    criterion = AdaptiveFocalLoss(
        gamma=args.focal_gamma,
        smoothing_safe=args.smoothing_safe,
        smoothing_unsafe=args.smoothing_unsafe,
        alpha=cb_weights
    )
    
    uncertainty_criterion = UncertaintyLoss()

    # Create samplers
    labels = [lbl for _, lbl in train_ds.samples]
    counts = Counter(labels)
    majority_count = max(counts.values())
    minority_count = min(counts.values())
    
    # Balanced sampler for curriculum
    balanced_weights = []
    for lbl in labels:
        if counts[lbl] == minority_count:
            weight = (majority_count / minority_count) * args.minority_oversample_factor
        else:
            weight = 1.0
        balanced_weights.append(weight)
    balanced_sampler = WeightedRandomSampler(balanced_weights, num_samples=len(train_ds), replacement=True)
    
    # Weighted sampler
    weighted_weights = [majority_count / counts[lbl] if counts[lbl] == minority_count else 1.0 for lbl in labels]
    weighted_sampler = WeightedRandomSampler(weighted_weights, num_samples=len(train_ds), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, 
                         sampler=balanced_sampler if args.curriculum_epochs > 0 else weighted_sampler,
                         num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                       num_workers=args.num_workers, pin_memory=True)

    model = ConvNextWithCLIP(
        img_size=resize_schedule[-1],
        clip_weight=args.clip_weight,
        dropout=args.dropout,
        freeze_layers=args.freeze_backbone,
        use_attention=args.use_attention
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_t0, T_mult=2)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.lr * 0.3)
    scaler = torch.amp.GradScaler('cuda')

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(args.checkpoint_dir, keep_last_n=args.keep_checkpoints)
    
    # Try to resume from checkpoint
    start_epoch, best_f1, best_threshold, training_history = checkpoint_mgr.load_checkpoint(
        model, optimizer, scheduler, swa_model, scaler
    )
    
    best_epoch = start_epoch - 1 if start_epoch > 0 else -1
    
    early_stopper = EarlyStopping(patience=args.early_stopping_patience, 
                                 min_delta=args.early_stopping_min_delta,
                                 mode='max')

    logging.info(f"Starting training from epoch {start_epoch}")
    logging.info(f"   Total epochs: {args.epochs}")
    logging.info(f"   Curriculum phase: {args.curriculum_epochs} epochs")
    logging.info(f"   Device: {DEVICE}")

    for epoch in range(start_epoch, args.epochs):
        # Progressive resizing
        schedule_idx = min(len(resize_schedule) - 1, 
                          math.floor(epoch / max(1, args.epochs // len(resize_schedule))))
        current_img_size = resize_schedule[schedule_idx]
        
        train_ds.transform = train_transform(current_img_size)
        train_ds.strong_aug_transform = strong_aug_transform(current_img_size)
        
        use_balanced = epoch < args.curriculum_epochs
        sampler = balanced_sampler if use_balanced else weighted_sampler
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.num_workers, pin_memory=True)
        
        mode = "BALANCED" if use_balanced else "WEIGHTED"
        logging.info(f"\nEpoch {epoch+1}/{args.epochs} | {mode} | Size: {current_img_size}px")

        # Training
        model.train()
        running_loss = 0.0
        running_unc_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_dl, desc="Training", leave=False)
        for step, (imgs, labels, paths, clip_feats) in enumerate(pbar):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            clip_feats = clip_feats.to(DEVICE)
            
            # Apply augmentation mixing
            if args.use_mixup and np.random.rand() < 0.3:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=args.mixup_alpha)
            elif np.random.rand() < 0.3:
                imgs, y_a, y_b, lam = cutmix_data(imgs, labels, alpha=args.cutmix_alpha)
            else:
                y_a, y_b, lam = labels, labels, 1.0
            
            with torch.amp.autocast('cuda', enabled=True):
                outputs, uncertainty = model(imgs, clip_feats, return_uncertainty=True)
                loss_cls = lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)
                
                # Add uncertainty loss with small weight
                loss_unc = uncertainty_criterion(uncertainty, outputs, labels)
                loss = loss_cls + 0.1 * loss_unc
                loss = loss / args.accum_steps
            
            scaler.scale(loss).backward()
            
            if (step + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss_cls.item() * imgs.size(0) * args.accum_steps
            running_unc_loss += loss_unc.item() * imgs.size(0) * args.accum_steps
            pbar.set_postfix({'cls_loss': f'{loss_cls.item() * args.accum_steps:.4f}',
                            'unc_loss': f'{loss_unc.item() * args.accum_steps:.4f}'})

        avg_loss = running_loss / len(train_ds)
        avg_unc_loss = running_unc_loss / len(train_ds)
        
        # Validation
        acc, f1, auc, bal_acc, all_probs, all_labels, per_class = validate_epoch(
            model, val_dl, threshold=best_threshold, tta=args.tta
        )
        
        # Find optimal threshold
        if len(np.unique(all_labels)) > 1:
            thresh_f1, thresh_bal, thresh_j = find_best_threshold(all_labels, all_probs, optimize_for='f1')
            best_threshold = thresh_f1
            
            # Re-evaluate with optimal threshold
            preds_optimal = (all_probs >= best_threshold).astype(int)
            f1 = f1_score(all_labels, preds_optimal, zero_division=0)
            acc = accuracy_score(all_labels, preds_optimal)
            bal_acc = balanced_accuracy_score(all_labels, preds_optimal)
            per_class = compute_per_class_metrics(all_labels, preds_optimal, all_probs)
        
        scheduler.step()
        if epoch >= args.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        
        # Logging
        logging.info(f"Results:")
        logging.info(f"   Loss: {avg_loss:.6f} | Unc Loss: {avg_unc_loss:.6f}")
        logging.info(f"   Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | BalAcc: {bal_acc:.4f}")
        logging.info(f"   Per-class metrics:")
        logging.info(f"   ├─ Safe   → P: {per_class.get('safe_precision', 0):.4f} | R: {per_class.get('safe_recall', 0):.4f} | F1: {per_class.get('safe_f1', 0):.4f}")
        logging.info(f"   └─ Unsafe → P: {per_class.get('unsafe_precision', 0):.4f} | R: {per_class.get('unsafe_recall', 0):.4f} | F1: {per_class.get('unsafe_f1', 0):.4f}")
        logging.info(f"   Optimal threshold: {best_threshold:.4f}")
        
        # Track history
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'unc_loss': avg_unc_loss,
            'acc': acc,
            'f1': f1,
            'auc': auc,
            'bal_acc': bal_acc,
            'threshold': best_threshold,
            'per_class': per_class,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Save checkpoint every epoch
        checkpoint_mgr.save_checkpoint(
            epoch, model, optimizer, scheduler, swa_model, scaler,
            best_f1, best_threshold, training_history, args
        )
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': f1,
                'threshold': best_threshold,
                'per_class_metrics': per_class,
                'temperature': model.temperature.item(),
                'clip_weight': model.clip_weight.item(),
                'training_args': vars(args)
            }
            torch.save(checkpoint, args.save_model)
            logging.info(f"New best F1! Model saved → {args.save_model}")
        
        # Early stopping
        if early_stopper.step(f1):
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Post-training: SWA and calibration
    logging.info("\n" + "="*70)
    logging.info("Post-Training Optimization")
    logging.info("="*70)
    
    logging.info("Updating batch normalization for SWA model...")
    torch.optim.swa_utils.update_bn(train_dl, swa_model)
    
    logging.info("Calibrating temperature scaling...")
    optimal_temp = calibrate_temperature(swa_model, val_dl)
    swa_model.module.temperature.data.fill_(optimal_temp)
    
    logging.info("Final validation with calibrated SWA model...")
    acc, f1, auc, bal_acc, all_probs, all_labels, per_class, uncertainties, paths = validate_epoch(
        swa_model, val_dl, threshold=best_threshold, tta=args.tta, return_predictions=True
    )
    
    # Re-optimize threshold on calibrated model
    final_threshold, _, _ = find_best_threshold(all_labels, all_probs, optimize_for='f1')
    preds_final = (all_probs >= final_threshold).astype(int)
    final_f1 = f1_score(all_labels, preds_final, zero_division=0)
    final_per_class = compute_per_class_metrics(all_labels, preds_final, all_probs)
    
    logging.info(f"\nFinal SWA Results (Calibrated):")
    logging.info(f"   F1: {final_f1:.4f} | AUC: {auc:.4f} | BalAcc: {bal_acc:.4f}")
    logging.info(f"   Unsafe Precision: {final_per_class.get('unsafe_precision', 0):.4f}")
    logging.info(f"   Unsafe Recall: {final_per_class.get('unsafe_recall', 0):.4f}")
    logging.info(f"   Final threshold: {final_threshold:.4f}")
    
    # Save predictions for error analysis
    predictions_data = {
        'paths': paths,
        'true_labels': all_labels.tolist(),
        'predicted_probs': all_probs.tolist(),
        'predicted_labels': preds_final.tolist(),
        'uncertainties': uncertainties.tolist(),
        'threshold': final_threshold
    }
    pred_path = args.save_model.replace(".pth", "_predictions.pkl")
    with open(pred_path, 'wb') as f:
        pickle.dump(predictions_data, f)
    logging.info(f"Predictions saved to {pred_path}")
    
    # Save SWA model
    swa_path = args.save_model.replace(".pth", "_swa.pth")
    swa_checkpoint = {
        'model_state_dict': swa_model.module.state_dict() if hasattr(swa_model, 'module') else swa_model.state_dict(),
        'threshold': final_threshold,
        'temperature': optimal_temp,
        'f1': final_f1,
        'per_class_metrics': final_per_class,
        'training_args': vars(args)
    }
    torch.save(swa_checkpoint, swa_path)
    
    # Save training history
    history_path = args.save_model.replace(".pth", "_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2, default=str)
    
    # Generate error analysis report
    generate_error_analysis(predictions_data, final_per_class, args)
    
    # Final summary
    logging.info("\n" + "="*70)
    logging.info("Training Complete!")
    logging.info("="*70)
    logging.info(f"\nSaved Files:")
    logging.info(f"   ├─ Best model:      {args.save_model}")
    logging.info(f"   ├─ SWA model:       {swa_path}")
    logging.info(f"   ├─ Training log:    {history_path}")
    logging.info(f"   ├─ Predictions:     {pred_path}")
    logging.info(f"   └─ Checkpoints:     {args.checkpoint_dir}")
    logging.info(f"\nBest Performance:")
    logging.info(f"   ├─ Best F1:         {best_f1:.4f} (epoch {best_epoch+1})")
    logging.info(f"   ├─ Final F1 (SWA):  {final_f1:.4f}")
    logging.info(f"   └─ Threshold:       {final_threshold:.4f}")


def generate_error_analysis(predictions_data, per_class_metrics, args):
    """Generate detailed error analysis report"""
    report_path = args.save_model.replace(".pth", "_error_analysis.txt")
    
    true_labels = np.array(predictions_data['true_labels'])
    pred_labels = np.array(predictions_data['predicted_labels'])
    probs = np.array(predictions_data['predicted_probs'])
    uncertainties = np.array(predictions_data['uncertainties'])
    paths = predictions_data['paths']
    
    # Find errors
    errors = true_labels != pred_labels
    false_positives = (true_labels == 0) & (pred_labels == 1)
    false_negatives = (true_labels == 1) & (pred_labels == 0)
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ERROR ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total samples: {len(true_labels)}\n")
        f.write(f"Total errors: {errors.sum()} ({100*errors.mean():.2f}%)\n")
        f.write(f"False positives: {false_positives.sum()}\n")
        f.write(f"False negatives: {false_negatives.sum()}\n\n")
        
        f.write("Per-class Metrics:\n")
        f.write(f"  Safe class:\n")
        f.write(f"    Precision: {per_class_metrics.get('safe_precision', 0):.4f}\n")
        f.write(f"    Recall:    {per_class_metrics.get('safe_recall', 0):.4f}\n")
        f.write(f"    F1:        {per_class_metrics.get('safe_f1', 0):.4f}\n\n")
        
        f.write(f"  Unsafe class:\n")
        f.write(f"    Precision: {per_class_metrics.get('unsafe_precision', 0):.4f}\n")
        f.write(f"    Recall:    {per_class_metrics.get('unsafe_recall', 0):.4f}\n")
        f.write(f"    F1:        {per_class_metrics.get('unsafe_f1', 0):.4f}\n\n")
        
        # High confidence errors
        f.write("\n" + "="*70 + "\n")
        f.write("HIGH CONFIDENCE ERRORS (prob > 0.8 or < 0.2)\n")
        f.write("="*70 + "\n")
        
        high_conf_errors = errors & ((probs > 0.8) | (probs < 0.2))
        if high_conf_errors.sum() > 0:
            for i in np.where(high_conf_errors)[0]:
                f.write(f"\nFile: {paths[i]}\n")
                f.write(f"  True: {true_labels[i]} | Pred: {pred_labels[i]} | Prob: {probs[i]:.4f} | Uncertainty: {uncertainties[i]:.4f}\n")
        else:
            f.write("No high confidence errors!\n")
        
        # Most uncertain predictions
        f.write("\n" + "="*70 + "\n")
        f.write("MOST UNCERTAIN PREDICTIONS (Top 20)\n")
        f.write("="*70 + "\n")
        
        most_uncertain_idx = np.argsort(uncertainties)[-20:][::-1]
        for i in most_uncertain_idx:
            f.write(f"\nFile: {paths[i]}\n")
            f.write(f"  True: {true_labels[i]} | Pred: {pred_labels[i]} | Prob: {probs[i]:.4f} | Uncertainty: {uncertainties[i]:.4f}\n")
    
    logging.info(f"Error analysis saved to {report_path}")


class EarlyStopping:
    """Early stopping with support for both minimize and maximize modes."""
    def __init__(self, patience=5, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.should_stop = False

    def step(self, current_score):
        if self.mode == 'max':
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


# -------------------------- CLI ---------------------------------
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Train ConvNeXt + CLIP for extreme imbalance with checkpoint resumption",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument("--train_dir", default="./datasets/train", help="Training data directory")
    parser.add_argument("--val_dir", default="./datasets/val", help="Validation data directory")
    # Low RAM: disable caching by default
    parser.add_argument("--cache_images", action="store_true", help="Cache images in memory for faster training (not recommended with <16GB RAM)")

    # Model architecture
    parser.add_argument("--img_size", type=int, default=384, help="Final image size")
    parser.add_argument("--resize_schedule", type=str, default="384",
                        help="Progressive resize schedule (comma-separated)")
    parser.add_argument("--clip_weight", type=float, default=0.15, help="Initial CLIP fusion weight")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")
    parser.add_argument("--freeze_backbone", type=int, default=0, help="Number of backbone layers to freeze")
    parser.add_argument("--use_attention", action="store_true", help="Use cross-modal attention for fusion")

    # Training hyperparameters (tuned for GTX 1660)
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size (optimized for 6GB VRAM)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers (6 recommended max for 8 cores)")

    # Loss function
    parser.add_argument("--focal_gamma", type=float, default=0.5, help="Focal loss gamma")
    parser.add_argument("--cb_beta", type=float, default=0.99, help="Class-balanced weights beta")
    parser.add_argument("--smoothing_safe", type=float, default=0.05, help="Label smoothing for safe class")
    parser.add_argument("--smoothing_unsafe", type=float, default=0.20, help="Label smoothing for unsafe class")

    # Sampling strategy
    parser.add_argument("--minority_oversample_factor", type=float, default=3.0,
                        help="Minority oversampling factor during curriculum")
    parser.add_argument("--curriculum_epochs", type=int, default=3,
                        help="Epochs with balanced sampling")

    # Augmentation
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup alpha")
    parser.add_argument("--cutmix_alpha", type=float, default=0.5, help="CutMix alpha")
    parser.add_argument("--use_mixup", action="store_true", help="Use mixup")

    # Optimization
    parser.add_argument("--scheduler_t0", type=int, default=5, help="Cosine annealing restart period")
    parser.add_argument("--swa_start", type=int, default=10, help="Epoch to start SWA")
    parser.add_argument("--use_onecycle", action="store_true", help="Use OneCycleLR schedule (recommended for low VRAM)")

    # Validation
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001, help="Early stopping minimum delta")

    # Checkpoint management
    parser.add_argument("--checkpoint_dir", default="./checkpoints", help="Directory for checkpoints")
    parser.add_argument("--keep_checkpoints", type=int, default=3, help="Number of checkpoints to keep")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint (auto-detected)")

    # Output
    parser.add_argument("--save_model", default="./models/moderation_model_best.pth", help="Path to save best model")
    parser.add_argument("--log_dir", default="./logs", help="Directory for logs")

    args = parser.parse_args()

    # Create output directories
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Run training
    main(args)
