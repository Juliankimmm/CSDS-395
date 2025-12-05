#!/usr/bin/env python3
"""
Comprehensive evaluation and visualization script for moderation model.
Analyzes predictions on a dataset and generates detailed visualizations.

Features:
- Confusion matrix and classification report
- ROC and Precision-Recall curves
- Confidence distribution analysis
- Per-class performance metrics
- Error analysis with sample visualizations
- Calibration plots
- Threshold sensitivity analysis
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, balanced_accuracy_score
)

from sklearn.calibration import calibration_curve
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["Safe", "Unsafe"]

# ========================= Model Architecture =========================
class ConvNextWithCLIP(torch.nn.Module):
    """Same architecture as training script"""
    def __init__(self, img_size=384, clip_weight=0.15, clip_dim=512, dropout=0.4, 
                 freeze_layers=0, use_attention=True):
        super().__init__()

        self.clip_weight = torch.nn.Parameter(torch.tensor(clip_weight, dtype=torch.float32))
        self.clip_dim = clip_dim
        self.temperature = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.use_attention = use_attention

        in_features = 1024  # ConvNeXt base output

        self.clip_proj = torch.nn.Linear(clip_dim, in_features, bias=False)

        self.backbone = models.convnext_base(weights="IMAGENET1K_V1")
        self.backbone.classifier = torch.nn.Identity()
        
        if self.use_attention:
            self.cross_attention = torch.nn.MultiheadAttention(
                embed_dim=in_features, num_heads=8, dropout=0.1, batch_first=True
            )
            self.attention_norm = torch.nn.LayerNorm(in_features)
        
        combined_features = in_features + clip_dim
        self.dropout = torch.nn.Dropout(dropout)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(combined_features),
            torch.nn.Linear(combined_features, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout * 0.5),
            torch.nn.Linear(256, NUM_CLASSES)
        )
        
        self.clip_head = torch.nn.Sequential(
            torch.nn.Linear(clip_dim, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, NUM_CLASSES)
        )
        
        self.uncertainty_head = torch.nn.Sequential(
            torch.nn.Linear(combined_features, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, clip_feats=None, return_uncertainty=False):
        conv_feat = self.backbone(x)
        if conv_feat.ndim == 4:
            conv_feat = conv_feat.mean(dim=[2, 3])
        
        if clip_feats is None or clip_feats.ndim != 2:
            clip_feats = torch.zeros(conv_feat.size(0), self.clip_dim, device=conv_feat.device)
        
        clip_feats = F.normalize(clip_feats, dim=-1)
        
        if self.use_attention:
            conv_feat_reshaped = conv_feat.unsqueeze(1)
            clip_key = clip_feats.unsqueeze(1)
            clip_proj = self.clip_proj(clip_key)
            attended_feat, _ = self.cross_attention(conv_feat_reshaped, clip_proj, clip_proj)
            conv_feat = self.attention_norm(conv_feat + attended_feat.squeeze(1))
        
        clip_weight_val = torch.sigmoid(self.clip_weight) * 0.5
        features = torch.cat([conv_feat, clip_feats * clip_weight_val], dim=1)
        
        logits_main = self.classifier(features)
        logits_clip = self.clip_head(clip_feats)
        fused_logits = 0.85 * logits_main + 0.15 * logits_clip
        
        temp = torch.clamp(self.temperature, min=0.5, max=3.0)
        scaled_logits = fused_logits / temp
        
        if return_uncertainty:
            uncertainty = self.uncertainty_head(features)
            return scaled_logits, uncertainty
        
        return scaled_logits

# ========================= Dataset =========================
class EvalDataset(Dataset):
    """Dataset for evaluation"""
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.samples = []
        self.transform = transform
        
        label_map = {"safe": 0, "unsafe": 1}
        for label_name, label_val in label_map.items():
            class_dir = os.path.join(root_dir, label_name)
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) 
                        if f.lower().endswith(("png", "jpg", "jpeg"))]
                
                # Limit samples if specified
                if max_samples:
                    files = files[:max_samples]
                
                for file in files:
                    path = os.path.join(class_dir, file)
                    self.samples.append((path, label_val))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        print(f"Loaded {len(self.samples)} images for evaluation")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return black image on error
            image = Image.new("RGB", (384, 384), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        # No CLIP features for evaluation
        clip_feat = torch.zeros(512, dtype=torch.float32)
        return image, torch.tensor(label, dtype=torch.long), path, clip_feat

# ========================= Evaluation Functions =========================
def load_model(checkpoint_path, device=DEVICE):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model config safely
    config = checkpoint.get('training_args', {})

    model = ConvNextWithCLIP(
        img_size=config.get('img_size', 384),
        clip_weight=config.get('clip_weight', 0.15),
        dropout=config.get('dropout', 0.4),
        freeze_layers=0,
        use_attention=config.get('use_attention', True)
    ).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    print("Resetting classifier, clip head, and uncertainty head to avoid corrupted weights...")

    # Only reset heads if checkpoint asks for it
    if checkpoint.get('reset_heads', False):
        print("Resetting classifier, clip head, and uncertainty head...")
        def reset_layer(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        model.classifier.apply(reset_layer)
        model.clip_head.apply(reset_layer)
        model.uncertainty_head.apply(reset_layer)

    threshold_to_use = checkpoint.get('threshold', 0.5)

    model.eval()
    print(f"Model loaded successfully! Threshold: {threshold_to_use:.4f}")

    return model, threshold_to_use, checkpoint

@torch.no_grad()
def evaluate_model(model, dataloader, threshold=0.5, tta=False):
    """Evaluate model and collect predictions"""
    model.eval()
    
    all_probs = []
    all_labels = []
    all_paths = []
    all_uncertainties = []
    
    print("Running evaluation...")
    for imgs, labels, paths, clip_feats in tqdm(dataloader, desc="Evaluating"):
        imgs = imgs.to(DEVICE)
        clip_feats = clip_feats.to(DEVICE)
        labels = labels.cpu()

        if tta:
            probs_list = []
            unc_list = []
            
            # Original
            out, unc = model(imgs, clip_feats, return_uncertainty=True)
            probs_list.append(F.softmax(out, dim=1)[:, 1].cpu())
            unc_list.append(unc.cpu())
            
            # Horizontal flip
            out, unc = model(torch.flip(imgs, dims=[3]), clip_feats, return_uncertainty=True)
            probs_list.append(F.softmax(out, dim=1)[:, 1].cpu())
            unc_list.append(unc.cpu())
            
            probs = torch.stack(probs_list).mean(dim=0)
            uncertainties = torch.stack(unc_list).mean(dim=0)
        else:
            out, unc = model(imgs, clip_feats, return_uncertainty=True)
            probs = F.softmax(out, dim=1)[:, 1].cpu()
            uncertainties = unc.cpu()
        
        probs = probs.view(-1).numpy()
        labels_np = labels.view(-1).numpy()
        uncertainties = uncertainties.view(-1).numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels_np.tolist())
        all_uncertainties.extend(uncertainties.tolist())
        all_paths.extend(paths)
    
    all_probs = np.atleast_1d(np.array(all_probs))
    all_labels = np.atleast_1d(np.array(all_labels))
    all_uncertainties = np.atleast_1d(np.array(all_uncertainties))

    from sklearn.metrics import roc_auc_score
    try:
        auc_normal = roc_auc_score(all_labels, all_probs)
        auc_inverted = roc_auc_score(all_labels, 1 - all_probs)

        print(f"ROC AUC (normal): {auc_normal:.4f}")
        print(f"ROC AUC (inverted): {auc_inverted:.4f}")

        if auc_inverted > auc_normal:
            print("Detected inverted probabilities — correcting!")
            all_probs = 1 - all_probs
    except Exception:
        pass

    all_preds = (all_probs >= threshold).astype(int)
    
    return {
        'probs': all_probs,
        'labels': all_labels,
        'preds': all_preds,
        'paths': all_paths,
        'uncertainties': all_uncertainties,
        'threshold': threshold
    }

# ========================= Visualization Functions =========================
def plot_confusion_matrix(results, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(results['labels'], results['preds'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")

def plot_roc_curve(results, save_path):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(results['labels'], results['probs'])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")

def plot_precision_recall_curve(results, save_path):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(results['labels'], results['probs'])
    ap = average_precision_score(results['labels'], results['probs'])
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {ap:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")

def plot_confidence_distribution(results, save_path):
    """Plot confidence distribution by class"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = results['labels'] == cls_idx
        probs = results['probs'][mask]
        
        axes[cls_idx].hist(probs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[cls_idx].axvline(results['threshold'], color='red', linestyle='--', 
                             linewidth=2, label=f'Threshold ({results["threshold"]:.3f})')
        axes[cls_idx].set_xlabel('Confidence Score', fontsize=11)
        axes[cls_idx].set_ylabel('Count', fontsize=11)
        axes[cls_idx].set_title(f'{cls_name} Class Distribution', 
                               fontsize=12, fontweight='bold')
        axes[cls_idx].legend()
        axes[cls_idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")

def plot_calibration_curve(results, save_path, n_bins=10):
    """Plot calibration curve"""
    prob_true, prob_pred = calibration_curve(
        results['labels'], results['probs'], n_bins=n_bins, strategy='uniform'
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")

def plot_threshold_sensitivity(results, save_path):
    """Plot metrics vs threshold"""
    thresholds = np.linspace(0.01, 0.99, 99)
    metrics = {'f1': [], 'precision': [], 'recall': [], 'accuracy': []}
    
    for thresh in thresholds:
        preds = (results['probs'] >= thresh).astype(int)
        
        from sklearn.metrics import precision_score, recall_score
        metrics['f1'].append(f1_score(results['labels'], preds, zero_division=0))
        metrics['precision'].append(precision_score(results['labels'], preds, zero_division=0))
        metrics['recall'].append(recall_score(results['labels'], preds, zero_division=0))
        metrics['accuracy'].append(accuracy_score(results['labels'], preds))
    
    plt.figure(figsize=(10, 6))
    for metric_name, values in metrics.items():
        plt.plot(thresholds, values, linewidth=2, label=metric_name.capitalize())
    
    plt.axvline(results['threshold'], color='red', linestyle='--', 
                linewidth=2, label=f'Current Threshold ({results["threshold"]:.3f})')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")

def plot_uncertainty_analysis(results, save_path):
    """Plot uncertainty vs correctness"""
    correct = results['labels'] == results['preds']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Uncertainty distribution by correctness
    axes[0].hist(results['uncertainties'][correct], bins=30, alpha=0.6, 
                label='Correct', color='green', edgecolor='black')
    axes[0].hist(results['uncertainties'][~correct], bins=30, alpha=0.6, 
                label='Incorrect', color='red', edgecolor='black')
    axes[0].set_xlabel('Uncertainty', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Uncertainty Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Scatter: confidence vs uncertainty
    colors = ['green' if c else 'red' for c in correct]
    axes[1].scatter(results['probs'], results['uncertainties'], 
                   c=colors, alpha=0.5, s=20)
    axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(results['threshold'], color='blue', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Confidence', fontsize=11)
    axes[1].set_ylabel('Uncertainty', fontsize=11)
    axes[1].set_title('Confidence vs Uncertainty', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.5, label='Correct'),
                      Patch(facecolor='red', alpha=0.5, label='Incorrect')]
    axes[1].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")

def visualize_error_samples(results, save_path, num_samples=16):
    """Visualize misclassified samples"""
    errors = results['labels'] != results['preds']
    error_indices = np.where(errors)[0]
    
    if len(error_indices) == 0:
        print("No errors found!")
        return
    
    # Sample random errors
    num_to_show = min(num_samples, len(error_indices))
    sampled_indices = np.random.choice(error_indices, num_to_show, replace=False)
    
    rows = int(np.sqrt(num_to_show))
    cols = int(np.ceil(num_to_show / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if num_to_show > 1 else [axes]
    
    for idx, ax in enumerate(axes):
        if idx < len(sampled_indices):
            sample_idx = sampled_indices[idx]
            img_path = results['paths'][sample_idx]
            true_label = CLASS_NAMES[results['labels'][sample_idx]]
            pred_label = CLASS_NAMES[results['preds'][sample_idx]]
            confidence = results['probs'][sample_idx]
            uncertainty = results['uncertainties'][sample_idx]
            
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
                ax.axis('off')
                
                title = f"True: {true_label} | Pred: {pred_label}\n"
                title += f"Conf: {confidence:.3f} | Unc: {uncertainty:.3f}"
                ax.set_title(title, fontsize=9, pad=5)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\n{os.path.basename(img_path)}", 
                       ha='center', va='center', fontsize=8)
                ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle('Misclassified Samples', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")

def generate_report(results, checkpoint, save_path):
    """Generate comprehensive text report"""
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("MODEL EVALUATION REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Dataset info
    report_lines.append("Dataset Statistics:")
    report_lines.append(f"  Total samples: {len(results['labels'])}")
    unique, counts = np.unique(results['labels'], return_counts=True)
    for cls_idx, count in zip(unique, counts):
        report_lines.append(f"  {CLASS_NAMES[cls_idx]}: {count} ({100*count/len(results['labels']):.2f}%)")
    report_lines.append("")
    
    # Model info
    report_lines.append("Model Configuration:")
    if 'training_args' in checkpoint:
        config = checkpoint['training_args']
        report_lines.append(f"  Image size: {config.get('img_size', 'N/A')}")
        report_lines.append(f"  Dropout: {config.get('dropout', 'N/A')}")
        report_lines.append(f"  CLIP weight: {config.get('clip_weight', 'N/A')}")
    report_lines.append(f"  Decision threshold: {results['threshold']:.4f}")
    if 'temperature' in checkpoint:
        report_lines.append(f"  Temperature: {checkpoint['temperature']:.4f}")
    report_lines.append("")
    
    # Overall metrics
    report_lines.append("Overall Performance:")
    acc = accuracy_score(results['labels'], results['preds'])
    bal_acc = balanced_accuracy_score(results['labels'], results['preds'])
    f1 = f1_score(results['labels'], results['preds'])
    
    report_lines.append(f"  Accuracy: {acc:.4f}")
    report_lines.append(f"  Balanced Accuracy: {bal_acc:.4f}")
    report_lines.append(f"  F1 Score: {f1:.4f}")
    
    if len(np.unique(results['labels'])) > 1:
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(results['labels'], results['probs'])
        report_lines.append(f"  ROC AUC: {auc_score:.4f}")
    report_lines.append("")
    
    # Per-class metrics
    report_lines.append("Per-Class Metrics:")
    report_lines.append(classification_report(results['labels'], results['preds'], 
                                              target_names=CLASS_NAMES, digits=4))
    report_lines.append("")
    
    # Confusion matrix
    cm = confusion_matrix(results['labels'], results['preds'])
    report_lines.append("Confusion Matrix:")
    report_lines.append(f"                  Predicted Safe  Predicted Unsafe")
    report_lines.append(f"  Actual Safe     {cm[0,0]:>14}  {cm[0,1]:>16}")
    report_lines.append(f"  Actual Unsafe   {cm[1,0]:>14}  {cm[1,1]:>16}")
    report_lines.append("")
    
    # Error analysis
    errors = results['labels'] != results['preds']
    report_lines.append("Error Analysis:")
    report_lines.append(f"  Total errors: {errors.sum()} ({100*errors.mean():.2f}%)")
    
    false_positives = (results['labels'] == 0) & (results['preds'] == 1)
    false_negatives = (results['labels'] == 1) & (results['preds'] == 0)
    report_lines.append(f"  False Positives: {false_positives.sum()}")
    report_lines.append(f"  False Negatives: {false_negatives.sum()}")
    
    if errors.sum() > 0:
        avg_error_conf = results['probs'][errors].mean()
        avg_error_unc = results['uncertainties'][errors].mean()
        report_lines.append(f"  Avg confidence on errors: {avg_error_conf:.4f}")
        report_lines.append(f"  Avg uncertainty on errors: {avg_error_unc:.4f}")
    report_lines.append("")
    
    # High confidence errors
    high_conf_errors = errors & ((results['probs'] > 0.8) | (results['probs'] < 0.2))
    report_lines.append(f"High Confidence Errors: {high_conf_errors.sum()}")
    if high_conf_errors.sum() > 0:
        report_lines.append("  (These require manual review)")
    report_lines.append("")
    
    report_lines.append("="*80)
    
    # Write report
    report_text = "\n".join(report_lines)
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    print(f" Saved: {save_path}")
    print("\n" + report_text)

def find_best_threshold(probs, labels):
    from sklearn.metrics import f1_score
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_f1 = 0.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    print(f"Best threshold for Unsafe class (max F1): {best_thresh:.3f} with F1={best_f1:.3f}")
    return best_thresh

def main(args):
    print("\n" + "="*80)
    print("MODEL EVALUATION AND VISUALIZATION")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}\n")
    
    # Load model
    model, threshold, checkpoint = load_model(args.model_path)
    
    # Setup data
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    dataset = EvalDataset(args.data_dir, transform=transform, max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers, 
                           pin_memory=True)
    
    # First evaluate with initial threshold
    results = evaluate_model(model, dataloader, threshold=threshold, tta=args.tta)
    
    # Optionally find best threshold based on F1
    best_thresh = find_best_threshold(results['probs'], results['labels'])
    results = evaluate_model(model, dataloader, threshold=best_thresh, tta=args.tta)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(results, os.path.join(args.output_dir, "confusion_matrix.png"))
    plot_roc_curve(results, os.path.join(args.output_dir, "roc_curve.png"))
    plot_precision_recall_curve(results, os.path.join(args.output_dir, "pr_curve.png"))
    plot_confidence_distribution(results, os.path.join(args.output_dir, "confidence_dist.png"))
    plot_calibration_curve(results, os.path.join(args.output_dir, "calibration_curve.png"))
    plot_threshold_sensitivity(results, os.path.join(args.output_dir, "threshold_sensitivity.png"))
    plot_uncertainty_analysis(results, os.path.join(args.output_dir, "uncertainty_analysis.png"))
    
    if args.visualize_errors:
        visualize_error_samples(results, os.path.join(args.output_dir, "error_samples.png"),
                               num_samples=args.num_error_samples)
    
    # Generate report
    generate_report(results, checkpoint, os.path.join(args.output_dir, "evaluation_report.txt"))
    
    # Save results as JSON
    results_json = {
        'threshold': float(results['threshold']),
        'accuracy': float(accuracy_score(results['labels'], results['preds'])),
        'balanced_accuracy': float(balanced_accuracy_score(results['labels'], results['preds'])),
        'f1_score': float(f1_score(results['labels'], results['preds'])),
        'num_samples': len(results['labels']),
        'num_errors': int((results['labels'] != results['preds']).sum())
    }
    
    with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f" Saved: {os.path.join(args.output_dir, 'results.json')}")
    
    # Save detailed predictions CSV
    import pandas as pd
    predictions_df = pd.DataFrame({
        'image_path': results['paths'],
        'true_label': [CLASS_NAMES[l] for l in results['labels']],
        'predicted_label': [CLASS_NAMES[p] for p in results['preds']],
        'confidence': results['probs'],
        'uncertainty': results['uncertainties'],
        'is_correct': results['labels'] == results['preds']
    })
    csv_path = os.path.join(args.output_dir, "predictions.csv")
    predictions_df.to_csv(csv_path, index=False)
    print(f" Saved: {csv_path}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {args.output_dir}")
    print(f"\nGenerated files:")
    print(f"  - confusion_matrix.png")
    print(f"  - roc_curve.png")
    print(f"  - pr_curve.png")
    print(f"  - confidence_dist.png")
    print(f"  - calibration_curve.png")
    print(f"  - threshold_sensitivity.png")
    print(f"  - uncertainty_analysis.png")
    if args.visualize_errors:
        print(f"  - error_samples.png")
    print(f"  - evaluation_report.txt")
    print(f"  - results.json")
    print(f"  - predictions.csv")
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize moderation model performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--model_path", required=True,
                       help="Path to trained model checkpoint (.pth file)")
    parser.add_argument("--data_dir", required=True,
                       help="Directory with test data (should contain 'safe' and 'unsafe' folders)")
    
    # Optional arguments
    parser.add_argument("--output_dir", default="./evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--img_size", type=int, default=384,
                       help="Image size for evaluation")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (None = all)")
    parser.add_argument("--tta", action="store_true",
                       help="Enable test-time augmentation")
    parser.add_argument("--visualize_errors", action="store_true",
                       help="Generate visualization of error samples")
    parser.add_argument("--num_error_samples", type=int, default=16,
                       help="Number of error samples to visualize")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    main(args)