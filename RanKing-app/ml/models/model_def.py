import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384
NUM_CLASSES = 2
CLIP_WEIGHT_DEFAULT = 0.3

class ConvNextWithCLIP(nn.Module):
    def __init__(self, clip_weight=CLIP_WEIGHT_DEFAULT, clip_dim=512, dropout=0.3, freeze_layers=0):
        super().__init__()
        self.clip_weight = nn.Parameter(torch.tensor(clip_weight, dtype=torch.float32))
        self.clip_dim = clip_dim
        self.backbone = models.convnext_base(weights="IMAGENET1K_V1")

        try:
            in_features = self.backbone.classifier[2].in_features
        except Exception:
            with torch.no_grad():
                dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
                feat = self.backbone(dummy)
                if feat.ndim == 4:
                    feat = feat.mean(dim=[2, 3])
                in_features = feat.shape[1]

        self.backbone.classifier = nn.Identity()

        combined_features = in_features + 512
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(combined_features, NUM_CLASSES)

    def forward(self, x, clip_feats=None):
        conv_feat = self.backbone(x)
        if conv_feat.ndim == 4:
            conv_feat = conv_feat.mean(dim=[2, 3])
        if clip_feats is None or clip_feats.ndim != 2:
            clip_feats = torch.zeros(conv_feat.size(0), 512, device=conv_feat.device)
        clip_feats = F.normalize(clip_feats, dim=-1)
        features = torch.cat([conv_feat, clip_feats * self.clip_weight], dim=1)
        return self.classifier(self.dropout(features))
