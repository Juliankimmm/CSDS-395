import torch, json
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384

def load_model(weights_path):
    model = models.convnext_base(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model

def predict(image_path, model, temp_file, use_clip=True, clip_weight=0.3):
    with open(temp_file) as f:
        T = json.load(f)["temperature"]

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")

    # TTA: flips
    ttas = [
        transform(image),
        transform(image.transpose(Image.FLIP_LEFT_RIGHT)),
        transform(image.transpose(Image.FLIP_TOP_BOTTOM))
    ]
    batch = torch.stack(ttas).to(DEVICE)
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits / T, dim=1)[:, 1].mean().item()

    # Optional CLIP ensemble
    if use_clip and CLIP_AVAILABLE:
        clip_model, preprocess = clip.load("ViT-B/16", device=DEVICE)
        text = clip.tokenize(["unsafe content", "safe content"]).to(DEVICE)
        clip_img = preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img_feat = clip_model.encode_image(clip_img)
            text_feat = clip_model.encode_text(text)
            clip_prob = (img_feat @ text_feat.T).softmax(dim=-1)[0, 0].item()
        probs = (1 - clip_weight) * probs + clip_weight * clip_prob

    return {"unsafe_prob": probs, "label": "unsafe" if probs > 0.5 else "safe"}


if __name__ == "__main__":
    model = load_model("ml/models/moderation_model.pth")
    result = predict("user_upload.jpg", model, "ml/models/temp.json", use_clip=True)
    print(result)
