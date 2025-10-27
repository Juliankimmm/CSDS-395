import os
import torch
from PIL import Image
from tqdm import tqdm
import clip
import numpy as np
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 384

def precompute_clip_embeddings(data_dir, save_path, clip_model_name="ViT-B/16"):
    # Load CLIP model
    model, preprocess = clip.load(clip_model_name, device=DEVICE)
    model.eval()
    
    # Collect image paths
    image_paths = []
    label_map = {"safe": 0, "unsafe": 1}
    for label_name in label_map.keys():
        class_dir = os.path.join(data_dir, label_name)
        if os.path.exists(class_dir):
            for file in os.listdir(class_dir):
                if file.lower().endswith(("png", "jpg", "jpeg")):
                    image_paths.append(os.path.join(class_dir, file))

    embeddings = {}
    for path in tqdm(image_paths, desc=f"Precomputing CLIP embeddings for {data_dir}"):
        try:
            image = Image.open(path).convert("RGB")
            image = preprocess(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = model.encode_image(image)
                feat = feat / feat.norm(dim=-1, keepdim=True)  # normalize
            embeddings[path] = feat.cpu().numpy()
        except Exception as e:
            print(f"Skipping {path}: {e}")

    # Save as a dictionary
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # <--- add this
    np.save(save_path, embeddings)
    print(f"Saved embeddings to {save_path}, total images: {len(embeddings)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="CSDS-395/RanKing-app/ml/datasets/train")
    parser.add_argument("--val_dir", default="CSDS-395/RanKing-app/ml/datasets/val")
    parser.add_argument("--train_save", default="CSDS-395/RanKing-app/ml/models/train_clip_embeddings.npy")
    parser.add_argument("--val_save", default="CSDS-395/RanKing-app/ml/models/val_clip_embeddings.npy")
    parser.add_argument("--clip_model", default="ViT-B/16")
    args = parser.parse_args()

    precompute_clip_embeddings(args.train_dir, args.train_save, args.clip_model)
    precompute_clip_embeddings(args.val_dir, args.val_save, args.clip_model)
