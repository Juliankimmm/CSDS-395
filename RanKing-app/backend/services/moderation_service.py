import torch
from torchvision import transforms, models
from PIL import Image
import io

class ModerationService:
    def __init__(self, model_path="moderation_model.pth"):
        # Load model
        self.model = models.mobilenet_v3_small(pretrained=False)
        self.model.classifier[3] = torch.nn.Linear(self.model.classifier[3].in_features, 2)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def analyze_image(self, image_bytes: bytes) -> str:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = outputs.max(1)
        
        return "safe" if predicted.item() == 0 else "unsafe"
