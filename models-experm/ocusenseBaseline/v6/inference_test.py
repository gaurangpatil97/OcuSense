# OcuSense V6 — Standalone Inference Script
# 6 Classes: Normal, Diabetic Retinopathy, Glaucoma, Cataract, Myopia, Other
# AMD and Hypertensive Retinopathy merged into Other

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
import os

# ── Config ─────────────────────────────────────────────────────────────────────
DISEASE_NAMES = [
    'Normal',
    'Diabetic Retinopathy',
    'Glaucoma',
    'Cataract',
    'Myopia',
    'Other'
]

# ── Model Architecture (must match V6 exactly) ─────────────────────────────────
class OcuSenseV6(nn.Module):
    def __init__(self, num_classes=6):
        super(OcuSenseV6, self).__init__()
        self.backbone = models.efficientnet_b3(weights=None)
        in_features   = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ── Inference Transform (must match training exactly) ──────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── Load Model ─────────────────────────────────────────────────────────────────
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = OcuSenseV6(num_classes=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded on {device}")
    return model, device

# ── Predict ────────────────────────────────────────────────────────────────────
def predict(model, device, image_path):
    image  = Image.open(image_path).convert('RGB')
    tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs     = model(tensor)
        probs       = torch.softmax(outputs, dim=1)[0]
        conf, pred  = probs.max(0)

    predicted_disease = DISEASE_NAMES[pred.item()]
    confidence_pct    = conf.item() * 100

    print(f"\n{'='*50}")
    print(f"OcuSense V6 — Prediction")
    print(f"{'='*50}")
    print(f"Image      : {os.path.basename(image_path)}")
    print(f"Prediction : {predicted_disease}")
    print(f"Confidence : {confidence_pct:.2f}%")
    print(f"\nAll class probabilities:")
    for name, prob in zip(DISEASE_NAMES, probs):
        bar = '█' * int(prob.item() * 40)
        print(f"  {name:<28} {prob.item()*100:5.2f}%  {bar}")
    print(f"{'='*50}")

    return predicted_disease, confidence_pct

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    MODEL_PATH = r'C:\Users\gaura\Desktop\Eye Disease Detection\models-experm\ocusenseBaseline\v6\ocusense_v6_best.pth'
    IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else None

    if not IMAGE_PATH:
        print("Usage: python inference_v6.py <path_to_image>")
        print("Example: python inference_v6.py retinal_image.jpg")
        sys.exit(1)

    model, device = load_model(MODEL_PATH)
    predict(model, device, IMAGE_PATH)