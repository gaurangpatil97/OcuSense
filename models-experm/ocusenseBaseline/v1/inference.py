import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────
DISEASE_NAMES = [
    'Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Cataract',
    'AMD', 'Hypertensive Retinopathy', 'Myopia', 'Other'
]

# ── Model Architecture (must match exactly what was trained) ───────────────────
class OcuSenseModel(nn.Module):
    def __init__(self, num_classes=8):
        super(OcuSenseModel, self).__init__()
        self.backbone = models.efficientnet_b3(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ── Inference Transform ────────────────────────────────────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Load Model ─────────────────────────────────────────────────────────────────
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = OcuSenseModel(num_classes=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded on {device}")
    return model, device

# ── Predict ────────────────────────────────────────────────────────────────────
def predict(model, device, image_path):
    image  = Image.open(image_path).convert('RGB')
    tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs               = model(tensor)
        probs                 = torch.softmax(outputs, dim=1)[0]
        confidence, predicted = probs.max(0)

    predicted_disease = DISEASE_NAMES[predicted.item()]
    confidence_pct    = confidence.item() * 100

    print(f"\nPrediction   : {predicted_disease}")
    print(f"Confidence   : {confidence_pct:.2f}%")
    print("\nAll class probabilities:")
    for name, prob in zip(DISEASE_NAMES, probs):
        bar = '█' * int(prob.item() * 30)
        print(f"  {name:<28} {prob.item()*100:5.2f}%  {bar}")

    return predicted_disease, confidence_pct

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model_path = r'C:\Users\gaura\Desktop\Eye Disease Detection\models-experm\ocusenseBaseline\v1\ocusense_best.pth'
    image_path = r'C:\Users\gaura\Desktop\Eye Disease Detection\models-experm\ocusenseBaseline\v1\image.png'

    model, device = load_model(model_path)
    predict(model, device, image_path)