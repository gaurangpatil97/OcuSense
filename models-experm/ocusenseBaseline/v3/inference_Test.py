import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import ast

# ── Config ─────────────────────────────────────────────────────────────────────
DISEASE_NAMES = [
    'Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Cataract',
    'AMD', 'Hypertensive Retinopathy', 'Myopia', 'Other'
]
DISEASE_MAP = {
    'N': 'Normal', 'D': 'Diabetic Retinopathy', 'G': 'Glaucoma',
    'C': 'Cataract', 'A': 'AMD', 'H': 'Hypertensive Retinopathy',
    'M': 'Myopia', 'O': 'Other'
}

MODEL_PATH        = r'C:\Users\gaura\Desktop\Eye Disease Detection\models-experm\ocusenseBaseline\v3\ocusense_v3_best.pth'
IMG_DIR           = r'C:\Users\gaura\Desktop\Eye Disease Detection\ocular-disease-recognition-odir5k\preprocessed_images'
CSV_PATH          = r'C:\Users\gaura\Desktop\Eye Disease Detection\ocular-disease-recognition-odir5k\full_df.csv'
SAMPLES_PER_CLASS = 4

# ── Model Architecture (V3) ────────────────────────────────────────────────────
class OcuSenseModel(nn.Module):
    def __init__(self, num_classes=8):
        super(OcuSenseModel, self).__init__()
        self.backbone = models.efficientnet_b3(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = OcuSenseModel(num_classes=8).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded on {device}\n")

# ── Load CSV ───────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df['label_str'] = df['labels'].apply(lambda x: ast.literal_eval(x)[0])

# ── Run Per Class ──────────────────────────────────────────────────────────────
correct = 0
total   = 0

for label_code, disease_name in DISEASE_MAP.items():
    class_df = df[df['label_str'] == label_code]
    samples  = class_df.sample(min(SAMPLES_PER_CLASS, len(class_df)), random_state=42)

    print(f"\n{'='*60}")
    print(f"CLASS: {disease_name} ({label_code})")
    print(f"{'='*60}")

    for _, row in samples.iterrows():
        img_path = os.path.join(IMG_DIR, row['filename'])

        try:
            image  = Image.open(img_path).convert('RGB')
            tensor = inference_transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs    = model(tensor)
                probs      = torch.softmax(outputs, dim=1)[0]
                conf, pred = probs.max(0)

            predicted = DISEASE_NAMES[pred.item()]
            correct  += 1 if predicted == disease_name else 0
            total    += 1
            status    = '✓' if predicted == disease_name else '✗'

            print(f"  {status} File: {row['filename']}")
            print(f"    Real     : {disease_name}")
            print(f"    Predicted: {predicted} ({conf.item()*100:.2f}%)")

        except Exception as e:
            print(f"  Error loading {row['filename']}: {e}")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Correct  : {correct}/{total}")
print(f"Accuracy : {correct/total*100:.2f}%")