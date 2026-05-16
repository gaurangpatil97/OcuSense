# OcuSense V5 — Inference Test Script
# 8 Classes — Stronger head Linear(512), capped sampler
# Tests 4 images per class from the dataset

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
    'N': 0, 'D': 1, 'G': 2, 'C': 3,
    'A': 4, 'H': 5, 'M': 6, 'O': 7
}

MODEL_PATH        = r'C:\Users\gaura\Desktop\Eye Disease Detection\models-experm\ocusenseBaseline\v5\ocusense_v5_best.pth'
IMG_DIR           = r'C:\Users\gaura\Desktop\Eye Disease Detection\ocular-disease-recognition-odir5k\preprocessed_images'
CSV_PATH          = r'C:\Users\gaura\Desktop\Eye Disease Detection\ocular-disease-recognition-odir5k\full_df.csv'
SAMPLES_PER_CLASS = 4

# ── Model Architecture (V5) ────────────────────────────────────────────────────
class OcuSenseV5(nn.Module):
    def __init__(self, num_classes=8):
        super(OcuSenseV5, self).__init__()
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

# ── Inference Transform ────────────────────────────────────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── Load Model ─────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = OcuSenseV5(num_classes=8).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded on {device}\n")

# ── Load CSV ───────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df['label_str']     = df['labels'].apply(lambda x: ast.literal_eval(x)[0])
df['label_encoded'] = df['label_str'].apply(lambda x: DISEASE_MAP[x])

# ── Test Per Class ─────────────────────────────────────────────────────────────
correct = 0
total   = 0

for label_code, class_idx in DISEASE_MAP.items():
    class_name = DISEASE_NAMES[class_idx]
    class_df   = df[df['label_str'] == label_code]
    samples    = class_df.sample(min(SAMPLES_PER_CLASS, len(class_df)), random_state=42)

    print(f"\n{'='*60}")
    print(f"CLASS: {class_name} ({label_code})")
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

            predicted  = DISEASE_NAMES[pred.item()]
            is_correct = pred.item() == class_idx
            correct   += 1 if is_correct else 0
            total     += 1
            status     = '✓' if is_correct else '✗'

            print(f"  {status} File: {row['filename']}")
            print(f"    Real      : {class_name}")
            print(f"    Predicted : {predicted} ({conf.item()*100:.2f}%)")

        except Exception as e:
            print(f"  Error loading {row['filename']}: {e}")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"SUMMARY — OcuSense V5")
print(f"{'='*60}")
print(f"Correct  : {correct}/{total}")
print(f"Accuracy : {correct/total*100:.2f}%")