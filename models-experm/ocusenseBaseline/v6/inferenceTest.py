# OcuSense V6 — Inference Test Script
# Tests 4 images per class from the dataset
# 6 Classes: Normal, Diabetic Retinopathy, Glaucoma, Cataract, Myopia, Other
# AMD and Hypertensive Retinopathy merged into Other

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import ast

# ── Config ─────────────────────────────────────────────────────────────────────
DISEASE_NAMES = [
    'Normal',
    'Diabetic Retinopathy',
    'Glaucoma',
    'Cataract',
    'Myopia',
    'Other'
]

# Merge map — same as training
MERGE_MAP = {
    'N': 0,  # Normal
    'D': 1,  # Diabetic Retinopathy
    'G': 2,  # Glaucoma
    'C': 3,  # Cataract
    'M': 4,  # Myopia
    'A': 5,  # AMD → Other
    'H': 5,  # Hypertensive Retinopathy → Other
    'O': 5,  # Other
}

MODEL_PATH        = r'C:\Users\gaura\Desktop\Eye Disease Detection\models-experm\ocusenseBaseline\v6\ocusense_v6_best.pth'
IMG_DIR           = r'C:\Users\gaura\Desktop\Eye Disease Detection\ocular-disease-recognition-odir5k\preprocessed_images'
CSV_PATH          = r'C:\Users\gaura\Desktop\Eye Disease Detection\ocular-disease-recognition-odir5k\full_df.csv'
SAMPLES_PER_CLASS = 4

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
model  = OcuSenseV6(num_classes=6).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded on {device}\n")

# ── Load CSV ───────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df['label_str']     = df['labels'].apply(lambda x: ast.literal_eval(x)[0])
df['label_encoded'] = df['label_str'].apply(lambda x: MERGE_MAP[x])

# ── Test Per Class ─────────────────────────────────────────────────────────────
# Test each of the 6 merged classes
TEST_CLASSES = {
    0: ('Normal',                ['N']),
    1: ('Diabetic Retinopathy',  ['D']),
    2: ('Glaucoma',              ['G']),
    3: ('Cataract',              ['C']),
    4: ('Myopia',                ['M']),
    5: ('Other',                 ['O', 'A', 'H']),  # includes merged classes
}

correct = 0
total   = 0

for class_idx, (class_name, source_codes) in TEST_CLASSES.items():
    # Sample from all source codes for this class
    class_df = df[df['label_str'].isin(source_codes)]
    samples  = class_df.sample(min(SAMPLES_PER_CLASS, len(class_df)), random_state=42)

    print(f"\n{'='*60}")
    print(f"CLASS: {class_name} (encoded: {class_idx})")
    if len(source_codes) > 1:
        print(f"  Includes merged: {source_codes}")
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

            predicted     = DISEASE_NAMES[pred.item()]
            is_correct    = pred.item() == class_idx
            correct      += 1 if is_correct else 0
            total        += 1
            status        = '✓' if is_correct else '✗'
            original_code = row['label_str']

            print(f"  {status} File: {row['filename']} (original: {original_code})")
            print(f"    Real      : {class_name}")
            print(f"    Predicted : {predicted} ({conf.item()*100:.2f}%)")

        except Exception as e:
            print(f"  Error loading {row['filename']}: {e}")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"SUMMARY — OcuSense V6")
print(f"{'='*60}")
print(f"Correct  : {correct}/{total}")
print(f"Accuracy : {correct/total*100:.2f}%")
print(f"\nNote: AMD and Hypertensive Retinopathy are merged into Other")
print(f"      They are tested under the Other class bucket")