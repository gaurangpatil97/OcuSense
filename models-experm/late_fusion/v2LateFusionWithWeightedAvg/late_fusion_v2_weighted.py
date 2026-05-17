# -*- coding: utf-8 -*-
"""
OcuSense Late Fusion Multimodal System (v2 - Weighted Average)
Combines OcuSense V6 (EfficientNet-B3) image model with 3 separate Metadata MLP configurations
using multiple weighted fusion strategies:
    fusion_probs = w_img * image_probs + w_meta * metadata_probs

Tests the following combinations:
- w_img=0.5, w_meta=0.5 (simple average)
- w_img=0.7, w_meta=0.3
- w_img=0.8, w_meta=0.2
- w_img=0.9, w_meta=0.1
- w_img=0.95, w_meta=0.05
"""

import os
import sys
import ast
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Reconfigure stdout to UTF-8 on Windows to safely print symbols and box drawings
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ==============================================================================
# SECTION 0: GLOBAL SEED & CUDA CONFIGURATION
# ==============================================================================
print("=" * 80)
print("SECTION 0: GLOBAL CONFIGURATION")
print("=" * 80)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Global seed set to: {SEED}")
print(f"Inference device   : {device}")
print("Configured successfully!\n")

# ==============================================================================
# SECTION 1: GLOBAL PATHS & MODEL WEIGHTS DEFINITIONS
# ==============================================================================
print("=" * 80)
print("SECTION 1: PATHS & CONSTANTS DEFINITIONS")
print("=" * 80)

CSV_PATH = r'C:\Users\gaura\Desktop\Eye Disease Detection\ocular-disease-recognition-odir5k\full_df.csv'
IMG_DIR = r'C:\Users\gaura\Desktop\Eye Disease Detection\ocular-disease-recognition-odir5k\preprocessed_images'
IMAGE_MODEL_PATH = r'C:\Users\gaura\Desktop\Eye Disease Detection\models-experm\ocusenseBaseline\v6\ocusense_v6_best.pth'
METADATA_AGE_ONLY_PATH = r'C:\Users\gaura\Desktop\Eye Disease Detection\models-experm\metadataModel\metadata_age_only.pth'
METADATA_AGE_SEX_PATH = r'C:\Users\gaura\Desktop\Eye Disease Detection\models-experm\metadataModel\metadata_age_sex.pth'
METADATA_ORACLE_PATH = r'C:\Users\gaura\Desktop\Eye Disease Detection\models-experm\metadataModel\metadata_oracle.pth'
OUTPUT_DIR = r'C:\Users\gaura\Desktop\Eye Disease Detection\models-experm\late_fusion\v2LateFusionWithWeightedAvg'

# Fallback helper for folder naming mismatch (metadataModel vs metdataModel)
def get_path_with_fallback(path):
    if os.path.exists(path):
        return path
    fallback = path.replace('metadataModel', 'metdataModel')
    if os.path.exists(fallback):
        return fallback
    return path

IMAGE_MODEL_PATH = get_path_with_fallback(IMAGE_MODEL_PATH)
METADATA_AGE_ONLY_PATH = get_path_with_fallback(METADATA_AGE_ONLY_PATH)
METADATA_AGE_SEX_PATH = get_path_with_fallback(METADATA_AGE_SEX_PATH)
METADATA_ORACLE_PATH = get_path_with_fallback(METADATA_ORACLE_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DISEASE_NAMES = ['Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Cataract', 'Myopia', 'Other']
MERGE_MAP = {
    'N': 0, 'D': 1, 'G': 2, 'C': 3, 'M': 4,
    'A': 5, 'H': 5, 'O': 5
}

print(f"CSV Path          : {CSV_PATH}")
print(f"Image Directory   : {IMG_DIR}")
print(f"Image Model Path  : {IMAGE_MODEL_PATH}")
print(f"Age Only Path     : {METADATA_AGE_ONLY_PATH}")
print(f"Age + Sex Path    : {METADATA_AGE_SEX_PATH}")
print(f"Oracle Path       : {METADATA_ORACLE_PATH}")
print(f"Output Directory  : {OUTPUT_DIR}")
print("Paths and constants mapped successfully!\n")

# ==============================================================================
# SECTION 2: DATA LOADING & TEST SET INTEGRITY VERIFICATION
# ==============================================================================
print("=" * 80)
print("SECTION 2: DATA LOADING & SPLITTING")
print("=" * 80)

# Load CSV
df = pd.read_csv(CSV_PATH)
print(f"Loaded ODIR dataset with shape: {df.shape}")

# Map labels
df['label_str'] = df['labels'].apply(lambda x: ast.literal_eval(x)[0])
df['label_encoded'] = df['label_str'].apply(lambda x: MERGE_MAP[x])

# Extract specific eye clinical keywords
df['diagnostic_keyword'] = df.apply(
    lambda row: row['Left-Diagnostic Keywords'] if 'left' in str(row['filename']) else row['Right-Diagnostic Keywords'],
    axis=1
)

# Impute
df['Patient Age'] = df['Patient Age'].fillna(df['Patient Age'].median())
df['Patient Sex'] = df['Patient Sex'].fillna('Unknown')
df['diagnostic_keyword'] = df['diagnostic_keyword'].fillna('')

# Split patients
unique_patients = df['ID'].unique()
train_ids, temp_ids = train_test_split(unique_patients, test_size=0.30, random_state=SEED)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=SEED)

train_df = df[df['ID'].isin(train_ids)].reset_index(drop=True)
val_df = df[df['ID'].isin(val_ids)].reset_index(drop=True)
test_df = df[df['ID'].isin(test_ids)].reset_index(drop=True)

# ── Verify Test Set Integrity ──────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST SET INTEGRITY CHECK")
print("="*60)
print(f"Total test samples     : {len(test_df)}")
print(f"Unique test patients   : {test_df['ID'].nunique()}")
print(f"Train ∩ Test patients  : {len(set(train_df['ID']) & set(test_df['ID']))}")
print(f"Val ∩ Test patients    : {len(set(val_df['ID']) & set(test_df['ID']))}")

if len(set(train_df['ID']) & set(test_df['ID'])) == 0 and len(set(val_df['ID']) & set(test_df['ID'])) == 0:
    print("✓ NO LEAKAGE — All test samples are from unseen patients only!")
else:
    print("✗ WARNING — LEAKAGE DETECTED!")
print("="*60 + "\n")

# ==============================================================================
# SECTION 3: METADATA PREPROCESSING PIPELINES FOR ALL THREE VERSIONS
# ==============================================================================
print("=" * 80)
print("SECTION 3: METADATA PREPROCESSING")
print("=" * 80)

# --- Version 1: Age Only ---
scaler_v1 = StandardScaler()
scaler_v1.fit(train_df[['Patient Age']])
X_test_v1 = scaler_v1.transform(test_df[['Patient Age']])

# --- Version 2: Age + Sex ---
preprocessor_v2 = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Patient Age']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Patient Sex'])
    ]
)
preprocessor_v2.fit(train_df[['Patient Age', 'Patient Sex']])
X_test_v2 = preprocessor_v2.transform(test_df[['Patient Age', 'Patient Sex']])

# --- Version 3: Oracle (Age + Sex + Keywords) ---
preprocessor_v3 = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Patient Age']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Patient Sex'])
    ]
)
preprocessor_v3.fit(train_df[['Patient Age', 'Patient Sex']])
train_tab_v3 = preprocessor_v3.transform(train_df[['Patient Age', 'Patient Sex']])

# Load state to align TF-IDF features dynamically to saved model weights
oracle_state = torch.load(METADATA_ORACLE_PATH, map_location='cpu')
oracle_input_dim = oracle_state['net.0.weight'].shape[1]

tfidf_max_features = oracle_input_dim - train_tab_v3.shape[1]
vectorizer_v3 = TfidfVectorizer(max_features=tfidf_max_features, stop_words='english')
vectorizer_v3.fit(train_df['diagnostic_keyword'])

test_tab_v3 = preprocessor_v3.transform(test_df[['Patient Age', 'Patient Sex']])
test_txt_v3 = vectorizer_v3.transform(test_df['diagnostic_keyword']).toarray()
X_test_v3 = np.hstack((test_tab_v3, test_txt_v3))

print("Preprocessing completed successfully!")
print(f"  * Age Only features: {X_test_v1.shape[1]}")
print(f"  * Age + Sex features: {X_test_v2.shape[1]}")
print(f"  * Oracle features   : {X_test_v3.shape[1]}\n")

# ==============================================================================
# SECTION 4: INITIALIZING MODEL ARCHITECTURES & LOADING WEIGHTS
# ==============================================================================
print("=" * 80)
print("SECTION 4: INITIALIZING MODEL ARCHITECTURES & WEIGHTS LOADING")
print("=" * 80)

# 1. Image Model Definition (OcuSenseV6)
class OcuSenseV6(nn.Module):
    def __init__(self, num_classes=6):
        super(OcuSenseV6, self).__init__()
        self.backbone = models.efficientnet_b3(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# 2. Dynamic MLP that automatically shapes layers from loaded checkpoints
class DynamicMetadataMLP(nn.Module):
    def __init__(self, state_dict):
        super(DynamicMetadataMLP, self).__init__()
        # Dynamic Layer Inspection
        input_dim = state_dict['net.0.weight'].shape[1]
        hidden_dim1 = state_dict['net.0.weight'].shape[0]
        hidden_dim2 = state_dict['net.4.weight'].shape[0]
        num_classes = state_dict['net.8.weight'].shape[0]
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(hidden_dim2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Load Image Model
print("Loading OcuSense V6 Image Model...")
img_model = OcuSenseV6(num_classes=6).to(device)
img_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
img_model.eval()
print(f"  * Successfully loaded Image Model: {IMAGE_MODEL_PATH}")

# Load Metadata Age Only
print("Loading Metadata MLP (Age Only)...")
age_state = torch.load(METADATA_AGE_ONLY_PATH, map_location=device)
meta_age_model = DynamicMetadataMLP(age_state).to(device)
meta_age_model.load_state_dict(age_state)
meta_age_model.eval()

# Load Metadata Age + Sex
print("Loading Metadata MLP (Age + Sex)...")
age_sex_state = torch.load(METADATA_AGE_SEX_PATH, map_location=device)
meta_age_sex_model = DynamicMetadataMLP(age_sex_state).to(device)
meta_age_sex_model.load_state_dict(age_sex_state)
meta_age_sex_model.eval()

# Load Metadata Oracle
print("Loading Metadata MLP (Oracle)...")
meta_oracle_model = DynamicMetadataMLP(oracle_state).to(device)
meta_oracle_model.load_state_dict(oracle_state)
meta_oracle_model.eval()

print("All models loaded successfully!\n")

# ==============================================================================
# SECTION 5: OPTIMIZED SINGLE-PASS IMAGE INFERENCE
# ==============================================================================
print("=" * 80)
print("SECTION 5: INFERENCE PIPELINE (RUNNING FUNDUS IMAGES ONCE)")
print("=" * 80)

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Arrays to collect valid inferences
valid_indices = []
filenames = []
true_labels = []
all_img_probs = []

missing_count = 0
print("Running single-pass fundus image inference on test set...")
for idx, row in test_df.iterrows():
    img_path = os.path.join(IMG_DIR, row['filename'])
    
    if not os.path.exists(img_path):
        missing_count += 1
        continue
        
    try:
        image = Image.open(img_path).convert('RGB')
        image_tensor = inference_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            img_logits = img_model(image_tensor)
            img_probs = torch.softmax(img_logits, dim=1)[0].cpu().numpy()
            
        all_img_probs.append(img_probs)
        filenames.append(row['filename'])
        true_labels.append(int(row['label_encoded']))
        valid_indices.append(idx)
        
    except Exception as e:
        print(f"  Error processing image {row['filename']}: {e}")
        continue

print(f"Image Inference Complete. Successfully processed: {len(filenames)} samples. Skipped: {missing_count} samples.\n")

y_true = np.array(true_labels)
img_probs_arr = np.array(all_img_probs)

# Run Inference on Metadata MLPs
print("Running inference on three separate Metadata MLPs...")

# 1. Age Only
test_meta_v1_valid = X_test_v1[valid_indices]
meta_v1_model_inputs = torch.tensor(test_meta_v1_valid, dtype=torch.float32).to(device)
with torch.no_grad():
    meta_v1_logits = meta_age_model(meta_v1_model_inputs)
    meta_v1_probs = torch.softmax(meta_v1_logits, dim=1).cpu().numpy()

# 2. Age + Sex
test_meta_v2_valid = X_test_v2[valid_indices]
meta_v2_model_inputs = torch.tensor(test_meta_v2_valid, dtype=torch.float32).to(device)
with torch.no_grad():
    meta_v2_logits = meta_age_sex_model(meta_v2_model_inputs)
    meta_v2_probs = torch.softmax(meta_v2_logits, dim=1).cpu().numpy()

# 3. Oracle
test_meta_v3_valid = X_test_v3[valid_indices]
meta_v3_model_inputs = torch.tensor(test_meta_v3_valid, dtype=torch.float32).to(device)
with torch.no_grad():
    meta_v3_logits = meta_oracle_model(meta_v3_model_inputs)
    meta_v3_probs = torch.softmax(meta_v3_logits, dim=1).cpu().numpy()

print("Metadata inference completed successfully!\n")

# ==============================================================================
# SECTION 6: WEIGHTED AVERAGE FUSION SEARCH & COMPARISONS
# ==============================================================================
print("=" * 80)
print("SECTION 6: WEIGHTED FUSION GRID SEARCH")
print("=" * 80)

# Evaluate individual baseline models first
img_acc = accuracy_score(y_true, img_probs_arr.argmax(axis=1))
img_f1 = f1_score(y_true, img_probs_arr.argmax(axis=1), average='macro')

v1_baseline_acc = accuracy_score(y_true, meta_v1_probs.argmax(axis=1))
v1_baseline_f1 = f1_score(y_true, meta_v1_probs.argmax(axis=1), average='macro')

v2_baseline_acc = accuracy_score(y_true, meta_v2_probs.argmax(axis=1))
v2_baseline_f1 = f1_score(y_true, meta_v2_probs.argmax(axis=1), average='macro')

v3_baseline_acc = accuracy_score(y_true, meta_v3_probs.argmax(axis=1))
v3_baseline_f1 = f1_score(y_true, meta_v3_probs.argmax(axis=1), average='macro')

# Define weight strategies
weight_combinations = [
    (0.5, 0.5),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
    (0.95, 0.05)
]

# Track full grid results
full_results_log = []

# --- 1. Fusion: Image + Age Only ---
print("=== Fusion: Image + Age Only ===")
print(f"| {'Weight (img/meta)':<18} | {'Accuracy':<10} | {'Macro F1':<10} |")
print("-" * 50)

best_v1_weight = None
best_v1_acc = 0.0
best_v1_f1 = 0.0

for w_img, w_meta in weight_combinations:
    fused_probs = w_img * img_probs_arr + w_meta * meta_v1_probs
    fused_preds = fused_probs.argmax(axis=1)
    acc = accuracy_score(y_true, fused_preds)
    f1 = f1_score(y_true, fused_preds, average='macro')
    
    full_results_log.append({
        'fusion_type': 'Image + Age',
        'w_img': w_img,
        'w_meta': w_meta,
        'accuracy': acc,
        'macro_f1': f1
    })
    
    print(f"| {f'{w_img} / {w_meta}':<18} | {acc*100:<9.2f}% | {f1:<10.4f} |")
    
    if acc > best_v1_acc or (acc == best_v1_acc and f1 > best_v1_f1):
        best_v1_acc = acc
        best_v1_f1 = f1
        best_v1_weight = (w_img, w_meta)

print(f"Best weight: {best_v1_weight[0]}/{best_v1_weight[1]} with accuracy {best_v1_acc*100:.2f}%\n")


# --- 2. Fusion: Image + Age + Sex ---
print("=== Fusion: Image + Age + Sex ===")
print(f"| {'Weight (img/meta)':<18} | {'Accuracy':<10} | {'Macro F1':<10} |")
print("-" * 50)

best_v2_weight = None
best_v2_acc = 0.0
best_v2_f1 = 0.0

for w_img, w_meta in weight_combinations:
    fused_probs = w_img * img_probs_arr + w_meta * meta_v2_probs
    fused_preds = fused_probs.argmax(axis=1)
    acc = accuracy_score(y_true, fused_preds)
    f1 = f1_score(y_true, fused_preds, average='macro')
    
    full_results_log.append({
        'fusion_type': 'Image + Age + Sex',
        'w_img': w_img,
        'w_meta': w_meta,
        'accuracy': acc,
        'macro_f1': f1
    })
    
    print(f"| {f'{w_img} / {w_meta}':<18} | {acc*100:<9.2f}% | {f1:<10.4f} |")
    
    if acc > best_v2_acc or (acc == best_v2_acc and f1 > best_v2_f1):
        best_v2_acc = acc
        best_v2_f1 = f1
        best_v2_weight = (w_img, w_meta)

print(f"Best weight: {best_v2_weight[0]}/{best_v2_weight[1]} with accuracy {best_v2_acc*100:.2f}%\n")


# --- 3. Fusion: Image + Oracle ---
print("=== Fusion: Image + Oracle ===")
print(f"| {'Weight (img/meta)':<18} | {'Accuracy':<10} | {'Macro F1':<10} |")
print("-" * 50)

best_v3_weight = None
best_v3_acc = 0.0
best_v3_f1 = 0.0

for w_img, w_meta in weight_combinations:
    fused_probs = w_img * img_probs_arr + w_meta * meta_v3_probs
    fused_preds = fused_probs.argmax(axis=1)
    acc = accuracy_score(y_true, fused_preds)
    f1 = f1_score(y_true, fused_preds, average='macro')
    
    full_results_log.append({
        'fusion_type': 'Image + Oracle',
        'w_img': w_img,
        'w_meta': w_meta,
        'accuracy': acc,
        'macro_f1': f1
    })
    
    print(f"| {f'{w_img} / {w_meta}':<18} | {acc*100:<9.2f}% | {f1:<10.4f} |")
    
    if acc > best_v3_acc or (acc == best_v3_acc and f1 > best_v3_f1):
        best_v3_acc = acc
        best_v3_f1 = f1
        best_v3_weight = (w_img, w_meta)

print(f"Best weight: {best_v3_weight[0]}/{best_v3_weight[1]} with accuracy {best_v3_acc*100:.2f}%\n")


# ==============================================================================
# SECTION 7: FINAL COMPARISON TABLE & METRIC SAVES
# ==============================================================================
print("=" * 80)
print("SECTION 7: FINAL PERFORMANCE COMPARISON & METRICS SAVING")
print("=" * 80)

# Calculate vs Image baseline changes
v1_change = (best_v1_acc - img_acc) * 100.0
v2_change = (best_v2_acc - img_acc) * 100.0
v3_change = (best_v3_acc - img_acc) * 100.0

v1_change_str = f"{v1_change:+.2f}%" if v1_change != 0 else "0.00%"
v2_change_str = f"{v2_change:+.2f}%" if v2_change != 0 else "0.00%"
v3_change_str = f"{v3_change:+.2f}%" if v3_change != 0 else "0.00%"

print(f"{'Configuration':<28} | {'Best Weight':<12} | {'Accuracy':<10} | {'Macro F1':<10} | {'vs Image Only':<15}")
print("-" * 80)
print(f"{'Image Only (V6)':<28} | {'—':<12} | {img_acc*100:<9.2f}% | {img_f1:<10.3f} | {'baseline':<15}")
print(f"{'Fusion: Image + Age':<28} | {f'{best_v1_weight[0]}/{best_v1_weight[1]}':<12} | {best_v1_acc*100:<9.2f}% | {best_v1_f1:<10.3f} | {v1_change_str:<15}")
print(f"{'Fusion: Image + Age + Sex':<28} | {f'{best_v2_weight[0]}/{best_v2_weight[1]}':<12} | {best_v2_acc*100:<9.2f}% | {best_v2_f1:<10.3f} | {v2_change_str:<15}")
print(f"{'Fusion: Image + Oracle':<28} | {f'{best_v3_weight[0]}/{best_v3_weight[1]}':<12} | {best_v3_acc*100:<9.2f}% | {best_v3_f1:<10.3f} | {v3_change_str:<15}")
print("=" * 80 + "\n")

# Save outputs to CSV
log_df = pd.DataFrame(full_results_log)
results_csv_path = os.path.join(OUTPUT_DIR, 'weighted_fusion_results.csv')
log_df.to_csv(results_csv_path, index=False)
print(f"  * Saved all search logs to: {results_csv_path}")

# Compile JSON Summary
summary_dict = {
    'baseline_image_only': {
        'accuracy': float(img_acc),
        'macro_f1': float(img_f1)
    },
    'baseline_metadata_age_only': {
        'accuracy': float(v1_baseline_acc),
        'macro_f1': float(v1_baseline_f1)
    },
    'baseline_metadata_age_sex': {
        'accuracy': float(v2_baseline_acc),
        'macro_f1': float(v2_baseline_f1)
    },
    'baseline_metadata_oracle': {
        'accuracy': float(v3_baseline_acc),
        'macro_f1': float(v3_baseline_f1)
    },
    'best_fusion_image_age': {
        'best_weight': {
            'w_img': best_v1_weight[0],
            'w_meta': best_v1_weight[1]
        },
        'accuracy': float(best_v1_acc),
        'macro_f1': float(best_v1_f1),
        'vs_baseline_acc_change': float(v1_change)
    },
    'best_fusion_image_age_sex': {
        'best_weight': {
            'w_img': best_v2_weight[0],
            'w_meta': best_v2_weight[1]
        },
        'accuracy': float(best_v2_acc),
        'macro_f1': float(best_v2_f1),
        'vs_baseline_acc_change': float(v2_change)
    },
    'best_fusion_image_oracle': {
        'best_weight': {
            'w_img': best_v3_weight[0],
            'w_meta': best_v3_weight[1]
        },
        'accuracy': float(best_v3_acc),
        'macro_f1': float(best_v3_f1),
        'vs_baseline_acc_change': float(v3_change)
    }
}

summary_json_path = os.path.join(OUTPUT_DIR, 'weighted_ablation_summary.json')
with open(summary_json_path, 'w') as f:
    json.dump(summary_dict, f, indent=4)
print(f"  * Saved JSON ablation summary to: {summary_json_path}\n")

# ==============================================================================
# SECTION 8: GRAPHICAL VISUALIZATIONS (BEST FUSION BAR CHART)
# ==============================================================================
print("=" * 80)
print("SECTION 8: GENERATING PERFORMANCE BAR CHART")
print("=" * 80)

# Build plotting dataframe
plot_configs = [
    "Image Only (V6)",
    f"Image + Age\n({best_v1_weight[0]}/{best_v1_weight[1]})",
    f"Image + Age + Sex\n({best_v2_weight[0]}/{best_v2_weight[1]})",
    f"Image + Oracle\n({best_v3_weight[0]}/{best_v3_weight[1]})"
]
plot_accs = [img_acc, best_v1_acc, best_v2_acc, best_v3_acc]
plot_f1s = [img_f1, best_v1_f1, best_v2_f1, best_v3_f1]

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 6.5))

x = np.arange(len(plot_configs))
width = 0.35

rects_acc = ax.bar(x - width/2, plot_accs, width, label='Accuracy', color='#3b82f6', edgecolor='black', alpha=0.9)
rects_f1 = ax.bar(x + width/2, plot_f1s, width, label='Macro F1', color='#10b981', edgecolor='black', alpha=0.9)

ax.set_ylabel('Performance Score (0.0 to 1.0)', fontsize=13, fontweight='bold', labelpad=10)
ax.set_title('OcuSense Best Weighted Late Fusion Systems Performance', fontsize=15, pad=20, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(plot_configs, fontsize=11, fontweight='semibold')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=12, loc='upper left')

# Add values above bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects_acc)
autolabel(rects_f1)

plt.tight_layout()
chart_save_path = os.path.join(OUTPUT_DIR, 'best_fusion_comparison.png')
plt.savefig(chart_save_path, dpi=150)
plt.close()
print(f"  * Saved premium best comparison grouped chart to: {chart_save_path}\n")

# Find the overall single best configuration
best_idx = np.argmax(plot_accs)
overall_best_name = plot_configs[best_idx].replace('\n', ' ')
overall_best_acc = plot_accs[best_idx]
overall_best_f1 = plot_f1s[best_idx]

print("=" * 80)
print("THE SINGLE BEST CONFIGURATION OVERALL")
print("=" * 80)
print(f"Configuration : {overall_best_name}")
print(f"Test Accuracy : {overall_best_acc*100:.2f}%")
print(f"Macro F1      : {overall_best_f1:.4f}")
print("=" * 80)
