# -*- coding: utf-8 -*-
"""
OcuSense Late Fusion Multimodal System (v1 - Full Comparative)
Combines OcuSense V6 (EfficientNet-B3) image model with 3 separate Metadata MLP configurations:
1. Age Only
2. Age + Sex
3. Oracle (Age + Sex + Keywords)

Executes test set integrity checks, individual models evaluation, multimodal fusion evaluations,
saves the comparative ablation study to CSV, and plots grouped performance charts.
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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Reconfigure stdout to UTF-8 on Windows environments to safely support checkmarks and lines
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
# SECTION 1: GLOBAL PATHS & MODEL WEIGHTS DEFINITION
# ==============================================================================
print("=" * 80)
print("SECTION 1: PATHS & CONSTANTS DEFINITIONS")
print("=" * 80)

CSV_PATH = r'C:\Users\gaura\Desktop\Eye Disease Detection\ocular-disease-recognition-odir5k\full_df.csv'
IMG_DIR = r'C:\Users\gaura\Desktop\Eye Disease Detection\ocular-disease-recognition-odir5k\preprocessed_images'
IMAGE_MODEL_PATH = r'C:\Users\gaura\Desktop\Eye Disease Detection\models-experm\ocusenseBaseline\v6\ocusense_v6_best.pth'
OUTPUT_DIR = r'C:\Users\gaura\Desktop\Eye Disease Detection\models-experm\late_fusion\v1'

# 3 Metadata Weight Paths
METADATA_AGE_ONLY_PATH = r'C:\Users\gaura\Desktop\Eye Disease Detection\models-experm\metadataModel\metadata_age_only.pth'
METADATA_AGE_SEX_PATH = r'C:\Users\gaura\Desktop\Eye Disease Detection\models-experm\metadataModel\metadata_age_sex.pth'
METADATA_ORACLE_PATH = r'C:\Users\gaura\Desktop\Eye Disease Detection\models-experm\metadataModel\metadata_oracle.pth'

# Self-healing helper for folder naming mismatch (metadataModel vs metdataModel)
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
print("SECTION 2: DATA LOADING & PATIENT-LEVEL SPLITTING")
print("=" * 80)

# Load CSV
df = pd.read_csv(CSV_PATH)
print(f"Loaded ODIR dataset with shape: {df.shape}")

# Map labels
df['label_str'] = df['labels'].apply(lambda x: ast.literal_eval(x)[0])
df['label_encoded'] = df['label_str'].apply(lambda x: MERGE_MAP[x])

# Extract ocular specific keyword
df['diagnostic_keyword'] = df.apply(
    lambda row: row['Left-Diagnostic Keywords'] if 'left' in str(row['filename']) else row['Right-Diagnostic Keywords'],
    axis=1
)

# Impute
df['Patient Age'] = df['Patient Age'].fillna(df['Patient Age'].median())
df['Patient Sex'] = df['Patient Sex'].fillna('Unknown')
df['diagnostic_keyword'] = df['diagnostic_keyword'].fillna('')

# Patient Split
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
print("SECTION 3: METADATA PREPROCESSING FIT & TRANSFORM")
print("=" * 80)

# --- Version 1: Age Only ---
scaler_v1 = StandardScaler()
scaler_v1.fit(train_df[['Patient Age']])
X_test_v1 = scaler_v1.transform(test_df[['Patient Age']])
print(f"V1 Age Only Test Feature Shape: {X_test_v1.shape}")

# --- Version 2: Age + Sex ---
preprocessor_v2 = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Patient Age']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Patient Sex'])
    ]
)
preprocessor_v2.fit(train_df[['Patient Age', 'Patient Sex']])
X_test_v2 = preprocessor_v2.transform(test_df[['Patient Age', 'Patient Sex']])
print(f"V2 Age + Sex Test Feature Shape: {X_test_v2.shape}")

# --- Version 3: Oracle (Age + Sex + Keywords) ---
preprocessor_v3 = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Patient Age']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Patient Sex'])
    ]
)
preprocessor_v3.fit(train_df[['Patient Age', 'Patient Sex']])
train_tab_v3 = preprocessor_v3.transform(train_df[['Patient Age', 'Patient Sex']])

# Check the exact dimensions of saved checkpoint weights to maintain size alignment
oracle_state = torch.load(METADATA_ORACLE_PATH, map_location='cpu')
oracle_input_dim = oracle_state['net.0.weight'].shape[1]
print(f"Oracle weight file expects input features: {oracle_input_dim}")

# Fit TF-IDF matching the remainder features needed (normally max_features=250, falls back to fit exact shape)
tfidf_max_features = oracle_input_dim - train_tab_v3.shape[1]
vectorizer_v3 = TfidfVectorizer(max_features=tfidf_max_features, stop_words='english')
vectorizer_v3.fit(train_df['diagnostic_keyword'])

test_tab_v3 = preprocessor_v3.transform(test_df[['Patient Age', 'Patient Sex']])
test_txt_v3 = vectorizer_v3.transform(test_df['diagnostic_keyword']).toarray()
X_test_v3 = np.hstack((test_tab_v3, test_txt_v3))
print(f"V3 Oracle Test Feature Shape  : {X_test_v3.shape} (Tabular: {test_tab_v3.shape[1]}, Keywords: {test_txt_v3.shape[1]})")
print("Preprocessing completed successfully!\n")

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

# 2. Flexible Metadata Model Definition supporting variable layer dimensions
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

print("All models loaded and placed in evaluation mode!\n")

# ==============================================================================
# SECTION 5: INFERENCE PIPELINE (RUNNING IMAGE MODALITY ONCE FOR SPEED)
# ==============================================================================
print("=" * 80)
print("SECTION 5: INFERENCE PIPELINE (OPTIMIZED IMAGE + 3x METADATA)")
print("=" * 80)

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Arrays to collect predictions and targets
valid_indices = []
filenames = []
true_labels = []
all_img_probs = []

# Processing Image Predictions ONCE to avoid redundant ConvNet runs
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

# Extract metadata inputs corresponding only to successfully processed image rows
y_true = np.array(true_labels)
img_probs_arr = np.array(all_img_probs)

# Run Inference on the 3 Metadata Configurations for the valid samples
print("Evaluating three separate Metadata MLP and Late Fusion configurations...")

# 1. Version 1: Age Only
test_meta_v1_valid = X_test_v1[valid_indices]
meta_v1_probs = []
meta_v1_model_inputs = torch.tensor(test_meta_v1_valid, dtype=torch.float32).to(device)
with torch.no_grad():
    meta_v1_logits = meta_age_model(meta_v1_model_inputs)
    meta_v1_probs = torch.softmax(meta_v1_logits, dim=1).cpu().numpy()

# 2. Version 2: Age + Sex
test_meta_v2_valid = X_test_v2[valid_indices]
meta_v2_probs = []
meta_v2_model_inputs = torch.tensor(test_meta_v2_valid, dtype=torch.float32).to(device)
with torch.no_grad():
    meta_v2_logits = meta_age_sex_model(meta_v2_model_inputs)
    meta_v2_probs = torch.softmax(meta_v2_logits, dim=1).cpu().numpy()

# 3. Version 3: Oracle
test_meta_v3_valid = X_test_v3[valid_indices]
meta_v3_probs = []
meta_v3_model_inputs = torch.tensor(test_meta_v3_valid, dtype=torch.float32).to(device)
with torch.no_grad():
    meta_v3_logits = meta_oracle_model(meta_v3_model_inputs)
    meta_v3_probs = torch.softmax(meta_v3_logits, dim=1).cpu().numpy()

# Compute Fusion Predictions (Average Probability Vectors)
fusion_age_probs = (img_probs_arr + meta_v1_probs) / 2.0
fusion_age_sex_probs = (img_probs_arr + meta_v2_probs) / 2.0
fusion_oracle_probs = (img_probs_arr + meta_v3_probs) / 2.0

print("Late Fusion probabilities computed successfully!\n")

# ==============================================================================
# SECTION 6: METRICS COMPUTATION & CLASSIFICATION REPORTS
# ==============================================================================
print("=" * 80)
print("SECTION 6: PERFORMANCE METRICS & REPORTS")
print("=" * 80)

# Predictions
y_pred_img = img_probs_arr.argmax(axis=1)

y_pred_v1 = meta_v1_probs.argmax(axis=1)
y_pred_v2 = meta_v2_probs.argmax(axis=1)
y_pred_v3 = meta_v3_probs.argmax(axis=1)

y_pred_fuse_v1 = fusion_age_probs.argmax(axis=1)
y_pred_fuse_v2 = fusion_age_sex_probs.argmax(axis=1)
y_pred_fuse_v3 = fusion_oracle_probs.argmax(axis=1)

# Compiling metric score lists
configs = [
    "Image Only (V6)",
    "Metadata Age Only",
    "Metadata Age + Sex",
    "Metadata Oracle",
    "Fusion: Image + Age",
    "Fusion: Image + Age + Sex",
    "Fusion: Image + Oracle"
]

preds_list = [
    y_pred_img,
    y_pred_v1,
    y_pred_v2,
    y_pred_v3,
    y_pred_fuse_v1,
    y_pred_fuse_v2,
    y_pred_fuse_v3
]

accuracies = [accuracy_score(y_true, pred) for pred in preds_list]
macro_f1s = [f1_score(y_true, pred, average='macro') for pred in preds_list]
notes = [
    "Baseline",
    "Pre-diagnostic",
    "Pre-diagnostic",
    "Post-diagnostic",
    "True Multimodal",
    "True Multimodal",
    "Oracle Upper Bound"
]

# Print reports for the fusion versions
print("\n" + "=" * 60)
print("FUSION CONFIGURATION 1: IMAGE + AGE ONLY")
print("=" * 60)
print(f"Accuracy: {accuracies[4]*100:.2f}% | Macro F1: {macro_f1s[4]:.4f}")
print(classification_report(y_true, y_pred_fuse_v1, target_names=DISEASE_NAMES, zero_division=0))

print("\n" + "=" * 60)
print("FUSION CONFIGURATION 2: IMAGE + AGE & SEX")
print("=" * 60)
print(f"Accuracy: {accuracies[5]*100:.2f}% | Macro F1: {macro_f1s[5]:.4f}")
print(classification_report(y_true, y_pred_fuse_v2, target_names=DISEASE_NAMES, zero_division=0))

print("\n" + "=" * 60)
print("FUSION CONFIGURATION 3: IMAGE + ORACLE")
print("=" * 60)
print(f"Accuracy: {accuracies[6]*100:.2f}% | Macro F1: {macro_f1s[6]:.4f}")
print(classification_report(y_true, y_pred_fuse_v3, target_names=DISEASE_NAMES, zero_division=0))

# Print final comparison ablation table
print("\n" + "=" * 80)
print("FINAL ABLATION TABLE")
print("=" * 80)
print(f"{'Configuration':<28} | {'Accuracy (%)':<15} | {'Macro F1':<12} | {'Notes':<20}")
print("-" * 80)
for i in range(7):
    print(f"{configs[i]:<28} | {accuracies[i]*100:<15.2f} | {macro_f1s[i]:<12.4f} | {notes[i]:<20}")
print("=" * 80 + "\n")

# ==============================================================================
# SECTION 7: PERSISTING ABLATION TABLES
# ==============================================================================
print("=" * 80)
print("SECTION 7: SAVING STATISTICAL COMPARISONS & LOGGING METRICS")
print("=" * 80)

ablation_df = pd.DataFrame({
    'Configuration': configs,
    'Accuracy': accuracies,
    'Macro F1': macro_f1s,
    'Notes': notes
})
csv_save_path = os.path.join(OUTPUT_DIR, 'ablation_full_results.csv')
ablation_df.to_csv(csv_save_path, index=False)
print(f"  * Saved comprehensive ablation CSV to: {csv_save_path}\n")

# ==============================================================================
# SECTION 8: GRAPHICAL VISUALIZATIONS
# ==============================================================================
print("=" * 80)
print("SECTION 8: GENERATING MULTI-CONFIGURATION COMPARISON CHART")
print("=" * 80)

# Set styled palette and design aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(configs))
width = 0.35

# Plot side-by-side grouped bars for Accuracy and Macro F1
rects_acc = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#3b82f6', edgecolor='black', alpha=0.95)
rects_f1 = ax.bar(x + width/2, macro_f1s, width, label='Macro F1', color='#f59e0b', edgecolor='black', alpha=0.95)

# Labeling and styling
ax.set_ylabel('Performance Score (0.0 to 1.0)', fontsize=13, fontweight='bold', labelpad=10)
ax.set_title('OcuSense Late Fusion Ablation Study — 7 System Configurations Performance', fontsize=15, pad=20, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=15, ha='right', fontsize=11, fontweight='semibold')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=12, loc='upper left')

# Add score labels on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects_acc)
autolabel(rects_f1)

plt.tight_layout()
chart_save_path = os.path.join(OUTPUT_DIR, 'ablation_full_comparison.png')
plt.savefig(chart_save_path, dpi=150)
plt.close()
print(f"  * Saved premium comparison grouped chart to: {chart_save_path}")

print("\n" + "=" * 80)
print("OCUSENSE LATE FUSION MULTIMODAL SYSTEM WORKFLOW COMPLETED SUCCESSFULLY!")
print("=" * 80)
