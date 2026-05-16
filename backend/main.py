# main.py — OcuSense V6 Flask API

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import os

app = Flask(__name__)
CORS(app)

# ── Config ─────────────────────────────────────────────────────────────────────
DISEASE_NAMES = [
    'Normal',
    'Diabetic Retinopathy',
    'Glaucoma',
    'Cataract',
    'Myopia',
    'Other'
]

DISEASE_INFO = {
    'Normal': 'No signs of retinal disease detected.',
    'Diabetic Retinopathy': 'Damage to blood vessels in the retina caused by diabetes.',
    'Glaucoma': 'Damage to the optic nerve, often caused by high eye pressure.',
    'Cataract': 'Clouding of the eye lens causing blurred vision.',
    'Myopia': 'Nearsightedness — difficulty seeing distant objects clearly.',
    'Other': 'Abnormality detected. Please consult a specialist for further evaluation.'
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ocusense_v6_best.pth')

# ── Model Architecture ─────────────────────────────────────────────────────────
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

# ── Load Model Once at Startup ─────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = OcuSenseV6(num_classes=6).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"OcuSense V6 loaded on {device}")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'OcuSense V6'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Load and preprocess image
        image_bytes = file.read()
        image       = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor      = inference_transform(image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(tensor)
            probs   = torch.softmax(outputs, dim=1)[0]

        # Build response
        predicted_idx     = probs.argmax().item()
        predicted_disease = DISEASE_NAMES[predicted_idx]
        confidence        = probs[predicted_idx].item() * 100

        all_probs = {
            name: round(prob.item() * 100, 2)
            for name, prob in zip(DISEASE_NAMES, probs)
        }

        return jsonify({
            'prediction'  : predicted_disease,
            'confidence'  : round(confidence, 2),
            'info'        : DISEASE_INFO[predicted_disease],
            'all_probs'   : all_probs,
            'model'       : 'OcuSense V6'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)