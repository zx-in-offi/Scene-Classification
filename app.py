"""
Scene Classification Web App
Classify images into beach, forest, office, or street using ResNet50 or EfficientNet-B0.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
from flask import Flask, render_template, request, jsonify
import timm
import io
import base64

app = Flask(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────
CLASS_NAMES = ["beach", "forest", "office", "street"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization (same as training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Inference transform (no augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ─── Model Loading ───────────────────────────────────────────────────────────
models = {}


def load_resnet50():
    """Load ResNet50 model with trained weights."""
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    weight_path = os.path.join(os.path.dirname(__file__), "resnet50_scene_model.pth")
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def load_efficientnet():
    """Load EfficientNet-B0 model with trained weights."""
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
    weight_path = os.path.join(os.path.dirname(__file__), "efficientnet_b0_scene_model.pth")
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def get_model(model_name):
    """Get or lazily load a model by name."""
    if model_name not in models:
        if model_name == "resnet50":
            models[model_name] = load_resnet50()
            print("Loaded ResNet50")
        elif model_name == "efficientnet":
            models[model_name] = load_efficientnet()
            print("Loaded EfficientNet-B0")
    return models[model_name]


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept image + model choice, return predictions."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    model_name = request.form.get("model", "efficientnet")

    try:
        # Load and preprocess image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Run inference
        model = get_model(model_name)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        # Build response
        results = []
        for i, class_name in enumerate(CLASS_NAMES):
            results.append({
                "class": class_name,
                "confidence": round(probabilities[i].item() * 100, 2),
            })

        # Sort by confidence descending
        results.sort(key=lambda x: x["confidence"], reverse=True)

        return jsonify({
            "prediction": results[0]["class"],
            "confidence": results[0]["confidence"],
            "all_scores": results,
            "model_used": "ResNet50" if model_name == "resnet50" else "EfficientNet-B0",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Scene Classification Web App ===")
    print(f"Device: {DEVICE}")
    print("Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
