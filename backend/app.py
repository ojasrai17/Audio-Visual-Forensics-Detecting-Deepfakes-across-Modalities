import torch
import torch.nn as nn
import timm
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
import io
import logging
import os

# --- 1. Setup Logging for Better Debugging ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")

# --- 2. Define the Correct Hybrid Model Architecture ---
class HybridModel(nn.Module):
    def __init__(self, effnet_backbone, swin_backbone, num_classes=2):
        super(HybridModel, self).__init__()
        self.effnet = effnet_backbone
        self.swin = swin_backbone

        # Get the number of output features from each backbone
        effnet_features = self.effnet.num_features
        swin_features = self.swin.num_features
        concatenated_features = effnet_features + swin_features

        logging.info(f"EfficientNet feature size: {effnet_features}")
        logging.info(f"Swin Transformer feature size: {swin_features}")
        logging.info(f"Concatenated feature size: {concatenated_features}")

        # Define the classifier head, including the BatchNorm layer
        self.classifier = nn.Sequential(
            nn.Linear(concatenated_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features_effnet = self.effnet(x)
        features_swin = self.swin(x)
        combined_features = torch.cat((features_effnet, features_swin), dim=1)
        return self.classifier(combined_features)

# --- 3. Load Model and Define Preprocessing ---
logging.info("Loading model...")

# Path to your single hybrid model file
MODEL_PATH = 'best_hybrid_model_head_only.pth'

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at: {MODEL_PATH}")

# Create the backbones with ImageNet pretrained weights and num_classes=0 (feature extractor mode)
logging.info("Creating model architecture with ImageNet pretrained weights...")
logging.info("Downloading ImageNet pretrained weights (first run only)...")

effnet_backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
logging.info("✅ EfficientNet-B0 pretrained weights loaded")

swin_backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
logging.info("✅ Swin Transformer pretrained weights loaded")

# Create the hybrid model
model = HybridModel(effnet_backbone, swin_backbone, num_classes=2).to(DEVICE)

# Load your fine-tuned weights (this overwrites ImageNet weights with your trained weights)
logging.info(f"Loading your fine-tuned weights from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Try loading full model, if that fails, load only classifier
try:
    model.load_state_dict(checkpoint)
    logging.info(f"✅ Loaded full fine-tuned model weights!")
except RuntimeError as e:
    logging.warning(f"Could not load full model, attempting to load classifier only...")
    try:
        model.classifier.load_state_dict(checkpoint)
        logging.info(f"✅ Loaded classifier weights (using pretrained ImageNet backbones)!")
    except Exception as e2:
        logging.error(f"❌ Failed to load model: {e2}")
        raise

model.eval()
logging.info(f"✅ Model ready for inference!")

# The image transformations must match your training validation transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- 4. Initialize Flask App ---
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- 5. Define Prediction Endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    logging.info("Received a new /predict request")
    
    if 'file' not in request.files:
        logging.warning("Request rejected: No file part found.")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        logging.warning("Request rejected: No file was selected.")
        return jsonify({"error": "No file selected"}), 400

    if not file.content_type.startswith('image/'):
        logging.warning(f"Request rejected: Unsupported file type '{file.content_type}'.")
        return jsonify({"error": "Unsupported file type. Please upload an image."}), 400

    try:
        logging.info(f"Processing file: {file.filename}")
        
        # Load and preprocess image
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        # Make prediction
        with torch.no_grad():
            output = model(tensor)
            
            # Get probabilities and prediction
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)
            
            # Define class names - adjust based on your training data
            # If your folders were named "Fake" and "Real", ImageFolder sorts alphabetically
            # So classes would be: ["Fake", "Real"]
            classes = ["Fake", "Real"]  # Change to ["Real", "Fake"] if your training had different order
            prediction = classes[pred.item()]
            confidence = conf.item()

        confidence_percent = f"{confidence*100:.2f}%"
        logging.info(f"Prediction successful: {prediction} with {confidence_percent} confidence.")

        return jsonify({
            "prediction": prediction,
            "confidence": confidence_percent,
            "class_index": pred.item(),
            "probabilities": {
                classes[0]: f"{probs[0][0].item()*100:.2f}%",
                classes[1]: f"{probs[0][1].item()*100:.2f}%"
            }
        })

    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# --- 6. Health Check Endpoint ---
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy", 
        "device": DEVICE,
        "model_loaded": True,
        "pretrained": "ImageNet-1K",
        "architecture": "Hybrid EfficientNet-B0 + Swin Transformer"
    })

# --- 7. Root Endpoint ---
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "Deepfake Detection API - ImageNet Pretrained Fine-tuned Model",
        "model": {
            "backbone_1": "EfficientNet-B0 (ImageNet pretrained)",
            "backbone_2": "Swin Transformer Tiny (ImageNet pretrained)",
            "architecture": "Hybrid Dual-Backbone CNN-Transformer",
            "classes": ["Fake", "Real"]
        },
        "endpoints": {
            "/": "GET - API information",
            "/health": "GET - Check API health",
            "/predict": "POST - Upload image for deepfake detection"
        }
    })

# --- 8. Run the Flask App ---
if __name__ == "__main__":
    logging.info("="*60)
    logging.info("🚀 Deepfake Detection API with ImageNet Pretrained Models")
    logging.info("="*60)
    logging.info(f"📊 Model: Hybrid EfficientNet-B0 + Swin Transformer")
    logging.info(f"🎯 Pretrained: ImageNet-1K → Fine-tuned on Deepfakes")
    logging.info(f"🖥️  Device: {DEVICE}")
    logging.info(f"🌐 API URL: http://0.0.0.0:5000")
    logging.info("="*60)
    
    # Get port from environment variable (for deployment) or use 5000
    port = int(os.environ.get("PORT", 5000))
    
    # Run Flask app
    # debug=True for development, debug=False for production
    app.run(host="0.0.0.0", port=port, debug=True)