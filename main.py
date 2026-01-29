# Save as: main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import json
import os
import uvicorn

app = FastAPI(title="E-commerce Product Classifier API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
metadata = None
transform = None
device = None

def load_model():
    """Load the trained model"""
    global model, metadata, transform, device
    
    try:
        # Load metadata
        with open("models/model_metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        model = models.mobilenet_v2()
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, metadata["num_classes"])
        
        # Load weights
        model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
        model.eval()
        model.to(device)
        
        # Setup transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Model loaded with {metadata['num_classes']} classes")
        print(f"✅ Categories: {metadata['classes']}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    print("🚀 Starting E-commerce Classifier API...")
    if not load_model():
        raise RuntimeError("Failed to load model")

@app.get("/")
async def root():
    return {
        "message": "E-commerce Product Classifier API",
        "status": "running",
        "model": "MobileNetV2",
        "num_classes": metadata["num_classes"] if metadata else 0,
        "endpoints": [
            "GET / - API info",
            "GET /categories - List categories",
            "POST /predict - Classify image",
            "GET /health - Health check"
        ]
    }

@app.get("/categories")
async def get_categories():
    if not metadata:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "categories": metadata["classes"],
        "count": len(metadata["classes"])
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Classify a product image"""
    if not model or not metadata:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only JPG/PNG images allowed")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Get class name
        predicted_class = metadata["classes"][predicted_idx.item()]
        
        # Get top 3 predictions
        top3_probs, top3_idx = torch.topk(probabilities, 3)
        
        return {
            "success": True,
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": float(confidence.item()),
            "top_predictions": [
                {
                    "class": metadata["classes"][idx.item()],
                    "confidence": float(prob.item())
                }
                for prob, idx in zip(top3_probs[0], top3_idx[0])
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)