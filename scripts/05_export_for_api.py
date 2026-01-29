#!/usr/bin/env python3
"""
Export model for FastAPI deployment
"""

import torch
import torch.nn as nn
from torchvision import models
import json
import os
import argparse

def export_model_for_api(model_path, metadata_path, output_dir="api_models"):
    """Export model and create API-ready files"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Exporting model with {metadata['num_classes']} classes")
    print(f"Classes: {metadata['classes']}")
    
    # Create a simpler model for API
    device = torch.device("cpu")
    
    # Create model
    model = models.mobilenet_v2()
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, metadata['num_classes'])
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Save regular model
    regular_model_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), regular_model_path)
    print(f"✓ Model saved: {regular_model_path}")
    
    # Create simplified metadata for API
    api_metadata = {
        "classes": metadata["classes"],
        "class_to_idx": metadata["class_to_idx"],
        "num_classes": metadata["num_classes"],
        "input_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
    
    api_metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(api_metadata_path, 'w') as f:
        json.dump(api_metadata, f, indent=2)
    
    print(f"✓ Metadata saved: {api_metadata_path}")
    
    # Create requirements file for API
    requirements = [
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "Pillow>=10.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "python-multipart>=0.0.6",
        "numpy>=1.24.0"
    ]
    
    requirements_path = os.path.join(output_dir, "requirements.txt")
    with open(requirements_path, 'w') as f:
        f.write("\n".join(requirements))
    
    print(f"✓ Requirements saved: {requirements_path}")
    
    # Create simple API script
    api_script = '''from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import json
import os
import uvicorn

app = FastAPI(
    title="E-commerce Product Classifier API",
    description="API for classifying product images into 9 categories",
    version="1.0.0"
)

# CORS middleware
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
    """Load model and metadata"""
    global model, metadata, transform, device
    
    try:
        # Load metadata
        with open("model_metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        model = models.mobilenet_v2()
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, metadata["num_classes"])
        
        # Load weights
        model.load_state_dict(torch.load("model.pth", map_location=device))
        model.eval()
        model.to(device)
        
        # Setup transform
        transform = transforms.Compose([
            transforms.Resize((metadata["input_size"], metadata["input_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=metadata["mean"], std=metadata["std"])
        ])
        
        print(f"Model loaded successfully with {metadata['num_classes']} classes")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    if not load_model():
        raise RuntimeError("Failed to load model on startup")

@app.get("/")
async def root():
    return {
        "message": "E-commerce Product Classifier API",
        "status": "running",
        "version": "1.0.0",
        "model": "MobileNetV2",
        "num_classes": metadata["num_classes"] if metadata else "Loading..."
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
    """Predict category for uploaded image"""
    if not model or not metadata:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        raise HTTPException(status_code=400, detail="Invalid image format")
    
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
        idx_to_class = {v: k for k, v in metadata["class_to_idx"].items()}
        predicted_class = idx_to_class[predicted_idx.item()]
        
        # Get top 3 predictions
        top3_probs, top3_idx = torch.topk(probabilities, 3)
        
        # Prepare response
        response = {
            "success": True,
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": float(confidence.item()),
            "top_predictions": [
                {
                    "class": idx_to_class[idx.item()],
                    "confidence": float(prob.item())
                }
                for prob, idx in zip(top3_probs[0], top3_idx[0])
            ]
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Predict categories for multiple images"""
    if not model or not metadata:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for file in files:
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
            idx_to_class = {v: k for k, v in metadata["class_to_idx"].items()}
            predicted_class = idx_to_class[predicted_idx.item()]
            
            results.append({
                "filename": file.filename,
                "predicted_class": predicted_class,
                "confidence": float(confidence.item()),
                "success": True
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    
    return {
        "success": True,
        "total_images": len(files),
        "predictions": results
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }

@app.get("/info")
async def model_info():
    """Get model information"""
    if not metadata:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "MobileNetV2",
        "num_classes": metadata["num_classes"],
        "input_size": metadata["input_size"],
        "classes": metadata["classes"],
        "class_to_idx": metadata["class_to_idx"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
'''
    
    api_script_path = os.path.join(output_dir, "main.py")
    with open(api_script_path, 'w') as f:
        f.write(api_script)
    
    print(f"✓ API script saved: {api_script_path}")
    
    # Create Dockerfile
    dockerfile = '''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY model.pth .
COPY model_metadata.json .
COPY main.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
'''
    
    dockerfile_path = os.path.join(output_dir, "Dockerfile")
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile)
    
    print(f"✓ Dockerfile saved: {dockerfile_path}")
    
    # Create README
    readme = f'''# E-commerce Product Classifier API

## Model Information
- **Model**: MobileNetV2
- **Input Size**: 224x224 pixels
- **Number of Classes**: {metadata['num_classes']}
- **Categories**: {', '.join(metadata['classes'])}

## Setup

# ### 1. Install dependencies:
# ```bash
# pip install -r requirements.txt