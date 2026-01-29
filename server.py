# server.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import io
from PIL import Image
import torch
import uvicorn
import os
from src.model.predict import ProductPredictor

app = FastAPI(title="E-commerce Product Classifier API")

# CORS for MERN integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor
predictor = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global predictor
    model_path = "models/best_model.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if os.path.exists(model_path):
        predictor = ProductPredictor(model_path, device)
        print(f"Model loaded successfully on {device}")
    else:
        print(f"Warning: Model not found at {model_path}")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Predict product category from uploaded image"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Save temp file for prediction
        temp_path = f"temp_{file.filename}"
        image.save(temp_path)
        
        # Predict
        result = predictor.predict(temp_path)
        
        # Cleanup
        os.remove(temp_path)
        
        if result:
            return {
                "success": True,
                "filename": file.filename,
                "prediction": result,
                "message": "Prediction successful"
            }
        else:
            raise HTTPException(status_code=500, detail="Prediction failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """Predict categories for multiple images"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []
    temp_files = []
    
    try:
        for file in files:
            # Read and save each image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            temp_path = f"temp_batch_{file.filename}"
            image.save(temp_path)
            temp_files.append(temp_path)
            
            # Predict
            result = predictor.predict(temp_path)
            if result:
                results.append({
                    "filename": file.filename,
                    **result
                })
        
        # Cleanup
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return {
            "success": True,
            "predictions": results,
            "total": len(results)
        }
        
    except Exception as e:
        # Cleanup on error
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
async def get_categories():
    """Get list of all product categories"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "categories": predictor.categories,
        "count": len(predictor.categories)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if predictor else "model_not_loaded",
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)