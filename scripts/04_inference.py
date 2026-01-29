#!/usr/bin/env python3
"""
Inference script for single image prediction
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import argparse
import os
import sys

class ProductClassifier:
    def __init__(self, model_path, metadata_path):
        """Initialize classifier"""
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path):
        """Load trained model"""
        model = models.mobilenet_v2()
        
        # Adjust classifier for correct number of classes
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, self.metadata['num_classes'])
        
        # Load state dict
        if 'model_state_dict' in torch.load(model_path, map_location=self.device):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        model = model.to(self.device)
        return model
    
    def predict(self, image_path, top_k=3):
        """Predict category for an image"""
        
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            return {"error": f"Cannot open image: {image_path}"}
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Get class names
        idx_to_class = {v: k for k, v in self.metadata['class_to_idx'].items()}
        predicted_class = idx_to_class[predicted_idx.item()]
        
        # Get top-k predictions
        topk_probs, topk_indices = torch.topk(probabilities, min(top_k, self.metadata['num_classes']))
        
        # Prepare results
        result = {
            "image": os.path.basename(image_path),
            "predicted_class": predicted_class,
            "confidence": float(confidence.item()),
            "top_predictions": [
                {
                    "class": idx_to_class[idx.item()],
                    "confidence": float(prob.item())
                }
                for prob, idx in zip(topk_probs[0], topk_indices[0])
            ],
            "all_classes": self.metadata['classes']
        }
        
        return result
    
    def predict_batch(self, image_paths, top_k=3):
        """Predict for multiple images"""
        results = []
        for img_path in image_paths:
            result = self.predict(img_path, top_k)
            results.append(result)
        return results

def display_results(result):
    """Display prediction results nicely"""
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Image: {result['image']}")
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.3%}")
    
    print("\nTop Predictions:")
    for i, pred in enumerate(result['top_predictions'], 1):
        print(f"  {i}. {pred['class']}: {pred['confidence']:.3%}")
    
    print("\nAll Categories:")
    for i, cls in enumerate(result['all_classes'], 1):
        print(f"  {i}. {cls}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Predict product category')
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--model', default='models/best_model.pth',
                       help='Path to model file')
    parser.add_argument('--metadata', default='models/model_metadata.json',
                       help='Path to metadata file')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("Please train a model first or specify correct path with --model")
        sys.exit(1)
    
    if not os.path.exists(args.metadata):
        print(f"Error: Metadata file '{args.metadata}' not found!")
        sys.exit(1)
    
    # Initialize classifier
    print("Loading model...")
    classifier = ProductClassifier(args.model, args.metadata)
    
    # Make prediction
    print(f"Predicting for: {args.image}")
    result = classifier.predict(args.image, args.top_k)
    
    # Display results
    display_results(result)
    
    # Save to JSON
    output_file = "prediction_result.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()