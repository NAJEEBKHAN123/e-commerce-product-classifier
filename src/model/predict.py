# src/model/predict.py
import torch
from PIL import Image
from src.data.transforms import val_transform
import torch.nn.functional as F

class ProductPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()
        
        # Your 9 categories
        self.categories = [
            'BABY_PRODUCTS',
            'BEAUTY_HEALTH', 
            'CLOTHING_ACCESSORIES_JEWELLERY',
            'ELECTRONICS',
            'GROCERY',
            'HOBBY_ARTS_STATIONERY',
            'HOME_KITCHEN_TOOLS',
            'PET_SUPPLIES',
            'SPORTS_OUTDOOR'
        ]
        
    def predict(self, image_path):
        """Predict category for a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = val_transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
            result = {
                'category': self.categories[predicted_idx.item()],
                'confidence': confidence.item(),
                'category_id': predicted_idx.item(),
                'all_probabilities': probabilities[0].tolist()
            }
            
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def predict_batch(self, image_paths):
        """Predict categories for multiple images"""
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            if result:
                results.append({
                    'image_path': img_path,
                    **result
                })
        return results