# src/data/dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class EcommerceDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train'):
        """
        Args:
            data_dir: Path to main dataset folder (contains train, val, check)
            transform: Optional transform to be applied
            split: 'train', 'val', or 'check'
        """
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.split = split
        
        # Get all categories
        self.categories = sorted([d for d in os.listdir(self.data_dir) 
                                if os.path.isdir(os.path.join(self.data_dir, d))])
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []
        
        for category in self.categories:
            category_path = os.path.join(self.data_dir, category)
            for img_name in os.listdir(category_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_paths.append(os.path.join(category_path, img_name))
                    self.labels.append(self.category_to_idx[category])
        
        print(f"{split} dataset: {len(self.image_paths)} images, {len(self.categories)} categories")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label, img_path  # Return path for debugging
        
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy image
            dummy_image = torch.zeros((3, 224, 224))
            return dummy_image, label, img_path