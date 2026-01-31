# preprocess_images.py
import cv2
import os
from PIL import Image
import numpy as np

def normalize_brightness_contrast(image_path, output_path):
    """Normalize brightness and contrast of an image"""
    img = cv2.imread(image_path)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    lab = cv2.merge([l, a, b])
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Save
    cv2.imwrite(output_path, normalized)

def preprocess_entire_dataset():
    """Preprocess all images in dataset"""
    for split in ['train', 'val', 'check']:
        for category in categories:
            input_dir = f"dataset/{split}/{category}"
            output_dir = f"dataset_normalized/{split}/{category}"
            
            os.makedirs(output_dir, exist_ok=True)
            
            for img_file in os.listdir(input_dir):
                input_path = os.path.join(input_dir, img_file)
                output_path = os.path.join(output_dir, img_file)
                normalize_brightness_contrast(input_path, output_path)