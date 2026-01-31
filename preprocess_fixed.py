import cv2
import os
import numpy as np
from tqdm import tqdm
import shutil

def normalize_brightness_contrast(image_path):
    """Normalize brightness and contrast using CLAHE"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Could not read: {image_path}")
            return None
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Split LAB channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        
        # Convert back to BGR
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return normalized
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None

def preprocess_dataset(input_root="dataset", output_root="dataset_normalized"):
    """Preprocess entire dataset"""
    
    print("🚀 STARTING DATASET PREPROCESSING")
    print("="*60)
    
    # Check if input exists
    if not os.path.exists(input_root):
        print(f"❌ Input dataset not found: {input_root}")
        return False
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    total_images = 0
    processed_images = 0
    
    # Count total images first
    print("📊 Counting images...")
    for split in ['train', 'val', 'check']:
        split_path = os.path.join(input_root, split)
        if not os.path.exists(split_path):
            continue
            
        for category in os.listdir(split_path):
            category_path = os.path.join(split_path, category)
            if os.path.isdir(category_path):
                images = [f for f in os.listdir(category_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                total_images += len(images)
    
    print(f"📁 Total images to process: {total_images}")
    
    # Process images
    print("\n🔄 Processing images...")
    with tqdm(total=total_images, desc="Progress") as pbar:
        for split in ['train', 'val', 'check']:
            split_input = os.path.join(input_root, split)
            split_output = os.path.join(output_root, split)
            
            if not os.path.exists(split_input):
                continue
                
            for category in os.listdir(split_input):
                category_input = os.path.join(split_input, category)
                category_output = os.path.join(split_output, category)
                
                if not os.path.isdir(category_input):
                    continue
                
                # Create output directory
                os.makedirs(category_output, exist_ok=True)
                
                # Process each image
                for img_name in os.listdir(category_input):
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        continue
                    
                    input_path = os.path.join(category_input, img_name)
                    output_path = os.path.join(category_output, img_name)
                    
                    # Process image
                    normalized = normalize_brightness_contrast(input_path)
                    
                    if normalized is not None:
                        # Save normalized image
                        cv2.imwrite(output_path, normalized)
                        processed_images += 1
                    else:
                        # Copy original if processing failed
                        shutil.copy2(input_path, output_path)
                        print(f"⚠️ Copied original: {img_name}")
                    
                    pbar.update(1)
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE")
    print(f"📊 Processed: {processed_images}/{total_images} images")
    print(f"📁 Output: {output_root}")
    
    # Verify output
    verify_preprocessing(output_root)
    
    return True

def verify_preprocessing(dataset_path):
    """Verify preprocessing worked"""
    print("\n🔍 VERIFYING PREPROCESSING...")
    
    # Check structure
    required_folders = ['train', 'val', 'check']
    for folder in required_folders:
        folder_path = os.path.join(dataset_path, folder)
        if os.path.exists(folder_path):
            categories = os.listdir(folder_path)
            print(f"  {folder}: {len(categories)} categories")
        else:
            print(f"  ⚠️ Missing: {folder}")
    
    # Check sample images
    print("\n📊 CHECKING SAMPLE IMAGES:")
    
    # Find a sample image
    sample_path = None
    for split in ['train', 'val', 'check']:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            categories = os.listdir(split_path)
            if categories:
                cat_path = os.path.join(split_path, categories[0])
                images = os.listdir(cat_path)
                if images:
                    sample_path = os.path.join(cat_path, images[0])
                    break
    
    if sample_path and os.path.exists(sample_path):
        img = cv2.imread(sample_path)
        print(f"  Sample image: {sample_path}")
        print(f"  Shape: {img.shape}")
        print(f"  Mean brightness: {img.mean():.1f}")
        print(f"  Std: {img.std():.1f}")
    else:
        print("  ⚠️ No sample image found")

if __name__ == "__main__":
    # Run preprocessing
    success = preprocess_dataset()
    
    if success:
        print("\n🎉 Ready to train with normalized dataset!")
        print("\nNext steps:")
        print("1. Update transforms.py to use simple transforms:")
        print("   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
        print("2. Run: python train_normalized.py")
    else:
        print("\n❌ Preprocessing failed")