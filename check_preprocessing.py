# check_preprocessing.py
import os
import cv2
import numpy as np
from PIL import Image

print("🔍 CHECKING PREPROCESSED IMAGES")
print("="*60)

# Check if normalized dataset exists
normalized_path = "dataset_normalized"
if not os.path.exists(normalized_path):
    print("❌ Normalized dataset not found!")
    print("   Make sure preprocess_images.py created 'dataset_normalized/' folder")
    exit()

# Compare original vs normalized
original_img = "dataset/train/BABY_PRODUCTS/1000_BABY_P_train.jpeg"
normalized_img = "dataset_normalized/train/BABY_PRODUCTS/1000_BABY_P_train.jpeg"

if os.path.exists(original_img) and os.path.exists(normalized_img):
    # Load images
    orig = cv2.imread(original_img)
    norm = cv2.imread(normalized_img)
    
    # Convert to RGB for PIL
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    norm_rgb = cv2.cvtColor(norm, cv2.COLOR_BGR2RGB)
    
    print(f"✅ Found both original and normalized images")
    print(f"\nOriginal image:")
    print(f"  Shape: {orig.shape}")
    print(f"  Mean brightness: {orig.mean():.1f}")
    print(f"  Std: {orig.std():.1f}")
    
    print(f"\nNormalized image:")
    print(f"  Shape: {norm.shape}")
    print(f"  Mean brightness: {norm.mean():.1f}")
    print(f"  Std: {norm.std():.1f}")
    
    # Check improvement
    orig_std = orig.std()
    norm_std = norm.std()
    
    if norm_std > orig_std * 1.5:
        print(f"\n✅ GOOD: Contrast improved! (std: {orig_std:.1f} → {norm_std:.1f})")
    else:
        print(f"\n⚠️  Contrast not significantly improved")
    
    # Visual check (optional)
    print(f"\n📊 Brightness distribution:")
    print(f"  Original range: [{orig.min()}, {orig.max()}]")
    print(f"  Normalized range: [{norm.min()}, {norm.max()}]")
    
else:
    print(f"❌ Could not find test images")
    print(f"   Original: {original_img} - Exists: {os.path.exists(original_img)}")
    print(f"   Normalized: {normalized_img} - Exists: {os.path.exists(normalized_img)}")

# Check multiple images
print(f"\n" + "="*60)
print("CHECKING MULTIPLE IMAGES:")

brightness_values = []
for i in range(3):
    orig_path = f"dataset/train/BABY_PRODUCTS/100{i}_BABY_P_train.jpeg"
    norm_path = f"dataset_normalized/train/BABY_PRODUCTS/100{i}_BABY_P_train.jpeg"
    
    if os.path.exists(orig_path) and os.path.exists(norm_path):
        orig = cv2.imread(orig_path)
        norm = cv2.imread(norm_path)
        
        brightness_values.append({
            'original': orig.mean(),
            'normalized': norm.mean()
        })
        
        print(f"Image {i}: Original={orig.mean():.1f}, Normalized={norm.mean():.1f}")

if brightness_values:
    orig_avg = np.mean([b['original'] for b in brightness_values])
    norm_avg = np.mean([b['normalized'] for b in brightness_values])
    orig_std = np.std([b['original'] for b in brightness_values])
    norm_std = np.std([b['normalized'] for b in brightness_values])
    
    print(f"\n📈 SUMMARY:")
    print(f"  Original - Mean: {orig_avg:.1f}, Std: {orig_std:.1f}")
    print(f"  Normalized - Mean: {norm_avg:.1f}, Std: {norm_std:.1f}")
    
    if norm_std < orig_std * 0.7:
        print(f"  ✅ SUCCESS: Brightness variation reduced by {100*(1-norm_std/orig_std):.0f}%")
    else:
        print(f"  ⚠️  Brightness variation not significantly reduced")