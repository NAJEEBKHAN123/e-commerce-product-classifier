import torch
import sys
import os
from PIL import Image
import numpy as np

sys.path.append('.')

print("🔍 CHECKING NORMALIZATION ISSUES")
print("="*60)

# 1. Import your transforms
from src.data.transforms import train_transform, val_transform
print("1. TRANSFORMS:")
print(f"   Train: {train_transform}")
print(f"   Val: {val_transform}")

# 2. Test on actual dataset images
from src.data.loader import create_dataloaders
train_loader, _, _, _, categories = create_dataloaders(
    data_dir="d:\\ecommerce-product-classifier\\dataset",
    batch_size=4
)

# 3. Check first batch statistics
print("\n2. BATCH STATISTICS (from dataloader):")
for images, labels, paths in train_loader:
    print(f"   Batch shape: {images.shape}")
    print(f"   Data type: {images.dtype}")
    
    # Calculate statistics
    mean = images.mean(dim=[0, 2, 3])  # Mean per channel
    std = images.std(dim=[0, 2, 3])    # Std per channel
    
    print(f"   Actual means:  [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")
    print(f"   Actual stds:   [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")
    print(f"   Expected means: [0.485, 0.456, 0.406]")
    print(f"   Expected stds:  [0.229, 0.224, 0.225]")
    
    # Check if normalized correctly
    target_mean = torch.tensor([0.485, 0.456, 0.406])
    target_std = torch.tensor([0.229, 0.224, 0.225])
    
    mean_diff = torch.abs(mean - target_mean)
    std_diff = torch.abs(std - target_std)
    
    print(f"   Mean difference: [{mean_diff[0]:.3f}, {mean_diff[1]:.3f}, {mean_diff[2]:.3f}]")
    print(f"   Std difference:  [{std_diff[0]:.3f}, {std_diff[1]:.3f}, {std_diff[2]:.3f}]")
    
    # Value range
    print(f"   Min value: {images.min():.3f}")
    print(f"   Max value: {images.max():.3f}")
    print(f"   Value range should be approximately [-2, 2] for normalized images")
    
    break

# 4. Test raw image before transform
print("\n3. RAW IMAGE CHECK (before transform):")
# Get first image path
for _, _, paths in train_loader:
    test_path = paths[0]
    break

img = Image.open(test_path).convert('RGB')
print(f"   Image path: {test_path}")
print(f"   Original size: {img.size}")
print(f"   Original mode: {img.mode}")

# Convert to numpy to check raw values
img_array = np.array(img)
print(f"   Raw array shape: {img_array.shape}")
print(f"   Raw min value: {img_array.min()}")
print(f"   Raw max value: {img_array.max()}")
print(f"   Raw mean: [{img_array[:,:,0].mean():.1f}, {img_array[:,:,1].mean():.1f}, {img_array[:,:,2].mean():.1f}]")

# 5. Step-by-step transform check
print("\n4. STEP-BY-STEP TRANSFORM CHECK:")

# Test resize
from torchvision import transforms
resize_transform = transforms.Resize((256, 256))
resized = resize_transform(img)
print(f"   After resize(256,256): {resized.size}")

# Test crop
crop_transform = transforms.RandomCrop((224, 224))
cropped = crop_transform(resized)
print(f"   After crop(224,224): {cropped.size}")

# Test ToTensor
tensor_transform = transforms.ToTensor()
tensor = tensor_transform(cropped)
print(f"   After ToTensor: shape={tensor.shape}, range=[{tensor.min():.3f}, {tensor.max():.3f}]")

# Test normalization
normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])
normalized = normalize_transform(tensor)
print(f"   After Normalize: range=[{normalized.min():.3f}, {normalized.max():.3f}]")
print(f"   Normalized mean: [{normalized.mean(dim=(1,2))[0]:.3f}, {normalized.mean(dim=(1,2))[1]:.3f}, {normalized.mean(dim=(1,2))[2]:.3f}]")
print(f"   Normalized std:  [{normalized.std(dim=(1,2))[0]:.3f}, {normalized.std(dim=(1,2))[1]:.3f}, {normalized.std(dim=(1,2))[2]:.3f}]")

# 6. Check if images are already normalized
print("\n5. CHECK IF IMAGES ARE PRE-PROCESSED:")
print("   If images are already 224x224 and low file size (7.8KB from EDA),")
print("   they might be already normalized or compressed.")

# Check multiple images
print("\n6. CHECKING MULTIPLE IMAGES:")
batch_count = 0
all_means = []
all_stds = []

for images, labels, paths in train_loader:
    mean = images.mean(dim=[0, 2, 3])
    std = images.std(dim=[0, 2, 3])
    all_means.append(mean)
    all_stds.append(std)
    
    batch_count += 1
    if batch_count >= 3:  # Check 3 batches
        break

avg_mean = torch.stack(all_means).mean(dim=0)
avg_std = torch.stack(all_stds).mean(dim=0)

print(f"   Average over {batch_count} batches:")
print(f"   Mean: [{avg_mean[0]:.3f}, {avg_mean[1]:.3f}, {avg_mean[2]:.3f}]")
print(f"   Std:  [{avg_std[0]:.3f}, {avg_std[1]:.3f}, {avg_std[2]:.3f}]")

print("\n" + "="*60)
print("DIAGNOSIS:")

# Determine issue
target_mean = torch.tensor([0.485, 0.456, 0.406])
target_std = torch.tensor([0.229, 0.224, 0.225])

mean_error = torch.abs(avg_mean - target_mean).mean().item()
std_error = torch.abs(avg_std - target_std).mean().item()

if mean_error > 0.1 or std_error > 0.1:
    print("❌ NORMALIZATION IS WRONG!")
    print(f"   Mean error: {mean_error:.3f} (should be < 0.1)")
    print(f"   Std error: {std_error:.3f} (should be < 0.1)")
    print("\n   SOLUTION: Try without normalization first:")
    print("   transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])")
else:
    print("✅ NORMALIZATION LOOKS CORRECT")
    print("   The issue might be elsewhere (model, labels, etc.)")