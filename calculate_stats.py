# calculate_normalized_stats.py
import torch
from torch.utils.data import DataLoader
from src.data.dataset import EcommerceDataset
import torchvision.transforms as transforms
import numpy as np

print("📊 CALCULATING NORMALIZED DATASET STATISTICS")
print("="*60)

# Simple transform - NO NORMALIZATION
simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load NORMALIZED dataset
dataset = EcommerceDataset(
    data_dir="dataset_normalized",  # Use normalized dataset!
    transform=simple_transform,
    split='train'
)

dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

# Calculate statistics
mean = 0.0
meansq = 0.0
total_pixels = 0

for batch_idx, (images, _, _) in enumerate(dataloader):
    # images shape: [batch, channels, height, width]
    batch_pixels = images.size(0) * images.size(2) * images.size(3)
    
    # Sum for mean
    mean += images.sum(dim=[0, 2, 3])  # Sum per channel
    
    # Sum for variance
    meansq += (images ** 2).sum(dim=[0, 2, 3])
    
    total_pixels += batch_pixels
    
    if batch_idx % 20 == 0:
        print(f"Processed {batch_idx * 64}/{len(dataset)} images...")

# Calculate final statistics
mean /= total_pixels
std = torch.sqrt(meansq / total_pixels - mean ** 2)

print("\n" + "="*60)
print("YOUR NORMALIZED DATASET STATISTICS:")
print(f"Mean (per channel): [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
print(f"Std  (per channel): [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
print("\nRecommended transforms:")
print(f"transforms.Normalize(mean=[{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}],")
print(f"                     std=[{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}])")