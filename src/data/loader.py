# Update your loader.py with this optimized version
optimized_loader_code = '''import torch
from torch.utils.data import DataLoader
from src.data.dataset import EcommerceDataset
from src.data.transforms import train_transform, val_transform
import os

def create_dataloaders(data_dir, batch_size=32, num_workers=None):
    """
    Create dataloaders for train, val, and check splits
    OPTIMIZED FOR GOOGLE COLAB
    """
    # COLAB OPTIMIZATION
    if num_workers is None:
        if 'COLAB_GPU' in os.environ:  # Detect Colab
            num_workers = 4  # Optimal for Colab T4
            print(f"🚀 Colab detected: Using {num_workers} workers for optimal performance")
        elif os.name == 'nt':  # Windows
            num_workers = 0
            print("   Windows detected: Using 0 workers")
        else:  # Linux (local)
            num_workers = min(4, os.cpu_count() // 2)
            print(f"   Linux: Using {num_workers} workers")
    
    print(f"📊 Creating dataloaders with num_workers={num_workers}")
    
    train_dataset = EcommerceDataset(
        data_dir=data_dir,
        transform=train_transform,
        split='train'
    )
    
    val_dataset = EcommerceDataset(
        data_dir=data_dir,
        transform=val_transform,
        split='val'
    )
    
    check_dataset = EcommerceDataset(
        data_dir=data_dir,
        transform=val_transform,
        split='check'
    )
    
    # Get class weights for imbalance handling
    class_counts = torch.bincount(torch.tensor(train_dataset.labels))
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    
    # COLAB OPTIMIZATION: Enable pin_memory for faster GPU transfer
    pin_memory = torch.cuda.is_available()
    
    # COLAB OPTIMIZATION: Use persistent workers to avoid restarting workers
    persistent_workers = num_workers > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None  # Prefetch 2 batches per worker
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    check_loader = DataLoader(
        check_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(2, num_workers) if num_workers > 0 else 0,
        pin_memory=False  # No need for test set
    )
    
    return train_loader, val_loader, check_loader, class_weights, train_dataset.categories
'''

# Save the optimized loader
loader_path = "/content/e-commerce-product-classifier/src/data/loader.py"
with open(loader_path, 'w') as f:
    f.write(optimized_loader_code)

print("✅ Updated loader.py with Colab optimizations")
print("   - Detects Colab environment")
print("   - Uses 4 workers for Colab (was 2)")
print("   - Enables persistent workers")
print("   - Adds prefetch for faster loading")