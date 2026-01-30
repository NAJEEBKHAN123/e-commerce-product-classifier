# src/data/loader.py - UPDATED FOR WINDOWS
import torch
from torch.utils.data import DataLoader
from src.data.dataset import EcommerceDataset
from src.data.transforms import train_transform, val_transform
import os

def create_dataloaders(data_dir, batch_size=32, num_workers=None):
    """
    Create dataloaders for train, val, and check splits
    Windows compatibility: num_workers must be 0 on Windows
    """
    # WINDOWS FIX: num_workers must be 0 on Windows
    if num_workers is None:
        # Auto-detect: 0 for Windows, 2 for Linux/Colab
        num_workers = 0 if os.name == 'nt' else 2
    
    print(f"📊 Creating dataloaders with num_workers={num_workers} "
          f"{'(Windows mode)' if os.name == 'nt' else '(Linux/Colab mode)'}")
    
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
    
    # Pin memory only if CUDA is available (not on Windows CPU)
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False if num_workers > 0 else False  # Disable for Windows
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False if num_workers > 0 else False  # Disable for Windows
    )
    
    check_loader = DataLoader(
        check_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False if num_workers > 0 else False  # Disable for Windows
    )
    
    return train_loader, val_loader, check_loader, class_weights, train_dataset.categories