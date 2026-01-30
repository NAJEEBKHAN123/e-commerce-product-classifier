# src/data/loader.py - CLEAN VERSION
import torch
from torch.utils.data import DataLoader
from src.data.dataset import EcommerceDataset
from src.data.transforms import train_transform, val_transform
import os

def create_dataloaders(data_dir, batch_size=32, num_workers=None):
    """
    Create dataloaders for train, val, and check splits
    Works on both Windows and Colab
    """
    # Auto-detect optimal settings
    if num_workers is None:
        if os.name == 'nt':  # Windows
            num_workers = 0
            print(f"📊 Windows detected: Using 0 workers")
        else:  # Linux/Colab
            # For Colab, use more workers
            cpu_count = os.cpu_count()
            if cpu_count:
                num_workers = min(4, cpu_count // 2)
            else:
                num_workers = 2
            print(f"📊 Linux/Colab: Using {num_workers} workers")
    
    print(f"Creating dataloaders with num_workers={num_workers}")
    
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
    
    # Pin memory for faster GPU transfer (if available)
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    check_loader = DataLoader(
        check_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, check_loader, class_weights, train_dataset.categories