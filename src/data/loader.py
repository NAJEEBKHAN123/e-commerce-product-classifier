# src/data/loader.py
import torch
from torch.utils.data import DataLoader
from src.data.dataset import EcommerceDataset
from src.data.transforms import train_transform, val_transform

def create_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Create dataloaders for train, val, and check splits
    """
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    check_loader = DataLoader(
        check_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, check_loader, class_weights, train_dataset.categories