import torch
from torch.utils.data import DataLoader
from src.data.dataset import EcommerceDataset
from src.data.transforms import train_transform, val_transform
import os

def create_dataloaders(data_dir, batch_size=64, num_workers=None):
    """
    Create dataloaders for train, val, and check splits
    Optimized for Colab GPU.
    """
    # Auto-detect optimal settings
    if num_workers is None:
        if os.name == 'nt':  # Windows
            num_workers = 0
        else:  # Linux / Colab
            cpu_count = os.cpu_count()
            num_workers = min(8, cpu_count // 2)  # Use up to 8 workers for speed
    print(f"📊 Using num_workers={num_workers} for DataLoader")

    # Pin memory if GPU is available
    pin_memory = torch.cuda.is_available()

    # Create datasets
    train_dataset = EcommerceDataset(data_dir, transform=train_transform, split='train')
    val_dataset   = EcommerceDataset(data_dir, transform=val_transform, split='val')
    check_dataset = EcommerceDataset(data_dir, transform=val_transform, split='check')

    # Compute class weights for imbalanced dataset
    class_counts = torch.bincount(torch.tensor(train_dataset.labels))
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    check_loader = DataLoader(
        check_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, check_loader, class_weights, train_dataset.categories
