import torch
from torch.utils.data import DataLoader
from src.data.dataset import EcommerceDataset
from src.data.transforms import train_transform, val_transform
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def create_dataloaders(data_dir, batch_size=32, num_workers=None):
    """
    Create dataloaders with proper class balancing
    """
    # Auto-detect workers
    if num_workers is None:
        num_workers = 0 if os.name == 'nt' else min(4, os.cpu_count() // 2)
    
    print(f"📊 Creating dataloaders with batch_size={batch_size}, workers={num_workers}")
    
    # Create datasets
    train_dataset = EcommerceDataset(data_dir, transform=train_transform, split='train')
    val_dataset = EcommerceDataset(data_dir, transform=val_transform, split='val')
    check_dataset = EcommerceDataset(data_dir, transform=val_transform, split='check')
    
    print(f"✅ Train: {len(train_dataset)} images")
    print(f"✅ Val: {len(val_dataset)} images")
    print(f"✅ Test: {len(check_dataset)} images")
    print(f"✅ Categories: {train_dataset.categories}")
    
    # PROPER class weights for imbalance
    train_labels = np.array(train_dataset.labels)
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(train_labels), 
        y=train_labels
    )
    class_weights = torch.FloatTensor(class_weights)
    
    # Print class distribution
    class_counts = torch.bincount(torch.tensor(train_labels))
    print(f"📈 Class distribution: {class_counts.tolist()}")
    print(f"⚖️  Class weights: {class_weights.tolist()}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    check_loader = DataLoader(
        check_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, check_loader, class_weights, train_dataset.categories