"""
Enhanced data transformations for better model generalization
"""

import torchvision.transforms as transforms

# ================= TRAINING TRANSFORMS =================
# Much stronger augmentation for better generalization
train_transform = transforms.Compose([
    # Random resized crop with varied scales
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    
    # Geometric transformations
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),  # For some products
    transforms.RandomRotation(degrees=20),  # Increased rotation
    
    # Color transformations
    transforms.ColorJitter(
        brightness=0.3, 
        contrast=0.3, 
        saturation=0.3, 
        hue=0.1
    ),
    
    # Advanced geometric transformations
    transforms.RandomAffine(
        degrees=0, 
        translate=(0.1, 0.1), 
        scale=(0.9, 1.1), 
        shear=5
    ),
    
    # Perspective transformation
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    
    # Convert to tensor
    transforms.ToTensor(),
    
    # Normalize with ImageNet stats
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
    
    # Random erasing (cutout)
    transforms.RandomErasing(
        p=0.2, 
        scale=(0.02, 0.1), 
        ratio=(0.3, 3.3), 
        value='random'
    )
])

# ================= VALIDATION TRANSFORMS =================
# Simpler transforms for validation (no augmentation)
val_transform = transforms.Compose([
    # Resize to 256 first
    transforms.Resize(256),
    
    # Center crop to 224
    transforms.CenterCrop(224),
    
    # Convert to tensor
    transforms.ToTensor(),
    
    # Normalize with ImageNet stats
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# ================= TEST TRANSFORMS =================
# Same as validation
test_transform = val_transform

# ================= SIMPLER TRANSFORMS (Fallback) =================
# If you have memory issues, use these simpler transforms
simple_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

simple_val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])