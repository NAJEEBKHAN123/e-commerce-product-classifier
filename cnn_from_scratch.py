"""
E-COMMERCE PRODUCT CLASSIFIER - CNN FROM SCRATCH
Perfect for Class Project - Shows Understanding of Fundamentals
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ================= CONFIGURATION =================
DATA_DIR = "dataset_normalized"
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
IMG_SIZE = 128  # Smaller for faster training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*70)
print("🎓 E-COMMERCE CLASSIFIER - CNN FROM SCRATCH (CLASS PROJECT)")
print("="*70)
print(f"📱 Device: {DEVICE}")
print(f"📊 Epochs: {EPOCHS}")
print(f"📦 Batch Size: {BATCH_SIZE}")
print(f"🎯 Learning Rate: {LEARNING_RATE}")
print(f"🖼️ Image Size: {IMG_SIZE}x{IMG_SIZE}")
print("="*70)

# ================= SIMPLE TRANSFORMS =================
# Basic augmentations for from-scratch training
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Simple normalization
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ================= CUSTOM DATASET =================
class EcommerceDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        
        print(f"\n📂 Loading {split.upper()} dataset...")
        
        # Get all class folders
        class_folders = sorted(os.listdir(self.data_dir))
        
        for class_idx, class_name in enumerate(class_folders):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                self.class_names.append(class_name)
                
                # Get all images in this class
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                
                for img_file in image_files:
                    self.images.append(os.path.join(class_path, img_file))
                    self.labels.append(class_idx)
        
        print(f"✅ Loaded {len(self.images)} images")
        print(f"📊 Classes: {self.class_names}")
        
        # Print class distribution
        class_counts = np.bincount(self.labels)
        print(f"📈 Class distribution:")
        for i, count in enumerate(class_counts):
            print(f"  {self.class_names[i]:30s}: {count:4d} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

# ================= SIMPLE CNN ARCHITECTURE =================
class SimpleCNN(nn.Module):
    """
    Simple CNN Architecture from Scratch
    Perfect for class project - shows understanding of layers
    """
    def __init__(self, num_classes=9):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 128x128x3 -> 128x128x32
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64x64x32 -> 64x64x64
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 32x32x64 -> 32x32x128
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 16x16x128 -> 16x16x256
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # Halves the spatial dimensions
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, 512)  # After 4 pooling layers: 128 -> 64 -> 32 -> 16 -> 8
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        print("\n🤖 CNN ARCHITECTURE (From Scratch):")
        print("  Input: 3 x 128 x 128")
        print("  Conv1: 32 filters, 3x3, padding=1")
        print("  Pool: 2x2 maxpool")
        print("  Conv2: 64 filters, 3x3, padding=1")
        print("  Pool: 2x2 maxpool")
        print("  Conv3: 128 filters, 3x3, padding=1")
        print("  Pool: 2x2 maxpool")
        print("  Conv4: 256 filters, 3x3, padding=1")
        print("  Pool: 2x2 maxpool")
        print("  FC1: 256*8*8 -> 512")
        print("  FC2: 512 -> 256")
        print("  FC3: 256 -> 9 (output)")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Conv block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, 256 * 8 * 8)
        
        # Fully connected with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

# ================= TRAINING FUNCTION =================
def train_model():
    """Main training loop"""
    
    # Load datasets
    print("\n" + "="*50)
    print("📊 LOADING DATASETS")
    print("="*50)
    
    train_dataset = EcommerceDataset(DATA_DIR, 'train', train_transform)
    val_dataset = EcommerceDataset(DATA_DIR, 'val', val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    print("\n" + "="*50)
    print("🏗️  BUILDING MODEL")
    print("="*50)
    
    model = SimpleCNN(num_classes=len(train_dataset.class_names)).to(DEVICE)
    
    # Loss function with class weights (for imbalance)
    class_counts = np.bincount(train_dataset.labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    
    print("\n" + "="*50)
    print("🚀 STARTING TRAINING")
    print("="*50)
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*40}")
        print(f"📅 EPOCH {epoch+1}/{EPOCHS}")
        print(f"{'='*40}")
        
        # ----- TRAINING PHASE -----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Progress bar
            if (batch_idx + 1) % 20 == 0:
                batch_acc = 100.0 * (predicted == labels).sum().item() / labels.size(0)
                print(f"  Batch {batch_idx+1:3d}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | Acc: {batch_acc:5.1f}%")
        
        train_accuracy = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # ----- VALIDATION PHASE -----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_accuracy = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(val_accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['learning_rate'].append(current_lr)
        
        # Print epoch results
        print(f"\n📊 EPOCH {epoch+1} SUMMARY:")
        print(f"  Training:   Loss = {avg_train_loss:.4f}, Accuracy = {train_accuracy:.1f}%")
        print(f"  Validation: Loss = {avg_val_loss:.4f}, Accuracy = {val_accuracy:.1f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'class_names': train_dataset.class_names,
                'architecture': 'SimpleCNN_Scratch'
            }, 'cnn_from_scratch_best.pth')
            print(f"💾 Saved best model (Accuracy: {val_accuracy:.1f}%)")
        
        # Early stopping if accuracy is good enough
        if val_accuracy > 80.0:
            print(f"\n🎉 EXCELLENT! Model reached {val_accuracy:.1f}% accuracy!")
            break
    
    # ================= FINAL RESULTS =================
    print("\n" + "="*70)
    print("🏆 TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"Final Training Accuracy: {history['train_acc'][-1]:.1f}%")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.1f}%")
    
    # Plot training history
    plot_training_history(history, train_dataset.class_names)
    
    # Generate confusion matrix
    generate_confusion_matrix(model, val_loader, train_dataset.class_names)
    
    return model, train_dataset.class_names

# ================= VISUALIZATION FUNCTIONS =================
def plot_training_history(history, class_names):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1, 0].plot(history['learning_rate'], marker='o', color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Class distribution
    class_counts = np.bincount([label for _, label in EcommerceDataset(DATA_DIR, 'train')])
    axes[1, 1].bar(range(len(class_names)), class_counts, color='skyblue')
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('Number of Images')
    axes[1, 1].set_title('Class Distribution in Training Set')
    axes[1, 1].set_xticks(range(len(class_names)))
    axes[1, 1].set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                                for name in class_names], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('training_history_from_scratch.png', dpi=150, bbox_inches='tight')
    print("✅ Saved training history: training_history_from_scratch.png")
    plt.show()

def generate_confusion_matrix(model, val_loader, class_names):
    """Generate and plot confusion matrix"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[name[:10] + '...' if len(name) > 10 else name 
                            for name in class_names],
                yticklabels=[name[:10] + '...' if len(name) > 10 else name 
                            for name in class_names])
    plt.title('Confusion Matrix - CNN from Scratch')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix_from_scratch.png', dpi=150, bbox_inches='tight')
    print("✅ Saved confusion matrix: confusion_matrix_from_scratch.png")
    
    # Classification report
    print("\n" + "="*70)
    print("📋 CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(all_labels, all_preds, target_names=class_names))

# ================= PREDICTION FUNCTION =================
def predict_image(model, image_path, class_names, transform=None):
    """Predict a single image"""
    if transform is None:
        transform = val_transform
    
    model.eval()
    img = Image.open(image_path).convert('RGB')
    
    if transform:
        img = transform(img)
    
    img = img.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence = probabilities[0][predicted.item()].item() * 100
    
    # Get top 3 predictions
    top3_probs, top3_indices = torch.topk(probabilities[0], 3)
    
    print(f"\n🔍 PREDICTION RESULTS:")
    print(f"   Image: {os.path.basename(image_path)}")
    print(f"   Predicted: {predicted_class} ({confidence:.1f}% confidence)")
    print(f"\n📊 Top 3 Predictions:")
    for i in range(3):
        idx = top3_indices[i].item()
        prob = top3_probs[i].item() * 100
        print(f"   {i+1}. {class_names[idx]:30s} ({prob:.1f}%)")
    
    return predicted_class, confidence

# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎓 CLASS PROJECT: CNN FROM SCRATCH FOR E-COMMERCE CLASSIFICATION")
    print("="*70)
    
    try:
        # Train the model
        trained_model, class_names = train_model()
        
        # Test prediction
        print("\n" + "="*70)
        print("🧪 TESTING THE MODEL")
        print("="*70)
        
        # Try to find a test image
        test_image = None
        search_paths = [
            "dataset_normalized/val/BABY_PRODUCTS",
            "dataset_normalized/val/ELECTRONICS",
            "dataset_normalized/val/CLOTHING_ACCESSORIES_JEWELLERY"
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                images = [f for f in os.listdir(path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if images:
                    test_image = os.path.join(path, images[0])
                    break
        
        if test_image:
            predict_image(trained_model, test_image, class_names)
        else:
            print("ℹ️  No test image found. To test, run:")
            print("   predict_image(model, 'path/to/image.jpg', class_names)")
        
        print("\n" + "="*70)
        print("✨ PROJECT COMPLETED SUCCESSFULLY! ✨")
        print("="*70)
        print("📁 Output Files:")
        print("   - cnn_from_scratch_best.pth (Best model)")
        print("   - training_history_from_scratch.png (Training plots)")
        print("   - confusion_matrix_from_scratch.png (Confusion matrix)")
        print("\n🎓 Perfect for class project submission!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()