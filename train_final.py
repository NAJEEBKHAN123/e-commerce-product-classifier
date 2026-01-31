# simple_classifier.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np

# 1. Simple transforms (NO fancy normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 2. Simple dataset
class SimpleDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        
        # Load all images
        for class_idx, class_name in enumerate(sorted(os.listdir(self.data_dir))):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                self.class_names.append(class_name)
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_path, img_name))
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# 3. SIMPLE MODEL (MobileNet - smaller, faster, works better)
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(1280, num_classes)
        
        # Freeze all layers except last
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.model(x)

# 4. TRAINING FUNCTION
def train_simple_model():
    # Data
    train_dataset = SimpleDataset('dataset_normalized', 'train', transform)
    val_dataset = SimpleDataset('dataset_normalized', 'val', transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleClassifier(num_classes=9).to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    print("🚀 Training MobileNetV2 (Simple & Fast)...")
    for epoch in range(5):  # Just 5 epochs
        model.train()
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accuracy
            preds = outputs.argmax(dim=1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
            
            if batch_idx % 50 == 0:
                batch_acc = 100 * correct / labels.size(0)
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss={loss.item():.3f}, Acc={batch_acc:.1f}%")
        
        # Epoch accuracy
        epoch_acc = 100 * total_correct / total_samples
        print(f"✅ Epoch {epoch+1} Complete: Train Acc = {epoch_acc:.1f}%")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        print(f"📊 Validation Acc = {val_acc:.1f}%\n")
        
        if val_acc > 70:  # Good enough!
            print(f"🎉 SUCCESS! Model reached {val_acc:.1f}% accuracy!")
            torch.save(model.state_dict(), 'simple_model.pth')
            break
    
    return model

if __name__ == "__main__":
    train_simple_model()