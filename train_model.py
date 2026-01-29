import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import os
import json

print("=" * 60)
print("TRAINING E-COMMERCE PRODUCT CLASSIFIER")
print("=" * 60)

# Check current directory
print(f"Current directory: {os.getcwd()}")

# Check if dataset exists
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    print(f"❌ ERROR: '{dataset_path}' folder not found!")
    print("Please make sure you have:")
    print("  dataset/train/  - training images")
    print("  dataset/val/    - validation images")
    print("  dataset/check/  - test images")
    exit()

print(f"✅ Found dataset folder: {dataset_path}")

# Check train folder
train_path = os.path.join(dataset_path, "train")
if not os.path.exists(train_path):
    print(f"❌ ERROR: '{train_path}' not found!")
    exit()

# List categories
categories = os.listdir(train_path)
print(f"✅ Found {len(categories)} categories:")
for cat in categories:
    print(f"  - {cat}")

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
print("\nLoading datasets...")
train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
val_dataset = datasets.ImageFolder(
    root=os.path.join(dataset_path, "val"),
    transform=val_transform
)

print(f"✅ Training samples: {len(train_dataset)}")
print(f"✅ Validation samples: {len(val_dataset)}")
print(f"✅ Classes: {train_dataset.classes}")

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✅ Using device: {device}")

model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_dataset.classes))
model = model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
print("\n" + "=" * 60)
print("STARTING TRAINING (5 epochs)")
print("=" * 60)

for epoch in range(5):
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    
    print(f"Epoch {epoch+1}/5:")
    print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

# Save model
os.makedirs("models", exist_ok=True)
model_path = "models/best_model.pth"
torch.save(model.state_dict(), model_path)

# Save metadata
metadata = {
    "classes": train_dataset.classes,
    "class_to_idx": train_dataset.class_to_idx,
    "num_classes": len(train_dataset.classes)
}

metadata_path = "models/model_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"Model saved to: {model_path}")
print(f"Metadata saved to: {metadata_path}")
print(f"Classes ({len(metadata['classes'])}): {metadata['classes']}")
print("\nNext step: Run the API with: python main.py")