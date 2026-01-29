#!/usr/bin/env python3
"""
Training script for E-commerce Product Classifier
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import wandb
import json
import os
from datetime import datetime
import argparse

def setup_data(data_dir="dataset", batch_size=32):
    """Setup data loaders"""
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=val_transform
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader, train_dataset.classes, train_dataset.class_to_idx

def create_model(num_classes, device):
    """Create MobileNetV2 model"""
    
    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(pretrained=True)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last 15 layers for fine-tuning
    for param in model.features[-15:].parameters():
        param.requires_grad = True
    
    # Replace classifier for 9 classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    model = model.to(device)
    return model

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), 100 * correct / total

def main():
    parser = argparse.ArgumentParser(description='Train E-commerce Classifier')
    parser.add_argument('--data_dir', default='dataset', help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_dir', default='models', help='Model save directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use Wandb logging')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Setup data
    print("Loading data...")
    train_loader, val_loader, classes, class_to_idx = setup_data(
        args.data_dir, args.batch_size
    )
    
    print(f"\nDataset Info:")
    print(f"  Classes: {len(classes)}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Class names: {classes}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(len(classes), device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Initialize Wandb
    if args.use_wandb:
        wandb.init(project="ecommerce-classifier", config=vars(args))
        wandb.config.update({
            "num_classes": len(classes),
            "classes": classes
        })
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0]
            })
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Save full model state
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': classes,
                'class_to_idx': class_to_idx,
                'config': vars(args)
            }, os.path.join(args.model_dir, 'best_model.pth'))
            
            print(f"✓ Saved best model with val_acc: {val_acc:.2f}%")
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # Save metadata
    metadata = {
        'classes': classes,
        'class_to_idx': class_to_idx,
        'num_classes': len(classes),
        'training_date': datetime.now().isoformat(),
        'best_val_acc': best_val_acc,
        'config': vars(args)
    }
    
    with open(os.path.join(args.model_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if args.use_wandb:
        wandb.finish()
    
    print(f"\n{'='*50}")
    print("TRAINING COMPLETED!")
    print(f"{'='*50}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {final_model_path}")
    print(f"Metadata saved to: {args.model_dir}/model_metadata.json")

if __name__ == "__main__":
    main()