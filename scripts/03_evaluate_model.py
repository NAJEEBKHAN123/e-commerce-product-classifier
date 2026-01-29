#!/usr/bin/env python3
"""
Evaluate trained model on test set
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse

def evaluate_model(model_path, metadata_path, data_dir="dataset"):
    """Evaluate model on test set"""
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = models.mobilenet_v2()
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, metadata['num_classes'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load test data (from check folder)
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "check"),
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2
    )
    
    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test samples: {len(test_dataset)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Classes: {metadata['classes']}")
    
    # Classification report
    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(
        all_labels, all_preds, 
        target_names=metadata['classes'],
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=metadata['classes'],
                yticklabels=metadata['classes'])
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Per-class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_acc)), class_acc)
    plt.xticks(range(len(class_acc)), metadata['classes'], rotation=45, ha='right')
    plt.title('Per-Class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.1])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, acc in zip(bars, class_acc):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('per_class_accuracy.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save results
    results = {
        'test_accuracy': float(accuracy),
        'per_class_accuracy': {metadata['classes'][i]: float(acc) 
                              for i, acc in enumerate(class_acc)},
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'labels': all_labels
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: evaluation_results.json")
    print(f"Visualizations saved: confusion_matrix.png, per_class_accuracy.png")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', default='models/best_model.pth', 
                       help='Model checkpoint path')
    parser.add_argument('--metadata', default='models/model_metadata.json',
                       help='Model metadata path')
    parser.add_argument('--data_dir', default='dataset',
                       help='Dataset directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        return
    
    if not os.path.exists(args.metadata):
        print(f"Error: Metadata file '{args.metadata}' not found!")
        return
    
    evaluate_model(args.model, args.metadata, args.data_dir)

if __name__ == "__main__":
    main()