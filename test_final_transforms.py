# test_final_transforms_fixed.py
import torch
from src.data.transforms import train_transform
from src.data.loader import create_dataloaders
import os

os.environ['DATASET_PATH'] = 'dataset_normalized'

print("🧪 TESTING FINAL TRANSFORMS")
print("="*60)

# Load data with new transforms - ADD data_dir parameter!
train_loader, _, _, _, categories = create_dataloaders(
    data_dir='dataset_normalized',  # ← ADD THIS
    batch_size=4
)

# Check first batch
for images, labels, _ in train_loader:
    print(f"Batch shape: {images.shape}")
    
    # Check normalization
    mean = images.mean(dim=[0, 2, 3])
    std = images.std(dim=[0, 2, 3])
    
    print(f"\nACTUAL (after your normalization):")
    print(f"  Mean: [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"  Std:  [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
    
    print(f"\nTARGET (should be close to):")
    print(f"  Mean: [0.000, 0.000, 0.000]")
    print(f"  Std:  [1.000, 1.000, 1.000]")
    
    # Calculate errors
    mean_error = torch.abs(mean).mean().item()
    std_error = torch.abs(std - 1.0).mean().item()
    
    print(f"\nNORMALIZATION ACCURACY:")
    print(f"  Mean error: {mean_error:.6f} (should be < 0.1)")
    print(f"  Std error:  {std_error:.6f} (should be < 0.2)")
    
    if mean_error < 0.1 and std_error < 0.2:
        print("\n✅ PERFECT NORMALIZATION! Ready for training.")
    else:
        print("\n⚠️  Normalization needs adjustment.")
        print(f"   Try adjusting mean/std values in transforms.py")
    
    # Test model
    from src.model.cnn import ProductClassifier
    model = ProductClassifier(num_classes=len(categories), use_pretrained=True)
    model.eval()
    
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        accuracy = (preds == labels).float().mean().item()
        
        print(f"\n🤖 MODEL TEST ON BATCH:")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Predictions: {preds.tolist()}")
        print(f"  True labels: {labels.tolist()}")
        
        if accuracy > 0.4:  # >40%
            print(f"  ✅ EXCELLENT! Should get 85-90%+ final accuracy.")
        elif accuracy > 0.2:  # >20%
            print(f"  ✅ GOOD! Should get 70-80%+ final accuracy.")
        else:
            print(f"  ⚠️  Low but should improve with training.")
    
    break

print("\n" + "="*60)
print("🚀 READY FOR TRAINING!")