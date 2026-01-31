"""
TEST SCRIPT - Perfect for Class Presentation
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

def visualize_model_predictions():
    """Visualize model predictions on sample images"""
    
    # Load model
    checkpoint = torch.load('cnn_from_scratch_best.pth', map_location='cpu')
    
    # Recreate model architecture
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=9):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(256 * 8 * 8, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, num_classes)
        
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            x = self.pool(torch.relu(self.conv4(x)))
            x = x.view(-1, 256 * 8 * 8)
            x = self.dropout(torch.relu(self.fc1(x)))
            x = self.dropout(torch.relu(self.fc2(x)))
            x = self.fc3(x)
            return x
    
    model = SimpleCNN(num_classes=len(checkpoint['class_names']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Test on sample images
    print("🧪 DEMONSTRATING MODEL PREDICTIONS")
    print("-" * 50)
    
    # Get sample images from each class
    base_path = "dataset_normalized/val"
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, class_name in enumerate(checkpoint['class_names'][:9]):  # Show first 9 classes
        class_path = os.path.join(base_path, class_name)
        
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if images:
                img_path = os.path.join(class_path, images[0])
                img = Image.open(img_path).convert('RGB')
                
                # Display original image
                axes[idx].imshow(img)
                
                # Make prediction
                img_tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    _, predicted = torch.max(output, 1)
                
                predicted_class = checkpoint['class_names'][predicted.item()]
                confidence = probabilities[0][predicted.item()].item() * 100
                
                # Set title with result
                title_color = 'green' if predicted_class == class_name else 'red'
                axes[idx].set_title(f"True: {class_name[:15]}\nPred: {predicted_class[:15]}\nConf: {confidence:.1f}%", 
                                   color=title_color, fontsize=10)
                axes[idx].axis('off')
                
                print(f"✓ {class_name:30s} → {predicted_class:30s} ({confidence:5.1f}%)")
    
    plt.suptitle("CNN From Scratch - Predictions on Validation Set", fontsize=16)
    plt.tight_layout()
    plt.savefig('model_predictions_demo.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved prediction visualization: model_predictions_demo.png")
    plt.show()

if __name__ == "__main__":
    visualize_model_predictions()