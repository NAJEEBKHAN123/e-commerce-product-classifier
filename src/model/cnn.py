import torch
import torch.nn as nn
import torchvision.models as models

class ProductClassifier(nn.Module):
    """
    E-commerce classifier using PRETRAINED ResNet50
    """
    
    def __init__(self, num_classes=9, use_pretrained=True):
        super(ProductClassifier, self).__init__()
        
        # LOAD PRETRAINED RESNET50 (NOT from scratch!)
        self.backbone = models.resnet50(pretrained=use_pretrained)
        
        # Freeze early layers (optional, speeds up training)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last few layers
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        
        # Replace classifier head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        print(f"✅ Loaded ResNet50 (pretrained={use_pretrained})")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔓 All layers unfrozen for fine-tuning")


# Optional: Simpler EfficientNet version
try:
    from efficientnet_pytorch import EfficientNet
    
    class EfficientProductClassifier(nn.Module):
        """Using EfficientNet-B0 (smaller, faster, often better)"""
        
        def __init__(self, num_classes=9):
            super(EfficientProductClassifier, self).__init__()
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
            
            # Replace classifier
            num_features = self.backbone._fc.in_features
            self.backbone._fc = nn.Linear(num_features, num_classes)
            
            print(f"✅ Loaded EfficientNet-B0 (pretrained)")
        
        def forward(self, x):
            return self.backbone(x)
            
except ImportError:
    print("⚠️ efficientnet-pytorch not installed. Using ResNet50 only.")


if __name__ == "__main__":
    # Test the model
    model = ProductClassifier(num_classes=9)
    test_input = torch.randn(2, 3, 224, 224)
    output = model(test_input)
    print(f"\nTest input: {test_input.shape}")
    print(f"Test output: {output.shape}")