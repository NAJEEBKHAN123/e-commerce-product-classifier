# /content/e-commerce-product-classifier/src/model/cnn.py
"""
E-commerce Product Classifier CNN Model
Optimized for 9 product categories with proper architecture
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ProductCNN(nn.Module):
    """
    CNN model for e-commerce product classification.
    Features:
    - Uses pretrained ResNet50 for better performance
    - Custom classifier head for 9 product categories
    - Proper weight initialization
    """
    
    def __init__(self, num_classes=9, use_pretrained=True):
        """
        Initialize the CNN model.
        
        Args:
            num_classes (int): Number of output classes (default: 9)
            use_pretrained (bool): Use pretrained ImageNet weights
        """
        super(ProductCNN, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=use_pretrained)
        
        # Freeze early layers if needed (optional)
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
        
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        
        # Custom classifier head
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Initialize new layers properly
        self._initialize_weights(self.resnet.fc)
    
    def _initialize_weights(self, module):
        """Initialize weights for new layers."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Raw logits of shape (batch_size, num_classes)
        """
        return self.resnet(x)
    
    def get_parameter_count(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self):
        """Freeze ResNet backbone layers."""
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Ensure classifier is trainable
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze all layers."""
        for param in self.parameters():
            param.requires_grad = True


# Alternative simpler model
class SimpleProductCNN(nn.Module):
    """Simpler CNN for product classification."""
    
    def __init__(self, num_classes=9):
        super(SimpleProductCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test function
if __name__ == "__main__":
    # Test ProductCNN
    model = ProductCNN(num_classes=9)
    print(f"✅ ProductCNN created")
    print(f"📊 Total parameters: {model.get_parameter_count():,}")
    
    # Test forward pass
    test_input = torch.randn(2, 3, 224, 224)
    output = model(test_input)
    print(f"📐 Input shape: {test_input.shape}")
    print(f"📐 Output shape: {output.shape}")
    
    # Test SimpleProductCNN
    simple_model = SimpleProductCNN(num_classes=9)
    print(f"\n✅ SimpleProductCNN created")
    print(f"📊 Total parameters: {simple_model.get_parameter_count():,}")
    
    print("\n🎯 All models working correctly!")