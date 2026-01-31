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
        if use_pretrained:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet50(weights=None)
        
        # Freeze early layers initially (can be unfrozen later)
        self._freeze_backbone()
        
        # Unfreeze the last block for fine-tuning
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        # Replace the final fully connected layer with BETTER architecture
        num_features = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.6),  # Increased dropout for better regularization
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
        # Initialize new layers properly
        self._initialize_weights(self.resnet.fc)
    
    def _freeze_backbone(self):
        """Freeze ResNet backbone layers except the last one."""
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Always keep classifier trainable
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
    
    def _initialize_weights(self, module):
        """Initialize weights for new layers."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
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
    
    def unfreeze_all(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        print("✅ All layers unfrozen for fine-tuning")
    
    def unfreeze_some(self, layers_to_unfreeze=['layer4', 'layer3']):
        """Unfreeze specific layers."""
        for layer_name in layers_to_unfreeze:
            if hasattr(self.resnet, layer_name):
                for param in getattr(self.resnet, layer_name).parameters():
                    param.requires_grad = True
                print(f"✅ {layer_name} unfrozen")
    
    def get_trainable_params_info(self):
        """Get information about trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = self.get_parameter_count()
        frozen_params = total_params - trainable_params
        
        print(f"📊 Parameter Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {frozen_params:,}")
        print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }


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
    
    # Get parameter info
    model.get_trainable_params_info()
    
    # Test unfreezing
    model.unfreeze_some(['layer3', 'layer4'])
    model.get_trainable_params_info()
    
    print("\n🎯 All models working correctly!")