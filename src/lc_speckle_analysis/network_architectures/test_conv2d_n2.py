#!/usr/bin/env python3
"""
TestConv2D_N2 - A lightweight CNN architecture with approximately 1,400 parameters.
This is a smaller version of TestConv2D designed for faster training and lower memory usage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class TestConv2D_N2(nn.Module):
    """
    Lightweight 2D CNN for patch classification with ~1,400 parameters.
    
    Architecture:
    - Input: 2 channels (VV, VH) × 10×10 patches
    - Conv1: 2→8 channels, 3×3 kernel
    - Conv2: 8→16 channels, 3×3 kernel  
    - Global Average Pooling
    - FC: 16→num_classes
    
    Total parameters: ~1,400
    """
    
    def __init__(self, patch_size: int = 10, num_classes: int = 4, dropout_rate: float = 0.2):
        """
        Initialize TestConv2D_N2 network.
        
        Args:
            patch_size: Size of input patches (not used in calculation, for compatibility)
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super(TestConv2D_N2, self).__init__()
        
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, padding=1)   # 2*3*3*8 + 8 = 152 params
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 8*3*3*16 + 16 = 1,168 params
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(8)   # 8*2 = 16 params
        self.bn2 = nn.BatchNorm2d(16)  # 16*2 = 32 params
        
        # Global average pooling (no parameters)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layer
        self.fc = nn.Linear(16, num_classes)  # 16*num_classes + num_classes params
        
        # Calculate total parameters
        total_params = 152 + 1168 + 16 + 32 + (16 * num_classes + num_classes)
        
        logger.info(f"TestConv2D_N2 network initialized:")
        logger.info(f"  Input size: 2×10×10")
        logger.info(f"  Conv layers: 2→8→16 channels")
        logger.info(f"  Kernel size: 3×3 with padding=1")
        logger.info(f"  Global average pooling: 16×H×W → 16×1×1")
        logger.info(f"  Output classes: {num_classes}")
        logger.info(f"  Dropout rate: {dropout_rate}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, height, width, channels) or (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Handle input format - convert (B,H,W,C) to (B,C,H,W) if needed
        if x.dim() == 4 and x.shape[1] != 2:  # If not in (B,C,H,W) format
            if x.shape[-1] == 2:  # (B,H,W,C) format
                x = x.permute(0, 3, 1, 2)  # Convert to (B,C,H,W)
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}. Expected (B,C,H,W) or (B,H,W,C)")
        
        # First convolutional block
        x = self.conv1(x)  # (batch, 8, 10, 10)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second convolutional block
        x = self.conv2(x)  # (batch, 16, 10, 10)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)  # (batch, 16, 1, 1)
        x = x.view(x.size(0), -1)    # (batch, 16)
        
        # Dropout and final classification
        x = self.dropout(x)
        x = self.fc(x)  # (batch, num_classes)
        
        return x
    
    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of feature maps
        """
        # Handle input format
        if x.dim() == 4 and x.shape[1] != 2:
            if x.shape[-1] == 2:
                x = x.permute(0, 3, 1, 2)
        
        features = {}
        
        # Conv1 features
        x = F.relu(self.bn1(self.conv1(x)))
        features['conv1'] = x
        
        # Conv2 features
        x = F.relu(self.bn2(self.conv2(x)))
        features['conv2'] = x
        
        return features


def create_test_conv2d_n2(patch_size: int = 10, num_classes: int = 4, **kwargs) -> TestConv2D_N2:
    """
    Factory function to create TestConv2D_N2 model.
    
    Args:
        patch_size: Size of input patches
        num_classes: Number of output classes
        **kwargs: Additional arguments (dropout_rate, etc.)
        
    Returns:
        TestConv2D_N2 model instance
    """
    return TestConv2D_N2(patch_size=patch_size, num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Test the network
    print("Testing TestConv2D_N2 architecture...")
    
    # Create model
    model = TestConv2D_N2(num_classes=4, dropout_rate=0.2)
    
    # Test with sample input (batch_size=2, height=10, width=10, channels=2)
    sample_input = torch.randn(2, 10, 10, 2)
    print(f"Input shape: {sample_input.shape}")
    
    # Forward pass
    output = model(sample_input)
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test feature maps
    features = model.get_feature_maps(sample_input)
    print("Feature map shapes:")
    for name, feature_map in features.items():
        print(f"  {name}: {feature_map.shape}")
    
    print("✅ TestConv2D_N2 test completed successfully!")