"""
Test Conv2D Network Architecture

A convolutional neural network for land cover classification using Sentinel-1 SAR data 
(VV and VH polarizations). Uses 2D convolutions to preserve spatial structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TestConv2D(nn.Module):
    """
    Convolutional neural network for SAR patch classification.
    
    Architecture:
    - Input: Patches (batch_size × 2 × patch_size × patch_size)
    - Conv Block 1: 2 → 32 channels, 3×3 kernel
    - Conv Block 2: 32 → 64 channels, 3×3 kernel  
    - Conv Block 3: 64 → 128 channels, 3×3 kernel
    - Global Average Pooling
    - Fully Connected: 128 → num_classes
    
    Features:
    - Batch normalization after each conv layer
    - ReLU activation
    - Dropout for regularization
    - Global average pooling to handle variable input sizes
    """
    
    def __init__(self, config):
        """
        Initialize the network.
        
        Args:
            config: Configuration object with neural_network section and classes list
        """
        super(TestConv2D, self).__init__()
        
        self.config = config
        self.patch_size = config.neural_network.patch_size
        self.dropout_rate = config.neural_network.dropout_rate
        self.num_classes = len(config.classes)
        
        # Input channels: 2 (VV and VH polarizations)
        self.input_channels = 2
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=self.input_channels, 
            out_channels=32, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(
            in_channels=64, 
            out_channels=128, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(128)
        
        # Global Average Pooling (adaptive to handle any spatial size)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout
        self.dropout = nn.Dropout(p=self.dropout_rate)
        
        # Final classification layer
        self.classifier = nn.Linear(128, self.num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        # Log network architecture
        self._log_architecture()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _log_architecture(self):
        """Log network architecture details."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info("TestConv2D network initialized:")
        logger.info(f"  Input size: {self.input_channels}×{self.patch_size}×{self.patch_size}")
        logger.info(f"  Conv layers: 2→32→64→128 channels")
        logger.info(f"  Kernel size: 3×3 with padding=1")
        logger.info(f"  Global average pooling: 128×H×W → 128×1×1")
        logger.info(f"  Output classes: {self.num_classes}")
        logger.info(f"  Dropout rate: {self.dropout_rate}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, patch_size, patch_size, 2)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Input shape: (batch_size, patch_size, patch_size, 2)
        # Rearrange to: (batch_size, 2, patch_size, patch_size)
        if x.dim() == 4 and x.shape[-1] == 2:
            x = x.permute(0, 3, 1, 2)
        
        # Conv Block 1: 2 → 32 channels
        x = self.conv1(x)  # (batch_size, 32, patch_size, patch_size)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        # Conv Block 2: 32 → 64 channels
        x = self.conv2(x)  # (batch_size, 64, patch_size, patch_size)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        
        # Conv Block 3: 64 → 128 channels
        x = self.conv3(x)  # (batch_size, 128, patch_size, patch_size)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        
        # Global Average Pooling: 128×H×W → 128×1×1
        x = self.global_avg_pool(x)  # (batch_size, 128, 1, 1)
        
        # Flatten: 128×1×1 → 128
        x = x.view(x.size(0), -1)  # (batch_size, 128)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification
        x = self.classifier(x)  # (batch_size, num_classes)
        
        return x
    
    def get_feature_maps(self, x, layer_name=None):
        """
        Extract intermediate feature maps for visualization.
        
        Args:
            x: Input tensor
            layer_name: Specific layer to extract ('conv1', 'conv2', 'conv3', or None for all)
        
        Returns:
            Dictionary of feature maps or specific layer output
        """
        feature_maps = {}
        
        # Rearrange input if needed
        if x.dim() == 4 and x.shape[-1] == 2:
            x = x.permute(0, 3, 1, 2)
        
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        feature_maps['conv1'] = x.clone()
        if layer_name == 'conv1':
            return x
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        feature_maps['conv2'] = x.clone()
        if layer_name == 'conv2':
            return x
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        feature_maps['conv3'] = x.clone()
        if layer_name == 'conv3':
            return x
        
        return feature_maps if layer_name is None else feature_maps.get(layer_name)
    
    def visualize_filters(self, save_path=None):
        """
        Visualize the learned convolutional filters.
        
        Args:
            save_path: Optional path to save the visualization
        """
        # Get first conv layer weights
        conv1_weights = self.conv1.weight.data.cpu()  # Shape: (32, 2, 3, 3)
        
        # Create visualization
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        fig.suptitle('Conv2D Layer 1 Filters (32 filters, 2 input channels)', fontsize=14)
        
        for i in range(min(32, 32)):  # Show all 32 filters
            row, col = i // 8, i % 8
            
            # Average across input channels for visualization
            filter_vis = conv1_weights[i].mean(dim=0)  # Average VV and VH channels
            
            im = axes[row, col].imshow(filter_vis, cmap='RdBu', vmin=-filter_vis.abs().max(), vmax=filter_vis.abs().max())
            axes[row, col].set_title(f'Filter {i+1}', fontsize=8)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Filter visualization saved to: {save_path}")
        
        return fig
    
    def get_receptive_field_info(self):
        """
        Calculate and return receptive field information.
        
        Returns:
            Dictionary with receptive field details
        """
        # For this simple architecture with 3×3 kernels and no pooling:
        # Conv1: RF = 3×3
        # Conv2: RF = 5×5  
        # Conv3: RF = 7×7
        # Global pooling: RF = entire feature map
        
        info = {
            'conv1_rf': (3, 3),
            'conv2_rf': (5, 5), 
            'conv3_rf': (7, 7),
            'final_rf': f'{self.patch_size}×{self.patch_size} (global)',
            'description': 'Each neuron in conv3 sees a 7×7 region of the original input. Global pooling sees the entire patch.'
        }
        
        return info


def test_network():
    """Test the network with dummy data."""
    # Create a dummy config
    class DummyConfig:
        def __init__(self):
            self.classes = [1, 4, 6, 12]
            self.neural_network = DummyNeuralNetworkConfig()
    
    class DummyNeuralNetworkConfig:
        def __init__(self):
            self.patch_size = 10
            self.dropout_rate = 0.2
    
    config = DummyConfig()
    
    # Create network
    model = TestConv2D(config)
    
    # Create dummy input: (batch_size=4, patch_size=10, patch_size=10, channels=2)
    batch_size = 4
    dummy_input = torch.randn(batch_size, config.neural_network.patch_size, config.neural_network.patch_size, 2)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output logits (first sample): {output[0]}")
    
    # Test feature map extraction
    feature_maps = model.get_feature_maps(dummy_input[:1])  # Use first sample
    for name, fmap in feature_maps.items():
        print(f"{name} feature map shape: {fmap.shape}")
    
    # Get receptive field info
    rf_info = model.get_receptive_field_info()
    print("\nReceptive Field Information:")
    for key, value in rf_info.items():
        print(f"  {key}: {value}")
    
    return model


if __name__ == "__main__":
    # Test the network
    test_network()
