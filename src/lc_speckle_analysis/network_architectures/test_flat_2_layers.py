"""
Test Flat 2 Layers Network Architecture

A simple fully connected network with 2 hidden layers for land cover classification
using Sentinel-1 SAR data (VV and VH polarizations).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TestFlat2Layers(nn.Module):
    """
    Simple fully connected network with 2 hidden layers.
    
    Architecture:
    - Input: Flattened patches (patch_size × patch_size × 2 channels) 
    - Hidden Layer 1: Configurable size (default: 64)
    - Hidden Layer 2: Configurable size (default: 128) 
    - Output: Number of classes
    
    Features:
    - Dropout for regularization
    - ReLU activation (configurable)
    - Designed for small patches (10×10 pixels)
    """
    
    def __init__(self, config):
        """
        Initialize the network.
        
        Args:
            config: Configuration object with neural_network section containing:
                - patch_size: Size of input patches (e.g., 10)
                - layer_sizes: List of hidden layer sizes (e.g., [64, 128])
                - dropout_rate: Dropout probability (e.g., 0.2)
                - activation_function: Activation function name (e.g., 'relu')
                - num_classes: Number of output classes
        """
        super(TestFlat2Layers, self).__init__()
        
        self.config = config
        self.patch_size = config.neural_network.patch_size
        self.layer_sizes = config.neural_network.layer_sizes
        self.dropout_rate = config.neural_network.dropout_rate
        self.activation_name = config.neural_network.activation_function
        self.num_classes = len(config.classes)
        
        # Calculate input size: patch_size × patch_size × 2 channels (VV, VH)
        self.input_size = self.patch_size * self.patch_size * 2
        
        # Get activation function
        self.activation_fn = self._get_activation_function(self.activation_name)
        
        # Define layers
        self.flatten = nn.Flatten()
        
        # Input layer to first hidden layer
        self.fc1 = nn.Linear(self.input_size, self.layer_sizes[0])
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
        # First hidden layer to second hidden layer
        self.fc2 = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.dropout2 = nn.Dropout(self.dropout_rate)
        
        # Second hidden layer to output
        self.fc3 = nn.Linear(self.layer_sizes[1], self.num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"TestFlat2Layers network initialized:")
        logger.info(f"  Input size: {self.input_size} (patch {self.patch_size}×{self.patch_size}×2)")
        logger.info(f"  Hidden layers: {self.layer_sizes}")
        logger.info(f"  Output classes: {self.num_classes}")
        logger.info(f"  Dropout rate: {self.dropout_rate}")
        logger.info(f"  Activation: {self.activation_name}")
        
    def _get_activation_function(self, activation_name):
        """Get activation function by name."""
        activations = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'gelu': F.gelu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid
        }
        
        if activation_name.lower() not in activations:
            logger.warning(f"Unknown activation '{activation_name}', using ReLU")
            return F.relu
        
        return activations[activation_name.lower()]
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, patch_size, patch_size, 2)
               or (batch_size, 2, patch_size, patch_size)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Handle both channel formats (H,W,C) and (C,H,W)
        if x.dim() == 4:
            if x.shape[1] == 2:  # (B, C, H, W) format
                x = x.permute(0, 2, 3, 1)  # Convert to (B, H, W, C)
        
        # Flatten input: (batch_size, patch_size, patch_size, 2) -> (batch_size, input_size)
        x = self.flatten(x)
        
        # First hidden layer
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout1(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = self.activation_fn(x)
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x
    
    def get_model_summary(self):
        """Get a detailed summary of the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = {
            'architecture': 'TestFlat2Layers',
            'input_size': self.input_size,
            'patch_size': f"{self.patch_size}×{self.patch_size}",
            'channels': 2,
            'hidden_layers': self.layer_sizes,
            'output_classes': self.num_classes,
            'activation': self.activation_name,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
        
        return summary
    
    def plot_architecture(self, save_path=None, show_plot=True):
        """
        Create a visualization of the network architecture.
        
        Args:
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Define layer positions
        layers = [
            ('Input', self.input_size, 'lightblue'),
            ('Hidden 1', self.layer_sizes[0], 'lightgreen'),
            ('Hidden 2', self.layer_sizes[1], 'lightcoral'),
            ('Output', self.num_classes, 'lightyellow')
        ]
        
        # Draw layers
        layer_width = 1.5
        layer_spacing = 3
        
        for i, (name, size, color) in enumerate(layers):
            x_pos = i * layer_spacing
            
            # Draw layer rectangle
            rect_height = min(6, max(1, size / 50))  # Scale height based on layer size
            rect = plt.Rectangle((x_pos - layer_width/2, -rect_height/2), 
                               layer_width, rect_height, 
                               facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Add layer label
            ax.text(x_pos, rect_height/2 + 0.5, name, 
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # Add size info
            ax.text(x_pos, -rect_height/2 - 0.5, f'{size} nodes', 
                   ha='center', va='top', fontsize=10)
            
            # Draw connections to next layer
            if i < len(layers) - 1:
                next_x = (i + 1) * layer_spacing
                # Draw several connection lines to show fully connected nature
                for j in range(min(5, size)):
                    y_start = (j - 2) * rect_height / 10
                    for k in range(min(5, layers[i+1][1])):
                        y_end = (k - 2) * min(6, max(1, layers[i+1][1] / 50)) / 10
                        ax.plot([x_pos + layer_width/2, next_x - layer_width/2], 
                               [y_start, y_end], 
                               'gray', alpha=0.3, linewidth=0.5)
        
        # Add dropout annotations
        for i in range(1, 3):  # Hidden layers have dropout
            x_pos = i * layer_spacing
            ax.text(x_pos, -3.5, f'Dropout: {self.dropout_rate}', 
                   ha='center', va='center', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Add activation function annotation
        ax.text(layer_spacing * 1.5, 4, f'Activation: {self.activation_name.upper()}', 
               ha='center', va='center', fontsize=10, 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Set plot properties
        ax.set_xlim(-2, (len(layers)-1) * layer_spacing + 2)
        ax.set_ylim(-4, 5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title with model info
        summary = self.get_model_summary()
        title = (f"TestFlat2Layers Architecture\n"
                f"Input: {summary['patch_size']} patch × 2 channels = {summary['input_size']} features\n"
                f"Parameters: {summary['total_parameters']:,} ({summary['model_size_mb']:.2f} MB)")
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Architecture plot saved to: {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_model(config):
    """
    Factory function to create a TestFlat2Layers model.
    
    Args:
        config: Configuration object
        
    Returns:
        TestFlat2Layers model instance
    """
    return TestFlat2Layers(config)


def test_model_creation(config):
    """
    Test function to verify model creation and basic functionality.
    
    Args:
        config: Configuration object
    """
    logger.info("Testing TestFlat2Layers model creation...")
    
    # Create model
    model = create_model(config)
    
    # Print model summary
    summary = model.get_model_summary()
    logger.info("Model Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    # Test forward pass with dummy data
    batch_size = config.neural_network.batch_size
    patch_size = config.neural_network.patch_size
    
    # Create dummy input (batch_size, patch_size, patch_size, 2)
    dummy_input = torch.randn(batch_size, patch_size, patch_size, 2)
    
    logger.info(f"Testing forward pass with input shape: {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    expected_output_shape = (batch_size, len(config.classes))
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Expected shape: {expected_output_shape}")
    
    assert output.shape == expected_output_shape, f"Output shape mismatch: {output.shape} != {expected_output_shape}"
    
    logger.info("✓ Model creation and forward pass test successful!")
    
    return model


if __name__ == "__main__":
    # Basic test if run directly
    from ..data_config import TrainingDataConfig
    
    # Load config
    config = TrainingDataConfig()
    
    # Test model
    model = test_model_creation(config)
    
    # Create architecture plot
    fig=model.plot_architecture(
        save_path="test_flat_2layers_architecture.png",
        show_plot=False
    )
    # fig.savefig("test_flat_2layers_architecture.png", dpi=300)
