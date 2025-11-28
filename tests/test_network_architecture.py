#!/usr/bin/env python3
"""
Test script for TestFlat2Layers network architecture.
Creates the model and generates an architecture visualization.
"""

import sys
from pathlib import Path
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.lc_speckle_analysis.data_config import TrainingDataConfig
from src.lc_speckle_analysis.network_architectures.test_flat_2_layers import TestFlat2Layers, test_model_creation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to test the network architecture."""
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config_path = project_root / "data" / "config.conf"
        config = TrainingDataConfig.from_file(config_path)
        
        # Log key configuration parameters
        logger.info("Configuration loaded:")
        logger.info(f"  Patch size: {config.neural_network.patch_size}")
        logger.info(f"  Layer sizes: {config.neural_network.layer_sizes}")
        logger.info(f"  Dropout rate: {config.neural_network.dropout_rate}")
        logger.info(f"  Activation: {config.neural_network.activation_function}")
        logger.info(f"  Number of classes: {len(config.classes)}")
        logger.info(f"  Classes: {config.classes}")
        
        # Test model creation
        logger.info("\n" + "="*50)
        logger.info("TESTING MODEL CREATION")
        logger.info("="*50)
        
        model = test_model_creation(config)
        
        # Generate architecture plot
        logger.info("\n" + "="*50)
        logger.info("GENERATING ARCHITECTURE PLOT")
        logger.info("="*50)
        
        output_dir = project_root / "data" / "output"
        output_dir.mkdir(exist_ok=True)
        
        plot_path = output_dir / "test_flat_2layers_architecture.png"
        
        fig = model.plot_architecture(
            save_path=plot_path,
            show_plot=False  # Don't show in terminal environment
        )
        
        logger.info(f"Architecture visualization saved to: {plot_path}")
        
        # Print detailed model information
        logger.info("\n" + "="*50)
        logger.info("DETAILED MODEL INFORMATION")
        logger.info("="*50)
        
        summary = model.get_model_summary()
        
        print(f"""
Network Architecture: {summary['architecture']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input Configuration:
  • Patch size: {summary['patch_size']} pixels
  • Channels: {summary['channels']} (VV + VH polarizations)
  • Flattened input: {summary['input_size']} features

Network Layers:
  • Hidden Layer 1: {config.neural_network.layer_sizes[0]} nodes
  • Hidden Layer 2: {config.neural_network.layer_sizes[1]} nodes
  • Output Layer: {summary['output_classes']} classes

Regularization:
  • Activation: {summary['activation']}
  • Dropout rate: {summary['dropout_rate']}

Model Statistics:
  • Total parameters: {summary['total_parameters']:,}
  • Trainable parameters: {summary['trainable_parameters']:,}
  • Model size: {summary['model_size_mb']:.2f} MB

Target Classes: {config.classes}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)
        
        logger.info("✅ Network architecture test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during network testing: {e}")
        raise

if __name__ == "__main__":
    main()
