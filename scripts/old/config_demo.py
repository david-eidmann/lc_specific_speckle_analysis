#!/usr/bin/env python3
"""Example script showing configuration usage."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
from lc_speckle_analysis import load_training_config, get_training_config
from lc_speckle_analysis.config import logger

def main():
    """Demonstrate configuration loading and usage."""
    logger.info("Starting configuration demo")
    
    try:
        # Load configuration
        config = load_training_config()
        
        # Display configuration
        logger.info("=== Training Configuration ===")
        logger.info(f"Training data: {config.train_data_path}")
        logger.info(f"Column ID: {config.column_id}")
        logger.info(f"Classes: {config.classes}")
        logger.info(f"Orbits: {config.orbits}")
        logger.info(f"Dates: {config.dates}")
        logger.info(f"File pattern: {config.file_pattern}")
        
        # Validate paths
        logger.info("=== Path Validation ===")
        if config.validate_paths():
            logger.info("All paths are valid")
        else:
            logger.warning("Some paths are invalid - check configuration")
        
        # Get file paths
        logger.info("=== File Discovery ===")
        file_paths = config.get_file_paths()
        logger.info(f"Found {len(file_paths)} satellite data files")
        
        if file_paths:
            logger.info("First few files:")
            for i, path in enumerate(file_paths[:3]):
                logger.info(f"  {i+1}: {path}")
            if len(file_paths) > 3:
                logger.info(f"  ... and {len(file_paths) - 3} more")
        
        logger.info("Configuration demo completed successfully")
        
    except Exception as e:
        logger.error(f"Configuration demo failed: {e}")
        raise

if __name__ == "__main__":
    main()
