#!/usr/bin/env python3
"""Quick diagnostic for satellite image CRS and bounds."""

import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import rasterio
from lc_speckle_analysis.config import logger
from lc_speckle_analysis.data_config import TrainingDataConfig

def main():
    """Check satellite image metadata."""
    logger.info("=== Satellite Image Diagnostic ===")
    
    config_path = Path(__file__).parent.parent / "data" / "config.conf"
    config = TrainingDataConfig.from_file(config_path)
    
    # Get sample satellite file
    file_paths = config.get_file_paths()
    if file_paths:
        sample_file = file_paths[0]
        logger.info(f"Checking sample file: {sample_file.name}")
        
        with rasterio.open(sample_file) as src:
            logger.info(f"  CRS: {src.crs}")
            logger.info(f"  Bounds: {src.bounds}")
            logger.info(f"  Shape: {src.shape}")
            logger.info(f"  Transform: {src.transform}")
            logger.info(f"  Resolution: {src.res}")
    else:
        logger.warning("No satellite files found")

if __name__ == "__main__":
    main()
