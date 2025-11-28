#!/usr/bin/env python3
"""Demo script for PatchYielder functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lc_speckle_analysis import get_training_config
from lc_speckle_analysis.patch_yielder import PatchYielder, DataMode
from lc_speckle_analysis.config import logger


def main():
    """Demonstrate PatchYielder functionality."""
    logger.info("Starting PatchYielder demo")
    
    try:
        # Load configuration
        config = get_training_config()
        
        # Initialize PatchYielder
        logger.info("Initializing PatchYielder")
        patch_yielder = PatchYielder(config, seed=42)
        
        # Display data split information
        logger.info("=== Data Split Summary ===")
        for mode in DataMode:
            data = patch_yielder.data_splits[mode]
            logger.info(f"{mode.value}: {len(data)} polygons")
        
        # Display image tuple information
        logger.info("=== Available Image Tuples ===")
        for tuple_info in patch_yielder.image_tuples:
            logger.info(f"Date: {tuple_info.date}")
            logger.info(f"  VV: {tuple_info.vv_path.name}")
            logger.info(f"  VH: {tuple_info.vh_path.name}")
            aoi_info = patch_yielder.tuple_aois[tuple_info.date]
            logger.info(f"  AOI area: {aoi_info.geometry[0].area / 1e6:.1f} km²")
        
        # Test batch generation for training mode
        logger.info("=== Testing Batch Generation ===")
        batch_generator = patch_yielder.yield_batch(DataMode.TRAIN)
        
        # Generate and inspect first few batches
        for i, (patches, labels) in enumerate(batch_generator):
            logger.info(f"Batch {i+1}:")
            logger.info(f"  Patches shape: {patches.shape}")
            logger.info(f"  Labels shape: {labels.shape}")
            logger.info(f"  Unique labels: {list(set(labels))}")
            logger.info(f"  Patch value range: [{patches.min():.2f}, {patches.max():.2f}]")
            
            if i >= 2:  # Stop after 3 batches
                break
        
        logger.info("PatchYielder demo completed successfully")
        
    except Exception as e:
        logger.error(f"PatchYielder demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
