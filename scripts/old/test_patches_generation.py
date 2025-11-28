#!/usr/bin/env python3
"""Test script to check if patches can be generated."""

import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lc_speckle_analysis.config import logger
from lc_speckle_analysis.data_config import TrainingDataConfig
from lc_speckle_analysis.patch_yielder import PatchYielder, DataMode

def main():
    """Test patch generation."""
    logger.info("Testing patch generation")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "data" / "config.conf"
    config = TrainingDataConfig.from_file(config_path)
    
    logger.info("Initializing PatchYielder")
    
    try:
        # Initialize PatchYielder 
        patch_yielder = PatchYielder(config, seed=42)
        
        logger.info("=== Checking Data Status ===")
        logger.info(f"Total polygons after filtering: {len(patch_yielder.gdf)}")
        logger.info(f"Image tuples available: {len(patch_yielder.image_tuples)}")
        
        # Check train split
        train_data = patch_yielder.data_splits[DataMode.TRAIN]
        logger.info(f"Train polygons: {len(train_data)}")
        
        # Try to get just one patch to test the system
        logger.info("=== Attempting to extract ONE patch ===")
        batch_gen = patch_yielder.yield_batch(DataMode.TRAIN, n_samples_per_polygon=1)
        
        try:
            patches, labels = next(batch_gen)
            logger.info(f"SUCCESS: Generated batch with {len(patches)} patches!")
            logger.info(f"Patch shape: {patches[0].shape}")
            logger.info(f"Labels: {labels}")
            
            # Now check if patches directory was created
            cache_dir = Path(__file__).parent.parent / "data" / "cache" / "patches"
            if cache_dir.exists():
                logger.info(f"Patches cache directory created: {cache_dir}")
                for mode_dir in cache_dir.iterdir():
                    if mode_dir.is_dir():
                        logger.info(f"  Mode: {mode_dir.name}")
                        for hash_dir in mode_dir.iterdir():
                            if hash_dir.is_dir():
                                pkl_files = list(hash_dir.glob("*.pkl"))
                                gpkg_files = list(hash_dir.glob("*.gpkg"))
                                logger.info(f"    Hash {hash_dir.name}: {len(pkl_files)} pkl, {len(gpkg_files)} gpkg files")
            else:
                logger.warning("No patches cache directory found")
                
        except StopIteration:
            logger.warning("No batches generated - checking individual polygon extraction")
            
            # Try to extract from individual polygons to debug
            logger.info("Debugging: Checking polygon-AOI intersection manually")
            
            train_data = patch_yielder.data_splits[DataMode.TRAIN]
            class_col = patch_yielder.config.column_id
            
            for class_id in patch_yielder.config.classes:
                class_polygons = train_data[train_data[class_col] == class_id]
                logger.info(f"Class {class_id}: {len(class_polygons)} polygons")
                
                if len(class_polygons) > 0:
                    # Check intersection with first image AOI
                    first_tuple = patch_yielder.image_tuples[0]
                    aoi = patch_yielder.tuple_aois[first_tuple.date]
                    intersecting = class_polygons[class_polygons.geometry.intersects(aoi.geometry[0])]
                    logger.info(f"  Intersecting with AOI {first_tuple.date}: {len(intersecting)}")
                    
                    if len(intersecting) > 0:
                        logger.info("  Found intersecting polygons - the issue might be in patch extraction")
                        break
        
        logger.info("Patch generation test completed")
        
    except Exception as e:
        logger.error(f"Error during patch test: {e}")
        raise

if __name__ == "__main__":
    main()
