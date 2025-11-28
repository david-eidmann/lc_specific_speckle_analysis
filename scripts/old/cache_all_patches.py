#!/usr/bin/env python3
"""Cache patches for all data modes (train, validation, test)."""

import sys
from pathlib import Path
import logging
import time
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lc_speckle_analysis.data_config import TrainingDataConfig
from lc_speckle_analysis.patch_yielder import PatchYielder, DataMode
from lc_speckle_analysis.config import logger


def cache_patches_for_mode(yielder: PatchYielder, data_mode: DataMode, 
                          n_samples_per_polygon: int = 10, 
                          max_batches: int = None) -> Dict[str, int]:
    """Cache patches for a specific data mode.
    
    Args:
        yielder: PatchYielder instance
        data_mode: DataMode enum (TRAIN/VALIDATION/TEST)
        n_samples_per_polygon: Number of samples per polygon
        max_batches: Maximum number of batches to process (None = unlimited)
        
    Returns:
        Dictionary with caching statistics
    """
    logger.info(f"Starting patch caching for {data_mode.value} mode")
    logger.info(f"Samples per polygon: {n_samples_per_polygon}")
    logger.info(f"Maximum batches: {max_batches if max_batches else 'unlimited'}")
    
    stats = {
        'batches_processed': 0,
        'total_patches': 0,
        'patches_by_class': {},
        'processing_time': 0
    }
    
    start_time = time.time()
    
    try:
        # Get batch generator using correct method
        batch_generator = yielder.yield_batch(data_mode, n_samples_per_polygon=n_samples_per_polygon)
        
        batch_idx = 0
        for patches_array, labels_array in batch_generator:
            batch_idx += 1
            
            logger.info(f"Processing batch {batch_idx} for {data_mode.value}")
            
            stats['batches_processed'] += 1
            stats['total_patches'] += len(patches_array)
            
            # Count patches by class
            for label in labels_array:
                label_int = int(label)
                if label_int not in stats['patches_by_class']:
                    stats['patches_by_class'][label_int] = 0
                stats['patches_by_class'][label_int] += 1
            
            logger.info(f"Batch {batch_idx}: {len(patches_array)} patches, "
                       f"shape: {patches_array.shape}")
            
            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                logger.info(f"Progress: {batch_idx} batches, {stats['total_patches']} patches cached")
                
            # Break if we've reached max_batches
            if max_batches and batch_idx >= max_batches:
                logger.info(f"Reached maximum batches limit: {max_batches}")
                break
    
    except Exception as e:
        logger.error(f"Error in batch generation for {data_mode.value}: {e}")
        import traceback
        traceback.print_exc()
    
    stats['processing_time'] = time.time() - start_time
    
    logger.info(f"Caching completed for {data_mode.value} mode:")
    logger.info(f"  Batches processed: {stats['batches_processed']}")
    logger.info(f"  Total patches: {stats['total_patches']}")
    logger.info(f"  Patches by class: {stats['patches_by_class']}")
    logger.info(f"  Processing time: {stats['processing_time']:.1f} seconds")
    
    return stats


def main():
    """Main function to cache patches for all modes."""
    logger.info("Starting comprehensive patch caching")
    
    # Load configuration
    config = TrainingDataConfig.from_file(Path("data/config.conf"))
    logger.info(f"Loaded config with classes: {config.classes}")
    logger.info(f"Patch extraction settings - Per feature: {config.n_patches_per_feature}, Per area ratio: {config.n_patches_per_area}")
    
    # Create patch yielder with configurable cache size
    patch_cache_size = 5000  # Cache 5000 patches before flushing to disk
    logger.info(f"Creating PatchYielder with cache size: {patch_cache_size}")
    yielder = PatchYielder(config, seed=42, patch_cache_size=patch_cache_size)
    
    logger.info(f"PatchYielder created successfully:")
    logger.info(f"  Image tuples: {len(yielder.image_tuples)}")
    logger.info(f"  AOIs: {len(yielder.tuple_aois)}")
    logger.info(f"  Target classes: {config.classes}")
    
    # Print data split information
    from lc_speckle_analysis.patch_yielder import DataMode
    logger.info(f"  train: {len(yielder.data_splits[DataMode.TRAIN])} polygons")
    logger.info(f"  validation: {len(yielder.data_splits[DataMode.VALIDATION])} polygons")
    logger.info(f"  test: {len(yielder.data_splits[DataMode.TEST])} polygons")
    
    # Cache configuration - use more aggressive caching with new patch extraction parameters
    caching_config = {
        DataMode.TRAIN: {'n_samples_per_polygon': 25, 'max_batches': None},      # 25 samples per polygon for train
        DataMode.VALIDATION: {'n_samples_per_polygon': 15, 'max_batches': None}, # 15 samples per polygon for validation
        DataMode.TEST: {'n_samples_per_polygon': 15, 'max_batches': None}        # 15 samples per polygon for test
    }
    
    # Cache patches for each mode
    all_stats = {}
    total_start_time = time.time()
    
    for data_mode, config_params in caching_config.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"CACHING {data_mode.value.upper()} PATCHES")
        logger.info(f"{'='*60}")
        
        try:
            stats = cache_patches_for_mode(
                yielder, 
                data_mode, 
                n_samples_per_polygon=config_params['n_samples_per_polygon'],
                max_batches=config_params['max_batches']
            )
            all_stats[data_mode.value] = stats
            
        except Exception as e:
            logger.error(f"Failed to cache patches for {data_mode.value}: {e}")
            all_stats[data_mode.value] = {'error': str(e)}
    
    total_time = time.time() - total_start_time
    
    # Print final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL CACHING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total processing time: {total_time:.1f} seconds")
    
    total_patches = 0
    for mode, stats in all_stats.items():
        if 'error' in stats:
            logger.error(f"{mode}: ERROR - {stats['error']}")
        else:
            total_patches += stats['total_patches']
            logger.info(f"{mode}:")
            logger.info(f"  Batches: {stats['batches_processed']}")
            logger.info(f"  Patches: {stats['total_patches']}")
            logger.info(f"  Classes: {stats['patches_by_class']}")
            logger.info(f"  Time: {stats['processing_time']:.1f}s")
    
    logger.info(f"\nTotal patches cached across all modes: {total_patches}")
    
    # Ensure all remaining patches are flushed
    logger.info("Flushing any remaining patches in cache...")
    yielder.flush_all_caches()
    
    # Log settings used
    logger.info(f"\nPatch extraction settings used:")
    logger.info(f"  n_patches_per_feature: {config.n_patches_per_feature}")
    logger.info(f"  n_patches_per_area: {config.n_patches_per_area}")
    logger.info(f"  These settings allow more patches per polygon than the old hard-coded limit of 10")
    
    # Check cache directory
    cache_dir = Path("data/cache/patches")
    if cache_dir.exists():
        logger.info(f"\nCache directory contents:")
        for mode_dir in cache_dir.iterdir():
            if mode_dir.is_dir():
                cache_files = list(mode_dir.rglob("*.pkl"))
                logger.info(f"  {mode_dir.name}: {len(cache_files)} cache files")


if __name__ == "__main__":
    main()
