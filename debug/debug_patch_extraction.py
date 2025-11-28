#!/usr/bin/env python3
"""
Debug entry point for patch extraction issues.
Run with: poetry run python debug_patch_extraction.py
"""

import sys
from pathlib import Path
sys.path.insert(0, 'src')

from lc_speckle_analysis.data_config import TrainingDataConfig
from lc_speckle_analysis.patch_yielder import PatchYielder, DataMode
import time
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def debug_single_polygon():
    """Debug extraction from a single polygon to isolate the issue."""
    print("=== DEBUGGING SINGLE POLYGON EXTRACTION ===")
    
    config = TrainingDataConfig.from_file(Path('data/config.conf'))
    patch_yielder = PatchYielder(config, seed=42, patch_cache_size=10)
    
    # Get first train polygon
    train_data = patch_yielder.data_splits[DataMode.TRAIN]
    print(f"Train polygons available: {len(train_data)}")
    
    # Get first image tuple
    image_tuple = patch_yielder.image_tuples[0]
    print(f"Using image: {image_tuple.vv_path.name}")
    
    # Get a sample polygon
    sample_polygon = train_data.iloc[0]
    print(f"Sample polygon: class={sample_polygon[config.column_id]}, geometry={sample_polygon.geometry.geom_type}")
    
    # Time the extraction
    import rasterio
    start_time = time.time()
    
    try:
        with rasterio.open(image_tuple.vv_path) as vv_src, rasterio.open(image_tuple.vh_path) as vh_src:
            print("Opened raster files successfully")
            
            # Call the problematic method directly
            patch_objects = patch_yielder._extract_patches_to_objects(
                sample_polygon, vv_src, vh_src, image_tuple, DataMode.TRAIN
            )
            
            elapsed = time.time() - start_time
            print(f"SUCCESS: Extracted {len(patch_objects)} patches in {elapsed:.2f} seconds")
            
            if patch_objects:
                patch = patch_objects[0]
                print(f"First patch: shape={patch.data.shape}, class={patch.class_id}")
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"ERROR after {elapsed:.2f} seconds: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def debug_batch_generation():
    """Debug batch generation to see where it hangs."""
    print("\n=== DEBUGGING BATCH GENERATION ===")
    
    config = TrainingDataConfig.from_file(Path('data/config.conf'))
    patch_yielder = PatchYielder(config, seed=42, patch_cache_size=10)
    
    train_generator = patch_yielder.yield_batch(DataMode.TRAIN, n_samples_per_polygon=1)
    
    print("Starting batch generation...")
    start_time = time.time()
    
    try:
        for i, (patches, labels) in enumerate(train_generator):
            elapsed = time.time() - start_time
            print(f"Batch {i+1}: {patches.shape[0]} patches after {elapsed:.2f}s")
            
            if i >= 0:  # Just get first batch
                break
                
        print("SUCCESS: Batch generation working")
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"INTERRUPTED after {elapsed:.2f} seconds")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"ERROR after {elapsed:.2f} seconds: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run both debug functions
    debug_single_polygon()
    debug_batch_generation()
