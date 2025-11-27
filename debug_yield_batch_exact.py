#!/usr/bin/env python3
"""
EXACT YIELD_BATCH REPLICATION - NO EXCEPTION HANDLING
This replicates the exact same flow as yield_batch() to find where the error actually occurs.
"""

import sys
from pathlib import Path
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lc_speckle_analysis import get_training_config
from lc_speckle_analysis.patch_yielder import PatchYielder, DataMode
import logging
import rasterio
from rasterio.mask import mask

# Enable full debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')

def extract_time_from_filename(filename):
    """Extract time from filename like the original."""
    import re
    match = re.search(r'T(\d{6})', filename)
    return match.group(1) if match else "000000"

def main():
    """Replicate the exact yield_batch flow with no exception handling!"""
    
    print("🔍 EXACT YIELD_BATCH REPLICATION - NO TRY/CATCH!")
    print("=" * 60)
    print("Replicating the exact same flow as yield_batch()...\n")
    
    # Initialize exactly like yield_batch
    config = get_training_config()
    yielder = PatchYielder(config, seed=42)
    
    mode = DataMode.TRAIN
    n_samples_per_polygon = 1
    batch_size = 32
    
    data = yielder.data_splits[mode]
    current_image_idx = 0
    
    print(f"✓ Initialized: {len(data)} polygons in {mode.value} split")
    print(f"✓ Image tuples: {len(yielder.image_tuples)}")
    
    # Set random seed exactly like yield_batch
    random.seed(yielder.seed)
    
    # Open datasets dict (like in yield_batch)
    opened_datasets = {}
    
    def get_open_datasets(image_tuple):
        tuple_key = (str(image_tuple.vv_path), str(image_tuple.vh_path))
        if tuple_key not in opened_datasets:
            vv_src = rasterio.open(image_tuple.vv_path)
            vh_src = rasterio.open(image_tuple.vh_path)
            opened_datasets[tuple_key] = (vv_src, vh_src)
        return opened_datasets[tuple_key]
    
    # EXACT REPLICATION OF YIELD_BATCH LOOP
    batch_count = 0
    
    while batch_count < 3:  # Test 3 batches max
        print(f"\n🎯 BATCH {batch_count + 1}")
        print("-" * 40)
        
        # Cycle through available images (EXACT SAME LOGIC)
        current_tuple = yielder.image_tuples[current_image_idx]
        
        # Get time string for logging (EXACT SAME)
        time_str = extract_time_from_filename(current_tuple.vv_path.name)
        print(f"Using image tuple: {current_tuple.date}_{time_str} (index {current_image_idx})")
        
        vv_src, vh_src = get_open_datasets(current_tuple)
        print(f"VV path: {vv_src.name}")
        print(f"Raster bounds: {vv_src.bounds}")
        
        # Use stable unique key (EXACT SAME)
        unique_key = yielder._generate_stable_unique_key(current_tuple)
        intersecting_indices = yielder.polygon_image_intersections.get(unique_key, set())
        
        print(f"Processing image {unique_key}: {current_tuple.vv_path.name}")
        print(f"Intersection mapping has {len(intersecting_indices)} polygons")
        
        if not intersecting_indices:
            print(f"No intersection mapping found for {unique_key}")
            current_image_idx = (current_image_idx + 1) % len(yielder.image_tuples)
            continue
        
        # Get polygons that intersect with current image (EXACT SAME LOGIC)
        data_indices = set(data.index)
        valid_indices = intersecting_indices & data_indices
        
        print(f"Valid polygons for {unique_key}: {len(valid_indices)} out of {len(intersecting_indices)} mapped")
        print(f"  Intersecting indices: {list(intersecting_indices)[:5]}...")
        print(f"  Split data indices range: {min(data_indices) if data_indices else 'N/A'} to {max(data_indices) if data_indices else 'N/A'}")
        
        if not valid_indices:
            print(f"No valid polygons found for {unique_key}")
            current_image_idx = (current_image_idx + 1) % len(yielder.image_tuples)
            continue
            
        valid_polygons = data.loc[list(valid_indices)]
        
        patches_collected = 0
        max_attempts = min(len(valid_polygons) * 2, batch_size * 5)
        
        print(f"Starting patch collection: max_attempts={max_attempts}")
        
        for attempt in range(max_attempts):
            if patches_collected >= batch_size:
                break
            
            # Sample polygon (EXACT SAME)
            polygon_row = valid_polygons.sample(n=1, random_state=None).iloc[0]
            
            print(f"\nATTEMPT {attempt + 1}: Sampled polygon index {polygon_row.name}")
            print(f"  Polygon bounds: {polygon_row.geometry.bounds}")
            print(f"  Class: {polygon_row[yielder.config.column_id]}")
            
            # Check manual overlap
            geom = polygon_row.geometry
            poly_minx, poly_miny, poly_maxx, poly_maxy = geom.bounds
            raster_minx, raster_miny, raster_maxx, raster_maxy = vv_src.bounds
            
            overlap = (poly_minx < raster_maxx and poly_maxx > raster_minx and 
                      poly_miny < raster_maxy and poly_maxy > raster_miny)
            
            print(f"  Manual overlap check: {overlap}")
            
            if not overlap:
                print(f"  ❌ NO OVERLAP DETECTED!")
                print(f"     Polygon X: {poly_minx:.0f} to {poly_maxx:.0f}")  
                print(f"     Raster X:  {raster_minx:.0f} to {raster_maxx:.0f}")
                print(f"     Polygon Y: {poly_miny:.0f} to {poly_maxy:.0f}")
                print(f"     Raster Y:  {raster_miny:.0f} to {raster_maxy:.0f}")
                print(f"  💥 CALLING _extract_patches_to_objects - SHOULD CRASH HERE!")
            
            # Call the EXACT method that should fail - NO TRY/CATCH!
            patch_objects = yielder._extract_patches_to_objects(
                polygon_row, vv_src, vh_src, current_tuple, mode
            )
            
            print(f"  ✓ _extract_patches_to_objects succeeded: {len(patch_objects) if patch_objects else 0} patches")
            
            if patch_objects:
                patches_collected += len(patch_objects)
                print(f"  Added {len(patch_objects)} patches (total: {patches_collected})")
            
        print(f"\nBatch {batch_count + 1} completed: {patches_collected} patches")
        batch_count += 1
        
        # Advance to next image (EXACT SAME)
        current_image_idx = (current_image_idx + 1) % len(yielder.image_tuples)
    
    # Close all opened datasets
    for vv_src, vh_src in opened_datasets.values():
        vv_src.close()
        vh_src.close()
    
    print(f"\n🤔 All batches completed successfully - no crashes!")

if __name__ == "__main__":
    main()
