#!/usr/bin/env python3
"""
Test script to verify image cycling is working correctly.
"""
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lc_speckle_analysis.config import logger
from lc_speckle_analysis.data_config import TrainingDataConfig
from lc_speckle_analysis.patch_yielder import PatchYielder, DataMode
import re

def extract_time_from_filename(filename: str) -> str:
    """Extract HHMMSS from S1 filename."""
    time_match = re.search(r'T(\d{6})', filename)
    return time_match.group(1) if time_match else filename[17:23]

def test_image_cycling():
    """Test that the patch yielder cycles through multiple images."""
    config_path = Path("data/config.conf")
    config = TrainingDataConfig.from_file(config_path)
    
    patch_yielder = PatchYielder(config, seed=42)
    
    logger.info(f"Available image tuples: {len(patch_yielder.image_tuples)}")
    for i, img_tuple in enumerate(patch_yielder.image_tuples):
        time_str = extract_time_from_filename(img_tuple.vv_path.name)
        logger.info(f"  Image {i}: {img_tuple.date}_{time_str}")
    
    # Generate a few batches and track which images are used
    used_images = set()
    batch_count = 0
    max_batches = 10  # Test first 10 batches
    
    train_generator = patch_yielder.yield_batch(DataMode.TRAIN, n_samples_per_polygon=1)
    
    for patches, labels in train_generator:
        batch_count += 1
        
        # Check cache for most recent patches to see which image they came from
        cache_dir = Path("data/cache/train")
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.json"))
            if cache_files:
                # Read most recent cache file
                import json
                latest_cache = max(cache_files, key=lambda x: x.stat().st_mtime)
                with open(latest_cache, 'r') as f:
                    cache_data = json.load(f)
                
                # Extract source files from recent patches
                for patch_data in cache_data.get('patches', []):
                    source_path = patch_data.get('source_vv_file', '')
                    if source_path:
                        filename = Path(source_path).name
                        time_str = extract_time_from_filename(filename)
                        used_images.add(time_str)
        
        logger.info(f"Batch {batch_count}: {patches.shape}, Images used so far: {sorted(used_images)}")
        
        if batch_count >= max_batches:
            break
    
    logger.info(f"Final result: Used {len(used_images)} different images: {sorted(used_images)}")
    
    if len(used_images) > 1:
        logger.info("✅ SUCCESS: Image cycling is working! Multiple images used.")
    else:
        logger.warning("❌ ISSUE: Only one image used. Image cycling may not be working.")

if __name__ == "__main__":
    test_image_cycling()
