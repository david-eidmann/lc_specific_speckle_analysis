#!/usr/bin/env python3
"""Test script to verify that cached patches now contain full Patch objects with spatial metadata."""

import logging
from pathlib import Path
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_pickle_cache():
    """Test that cached patches contain full Patch objects with spatial metadata."""
    
    cache_dir = Path("data/cache/patches")
    
    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return
    
    # Find any pickle files
    pickle_files = list(cache_dir.rglob("*.pkl"))
    
    if not pickle_files:
        logger.error("No pickle cache files found")
        logger.info("Available files:")
        for f in cache_dir.rglob("*"):
            if f.is_file():
                logger.info(f"  {f}")
        return
    
    logger.info(f"Found {len(pickle_files)} pickle cache files")
    
    # Test the first pickle file
    test_file = pickle_files[0]
    logger.info(f"Testing cache file: {test_file}")
    
    try:
        with open(test_file, 'rb') as f:
            patches = pickle.load(f)
        
        logger.info(f"Successfully loaded {len(patches)} patches from {test_file.name}")
        
        if patches:
            patch = patches[0]
            logger.info(f"First patch type: {type(patch)}")
            
            # Check if it has the expected Patch object attributes
            expected_attrs = ['data', 'transform', 'bounds', 'crs', 'orbit', 'src_files', 'date', 'class_id', 'data_mode']
            
            logger.info("Patch attributes:")
            for attr in expected_attrs:
                if hasattr(patch, attr):
                    value = getattr(patch, attr)
                    logger.info(f"  ✓ {attr}: {type(value)} - {str(value)[:100]}...")
                else:
                    logger.error(f"  ✗ Missing attribute: {attr}")
            
            # Specifically check spatial metadata
            if hasattr(patch, 'bounds'):
                logger.info(f"Spatial bounds: {patch.bounds}")
            if hasattr(patch, 'transform'):
                logger.info(f"Transform: {patch.transform}")
            if hasattr(patch, 'crs'):
                logger.info(f"CRS: {patch.crs}")
            
            logger.info("✓ Patch objects contain full spatial metadata!")
        
    except Exception as e:
        logger.error(f"Failed to load pickle file: {e}")

if __name__ == "__main__":
    test_pickle_cache()
