#!/usr/bin/env python3
"""
DEBUG SCRIPT - NO TRY/CATCH BLOCKS!
This script will raise exceptions exactly where they occur to show the full stack trace.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lc_speckle_analysis import get_training_config
from lc_speckle_analysis.patch_yielder import PatchYielder, DataMode
import logging

# Enable full debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')

def main():
    """Debug script with no exception handling - let it crash where it fails!"""
    
    print("🐛 RAW DEBUG - NO EXCEPTION HANDLING")
    print("=" * 50)
    print("This will crash at the exact point of failure!\n")
    
    # Initialize without any error handling
    config = get_training_config()
    yielder = PatchYielder(config, seed=42)
    
    print(f"✓ Initialized successfully")
    print(f"  - GDF shape: {yielder.gdf.shape}")
    print(f"  - Train split: {len(yielder.data_splits[DataMode.TRAIN])}")
    print(f"  - Image tuples: {len(yielder.image_tuples)}")
    
    # Create batch generator - no error handling
    batch_generator = yielder.yield_batch(DataMode.TRAIN, n_samples_per_polygon=1)
    
    print("\n🎯 Generating batches - will crash on first error!")
    
    # Generate batches until it crashes
    for i in range(10):
        print(f"\n--- BATCH {i+1} ---")
        patches, labels = next(batch_generator)  # NO TRY/CATCH!
        print(f"✓ Success: {patches.shape}")

if __name__ == "__main__":
    main()
