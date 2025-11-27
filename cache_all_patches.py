#!/usr/bin/env python3
"""
Cache all patches for train/validation/test sets.
Run with: poetry run python cache_all_patches.py
"""

import sys
from pathlib import Path
sys.path.insert(0, 'src')

from lc_speckle_analysis.data_config import TrainingDataConfig
from lc_speckle_analysis.patch_yielder import PatchYielder, DataMode
import time

def cache_all_patches():
    """Cache substantial amounts of patches for all data modes."""
    print("=== CACHING ALL PATCHES ===")
    
    config = TrainingDataConfig.from_file(Path('data/config.conf'))
    patch_yielder = PatchYielder(config, seed=42, patch_cache_size=1000)  # Large cache for efficiency
    
    total_patches = 0
    start_time = time.time()
    
    # Cache train set
    print('\n=== CACHING TRAIN SET ===')
    train_generator = patch_yielder.yield_batch(DataMode.TRAIN, n_samples_per_polygon=1)
    train_count = 0
    
    # try:
    for i, (patches, labels) in enumerate(train_generator):
        train_count += patches.shape[0]
        elapsed = time.time() - start_time
        print(f'Train batch {i+1}: {patches.shape[0]} patches (total: {train_count}, time: {elapsed:.1f}s)')
        
        if i >= 100:  # Cache substantial amount (100+ batches)
            break
                
    # except KeyboardInterrupt:
    #     print(f'Train caching interrupted at {train_count} patches')
    # except Exception as e:
    #     print(f'Train caching error: {e}')
    #     import traceback
    #     traceback.print_exc()
    
    total_patches += train_count
    
    # Cache validation set
    print('\n=== CACHING VALIDATION SET ===')
    val_generator = patch_yielder.yield_batch(DataMode.VALIDATION, n_samples_per_polygon=1)
    val_count = 0
    
    # try:
    for i, (patches, labels) in enumerate(val_generator):
        val_count += patches.shape[0]
        elapsed = time.time() - start_time
        print(f'Validation batch {i+1}: {patches.shape[0]} patches (total: {val_count}, time: {elapsed:.1f}s)')
        
        if i >= 30:  # Cache substantial amount
            break
                
    # except KeyboardInterrupt:
    #     print(f'Validation caching interrupted at {val_count} patches')
    # except Exception as e:
    #     print(f'Validation caching error: {e}')
    #     import traceback
    #     traceback.print_exc()
    
    total_patches += val_count
    
    # Cache test set
    print('\n=== CACHING TEST SET ===')
    test_generator = patch_yielder.yield_batch(DataMode.TEST, n_samples_per_polygon=1)
    test_count = 0
    
    # try:
    for i, (patches, labels) in enumerate(test_generator):
        test_count += patches.shape[0]
        elapsed = time.time() - start_time
        print(f'Test batch {i+1}: {patches.shape[0]} patches (total: {test_count}, time: {elapsed:.1f}s)')
        
        if i >= 30:  # Cache substantial amount
            break
                
    # except KeyboardInterrupt:
    #     print(f'Test caching interrupted at {test_count} patches')
    # except Exception as e:
    #     print(f'Test caching error: {e}')
    #     import traceback
    #     traceback.print_exc()
    
    total_patches += test_count
    
    # Final summary
    total_elapsed = time.time() - start_time
    print(f'\n=== CACHING COMPLETE ===')
    print(f'Train patches cached: {train_count}')
    print(f'Validation patches cached: {val_count}')
    print(f'Test patches cached: {test_count}')
    print(f'Total patches cached: {total_patches}')
    print(f'Total time: {total_elapsed:.1f} seconds')
    print(f'Average rate: {total_patches/total_elapsed:.1f} patches/second')

if __name__ == "__main__":
    cache_all_patches()
