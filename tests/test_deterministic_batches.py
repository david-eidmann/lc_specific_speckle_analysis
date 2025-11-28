#!/usr/bin/env python3
"""
Test script to verify deterministic batch generation.
Generates 50 batches twice and ensures they are byte-identical.
"""

import sys
from pathlib import Path
import numpy as np
import hashlib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lc_speckle_analysis import get_training_config
from lc_speckle_analysis.patch_yielder import PatchYielder, DataMode
import logging

# Only show important messages
logging.basicConfig(level=logging.WARNING)

def generate_batches(seed, n_batches=50):
    """Generate n batches and return their concatenated data."""
    print(f"Generating {n_batches} batches with seed {seed}")
    
    config = get_training_config()
    yielder = PatchYielder(config, seed=seed)
    
    batch_generator = yielder.yield_batch(DataMode.TRAIN, n_samples_per_polygon=1)
    
    all_patches = []
    all_labels = []
    
    for i in range(n_batches):
        try:
            patches, labels = next(batch_generator)
            all_patches.append(patches)
            all_labels.append(labels)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated batch {i+1}/{n_batches}")
                
        except StopIteration:
            print(f"  StopIteration after {i+1} batches")
            break
        except Exception as e:
            print(f"  Error at batch {i+1}: {e}")
            break
    
    if all_patches:
        # Concatenate all batches
        final_patches = np.concatenate(all_patches, axis=0)
        final_labels = np.concatenate(all_labels, axis=0)
        
        print(f"  Total patches: {final_patches.shape}")
        print(f"  Total labels: {final_labels.shape}")
        
        return final_patches, final_labels
    else:
        return None, None

def compute_hash(patches, labels):
    """Compute hash of the data for comparison."""
    patches_bytes = patches.tobytes()
    labels_bytes = labels.tobytes()
    
    hasher = hashlib.md5()
    hasher.update(patches_bytes)
    hasher.update(labels_bytes)
    
    return hasher.hexdigest()

def main():
    """Test deterministic batch generation."""
    
    print("🔬 TESTING DETERMINISTIC BATCH GENERATION")
    print("=" * 60)
    print("Generating 50 batches twice with same seed...")
    print()
    
    seed = 42
    n_batches = 50
    
    # First run
    print("🎯 FIRST RUN:")
    patches1, labels1 = generate_batches(seed, n_batches)
    
    if patches1 is None:
        print("❌ First run failed!")
        return
    
    hash1 = compute_hash(patches1, labels1)
    print(f"  Hash: {hash1}")
    print()
    
    # Second run  
    print("🎯 SECOND RUN:")
    patches2, labels2 = generate_batches(seed, n_batches)
    
    if patches2 is None:
        print("❌ Second run failed!")
        return
        
    hash2 = compute_hash(patches2, labels2)
    print(f"  Hash: {hash2}")
    print()
    
    # Compare results
    print("🔍 COMPARISON:")
    print(f"  Shapes match: {patches1.shape == patches2.shape and labels1.shape == labels2.shape}")
    print(f"  Hashes match: {hash1 == hash2}")
    
    # Detailed comparison
    if patches1.shape == patches2.shape:
        patches_equal = np.array_equal(patches1, patches2)
        labels_equal = np.array_equal(labels1, labels2)
        
        print(f"  Patches identical: {patches_equal}")
        print(f"  Labels identical: {labels_equal}")
        
        if not patches_equal:
            diff_count = np.sum(patches1 != patches2)
            print(f"  Patch differences: {diff_count} elements")
            
        if not labels_equal:
            diff_count = np.sum(labels1 != labels2)
            print(f"  Label differences: {diff_count} elements")
    
    print()
    
    # Final verdict
    if hash1 == hash2:
        print("✅ SUCCESS! Batches are BYTE-IDENTICAL")
        print("   The patch generation is deterministic with the same seed")
    else:
        print("❌ FAILURE! Batches are DIFFERENT")
        print("   The patch generation is not deterministic")
        
        # Additional debugging info
        if patches1 is not None and patches2 is not None:
            print(f"   Run 1 - Min: {patches1.min():.6f}, Max: {patches1.max():.6f}")
            print(f"   Run 2 - Min: {patches2.min():.6f}, Max: {patches2.max():.6f}")
            
            # Check if it's just floating point precision
            if patches1.shape == patches2.shape:
                max_diff = np.max(np.abs(patches1 - patches2))
                print(f"   Maximum difference: {max_diff:.10f}")
                
                if max_diff < 1e-10:
                    print("   → Differences are likely floating-point precision")
                else:
                    print("   → Significant differences detected")

if __name__ == "__main__":
    main()
