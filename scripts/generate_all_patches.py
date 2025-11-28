#!/usr/bin/env python3
"""
Script to generate and cache all patches for train, validation, and test modes.
Designed to run with nohup for long-running background processing.
"""

import sys
import time
import signal
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lc_speckle_analysis import get_training_config
from lc_speckle_analysis.patch_yielder import PatchYielder, DataMode
import logging

# Configure logging for file output
log_file = Path("patch_generation.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)  # Also show on console
    ]
)

logger = logging.getLogger(__name__)

class PatchGenerationManager:
    """Manages the complete patch generation process."""
    
    def __init__(self):
        self.start_time = None
        self.yielder = None
        self.interrupted = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
        self.interrupted = True
    
    def generate_mode_patches(self, mode: DataMode, samples_per_polygon: int = 50) -> dict:
        """Generate all patches for a specific mode.
        
        Args:
            mode: DataMode to generate patches for
            samples_per_polygon: Number of samples to extract per polygon
            
        Returns:
            Statistics dictionary
        """
        logger.info(f"🚀 Starting patch generation for {mode.value.upper()} mode")
        logger.info(f"Target: {samples_per_polygon} samples per polygon")
        
        mode_start = time.time()
        
        # Get available polygons for this mode
        available_polygons = len(self.yielder.data_splits[mode])
        logger.info(f"Available polygons: {available_polygons}")
        
        # Calculate expected patches
        expected_batches = available_polygons * samples_per_polygon // self.yielder.config.neural_network.batch_size
        logger.info(f"Expected batches: ~{expected_batches}")
        
        # Generate batches
        batch_generator = self.yielder.yield_batch(mode, n_samples_per_polygon=samples_per_polygon)
        
        batch_count = 0
        patch_count = 0
        error_count = 0
        
        try:
            while not self.interrupted:
                try:
                    patches, labels = next(batch_generator)
                    batch_count += 1
                    patch_count += len(patches)
                    
                    if batch_count % 50 == 0:
                        elapsed = time.time() - mode_start
                        logger.info(f"  Batch {batch_count}: {patch_count} patches generated "
                                  f"({elapsed:.1f}s elapsed)")
                    
                except StopIteration:
                    logger.info(f"  Completed: All available data processed for {mode.value}")
                    break
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"  Error in batch {batch_count + 1}: {e}")
                    if error_count > 10:
                        logger.error(f"  Too many errors ({error_count}), stopping {mode.value} mode")
                        break
                    continue
        
        except KeyboardInterrupt:
            logger.warning(f"  Interrupted during {mode.value} mode generation")
            self.interrupted = True
        
        # All patches are now cached automatically per file
        
        mode_elapsed = time.time() - mode_start
        
        stats = {
            'mode': mode.value,
            'batches_generated': batch_count,
            'patches_generated': patch_count,
            'errors': error_count,
            'time_seconds': mode_elapsed,
            'patches_per_second': patch_count / mode_elapsed if mode_elapsed > 0 else 0
        }
        
        logger.info(f"✅ {mode.value.upper()} mode completed:")
        logger.info(f"  Batches: {batch_count}")
        logger.info(f"  Patches: {patch_count}")
        logger.info(f"  Errors: {error_count}")
        logger.info(f"  Time: {mode_elapsed:.1f}s ({stats['patches_per_second']:.1f} patches/sec)")
        
        return stats
    
    def generate_all_patches(self, samples_per_polygon: int = 50):
        """Generate patches for all modes (train, validation, test).
        
        Args:
            samples_per_polygon: Number of samples to extract per polygon
        """
        logger.info("=" * 80)
        logger.info("🎯 COMPREHENSIVE PATCH GENERATION STARTED")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {datetime.now()}")
        logger.info(f"Samples per polygon: {samples_per_polygon}")
        logger.info(f"Log file: {log_file.absolute()}")
        
        self.start_time = time.time()
        
        # Initialize PatchYielder
        try:
            logger.info("Initializing PatchYielder...")
            config = get_training_config()
            self.yielder = PatchYielder(config, seed=42)
            
            logger.info(f"Configuration loaded:")
            logger.info(f"  Image tuples: {len(self.yielder.image_tuples)}")
            logger.info(f"  Train polygons: {len(self.yielder.data_splits[DataMode.TRAIN])}")
            logger.info(f"  Validation polygons: {len(self.yielder.data_splits[DataMode.VALIDATION])}")
            logger.info(f"  Test polygons: {len(self.yielder.data_splits[DataMode.TEST])}")
            logger.info(f"  Patch size: {self.yielder.config.neural_network.patch_size}x{self.yielder.config.neural_network.patch_size}")
            logger.info(f"  Batch size: {self.yielder.config.neural_network.batch_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize PatchYielder: {e}")
            return
        
        # Generate patches for each mode
        all_stats = []
        modes_to_process = [DataMode.TRAIN, DataMode.VALIDATION, DataMode.TEST]
        
        for i, mode in enumerate(modes_to_process, 1):
            if self.interrupted:
                logger.warning("Interrupted before completing all modes")
                break
                
            logger.info(f"\n📦 Processing mode {i}/{len(modes_to_process)}: {mode.value.upper()}")
            logger.info("-" * 60)
            
            try:
                stats = self.generate_mode_patches(mode, samples_per_polygon)
                all_stats.append(stats)
            except Exception as e:
                logger.error(f"Failed to generate patches for {mode.value}: {e}")
                continue
        
        # Final summary
        self._print_final_summary(all_stats)
    
    def _print_final_summary(self, all_stats: list):
        """Print comprehensive summary of patch generation."""
        total_time = time.time() - self.start_time
        total_patches = sum(stats['patches_generated'] for stats in all_stats)
        total_batches = sum(stats['batches_generated'] for stats in all_stats)
        total_errors = sum(stats['errors'] for stats in all_stats)
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 PATCH GENERATION SUMMARY")
        logger.info("=" * 80)
        
        for stats in all_stats:
            logger.info(f"{stats['mode'].upper():>12}: {stats['patches_generated']:>8,} patches "
                       f"({stats['batches_generated']:>6,} batches, {stats['errors']:>3} errors)")
        
        logger.info("-" * 80)
        logger.info(f"{'TOTAL':>12}: {total_patches:>8,} patches "
                   f"({total_batches:>6,} batches, {total_errors:>3} errors)")
        
        logger.info(f"\n⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        if total_time > 0:
            logger.info(f"📊 Overall rate: {total_patches/total_time:.1f} patches/sec")
        
        # Cache information
        cache_dir = Path("data/cache/patches")
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("**/*.npz"))
            total_cache_size = sum(f.stat().st_size for f in cache_files)
            logger.info(f"💾 Cache files: {len(cache_files)} files, {total_cache_size/1024/1024:.1f} MB")
        
        if self.interrupted:
            logger.warning("⚠️  Generation was interrupted - results may be incomplete")
        else:
            logger.info("✅ All patches generated successfully!")
        
        logger.info(f"📝 Full log available at: {log_file.absolute()}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate all patches for training")
    parser.add_argument("--samples-per-polygon", type=int, default=50,
                       help="Number of samples to extract per polygon (default: 50)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with only 5 samples per polygon")
    
    args = parser.parse_args()
    
    samples = 5 if args.quick else args.samples_per_polygon
    
    manager = PatchGenerationManager()
    
    try:
        manager.generate_all_patches(samples_per_polygon=samples)
    except Exception as e:
        logger.error(f"Unexpected error during patch generation: {e}")
        # Patches are cached automatically, no manual flushing needed
        raise

if __name__ == "__main__":
    main()
