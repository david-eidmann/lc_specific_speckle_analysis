#!/usr/bin/env python3
"""
Run a single training with specific configuration:
- zero-mean normalization
- single date (20220611)
- TestConv2D_N2 architecture
- equal class distribution
- shuffled labels
"""

import sys
import json
import subprocess
import tempfile
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_base_config() -> str:
    """Load the base configuration file content."""
    config_path = Path(__file__).parent.parent / "data" / "config.conf"
    return config_path.read_text()

def create_shuffled_config() -> str:
    """Create configuration with shuffled labels and TestConv2D_N2."""
    base_config = load_base_config()
    lines = base_config.split('\n')
    
    # Modify the relevant lines
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped.startswith('shuffle_labels'):
            lines[i] = "shuffle_labels = True"
        elif line_stripped.startswith('data_with_zero_mean'):
            lines[i] = "data_with_zero_mean = True"
        elif line_stripped.startswith('dates'):
            lines[i] = "dates = 20220611"
        elif line_stripped.startswith('network_architecture_id'):
            lines[i] = "network_architecture_id = test_conv2d_n2"
        elif line_stripped.startswith('equal_class_dist'):
            lines[i] = "equal_class_dist = True"
    
    # If shuffle_labels line doesn't exist, add it after equal_class_dist
    if not any(line.strip().startswith('shuffle_labels') for line in lines):
        for i, line in enumerate(lines):
            if line.strip().startswith('equal_class_dist'):
                lines.insert(i + 1, "shuffle_labels = True")
                break
    
    return '\n'.join(lines)

def run_shuffled_training():
    """Run training with shuffled configuration."""
    
    logger.info("=" * 60)
    logger.info("RUNNING SHUFFLED LABELS TRAINING")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("  - Architecture: TestConv2D_N2 (1,436 params)")
    logger.info("  - Data Processing: Zero-mean normalization")
    logger.info("  - Dates: Single date (20220611)")
    logger.info("  - Equal Class Distribution: True")
    logger.info("  - Shuffle Labels: True (SHUFFLED)")
    logger.info("  - Run ID: shuffled_zeromean_single_20220611_conv2d_n2")
    
    # Create configuration
    config_content = create_shuffled_config()
    run_id = "shuffled_zeromean_single_20220611_conv2d_n2"
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_content)
        temp_config_path = f.name
    
    try:
        # Build command
        cmd = [
            "poetry", "run", "python", "-m", "lc_speckle_analysis.train_model",
            "--config", temp_config_path,
            "--run-id", run_id
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info("Starting training...")
        
        # Run training
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ Training completed successfully!")
            logger.info(f"Duration: {duration:.1f} seconds")
            logger.info(f"Run ID: {run_id}")
            
            # Try to find the output directory
            training_output_dir = Path(__file__).parent.parent / "data" / "training_output"
            run_dirs = [d for d in training_output_dir.iterdir() if d.is_dir() and run_id in d.name]
            
            if run_dirs:
                latest_run_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
                logger.info(f"Output directory: {latest_run_dir}")
                
                # Check for training summary
                summary_files = list(latest_run_dir.glob("training_summary_*.json"))
                if summary_files:
                    summary_file = summary_files[0]
                    logger.info(f"Training summary: {summary_file}")
                    
                    # Quick performance preview
                    try:
                        with open(summary_file, 'r') as f:
                            data = json.load(f)
                        
                        test_results = data.get("test_results", {})
                        oa = test_results.get("test_accuracy", 0.0)
                        macro_f1 = test_results.get("macro_avg", {}).get("f1-score", 0.0)
                        
                        logger.info("=" * 40)
                        logger.info("QUICK PERFORMANCE PREVIEW:")
                        logger.info("=" * 40)
                        logger.info(f"Overall Accuracy: {oa:.3f}")
                        logger.info(f"Macro F1 Score: {macro_f1:.3f}")
                        logger.info("=" * 40)
                        
                        # Compare with non-shuffled equivalent
                        logger.info("COMPARISON WITH NON-SHUFFLED:")
                        logger.info("Non-shuffled TestConv2D_N2 + Zero-mean + Single:")
                        logger.info("  - OA: 0.521, F1: 0.498")
                        logger.info(f"Shuffled TestConv2D_N2 + Zero-mean + Single:")
                        logger.info(f"  - OA: {oa:.3f}, F1: {macro_f1:.3f}")
                        
                        if oa < 0.3:  # Expected for shuffled labels
                            logger.info("✅ EXPECTED: Shuffled labels show poor performance (confirms no data leakage)")
                        else:
                            logger.warning("⚠️  UNEXPECTED: Shuffled labels show good performance (possible data leakage?)")
                        
                    except Exception as e:
                        logger.warning(f"Could not read performance metrics: {e}")
            
            return True, f"Success in {duration:.1f}s"
        else:
            logger.error(f"❌ Training failed!")
            logger.error(f"stdout: {result.stdout[-1000:]}")  # Last 1000 chars
            logger.error(f"stderr: {result.stderr[-1000:]}")  # Last 1000 chars
            return False, f"Failed: {result.stderr[-200:]}"
    
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Training timed out after 2 hours")
        return False, "Timeout after 2 hours"
    except Exception as e:
        logger.error(f"💥 Exception during training: {e}")
        return False, f"Exception: {str(e)[:200]}"
    
    finally:
        # Clean up temporary config file
        Path(temp_config_path).unlink(missing_ok=True)

def main():
    """Main function."""
    logger.info("Starting shuffled labels training run...")
    
    success, message = run_shuffled_training()
    
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULT")
    logger.info("=" * 60)
    
    if success:
        logger.info("🎉 Shuffled training completed successfully!")
        logger.info("This run tests for data leakage by using shuffled labels.")
        logger.info("Expected result: Poor performance (~25% OA for 4-class problem)")
        logger.info("If performance is good, there might be data leakage.")
    else:
        logger.error("❌ Shuffled training failed!")
    
    logger.info(f"Result: {message}")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
