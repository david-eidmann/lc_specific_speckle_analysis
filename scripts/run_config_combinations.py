#!/usr/bin/env python3
"""
Run training with all combinations of specified configurations.

This script tests different combinations of:
- shuffle_labels: True/False  
- data_with_zero_mean: True/False
- dates: single date vs multiple dates
- architectures: test_conv2d vs test_conv2d_n2

Usage: poetry run python scripts/run_config_combinations.py
"""

import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lc_speckle_analysis.config import logger

# Base configuration combinations to test
BASE_CONFIG_COMBINATIONS = [
    # Format: (shuffle_labels, data_with_zero_mean, dates)
    (False, True, "20220611"),           # Current config baseline
    (False, False, "20220611"),          # No zero-mean normalization
    (False, True, "20220611,20220623"),  # Multiple dates with zero-mean
    (False, False, "20220611,20220623"), # Multiple dates without zero-mean
    # (True, True, "20220611"),            # Shuffled labels baseline
    # (True, False, "20220611"),           # Shuffled labels, no zero-mean
    # (True, True, "20220611,20220623"),   # Shuffled labels, multiple dates, zero-mean
    # (True, False, "20220611,20220623"),  # Shuffled labels, multiple dates, no zero-mean
]

# Available network architectures
ARCHITECTURES = [
    "test_conv2d",      # ~93,700 parameters
    "test_conv2d_n2",   # ~1,400 parameters
]

# Generate all combinations including architectures
CONFIG_COMBINATIONS = []
for arch in ARCHITECTURES:
    for shuffle_labels, data_with_zero_mean, dates in BASE_CONFIG_COMBINATIONS:
        CONFIG_COMBINATIONS.append((shuffle_labels, data_with_zero_mean, dates, arch))

def load_base_config() -> str:
    """Load the base configuration file content."""
    config_path = Path(__file__).parent.parent / "data" / "config.conf"
    return config_path.read_text()

def create_temp_config(base_config: str, shuffle_labels: bool, 
                      data_with_zero_mean: bool, dates: str, architecture: str) -> str:
    """Create a temporary configuration file with specified parameters."""
    # Split config into lines for modification
    lines = base_config.split('\n')
    
    # Modify the relevant lines
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped.startswith('shuffle_labels'):
            lines[i] = f"shuffle_labels = {str(shuffle_labels).lower()}"
        elif line_stripped.startswith('data_with_zero_mean'):
            # Update to new modus parameter format
            modus_value = 'data_with_zero_mean' if data_with_zero_mean else 'raw'
            lines[i] = f"modus = '{modus_value}'"
        elif line_stripped.startswith('dates'):
            lines[i] = f"dates = {dates}"
        elif line_stripped.startswith('network_architecture_id'):
            lines[i] = f"network_architecture_id = {architecture}"
    
    # If shuffle_labels line doesn't exist, add it after equal_class_dist
    if not any(line.strip().startswith('shuffle_labels') for line in lines):
        for i, line in enumerate(lines):
            if line.strip().startswith('equal_class_dist'):
                lines.insert(i + 1, f"shuffle_labels = {str(shuffle_labels).lower()}")
                break
    
    return '\n'.join(lines)

def generate_run_id(shuffle_labels: bool, data_with_zero_mean: bool, dates: str, architecture: str) -> str:
    """Generate a unique run ID based on configuration parameters."""
    shuffle_str = "shuffled" if shuffle_labels else "normal"
    zeromean_str = "zeromean" if data_with_zero_mean else "raw"
    dates_str = "single" if "," not in dates else "multi"
    arch_str = architecture.replace("test_", "")  # conv2d or conv2d_n2
    
    return f"{shuffle_str}_{zeromean_str}_{dates_str}_{dates.replace(',', '_')}_{arch_str}"

def run_training(config_content: str, run_id: str) -> Tuple[bool, str]:
    """Run training with the given configuration."""
    logger.info(f"Starting training run: {run_id}")
    
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
        
        # Run training
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per run
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ Training completed successfully for {run_id} (duration: {duration:.1f}s)")
            return True, f"Success in {duration:.1f}s"
        else:
            logger.error(f"❌ Training failed for {run_id}")
            logger.error(f"stdout: {result.stdout[-1000:]}")  # Last 1000 chars
            logger.error(f"stderr: {result.stderr[-1000:]}")  # Last 1000 chars
            return False, f"Failed: {result.stderr[-200:]}"
    
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Training timed out for {run_id}")
        return False, "Timeout after 2 hours"
    except Exception as e:
        logger.error(f"💥 Exception during training for {run_id}: {e}")
        return False, f"Exception: {str(e)[:200]}"
    
    finally:
        # Clean up temporary config file
        Path(temp_config_path).unlink(missing_ok=True)

def main():
    """Main function to run all configuration combinations."""
    logger.info("=" * 80)
    logger.info("STARTING CONFIGURATION COMBINATION TRAINING RUNS WITH ARCHITECTURE ITERATION")
    logger.info("=" * 80)
    
    # Load base configuration
    try:
        base_config = load_base_config()
        logger.info("✓ Base configuration loaded")
    except Exception as e:
        logger.error(f"Failed to load base configuration: {e}")
        return 1
    
    # Track results
    results = []
    total_combinations = len(CONFIG_COMBINATIONS)
    
    logger.info(f"Running {total_combinations} configuration combinations:")
    logger.info(f"  - Architectures: {ARCHITECTURES}")
    logger.info(f"  - Base combinations: {len(BASE_CONFIG_COMBINATIONS)}")
    logger.info(f"  - Total: {len(ARCHITECTURES)} × {len(BASE_CONFIG_COMBINATIONS)} = {total_combinations}")
    
    for i, (shuffle_labels, data_with_zero_mean, dates, arch) in enumerate(CONFIG_COMBINATIONS, 1):
        logger.info(f"  {i:2d}. shuffle_labels={shuffle_labels}, data_with_zero_mean={data_with_zero_mean}, dates={dates}, arch={arch}")
    
    logger.info("")
    
    # Run each combination
    for i, (shuffle_labels, data_with_zero_mean, dates, arch) in enumerate(CONFIG_COMBINATIONS, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"COMBINATION {i}/{total_combinations}")
        logger.info(f"{'='*60}")
        logger.info(f"Configuration:")
        logger.info(f"  - shuffle_labels: {shuffle_labels}")
        logger.info(f"  - data_with_zero_mean: {data_with_zero_mean}")
        logger.info(f"  - dates: {dates}")
        logger.info(f"  - architecture: {arch}")
        
        # Generate run ID
        run_id = generate_run_id(shuffle_labels, data_with_zero_mean, dates, arch)
        logger.info(f"  - run_id: {run_id}")
        
        # Create configuration
        config_content = create_temp_config(base_config, shuffle_labels, data_with_zero_mean, dates, arch)
        
        # Run training
        success, message = run_training(config_content, run_id)
        
        # Store result
        results.append({
            'combination': i,
            'shuffle_labels': shuffle_labels,
            'data_with_zero_mean': data_with_zero_mean,
            'dates': dates,
            'architecture': arch,
            'run_id': run_id,
            'success': success,
            'message': message
        })
        
        logger.info(f"Result: {message}")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    
    successful_runs = sum(1 for r in results if r['success'])
    failed_runs = len(results) - successful_runs
    
    logger.info(f"Total combinations: {len(results)}")
    logger.info(f"Successful runs: {successful_runs}")
    logger.info(f"Failed runs: {failed_runs}")
    
    # Summary by architecture
    logger.info("\nResults by architecture:")
    for arch in ARCHITECTURES:
        arch_results = [r for r in results if r['architecture'] == arch]
        arch_successful = sum(1 for r in arch_results if r['success'])
        logger.info(f"  {arch}: {arch_successful}/{len(arch_results)} successful")
    
    logger.info("\nDetailed results:")
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        logger.info(f"  {result['combination']:2d}. {result['run_id']}: {status} - {result['message']}")
    
    # Save results to JSON
    results_file = Path(__file__).parent.parent / "data" / "combination_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nDetailed results saved to: {results_file}")
    
    if failed_runs > 0:
        logger.warning(f"⚠️  {failed_runs} runs failed. Check logs above for details.")
        return 1
    else:
        logger.info("🎉 All configuration combinations completed successfully!")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
