#!/usr/bin/env python3
"""
Python-based parallel training runner for all modular processing configurations.
Runs multiple training processes in parallel using multiprocessing.
"""

import os
import subprocess
import time
import logging
from pathlib import Path
from multiprocessing import Pool
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG_LIST_FILE = "configs/generated_configs_list.txt"
MAX_PARALLEL_PROCESSES = 4  # Use 4 parallel processes on GPU
RUN_ID = "parallel_training"
LOG_DIR = Path("logs")

def run_single_training(config_file):
    """Run training for a single configuration file."""
    config_name = Path(config_file).stem
    log_file = LOG_DIR / f"training_{config_name}.log"
    
    logger.info(f"Starting training: {config_name}")
    
    # Prepare command
    cmd = [
        "poetry", "run", "python", "-m", "lc_speckle_analysis.train_model",
        "--config", config_file,
        "--run-id", RUN_ID
    ]
    
    # Set environment with PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    
    start_time = time.time()
    
    try:
        # Run training process
        with open(log_file, "w") as f:
            f.write(f"Starting training: {config_name} at {datetime.now()}\n")
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=Path(__file__).parent
            )
            return_code = process.wait()
        
        duration = time.time() - start_time
        
        if return_code == 0:
            logger.info(f"✓ SUCCESS: {config_name} completed in {duration:.1f}s")
            return {"config": config_name, "status": "success", "duration": duration, "return_code": return_code}
        else:
            logger.error(f"❌ FAILED: {config_name} - process returned code {return_code}")
            return {"config": config_name, "status": "failed", "duration": duration, "return_code": return_code}
            
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"💥 ERROR: {config_name} - {str(e)}")
        return {"config": config_name, "status": "error", "duration": duration, "error": str(e)}

def main():
    """Main function to run parallel training."""
    logger.info(f"Starting parallel training with {MAX_PARALLEL_PROCESSES} processes")
    logger.info(f"Master script PID: {os.getpid()}")
    
    # Create logs directory
    LOG_DIR.mkdir(exist_ok=True)
    
    # Load configuration files
    config_list_path = Path(CONFIG_LIST_FILE)
    if not config_list_path.exists():
        logger.error(f"Configuration list file not found: {CONFIG_LIST_FILE}")
        return 1
    
    configs = []
    with open(config_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if Path(line).exists():
                    configs.append(line)
                else:
                    logger.warning(f"Config file not found: {line}")
    
    if not configs:
        logger.error("No valid configuration files found")
        return 1
    
    logger.info(f"Found {len(configs)} configuration files to process")
    
    # Run parallel training for all configs
    start_time = time.time()
    
    with Pool(processes=MAX_PARALLEL_PROCESSES) as pool:
        results = pool.map(run_single_training, configs)
    
    total_duration = time.time() - start_time
    
    # Analyze results
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    
    logger.info(f"All parallel training completed in {total_duration:.1f}s")
    logger.info(f"Results: {len(successful)} successful, {len(failed)} failed")
    
    if successful:
        logger.info("Successful configurations:")
        for result in successful:
            logger.info(f"  ✓ {result['config']} ({result['duration']:.1f}s)")
    
    if failed:
        logger.info("Failed configurations:")
        for result in failed:
            error_msg = result.get('error', f"return code {result.get('return_code', 'unknown')}")
            logger.info(f"  ❌ {result['config']} - {error_msg}")
    
    logger.info("Check logs/ directory for individual training logs")
    
    return 0 if not failed else 1

if __name__ == "__main__":
    exit(main())
