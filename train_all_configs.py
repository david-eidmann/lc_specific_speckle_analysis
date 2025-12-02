#!/usr/bin/env python3
"""
Systematic training runner for all modular processing configurations.
Trains all 24 configuration combinations and collects results.
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigTrainingRunner:
    """Manages systematic training of all modular processing configurations."""
    
    def __init__(self, base_dir: str = "/home/davideidmann/code/lc_specific_speckle_analysis"):
        self.base_dir = Path(base_dir)
        self.configs_dir = self.base_dir / "configs"
        self.results_dir = self.base_dir / "training_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize results tracking
        self.training_results = {}
        self.failed_configs = []
        self.successful_configs = []
        
    def discover_config_files(self) -> List[Path]:
        """Discover all config_*.conf files."""
        config_files = list(self.configs_dir.glob("config_*.conf"))
        config_files.sort()  # Ensure consistent ordering
        
        logger.info(f"Discovered {len(config_files)} configuration files:")
        for config_file in config_files:
            logger.info(f"  - {config_file.name}")
        
        return config_files
    
    def train_single_config(self, config_path: Path) -> Dict:
        """Train a single configuration and return results."""
        config_name = config_path.stem
        logger.info(f"🚀 Starting training for: {config_name}")
        
        # Prepare training command
        cmd = [
            "poetry", "run", "python", "-m", "src.lc_speckle_analysis.train_model",
            "--config", str(config_path)
        ]
        
        start_time = time.time()
        result = {
            'config_name': config_name,
            'config_path': str(config_path),
            'start_time': start_time,
            'status': 'unknown',
            'duration': 0,
            'error_message': None,
            'stdout': '',
            'stderr': ''
        }
        
        try:
            # Run training
            logger.info(f"Executing: {' '.join(cmd)}")
            process = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per config
            )
            
            end_time = time.time()
            result['duration'] = end_time - start_time
            result['stdout'] = process.stdout
            result['stderr'] = process.stderr
            result['return_code'] = process.returncode
            
            if process.returncode == 0:
                result['status'] = 'success'
                logger.info(f"✅ SUCCESS: {config_name} completed in {result['duration']:.1f}s")
                self.successful_configs.append(config_name)
            else:
                result['status'] = 'failed'
                result['error_message'] = f"Process returned code {process.returncode}"
                logger.error(f"❌ FAILED: {config_name} - {result['error_message']}")
                self.failed_configs.append(config_name)
                
        except subprocess.TimeoutExpired:
            result['status'] = 'timeout'
            result['error_message'] = "Training timed out after 1 hour"
            result['duration'] = 3600
            logger.error(f"⏰ TIMEOUT: {config_name} - exceeded 1 hour limit")
            self.failed_configs.append(config_name)
            
        except Exception as e:
            end_time = time.time()
            result['duration'] = end_time - start_time
            result['status'] = 'error'
            result['error_message'] = str(e)
            logger.error(f"💥 ERROR: {config_name} - {str(e)}")
            self.failed_configs.append(config_name)
        
        return result
    
    def save_results(self):
        """Save training results to JSON file."""
        results_file = self.results_dir / "training_results.json"
        
        summary = {
            'total_configs': len(self.training_results),
            'successful': len(self.successful_configs),
            'failed': len(self.failed_configs),
            'success_rate': len(self.successful_configs) / len(self.training_results) * 100 if self.training_results else 0,
            'successful_configs': self.successful_configs,
            'failed_configs': self.failed_configs,
            'detailed_results': self.training_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
    
    def print_summary(self):
        """Print training summary."""
        total = len(self.training_results)
        successful = len(self.successful_configs)
        failed = len(self.failed_configs)
        success_rate = (successful / total * 100) if total > 0 else 0
        
        logger.info(f"\n{'='*80}")
        logger.info("🏁 TRAINING SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total configurations: {total}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        if self.successful_configs:
            logger.info(f"\n✅ Successful configurations:")
            for config in self.successful_configs:
                duration = self.training_results[config]['duration']
                logger.info(f"  - {config} ({duration:.1f}s)")
        
        if self.failed_configs:
            logger.info(f"\n❌ Failed configurations:")
            for config in self.failed_configs:
                error = self.training_results[config]['error_message']
                logger.info(f"  - {config}: {error}")
        
        logger.info(f"{'='*80}")
    
    def run_all_training(self):
        """Run training for all discovered configurations."""
        logger.info("🎯 Starting systematic training of all modular processing configurations")
        
        # Discover configurations
        config_files = self.discover_config_files()
        
        if not config_files:
            logger.error("No configuration files found!")
            return
        
        total_configs = len(config_files)
        logger.info(f"📋 Will train {total_configs} configurations")
        
        # Train each configuration
        for i, config_path in enumerate(config_files, 1):
            config_name = config_path.stem
            
            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 PROGRESS: {i}/{total_configs} - Training {config_name}")
            logger.info(f"{'='*80}")
            
            # Train this configuration
            result = self.train_single_config(config_path)
            self.training_results[config_name] = result
            
            # Save intermediate results
            self.save_results()
            
            # Brief pause between trainings
            if i < total_configs:
                logger.info("⏱️  Pausing 5 seconds before next training...")
                time.sleep(5)
        
        # Final summary
        self.print_summary()
        logger.info("🎉 All training completed!")


def main():
    """Main entry point."""
    runner = ConfigTrainingRunner()
    
    try:
        runner.run_all_training()
    except KeyboardInterrupt:
        logger.info("\n🛑 Training interrupted by user")
        runner.print_summary()
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        runner.print_summary()
        raise


if __name__ == "__main__":
    main()
