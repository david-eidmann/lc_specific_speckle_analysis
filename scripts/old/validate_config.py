#!/usr/bin/env python3
"""Configuration validation script."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lc_speckle_analysis import get_training_config
from lc_speckle_analysis.config import logger


def main():
    """Validate configuration values."""
    logger.info("Starting configuration validation")
    
    try:
        config = get_training_config()
        
        # Test configuration values
        assert config.train_data_path.endswith('.gpkg'), f"Expected GPKG file, got: {config.train_data_path}"
        assert config.column_id == 'cora_id', f"Expected column_id='cora_id', got: {config.column_id}"
        assert len(config.classes) == 5, f"Expected 5 classes, got: {len(config.classes)}"
        assert config.orbits == ['D139'], f"Expected orbits=['D139'], got: {config.orbits}"
        assert config.dates == ['20230606'], f"Expected dates=['20230606'], got: {config.dates}"
        assert config.num_workers == 4, f"Expected num_workers=4, got: {config.num_workers}"
        assert config.max_memory_mb == 2048, f"Expected max_memory_mb=2048, got: {config.max_memory_mb}"
        assert config.output_format == 'npz', f"Expected output_format='npz', got: {config.output_format}"
        
        logger.info("✓ All configuration validation tests passed!")
        
        # Test path validation
        train_path = Path(config.train_data_path)
        if train_path.exists():
            logger.info(f"✓ Training data file exists: {train_path}")
        else:
            logger.warning(f"✗ Training data file not found: {train_path}")
        
        # Test satellite file discovery
        sat_files = config.get_file_paths()
        logger.info(f"✓ Found {len(sat_files)} satellite data files")
        
        if sat_files:
            # Check a few files exist
            existing_files = [f for f in sat_files[:5] if f.exists()]
            logger.info(f"✓ Verified {len(existing_files)} of first 5 satellite files exist")
        
        logger.info("Configuration validation completed successfully")
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


if __name__ == "__main__":
    main()
