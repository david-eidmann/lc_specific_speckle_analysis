#!/usr/bin/env python3
"""
Generate all 24 configuration combinations for LC speckle analysis.

This script creates all possible combinations of the modular data processing
parameters and writes config files with unique names and hash validation.
"""

import itertools
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ConfigParameters:
    """Data processing configuration parameters."""
    shuffled: bool
    zero_mean: bool
    normalized: bool
    quantiles: bool
    aggregation: Optional[str]

    def get_unique_name(self) -> str:
        """Generate unique name based on parameters."""
        parts = []
        
        # Base name
        if not any([self.shuffled, self.zero_mean, self.normalized, self.quantiles]):
            parts.append("base")
        else:
            if self.shuffled:
                parts.append("shuffled")
            if self.zero_mean:
                parts.append("zeromean")
            if self.normalized:
                parts.append("normalized")
            if self.quantiles:
                parts.append("quantiles")
        
        # Aggregation suffix
        if self.aggregation:
            parts.append(self.aggregation)
        
        return "_".join(parts)

    def get_description(self) -> str:
        """Generate human-readable description."""
        desc_parts = []
        
        if self.shuffled:
            desc_parts.append("spatially shuffled labels")
        if self.zero_mean:
            desc_parts.append("zero-mean normalization")
        if self.normalized:
            desc_parts.append("unit variance normalization")
        if self.quantiles:
            desc_parts.append("quantile transformation")
        if self.aggregation:
            desc_parts.append(f"{self.aggregation} aggregation")
        
        if not desc_parts:
            return "raw data processing"
        
        return " + ".join(desc_parts)

def generate_all_combinations() -> List[ConfigParameters]:
    """Generate all valid parameter combinations."""
    combinations = []
    
    #base_options
    zero_mean_options = [False, True]
    normalized_options = [False, True]
    # Define parameter options
    shuffled_options = [False, True]
    quantiles_options = [False, True]
    aggregation_options = [None, "mean", "std", "stdandmean"]
    
    # Generate all combinations
    for shuffled, zero_mean, normalized, quantiles, aggregation in itertools.product(
        shuffled_options, zero_mean_options, normalized_options, quantiles_options, aggregation_options
    ):
        nmod=sum([shuffled, quantiles, aggregation is not None])
        if nmod>1:
            continue  # Skip combinations with more than one modification
        config = ConfigParameters(
            shuffled=shuffled,
            zero_mean=zero_mean,
            normalized=normalized,
            quantiles=quantiles,
            aggregation=aggregation
        )
        combinations.append(config)
    
    logger.info(f"Generated {len(combinations)} total combinations")
    return combinations

def write_config_file(config: ConfigParameters, output_dir: Path) -> Path:
    """Write a single configuration file."""
    unique_name = config.get_unique_name()
    config_file = output_dir / f"config_{unique_name}.conf"
    
    # Determine architecture based on aggregation
    architecture = "linear_stats_net" if config.aggregation else "test_conv2d_n2"
    
    # Base template
    config_content = f"""# {config.get_description().title()} configuration
# New modular data processing format

[training_data]
# Unique identifier for this configuration (raises error if name exists with different hash)
unique_name = {unique_name}

# Path to the training dataset (GPKG file)
paths = /mnt/ssddata/PbNNMod/label_preparation/classification/crop_classification/cldef_5/Niedersachsen_2022/Niedersachsen_2022_InvekosDataset_c823cf0c.gpkg,/mnt/ssddata/PbNNMod/label_preparation/classification/crop_classification/cldef_5/NRW_2022/NRW_2022_InvekosDataset_*.gpkg

# Column name containing the classification IDs
column_id = cora_id

# Comma-separated list of classification classes to use
classes = 1,4,6,12

# Balance class distribution by subsampling to minimum class count
equal_class_dist = true

# Randomly shuffle patch labels after loading and balancing (for data leak testing and random baseline)
shuffle_labels = false

[data_processing]
# {config.get_description()}
# Applied in order: shuffle → zero_mean → normalize → quantiles → aggregation
shuffled = {str(config.shuffled).lower()}
zero_mean = {str(config.zero_mean).lower()}
normalized = {str(config.normalized).lower()}
quantiles = {str(config.quantiles).lower()}
aggregation = {config.aggregation if config.aggregation else ""}

[satellite_data]
# Sentinel-1 orbit identifiers (comma-separated)
orbits = D139

# Dates to process (YYYYMMDD format, comma-separated)
dates = 20220611

# File pattern for satellite data with {{orbit}} placeholder
file_pattern = /mnt/cephfs/data/CorDAu/S1/download/preproc_1/data/UTMZ_32N/2022/{{orbit}}/*.tif

[processing]
# Number of worker processes for data loading
num_workers = 4

# Maximum memory usage in MB
max_memory_mb = 2048

# Output format for processed data
output_format = npz

[patch_extraction]
# Number of patches to extract per feature
n_patches_per_feature = 50

# Area ratio factor for patch extraction
n_patches_per_area = 1.0

[neural_network]
# 10*10 pixel input patches - architecture depends on aggregation setting
patch_size = 10
batch_size = 32
network_architecture_id = {architecture}
dropout_rate = 0.2
activation_function = relu
optimizer = adam
layer_sizes = 64,32,16
n_epochs = 50
early_stopping_patience = 10

# Early stopping patience (epochs without improvement)
early_stopping_patience = 10
"""
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    logger.info(f"Created config: {config_file.name}")
    return config_file

def validate_unique_names_and_hashes(configs: List[ConfigParameters]) -> Dict[str, str]:
    """Validate unique names and get hashes for all configurations."""
    from src.lc_speckle_analysis.data_config import TrainingDataConfig
    
    name_hash_mapping = {}
    conflicts = []
    
    logger.info("Validating unique names and checking for hash conflicts...")
    
    for config in configs:
        unique_name = config.get_unique_name()
        
        # Create temporary config file to get hash
        temp_dir = Path("temp_configs")
        temp_dir.mkdir(exist_ok=True)
        try:
            write_config_file(config, temp_dir)
            temp_file = temp_dir / f"config_{unique_name}.conf"
            
            # Load config and get hash
            training_config = TrainingDataConfig.from_file(temp_file)
            config_hash = training_config.get_config_hash()
            
            # Check for conflicts
            if unique_name in name_hash_mapping:
                existing_hash = name_hash_mapping[unique_name]
                if existing_hash != config_hash:
                    conflicts.append(f"CONFLICT: '{unique_name}' has multiple hashes: {existing_hash} vs {config_hash}")
                else:
                    logger.debug(f"Duplicate name '{unique_name}' with same hash: {config_hash}")
            else:
                name_hash_mapping[unique_name] = config_hash
                logger.debug(f"Validated: {unique_name} → {config_hash}")
            
            # Clean up temp file
            temp_file.unlink()
            
        except Exception as e:
            logger.error(f"Error validating config '{unique_name}': {e}")
            conflicts.append(f"ERROR: Could not validate '{unique_name}': {e}")
    
    # Clean up temp directory
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    if conflicts:
        logger.error("Conflicts detected:")
        for conflict in conflicts:
            logger.error(f"  {conflict}")
        raise ValueError(f"Found {len(conflicts)} configuration conflicts")
    
    logger.info(f"All {len(name_hash_mapping)} configurations validated successfully")
    return name_hash_mapping

def write_summary_file(configs: List[ConfigParameters], name_hash_mapping: Dict[str, str], 
                      created_files: List[Path], output_dir: Path) -> Path:
    """Write summary file with all created configurations."""
    summary_file = output_dir / "generated_configs_summary.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("LC Speckle Analysis - Generated Configuration Files\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated {len(configs)} configuration files\n")
        f.write(f"Output directory: {output_dir.absolute()}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Configuration Summary:\n")
        f.write("-" * 20 + "\n")
        
        for config in sorted(configs, key=lambda x: x.get_unique_name()):
            unique_name = config.get_unique_name()
            config_hash = name_hash_mapping.get(unique_name, "UNKNOWN")
            
            f.write(f"\nConfig: {unique_name}\n")
            f.write(f"  File: config_{unique_name}.conf\n")
            f.write(f"  Hash: {config_hash}\n")
            f.write(f"  Description: {config.get_description()}\n")
            f.write(f"  Parameters:\n")
            f.write(f"    shuffled: {config.shuffled}\n")
            f.write(f"    zero_mean: {config.zero_mean}\n")
            f.write(f"    normalized: {config.normalized}\n")
            f.write(f"    quantiles: {config.quantiles}\n")
            f.write(f"    aggregation: {config.aggregation}\n")
            
            # Determine architecture
            architecture = "linear_stats_net" if config.aggregation else "test_conv2d_n2"
            f.write(f"    architecture: {architecture}\n")
        
        f.write(f"\n\nCreated Files ({len(created_files)}):\n")
        f.write("-" * 15 + "\n")
        for file_path in sorted(created_files):
            f.write(f"  {file_path.name}\n")
    
    logger.info(f"Summary written to: {summary_file}")
    return summary_file

def main():
    """Main function to generate all configuration files."""
    
    # Setup
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    logger.info("Starting configuration generation...")
    
    try:
        # Generate all combinations
        configs = generate_all_combinations()
        
        # Skip validation for now - create simple hash mapping
        name_hash_mapping = {config.get_unique_name(): f"hash_{i:04d}" for i, config in enumerate(configs)}
        
        logger.info(f"Creating {len(configs)} configuration files...")
        
        # Write all config files
        created_files = []
        for config in configs:
            config_file = write_config_file(config, configs_dir)
            created_files.append(config_file)
        
        # Write summary
        summary_file = write_summary_file(configs, name_hash_mapping, created_files, configs_dir)
        
        # Write simple list file for easy script usage
        list_file = configs_dir / "generated_configs_list.txt"
        with open(list_file, 'w', encoding='utf-8') as f:
            for config_file in sorted(created_files):
                f.write(f"{config_file}\n")
        logger.info(f"Config list written to: {list_file}")
        
        # Final summary
        logger.info("=" * 50)
        logger.info("Configuration generation completed successfully!")
        logger.info(f"Created {len(created_files)} config files in: {configs_dir}")
        logger.info(f"Summary file: {summary_file}")
        logger.info(f"Config list file: {list_file}")
        logger.info(f"Unique configurations: {len(name_hash_mapping)}")
        
        # Print unique names for reference
        logger.info("\nGenerated configurations:")
        for i, unique_name in enumerate(sorted(name_hash_mapping.keys()), 1):
            config_hash = name_hash_mapping[unique_name]
            logger.info(f"  {i:2d}. {unique_name} ({config_hash})")
        
    except Exception as e:
        logger.error(f"Configuration generation failed: {e}")
        raise

if __name__ == "__main__":
    main()
