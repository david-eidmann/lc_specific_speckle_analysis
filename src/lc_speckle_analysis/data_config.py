"""Data configuration parser for LC speckle analysis project."""

import configparser
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NeuralNetworkConfig:
    """Configuration for neural network parameters."""
    patch_size: int
    batch_size: int
    network_architecture_id: str
    dropout_rate: float
    activation_function: str
    optimizer: str
    layer_sizes: List[int]


@dataclass
class TrainingDataConfig:
    """Configuration for training data."""
    
    train_data_path: str
    column_id: str
    classes: List[int]
    orbits: List[str]
    dates: List[str]
    file_pattern: str
    num_workers: int
    max_memory_mb: int
    output_format: str
    neural_network: NeuralNetworkConfig

    @classmethod
    def from_file(cls, config_path: Path) -> "TrainingDataConfig":
        """Parse training data configuration from .conf file.
        
        Args:
            config_path: Path to the .conf configuration file
            
        Returns:
            TrainingDataConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file format is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if not config_path.suffix.lower() in ['.conf', '.ini']:
            raise ValueError(f"Configuration file must be .conf or .ini format: {config_path}")
        
        logger.info(f"Reading configuration from {config_path}")
        
        try:
            config = configparser.ConfigParser()
            config.read(config_path)
            
            # Parse training data section
            if 'training_data' not in config:
                raise ValueError("Missing [training_data] section in configuration")
            
            train_section = config['training_data']
            train_data_path = train_section.get('path')
            column_id = train_section.get('column_id')
            classes_str = train_section.get('classes')
            
            if not all([train_data_path, column_id, classes_str]):
                raise ValueError("Missing required fields in [training_data] section")
            
            classes = [int(x.strip()) for x in classes_str.split(',')]
            logger.info(f"Training data path: {train_data_path}")
            logger.info(f"Column ID: {column_id}")
            logger.info(f"Classes: {classes}")
            
            # Parse satellite data section
            if 'satellite_data' not in config:
                raise ValueError("Missing [satellite_data] section in configuration")
            
            sat_section = config['satellite_data']
            orbits_str = sat_section.get('orbits')
            dates_str = sat_section.get('dates')
            file_pattern = sat_section.get('file_pattern')
            
            if not all([orbits_str, dates_str, file_pattern]):
                raise ValueError("Missing required fields in [satellite_data] section")
            
            orbits = [x.strip() for x in orbits_str.split(',')]
            dates = [x.strip() for x in dates_str.split(',')]
            logger.info(f"Orbits: {orbits}")
            logger.info(f"Dates: {dates}")
            logger.info(f"File pattern: {file_pattern}")
            
            # Parse processing section (with defaults)
            if 'processing' in config:
                processing_section = config['processing']
                num_workers = processing_section.getint('num_workers', 4)
                max_memory_mb = processing_section.getint('max_memory_mb', 2048)
                output_format = processing_section.get('output_format', 'npz')
            else:
                # Use defaults if processing section is missing
                num_workers = 4
                max_memory_mb = 2048
                output_format = 'npz'
            
            logger.info(f"Processing config - Workers: {num_workers}, Memory: {max_memory_mb}MB, Format: {output_format}")
            
            # Parse neural network section
            if 'neural_network' not in config:
                raise ValueError("Missing [neural_network] section in configuration")
            
            nn_section = config['neural_network']
            patch_size = nn_section.getint('patch_size')
            batch_size = nn_section.getint('batch_size') 
            network_architecture_id = nn_section.get('network_architecture_id')
            dropout_rate = nn_section.getfloat('dropout_rate')
            activation_function = nn_section.get('activation_function')
            optimizer = nn_section.get('optimizer')
            layer_sizes_str = nn_section.get('layer_sizes')
            
            if not all([patch_size, batch_size, network_architecture_id, dropout_rate is not None, 
                       activation_function, optimizer, layer_sizes_str]):
                raise ValueError("Missing required fields in [neural_network] section")
            
            layer_sizes = [int(x.strip()) for x in layer_sizes_str.split(',')]
            
            neural_network_config = NeuralNetworkConfig(
                patch_size=patch_size,
                batch_size=batch_size,
                network_architecture_id=network_architecture_id,
                dropout_rate=dropout_rate,
                activation_function=activation_function,
                optimizer=optimizer,
                layer_sizes=layer_sizes
            )
            
            logger.info(f"Neural network config - Patch size: {patch_size}, Batch size: {batch_size}, Architecture: {network_architecture_id}")
            
            return cls(
                train_data_path=train_data_path,
                column_id=column_id,
                classes=classes,
                orbits=orbits,
                dates=dates,
                file_pattern=file_pattern,
                num_workers=num_workers,
                max_memory_mb=max_memory_mb,
                output_format=output_format,
                neural_network=neural_network_config
            )
            
        except Exception as e:
            logger.error(f"Failed to parse configuration file: {e}")
            raise ValueError(f"Invalid configuration file format: {e}")

    def get_file_paths(self) -> List[Path]:
        """Get list of actual file paths based on the pattern and orbits.
        
        Returns:
            List of Path objects for satellite data files
        """
        file_paths = []
        
        for orbit in self.orbits:
            pattern_path = self.file_pattern.replace("{orbit}", orbit)
            base_path = Path(pattern_path).parent
            
            if base_path.exists():
                # Find all .tif files in the directory
                tif_files = list(base_path.glob("*.tif"))
                file_paths.extend(tif_files)
                logger.info(f"Found {len(tif_files)} files for orbit {orbit}")
            else:
                logger.warning(f"Path does not exist for orbit {orbit}: {base_path}")
        
        return file_paths

    def validate_paths(self) -> bool:
        """Validate that all configured paths exist.
        
        Returns:
            True if all paths are valid, False otherwise
        """
        valid = True
        
        # Check training data path
        train_path = Path(self.train_data_path)
        if not train_path.exists():
            logger.error(f"Training data path does not exist: {train_path}")
            valid = False
        
        # Check satellite data paths
        file_paths = self.get_file_paths()
        if not file_paths:
            logger.error("No satellite data files found")
            valid = False
        
        return valid
