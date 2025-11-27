"""LC specific speckle analysis package."""

import logging
from pathlib import Path

from .data_config import TrainingDataConfig
from .config import PROJECT_ROOT

__version__ = "0.1.0"

# Configure logging
logger = logging.getLogger(__name__)

# Global configuration instance
training_config: TrainingDataConfig = None

def load_training_config(config_path: Path = None) -> TrainingDataConfig:
    """Load training configuration from file.
    
    Args:
        config_path: Path to config file. Defaults to data/config.conf
        
    Returns:
        TrainingDataConfig instance
    """
    global training_config
    
    if config_path is None:
        config_path = PROJECT_ROOT / "data" / "config.conf"
    
    
    training_config = TrainingDataConfig.from_file(config_path)
    logger.info("Training configuration loaded successfully")
    return training_config
    
def get_training_config() -> TrainingDataConfig:
    """Get the current training configuration.
    
    Returns:
        TrainingDataConfig instance
        
    Raises:
        RuntimeError: If configuration hasn't been loaded
    """
    if training_config is None:
        raise RuntimeError("Training configuration not loaded. Call load_training_config() first.")
    
    return training_config

# Auto-load configuration on import if file exists

default_config_path = PROJECT_ROOT / "data" / "config.conf"
if default_config_path.exists():
    load_training_config()
    logger.info("Auto-loaded training configuration on import")
