"""Tests for data configuration."""

import unittest
from pathlib import Path
import tempfile
import os

from src.lc_speckle_analysis.data_config import TrainingDataConfig


class TestDataConfig(unittest.TestCase):
    """Test data configuration parsing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config_content = """[training_data]
path = /path/to/test/data.gpkg
column_id = test_id
classes = 1,2,3

[satellite_data]
orbits = D139,A151
dates = 20230606,20230618
file_pattern = /path/to/data/{orbit}/*.tif

[processing]
num_workers = 2
max_memory_mb = 1024
output_format = npz

[neural_network]
patch_size = 8
batch_size = 16
network_architecture_id = test_arch
dropout_rate = 0.1
activation_function = relu
optimizer = adam
layer_sizes = 32,64
"""
    
    def test_config_parsing(self):
        """Test parsing of configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
            f.write(self.test_config_content)
            temp_path = f.name
        
        try:
            config = TrainingDataConfig.from_file(Path(temp_path))
            
            self.assertEqual(config.train_data_path, "/path/to/test/data.gpkg")
            self.assertEqual(config.column_id, "test_id")
            self.assertEqual(config.classes, [1, 2, 3])
            self.assertEqual(config.orbits, ["D139", "A151"])
            self.assertEqual(config.dates, ["20230606", "20230618"])
            self.assertEqual(config.file_pattern, "/path/to/data/{orbit}/*.tif")
            self.assertEqual(config.num_workers, 2)
            self.assertEqual(config.max_memory_mb, 1024)
            self.assertEqual(config.output_format, "npz")
            
            # Test neural network config
            self.assertEqual(config.neural_network.patch_size, 8)
            self.assertEqual(config.neural_network.batch_size, 16)
            self.assertEqual(config.neural_network.network_architecture_id, "test_arch")
            self.assertEqual(config.neural_network.dropout_rate, 0.1)
            self.assertEqual(config.neural_network.activation_function, "relu")
            self.assertEqual(config.neural_network.optimizer, "adam")
            self.assertEqual(config.neural_network.layer_sizes, [32, 64])
            
        finally:
            os.unlink(temp_path)
    
    def test_missing_file(self):
        """Test handling of missing config file."""
        with self.assertRaises(FileNotFoundError):
            TrainingDataConfig.from_file(Path("/nonexistent/path.txt"))
    
    def test_invalid_format(self):
        """Test handling of invalid config format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
            f.write("invalid config content")
            temp_path = f.name
        
        try:
            with self.assertRaises(ValueError):
                TrainingDataConfig.from_file(Path(temp_path))
        finally:
            os.unlink(temp_path)
    
    def test_wrong_file_extension(self):
        """Test rejection of non-.conf files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(self.test_config_content)
            temp_path = f.name
        
        try:
            with self.assertRaises(ValueError):
                TrainingDataConfig.from_file(Path(temp_path))
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
