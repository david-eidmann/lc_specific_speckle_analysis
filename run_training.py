#!/usr/bin/env python3
"""
Entry point script for training the TestFlat2Layers model.
This script runs the training module from the project root.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.lc_speckle_analysis.train_model import main

if __name__ == "__main__":
    main()
