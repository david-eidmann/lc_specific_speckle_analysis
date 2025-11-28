#!/usr/bin/env python3
"""Example analysis script."""

import logging
from pathlib import Path
import sys

# Add src to path to import our package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lc_speckle_analysis.analysis import SpeckleAnalyzer
from lc_speckle_analysis.config import logger, RAW_DATA_DIR, RESULTS_DIR


def main():
    """Run example analysis."""
    logger.info("Starting example analysis")
    
    # Initialize analyzer
    analyzer = SpeckleAnalyzer(data_path=RAW_DATA_DIR)
    
    # Example usage (you'll need actual data files)
    logger.info(f"Looking for data in: {RAW_DATA_DIR}")
    logger.info(f"Results will be saved to: {RESULTS_DIR}")
    
    # Add your analysis code here
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
