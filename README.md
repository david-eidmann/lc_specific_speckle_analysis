# LC Specific Speckle Analysis

A Python research project for analyzing LC-specific speckle patterns with Sentinel-1 data.

## Project Structure

```
lc_specific_speckle_analysis/
├── src/
│   └── lc_speckle_analysis/     # Main package
│       ├── __init__.py          # Package init with auto-config loading
│       ├── config.py            # Project paths and logging setup
│       ├── data_config.py       # Training data configuration parser
│       └── analysis.py          # Speckle analysis utilities
├── tests/                       # Unit tests
│   ├── test_basic.py           # Basic package tests
│   └── test_data_config.py     # Configuration tests
├── notebooks/                   # Jupyter notebooks for analysis
│   └── 01_project_setup.ipynb  # Project setup notebook
├── data/
│   ├── config.conf             # Structured configuration file (INI format)
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed data files
├── results/                    # Analysis results and outputs
├── docs/                       # Documentation
├── scripts/                    # Utility scripts
│   ├── config_demo.py          # Configuration system demo
│   └── example_analysis.py     # Example analysis script
└── pyproject.toml              # Project configuration

```

## Configuration

The project uses a structured configuration file format. All parameters are defined in `data/config.conf` using INI format:

### Configuration Sections

**[training_data]**
- **path**: Path to GPKG file with label data
- **column_id**: Column name for classification IDs  
- **classes**: Comma-separated list of classification classes

**[satellite_data]**
- **orbits**: Comma-separated Sentinel-1 orbit identifiers
- **dates**: Comma-separated acquisition dates (YYYYMMDD format)
- **file_pattern**: Template for satellite data file paths (use `{orbit}` placeholder)

**[processing]**
- **num_workers**: Number of parallel processing workers
- **max_memory_mb**: Maximum memory usage per worker
- **output_format**: Output format for processed data

Configuration is automatically loaded on package import and provides structured access to all parameters.

## Installation

1. Clone the repository
2. Install Poetry if you haven't already: `pip install poetry`
3. Install dependencies: `poetry install`
4. Activate the virtual environment: `poetry shell`

## Usage

```bash
# Run tests
poetry run pytest

# Start Jupyter notebook
poetry run jupyter notebook

# Run analysis scripts
poetry run python scripts/your_script.py
```

## Development

This project uses:
- **Poetry** for dependency management
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **pytest** for testing

Run all checks:
```bash
poetry run black src tests
poetry run isort src tests
poetry run flake8 src tests
poetry run mypy src
poetry run pytest
```
