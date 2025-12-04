#!/bin/bash
#
# Train all generated configurations using nohup and poetry
# This script runs in the background and logs all output
#

set -e

# Change to project directory
cd /home/davideidmann/code/lc_specific_speckle_analysis

# Check if config list exists
if [ ! -f "configs/generated_configs_list.txt" ]; then
    echo "Error: configs/generated_configs_list.txt not found"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Get total number of configs
total_configs=$(wc -l < configs/generated_configs_list.txt)
echo "Starting training for $total_configs configurations at $(date)"

# Counter for progress
counter=0

# Read each config file and train
while IFS= read -r config_file; do
    counter=$((counter + 1))
    
    # Extract config name for logging
    config_name=$(basename "$config_file" .conf)
    log_file="logs/training_${config_name}.log"
    
    echo "[$counter/$total_configs] Training $config_name at $(date)"
    echo "  Config: $config_file"
    echo "  Log: $log_file"
    
    # Run training with poetry and redirect output to log file
    poetry run python -m src.lc_speckle_analysis.train_model "$config_file" > "$log_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Completed successfully"
    else
        echo "  ✗ Failed - check $log_file"
    fi
    
    echo ""
    
done < configs/generated_configs_list.txt

echo "All training completed at $(date)"
echo "Check logs/ directory for individual training logs"
