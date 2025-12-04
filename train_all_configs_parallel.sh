#!/bin/bash
#
# Train all generated configurations in parallel using CPU
# Uses xargs to manage parallel processes without overwhelming GPU
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
echo "Starting parallel training for $total_configs configurations at $(date)"

# Function to train a single config
train_config() {
    config_file="$1"
    config_name=$(basename "$config_file" .conf)
    log_file="logs/training_${config_name}.log"
    
    echo "Starting training: $config_name at $(date)" >> "$log_file"
    
    # Run training with poetry, forcing CPU usage and redirect output to log file
    if CUDA_VISIBLE_DEVICES="" poetry run python -m src.lc_speckle_analysis.train_model --config "$config_file" >> "$log_file" 2>&1; then
        echo "Completed: $config_name at $(date)" >> "$log_file"
        echo "✓ $config_name completed"
    else
        echo "Failed: $config_name at $(date)" >> "$log_file"
        echo "✗ $config_name failed - check $log_file"
    fi
}

# Export the function so xargs can use it
export -f train_config

# Run training in parallel with 4 jobs at once (adjust -P value as needed)
# -P 4: run 4 parallel processes
# -I {}: replace {} with input line
cat configs/generated_configs_list.txt | xargs -P 4 -I {} bash -c 'train_config "{}"'

echo "All parallel training completed at $(date)"
echo "Check logs/ directory for individual training logs"
