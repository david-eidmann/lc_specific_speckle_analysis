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
echo "Master script PID: $$"
echo "To kill all training: pkill -P $$ -f train_model"

# Function to train a single config
train_config() {
    config_file="$1"
    config_name=$(basename "$config_file" .conf)
    log_file="logs/training_${config_name}.log"
    
    echo "Starting training: $config_name at $(date)" >> "$log_file"
    echo "Starting training: $config_name (PID will be logged)" 
    
    # Run training with poetry on GPU and redirect output to log file
    cd /home/davideidmann/code/lc_specific_speckle_analysis && PYTHONPATH=src poetry run python -m lc_speckle_analysis.train_model --config "$config_file" --run-id "parallel_training" >> "$log_file" 2>&1 &
    training_pid=$!
    echo "Training $config_name started with PID: $training_pid"
    
    # Wait for the process to complete and check result
    if wait $training_pid; then
        echo "Completed: $config_name at $(date)" >> "$log_file"
        echo "✓ $config_name completed (PID: $training_pid)"
    else
        echo "Failed: $config_name at $(date)" >> "$log_file"
        echo "✗ $config_name failed (PID: $training_pid) - check $log_file"
    fi
}

# Export the function so xargs can use it
export -f train_config

# Run training in parallel with 4 jobs at once (Tesla V100 GPU can handle it)
# -P 4: run 4 parallel processes on GPU
# -I {}: replace {} with input line
cat configs/generated_configs_list.txt | xargs -P 4 -I {} bash -c 'train_config "{}"'

echo "All parallel training completed at $(date)"
echo "Check logs/ directory for individual training logs"
