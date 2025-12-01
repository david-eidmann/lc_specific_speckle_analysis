#!/usr/bin/env python3
"""
Create visualization plots for configuration combinations results.

This script creates:
1. Heatmap of training times by configuration
2. Bar plot of parameter counts by architecture
3. Training time comparison plots
4. Success/failure grid visualization
"""

import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_results() -> List[Dict]:
    """Load the combination results JSON file."""
    results_path = Path(__file__).parent.parent / "data" / "combination_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Loaded {len(results)} results from {results_path}")
    return results

def extract_training_times(results: List[Dict]) -> Dict:
    """Extract training times from success messages."""
    times = {}
    for result in results:
        if result['success'] and 'Success in' in result['message']:
            # Extract time from message like "Success in 73.0s"
            time_str = result['message'].split('Success in ')[1].replace('s', '')
            times[result['run_id']] = float(time_str)
        else:
            times[result['run_id']] = np.nan
    return times

def create_heatmap_data(results: List[Dict]) -> pd.DataFrame:
    """Create DataFrame for heatmap visualization."""
    # Extract training times
    times = extract_training_times(results)
    
    # Create matrix structure
    data_matrix = []
    architectures = ['test_conv2d', 'test_conv2d_n2']
    
    # Configuration combinations
    configs = [
        ('Zero-mean', 'Single\n(20220611)'),
        ('Raw data', 'Single\n(20220611)'),
        ('Zero-mean', 'Multi\n(20220611,20220623)'),
        ('Raw data', 'Multi\n(20220611,20220623)')
    ]
    
    for arch in architectures:
        arch_times = []
        for result in results:
            if result['architecture'] == arch:
                time_val = times.get(result['run_id'], np.nan)
                arch_times.append(time_val)
        data_matrix.append(arch_times)
    
    # Create column labels
    col_labels = [f"{preprocessing}\n{dates}" for preprocessing, dates in configs]
    
    df = pd.DataFrame(data_matrix, 
                     index=['TestConv2D\n(93,700 params)', 'TestConv2D_N2\n(1,436 params)'],
                     columns=col_labels)
    
    return df

def plot_training_time_heatmap(results: List[Dict], output_dir: Path):
    """Create heatmap of training times."""
    logger.info("Creating training time heatmap...")
    
    df = create_heatmap_data(results)
    
    plt.figure(figsize=(12, 6))
    
    # Create heatmap
    sns.heatmap(df, 
                annot=True, 
                fmt='.1f',
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Training Time (seconds)'},
                linewidths=0.5)
    
    plt.title('Training Time Heatmap: Architecture vs Configuration', fontsize=14, fontweight='bold')
    plt.xlabel('Configuration (Data Processing + Temporal Coverage)', fontsize=12)
    plt.ylabel('Architecture (Parameter Count)', fontsize=12)
    plt.tight_layout()
    
    output_path = output_dir / "training_time_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved heatmap: {output_path}")
    plt.close()

def plot_parameter_comparison(output_dir: Path):
    """Create bar plot of parameter counts by architecture."""
    logger.info("Creating parameter comparison bar plot...")
    
    # Architecture data
    architectures = ['TestConv2D', 'TestConv2D_N2']
    parameters = [93700, 1436]
    colors = ['#FF6B6B', '#4ECDC4']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(architectures, parameters, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, param in zip(bars, parameters):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{param:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add reduction factor annotation
    ax.annotate('65x smaller', 
                xy=(1, 1436), xytext=(0.5, 30000),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=14, fontweight='bold', color='red',
                ha='center')
    
    ax.set_ylabel('Number of Parameters', fontsize=12)
    ax.set_title('Architecture Parameter Comparison', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Format y-axis labels
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()
    
    output_path = output_dir / "parameter_comparison_barplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved parameter comparison: {output_path}")
    plt.close()

def plot_success_failure_grid(results: List[Dict], output_dir: Path):
    """Create grid visualization of success/failure status."""
    logger.info("Creating success/failure grid...")
    
    # Create success matrix (1 = success, 0 = failure, -1 = N/A)
    success_matrix = []
    architectures = ['test_conv2d', 'test_conv2d_n2']
    
    configs = [
        'Zero-mean\nSingle',
        'Raw data\nSingle', 
        'Zero-mean\nMulti',
        'Raw data\nMulti'
    ]
    
    for arch in architectures:
        arch_success = []
        for result in results:
            if result['architecture'] == arch:
                arch_success.append(1 if result['success'] else 0)
        success_matrix.append(arch_success)
    
    df_success = pd.DataFrame(success_matrix,
                             index=['TestConv2D\n(93,700 params)', 'TestConv2D_N2\n(1,436 params)'],
                             columns=configs)
    
    plt.figure(figsize=(10, 5))
    
    # Custom colormap: Red for failure (0), Green for success (1)
    colors = ['#FF4444', '#44AA44']
    cmap = sns.color_palette(colors, as_cmap=True)
    
    sns.heatmap(df_success,
                annot=True,
                fmt='d',
                cmap=cmap,
                cbar_kws={'label': 'Status (0=Failed, 1=Success)'},
                linewidths=2,
                linecolor='white')
    
    plt.title('Training Success Grid: Architecture vs Configuration', fontsize=14, fontweight='bold')
    plt.xlabel('Configuration Type', fontsize=12)
    plt.ylabel('Architecture', fontsize=12)
    plt.tight_layout()
    
    output_path = output_dir / "success_failure_grid.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved success grid: {output_path}")
    plt.close()

def plot_training_time_comparison(results: List[Dict], output_dir: Path):
    """Create comparative bar plot of training times."""
    logger.info("Creating training time comparison plot...")
    
    times = extract_training_times(results)
    
    # Separate by architecture
    conv2d_times = []
    conv2d_n2_times = []
    labels = []
    
    for result in results:
        if result['success']:
            time_val = times[result['run_id']]
            
            # Create short label
            zero_mean = "ZM" if result['data_with_zero_mean'] else "Raw"
            dates = "Single" if "," not in result['dates'] else "Multi"
            label = f"{zero_mean}\n{dates}"
            
            if result['architecture'] == 'test_conv2d':
                conv2d_times.append(time_val)
                if label not in labels:
                    labels.append(label)
            else:
                conv2d_n2_times.append(time_val)
    
    # Create grouped bar plot
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, conv2d_times, width, 
                   label='TestConv2D (93,700 params)', 
                   color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, conv2d_n2_times, width,
                   label='TestConv2D_N2 (1,436 params)', 
                   color='#4ECDC4', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{height:.1f}s', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Training Time Comparison by Architecture and Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = output_dir / "training_time_comparison_barplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved training time comparison: {output_path}")
    plt.close()

def plot_efficiency_analysis(results: List[Dict], output_dir: Path):
    """Create efficiency analysis plot (parameters vs training time)."""
    logger.info("Creating efficiency analysis plot...")
    
    times = extract_training_times(results)
    
    # Data for scatter plot
    param_counts = []
    training_times = []
    arch_labels = []
    config_labels = []
    
    arch_params = {'test_conv2d': 93700, 'test_conv2d_n2': 1436}
    
    for result in results:
        if result['success']:
            param_counts.append(arch_params[result['architecture']])
            training_times.append(times[result['run_id']])
            arch_labels.append('TestConv2D' if result['architecture'] == 'test_conv2d' else 'TestConv2D_N2')
            
            # Configuration description
            zero_mean = "Zero-mean" if result['data_with_zero_mean'] else "Raw data"
            dates = "Single" if "," not in result['dates'] else "Multi-date"
            config_labels.append(f"{zero_mean}, {dates}")
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with different colors for architectures
    conv2d_mask = np.array(arch_labels) == 'TestConv2D'
    conv2d_n2_mask = np.array(arch_labels) == 'TestConv2D_N2'
    
    plt.scatter(np.array(param_counts)[conv2d_mask], 
               np.array(training_times)[conv2d_mask],
               c='#FF6B6B', s=150, alpha=0.8, 
               label='TestConv2D (93,700 params)', edgecolor='black', linewidth=1.5)
    
    plt.scatter(np.array(param_counts)[conv2d_n2_mask], 
               np.array(training_times)[conv2d_n2_mask],
               c='#4ECDC4', s=150, alpha=0.8, 
               label='TestConv2D_N2 (1,436 params)', edgecolor='black', linewidth=1.5)
    
    # Add annotations for each point
    for i, (params, time, config) in enumerate(zip(param_counts, training_times, config_labels)):
        plt.annotate(config, (params, time), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.xlabel('Number of Parameters', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.title('Architecture Efficiency: Parameters vs Training Time', fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / "efficiency_analysis_scatterplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved efficiency analysis: {output_path}")
    plt.close()

def main():
    """Main function to create all visualization plots."""
    logger.info("Starting results visualization script...")
    
    # Load results
    try:
        results = load_results()
    except FileNotFoundError as e:
        logger.error(f"Could not load results: {e}")
        return 1
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "results" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Set plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    try:
        # Create all plots
        plot_training_time_heatmap(results, output_dir)
        plot_parameter_comparison(output_dir)
        plot_success_failure_grid(results, output_dir)
        plot_training_time_comparison(results, output_dir)
        plot_efficiency_analysis(results, output_dir)
        
        logger.info("All plots created successfully!")
        logger.info(f"Plots saved in: {output_dir}")
        
        # List created files
        plot_files = list(output_dir.glob("*.png"))
        logger.info(f"Created {len(plot_files)} plot files:")
        for plot_file in sorted(plot_files):
            logger.info(f"  - {plot_file.name}")
            
        return 0
        
    except Exception as e:
        logger.error(f"Error creating plots: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
