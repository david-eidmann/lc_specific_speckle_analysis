#!/usr/bin/env python3
"""
Create comprehensive summary of all 24 training configurations
with Overall Accuracy (OA) bar plot and CSV results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_test_accuracy(run_dir):
    """Extract test accuracy from a training run directory."""
    try:
        results_file = run_dir / "latest_results.txt"
        if results_file.exists():
            with open(results_file, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if line.startswith('Test Accuracy:'):
                        return float(line.split(':')[1].strip())
        
        # Fallback to training_summary JSON
        summary_files = list(run_dir.glob("training_summary_*.json"))
        if summary_files:
            with open(summary_files[0], 'r') as f:
                summary = json.load(f)
                return summary.get('test_accuracy', None)
        
        return None
    except Exception as e:
        logger.warning(f"Could not extract accuracy from {run_dir}: {e}")
        return None

def get_config_name_from_hash(config_hash, training_results):
    """Get configuration name from hash using training results."""
    for config_name, result in training_results.get('detailed_results', {}).items():
        if config_name in result.get('stderr', ''):
            if config_hash in result.get('stderr', ''):
                return config_name
    return f"config_{config_hash}"

def main():
    # Project paths
    project_root = Path("/home/davideidmann/code/lc_specific_speckle_analysis")
    training_output_dir = project_root / "data" / "training_output"
    training_results_file = project_root / "training_results" / "training_results.json"
    
    # Load systematic training results
    training_results = {}
    if training_results_file.exists():
        with open(training_results_file, 'r') as f:
            training_results = json.load(f)
        logger.info(f"Loaded systematic training results with {len(training_results.get('detailed_results', {}))} configs")
    
    # Collect all results
    all_results = []
    
    # Get all training run directories
    run_dirs = [d for d in training_output_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    logger.info(f"Found {len(run_dirs)} training run directories")
    
    # Process each run
    for run_dir in sorted(run_dirs):
        config_hash = run_dir.name.replace('run_', '')
        test_accuracy = extract_test_accuracy(run_dir)
        
        if test_accuracy is not None:
            # Try to get config name from systematic results
            config_name = None
            for name, result in training_results.get('detailed_results', {}).items():
                if config_hash in result.get('stderr', ''):
                    config_name = name
                    break
            
            if config_name is None:
                # Check if it's our manual config_normalized_quantiles run
                if config_hash == '99ad6038':
                    config_name = 'config_normalized_quantiles'
                else:
                    config_name = f'config_{config_hash}'
            
            # Parse configuration parameters from name
            params = {
                'shuffled': 'shuffled' in config_name,
                'zero_mean': 'zeromean' in config_name,
                'normalized': 'normalized' in config_name,
                'quantiles': 'quantiles' in config_name,
                'aggregation': None
            }
            
            # Determine aggregation
            if 'stdandmean' in config_name:
                params['aggregation'] = 'stdandmean'
            elif 'std' in config_name:
                params['aggregation'] = 'std'
            elif 'mean' in config_name:
                params['aggregation'] = 'mean'
            
            # Determine architecture
            architecture = 'linear_stats_net' if params['aggregation'] else 'test_conv2d_n2'
            
            all_results.append({
                'config_name': config_name,
                'config_hash': config_hash,
                'test_accuracy': test_accuracy,
                'shuffled': params['shuffled'],
                'zero_mean': params['zero_mean'],
                'normalized': params['normalized'],
                'quantiles': params['quantiles'],
                'aggregation': params['aggregation'] or 'none',
                'architecture': architecture
            })
    
    # Sort by accuracy (descending)
    all_results.sort(key=lambda x: x['test_accuracy'], reverse=True)
    
    logger.info(f"Collected {len(all_results)} complete training results")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save CSV
    csv_file = project_root / "training_summary_all_configs.csv"
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved CSV results to {csv_file}")
    
    # Create bar plot
    plt.figure(figsize=(16, 10))
    
    # Color by architecture
    colors = ['#2E86AB' if arch == 'test_conv2d_n2' else '#A23B72' for arch in df['architecture']]
    
    bars = plt.bar(range(len(df)), df['test_accuracy'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize plot
    plt.xlabel('Configuration', fontsize=12, fontweight='bold')
    plt.ylabel('Test Accuracy (Overall Accuracy)', fontsize=12, fontweight='bold')
    plt.title('Modular Data Processing: Test Accuracy across All 24 Configurations', fontsize=14, fontweight='bold')
    plt.xticks(range(len(df)), [name.replace('config_', '') for name in df['config_name']], 
               rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, df['test_accuracy'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', alpha=0.8, label='Spatial (test_conv2d_n2)'),
        Patch(facecolor='#A23B72', alpha=0.8, label='Statistical (linear_stats_net)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add statistics text
    stats_text = f"""Statistics:
    Total Configurations: {len(df)}
    Best Accuracy: {df['test_accuracy'].max():.3f} ({df.iloc[0]['config_name']})
    Worst Accuracy: {df['test_accuracy'].min():.3f} ({df.iloc[-1]['config_name']})
    Mean Accuracy: {df['test_accuracy'].mean():.3f} ± {df['test_accuracy'].std():.3f}
    Spatial Configs: {sum(df['architecture'] == 'test_conv2d_n2')}
    Statistical Configs: {sum(df['architecture'] == 'linear_stats_net')}"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = project_root / "training_summary_all_configs_barplot.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved bar plot to {plot_file}")
    
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("MODULAR DATA PROCESSING: COMPREHENSIVE TRAINING RESULTS SUMMARY")
    print("="*80)
    print(f"Total Configurations Trained: {len(df)}")
    print(f"Best Overall Accuracy: {df['test_accuracy'].max():.3f} ({df.iloc[0]['config_name']})")
    print(f"Worst Overall Accuracy: {df['test_accuracy'].min():.3f} ({df.iloc[-1]['config_name']})")
    print(f"Mean ± Std: {df['test_accuracy'].mean():.3f} ± {df['test_accuracy'].std():.3f}")
    
    print(f"\nTop 5 Configurations:")
    for i in range(min(5, len(df))):
        config = df.iloc[i]
        print(f"  {i+1}. {config['config_name']}: {config['test_accuracy']:.3f} ({config['architecture']})")
    
    print(f"\nBottom 5 Configurations:")
    start_idx = max(0, len(df) - 5)
    for i in range(start_idx, len(df)):
        config = df.iloc[i]
        rank = len(df) - i
        print(f"  {rank}. {config['config_name']}: {config['test_accuracy']:.3f} ({config['architecture']})")
    
    print(f"\nArchitecture Performance:")
    spatial_df = df[df['architecture'] == 'test_conv2d_n2']
    stats_df = df[df['architecture'] == 'linear_stats_net']
    
    print(f"  Spatial (test_conv2d_n2): {len(spatial_df)} configs, mean={spatial_df['test_accuracy'].mean():.3f}")
    print(f"  Statistical (linear_stats_net): {len(stats_df)} configs, mean={stats_df['test_accuracy'].mean():.3f}")
    
    print(f"\nFiles Created:")
    print(f"  - CSV: {csv_file}")
    print(f"  - Plot: {plot_file}")
    print("="*80)

if __name__ == "__main__":
    main()
