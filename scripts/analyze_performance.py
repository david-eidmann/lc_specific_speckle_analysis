#!/usr/bin/env python3
"""
Extract and visualize test performance metrics (OA and F1 scores) from all training outputs.

This script:
1. Scans all training output directories
2. Extracts test accuracy (OA) and F1 scores
3. Creates performance comparison plots across configurations
4. Generates heatmaps and bar plots focused on model performance
"""

import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_performance_metrics() -> pd.DataFrame:
    """Extract performance metrics from all training output directories."""
    logger.info("Scanning training output directories for performance metrics...")
    
    training_dir = Path(__file__).parent.parent / "data" / "training_output"
    
    if not training_dir.exists():
        raise FileNotFoundError(f"Training output directory not found: {training_dir}")
    
    results = []
    
    for run_dir in training_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
            
        logger.info(f"Processing {run_dir.name}")
        
        # Find training summary JSON
        summary_files = list(run_dir.glob("training_summary_*.json"))
        if not summary_files:
            logger.warning(f"No training summary found in {run_dir.name}")
            continue
            
        summary_file = summary_files[0]  # Take the first/latest one
        
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
            
            # Extract run configuration from directory name
            run_name = run_dir.name.replace("run_", "").split("_")
            
            # Parse configuration from run name
            shuffle_labels = "shuffled" in run_name[0]
            data_processing = run_name[1]  # 'zeromean' or 'raw'
            temporal = run_name[2]  # 'single' or 'multi'
            architecture = run_name[-1].split("_")[0]  # 'conv2d' or 'conv2d_n2'
            
            # Extract dates
            if temporal == "single":
                dates = [item for item in run_name if len(item) == 8 and item.isdigit()]
                date_str = dates[0] if dates else "unknown"
            else:
                dates = [item for item in run_name if len(item) == 8 and item.isdigit()]
                date_str = ",".join(dates[:2]) if len(dates) >= 2 else "unknown"
            
            # Extract performance metrics
            test_results = data.get("test_results", {})
            overall_accuracy = test_results.get("test_accuracy", 0.0)
            
            # Extract per-class F1 scores
            per_class_metrics = test_results.get("per_class_metrics", {})
            class_f1_scores = {}
            for class_id, metrics in per_class_metrics.items():
                class_f1_scores[f"f1_class_{class_id}"] = metrics.get("f1_score", 0.0)
            
            # Macro-averaged F1
            macro_avg = test_results.get("macro_avg", {})
            macro_f1 = macro_avg.get("f1-score", 0.0)
            
            # Training info
            training_results = data.get("training_results", {})
            best_val_acc = training_results.get("best_validation_accuracy", 0.0)
            epochs_completed = training_results.get("epochs_completed", 0)
            
            # Configuration info
            config = data.get("configuration", {})
            actual_architecture = config.get("architecture", architecture)
            
            result = {
                "run_id": run_dir.name,
                "architecture": actual_architecture,
                "data_processing": data_processing,
                "temporal_coverage": temporal,
                "dates": date_str,
                "overall_accuracy": overall_accuracy,
                "macro_f1": macro_f1,
                "best_val_accuracy": best_val_acc,
                "epochs_completed": epochs_completed,
                **class_f1_scores
            }
            
            results.append(result)
            logger.info(f"  ✓ OA: {overall_accuracy:.3f}, Macro F1: {macro_f1:.3f}")
            
        except Exception as e:
            logger.error(f"Error processing {run_dir.name}: {e}")
            continue
    
    if not results:
        raise ValueError("No performance metrics found in any training output directory")
    
    df = pd.DataFrame(results)
    logger.info(f"Extracted metrics from {len(df)} training runs")
    return df

def create_performance_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create heatmap of Overall Accuracy by configuration."""
    logger.info("Creating Overall Accuracy heatmap...")
    
    # Create pivot table for heatmap
    # Rows: Architecture, Columns: Configuration
    pivot_data = []
    
    architectures = ["test_conv2d", "test_conv2d_n2"]
    arch_labels = ["TestConv2D\n(93,700 params)", "TestConv2D_N2\n(1,436 params)"]
    
    # Configuration combinations
    configs = []
    config_labels = []
    
    for _, row in df.iterrows():
        config_key = f"{row['data_processing']}_{row['temporal_coverage']}"
        config_label = f"{row['data_processing'].title()}\n{row['temporal_coverage'].title()}"
        
        if config_key not in configs:
            configs.append(config_key)
            config_labels.append(config_label)
    
    # Build matrix
    oa_matrix = []
    for arch in architectures:
        arch_oas = []
        for config in configs:
            # Find matching row
            matching_rows = df[
                (df['architecture'] == arch) & 
                (df['data_processing'] + '_' + df['temporal_coverage'] == config)
            ]
            if len(matching_rows) > 0:
                oa = matching_rows.iloc[0]['overall_accuracy']
                arch_oas.append(oa)
            else:
                arch_oas.append(np.nan)
        oa_matrix.append(arch_oas)
    
    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(oa_matrix, 
                             index=arch_labels,
                             columns=config_labels)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_df, 
                annot=True, 
                fmt='.3f',
                cmap='RdYlGn',
                vmin=0, vmax=1,
                cbar_kws={'label': 'Overall Accuracy (OA)'},
                linewidths=0.5)
    
    plt.title('Overall Accuracy (OA) Heatmap: Architecture vs Configuration', fontsize=14, fontweight='bold')
    plt.xlabel('Configuration (Data Processing + Temporal Coverage)', fontsize=12)
    plt.ylabel('Architecture (Parameter Count)', fontsize=12)
    plt.tight_layout()
    
    output_path = output_dir / "performance_oa_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved OA heatmap: {output_path}")
    plt.close()

def create_f1_comparison_plot(df: pd.DataFrame, output_dir: Path):
    """Create bar plot comparing Macro F1 scores across configurations."""
    logger.info("Creating Macro F1 comparison plot...")
    
    # Prepare data for grouped bar plot
    architectures = ["test_conv2d", "test_conv2d_n2"]
    arch_labels = ["TestConv2D\n(93,700 params)", "TestConv2D_N2\n(1,436 params)"]
    
    # Get unique configurations
    configs = []
    config_labels = []
    for _, row in df.iterrows():
        config_key = f"{row['data_processing']}_{row['temporal_coverage']}"
        config_label = f"{row['data_processing'].title()}\n{row['temporal_coverage'].title()}"
        if config_key not in configs:
            configs.append(config_key)
            config_labels.append(config_label)
    
    # Extract F1 scores for each architecture and configuration
    arch_f1_data = {}
    for arch, arch_label in zip(architectures, arch_labels):
        arch_f1_scores = []
        for config in configs:
            matching_rows = df[
                (df['architecture'] == arch) & 
                (df['data_processing'] + '_' + df['temporal_coverage'] == config)
            ]
            if len(matching_rows) > 0:
                f1 = matching_rows.iloc[0]['macro_f1']
                arch_f1_scores.append(f1)
            else:
                arch_f1_scores.append(0.0)
        arch_f1_data[arch_label] = arch_f1_scores
    
    # Create grouped bar plot
    x = np.arange(len(config_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#FF6B6B', '#4ECDC4']
    bars = []
    
    for i, (arch_label, f1_scores) in enumerate(arch_f1_data.items()):
        bars.append(ax.bar(x + i*width - width/2, f1_scores, width,
                          label=arch_label, color=colors[i], alpha=0.8, edgecolor='black'))
    
    # Add value labels on bars
    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Macro F1 Score', fontsize=12)
    ax.set_title('Macro F1 Score Comparison by Architecture and Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    output_path = output_dir / "performance_macro_f1_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved Macro F1 comparison: {output_path}")
    plt.close()

def create_oa_comparison_plot(df: pd.DataFrame, output_dir: Path):
    """Create bar plot comparing Overall Accuracy across configurations."""
    logger.info("Creating Overall Accuracy comparison plot...")
    
    # Prepare data for grouped bar plot
    architectures = ["test_conv2d", "test_conv2d_n2"]
    arch_labels = ["TestConv2D\n(93,700 params)", "TestConv2D_N2\n(1,436 params)"]
    
    # Get unique configurations
    configs = []
    config_labels = []
    for _, row in df.iterrows():
        config_key = f"{row['data_processing']}_{row['temporal_coverage']}"
        config_label = f"{row['data_processing'].title()}\n{row['temporal_coverage'].title()}"
        if config_key not in configs:
            configs.append(config_key)
            config_labels.append(config_label)
    
    # Extract OA scores for each architecture and configuration
    arch_oa_data = {}
    for arch, arch_label in zip(architectures, arch_labels):
        arch_oa_scores = []
        for config in configs:
            matching_rows = df[
                (df['architecture'] == arch) & 
                (df['data_processing'] + '_' + df['temporal_coverage'] == config)
            ]
            if len(matching_rows) > 0:
                oa = matching_rows.iloc[0]['overall_accuracy']
                arch_oa_scores.append(oa)
            else:
                arch_oa_scores.append(0.0)
        arch_oa_data[arch_label] = arch_oa_scores
    
    # Create grouped bar plot
    x = np.arange(len(config_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#FF6B6B', '#4ECDC4']
    bars = []
    
    for i, (arch_label, oa_scores) in enumerate(arch_oa_data.items()):
        bars.append(ax.bar(x + i*width - width/2, oa_scores, width,
                          label=arch_label, color=colors[i], alpha=0.8, edgecolor='black'))
    
    # Add value labels on bars
    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Overall Accuracy (OA)', fontsize=12)
    ax.set_title('Overall Accuracy (OA) Comparison by Architecture and Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    output_path = output_dir / "performance_oa_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved OA comparison: {output_path}")
    plt.close()

def create_per_class_f1_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create heatmap of per-class F1 scores."""
    logger.info("Creating per-class F1 heatmap...")
    
    # Get available classes
    f1_columns = [col for col in df.columns if col.startswith('f1_class_')]
    classes = sorted([col.split('_')[-1] for col in f1_columns])
    
    if not classes:
        logger.warning("No per-class F1 scores found, skipping per-class heatmap")
        return
    
    # Create subplot for each architecture
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Per-Class F1 Scores by Architecture and Configuration', fontsize=16, fontweight='bold')
    
    architectures = ["test_conv2d", "test_conv2d_n2"]
    arch_titles = ["TestConv2D (93,700 params)", "TestConv2D_N2 (1,436 params)"]
    
    for arch_idx, (arch, arch_title) in enumerate(zip(architectures, arch_titles)):
        arch_data = df[df['architecture'] == arch]
        
        # Build matrix: rows = configurations, columns = classes
        config_labels = []
        class_f1_matrix = []
        
        for _, row in arch_data.iterrows():
            config_label = f"{row['data_processing'].title()}\n{row['temporal_coverage'].title()}"
            config_labels.append(config_label)
            
            class_f1_row = []
            for class_id in classes:
                f1_col = f'f1_class_{class_id}'
                f1_score = row.get(f1_col, 0.0)
                class_f1_row.append(f1_score)
            class_f1_matrix.append(class_f1_row)
        
        # Create heatmap
        heatmap_df = pd.DataFrame(class_f1_matrix, 
                                 index=config_labels,
                                 columns=[f'Class {c}' for c in classes])
        
        sns.heatmap(heatmap_df, 
                    annot=True, 
                    fmt='.3f',
                    cmap='RdYlGn',
                    vmin=0, vmax=1,
                    ax=axes[arch_idx],
                    cbar=(arch_idx == 1),
                    cbar_kws={'label': 'F1 Score'} if arch_idx == 1 else None)
        
        axes[arch_idx].set_title(arch_title, fontsize=12, fontweight='bold')
        axes[arch_idx].set_xlabel('Class', fontsize=10)
        axes[arch_idx].set_ylabel('Configuration' if arch_idx == 0 else '', fontsize=10)
    
    plt.tight_layout()
    
    output_path = output_dir / "performance_per_class_f1_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved per-class F1 heatmap: {output_path}")
    plt.close()

def create_performance_summary_table(df: pd.DataFrame, output_dir: Path):
    """Create a summary table with all performance metrics."""
    logger.info("Creating performance summary table...")
    
    # Prepare summary data
    summary_rows = []
    
    for _, row in df.iterrows():
        arch_name = "TestConv2D" if row['architecture'] == 'test_conv2d' else "TestConv2D_N2"
        param_count = "93,700" if row['architecture'] == 'test_conv2d' else "1,436"
        config_desc = f"{row['data_processing'].title()} + {row['temporal_coverage'].title()}"
        
        summary_rows.append({
            'Architecture': f"{arch_name}\n({param_count} params)",
            'Configuration': config_desc,
            'Overall Accuracy': f"{row['overall_accuracy']:.3f}",
            'Macro F1': f"{row['macro_f1']:.3f}",
            'Best Val Acc': f"{row['best_val_accuracy']:.3f}",
            'Epochs': row['epochs_completed']
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(color='white')
        else:
            if j % 2 == 0:
                cell.set_facecolor('#F5F5F5')
            else:
                cell.set_facecolor('white')
    
    plt.title('Performance Summary: All Architecture and Configuration Combinations', 
              fontsize=16, fontweight='bold', pad=20)
    
    output_path = output_dir / "performance_summary_table.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved performance summary table: {output_path}")
    plt.close()
    
    # Also save as CSV
    csv_path = output_dir / "performance_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Saved performance summary CSV: {csv_path}")

def main():
    """Main function to extract and visualize performance metrics."""
    logger.info("Starting performance analysis script...")
    
    try:
        # Extract performance metrics
        df = extract_performance_metrics()
        
        # Create output directory
        output_dir = Path(__file__).parent.parent / "results" / "performance_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Create all performance plots
        create_performance_heatmap(df, output_dir)
        create_f1_comparison_plot(df, output_dir)
        create_oa_comparison_plot(df, output_dir)
        create_per_class_f1_heatmap(df, output_dir)
        create_performance_summary_table(df, output_dir)
        
        logger.info("All performance plots created successfully!")
        
        # Print summary to console
        logger.info("\n" + "="*50)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*50)
        
        for _, row in df.iterrows():
            arch = "TestConv2D" if row['architecture'] == 'test_conv2d' else "TestConv2D_N2"
            config = f"{row['data_processing']}-{row['temporal_coverage']}"
            logger.info(f"{arch:12} | {config:15} | OA: {row['overall_accuracy']:.3f} | F1: {row['macro_f1']:.3f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in performance analysis: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
