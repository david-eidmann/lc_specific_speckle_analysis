#!/usr/bin/env python3
"""
Create a bar plot of F1 scores for a specific training run.

This script creates a bar plot showing per-class F1 scores from the specified training output.
"""

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_f1_barplot_for_run(run_path: str, output_dir: Path):
    """Create F1 score bar plot for a specific training run."""
    
    run_dir = Path(run_path)
    logger.info(f"Creating F1 bar plot for: {run_dir.name}")
    
    # Find training summary JSON
    summary_files = list(run_dir.glob("training_summary_*.json"))
    if not summary_files:
        raise FileNotFoundError(f"No training summary found in {run_dir}")
    
    summary_file = summary_files[0]
    logger.info(f"Reading data from: {summary_file}")
    
    # Load the data
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    # Extract run information
    config = data.get("configuration", {})
    test_results = data.get("test_results", {})
    metadata = data.get("metadata", {})
    
    architecture = config.get("architecture", "unknown")
    # Handle both new 'modus' parameter and legacy 'data_with_zero_mean' boolean
    modus = config.get("modus", None)
    if modus:
        data_processing = "zero-mean" if modus == "data_with_zero_mean" else "raw"
    else:
        # Backward compatibility
        data_processing = "zero-mean" if config.get("data_with_zero_mean", False) else "raw"
    dates = ",".join(config.get("dates", []))
    
    # Extract per-class F1 scores
    per_class_metrics = test_results.get("per_class_metrics", {})
    
    classes = []
    f1_scores = []
    precisions = []
    recalls = []
    
    for class_id in sorted(per_class_metrics.keys(), key=int):
        metrics = per_class_metrics[class_id]
        classes.append(f"Class {class_id}")
        f1_scores.append(metrics.get("f1_score", 0.0))
        precisions.append(metrics.get("precision", 0.0))
        recalls.append(metrics.get("recall", 0.0))
    
    # Get overall metrics
    overall_accuracy = test_results.get("test_accuracy", 0.0)
    macro_f1 = test_results.get("macro_avg", {}).get("f1-score", 0.0)
    
    # Create the bar plot
    plt.figure(figsize=(8, 8))  # 1:1 ratio, square image
    
    x = np.arange(len(classes))
    width = 0.25
    
    # Create bars for F1, Precision, and Recall (blue shades)
    bars_f1 = plt.bar(x - width, f1_scores, width, label='F1 Score', color='#1f77b4', alpha=0.8, edgecolor='black')
    bars_precision = plt.bar(x, precisions, width, label='Precision', color='#2ca02c', alpha=0.8, edgecolor='black')
    bars_recall = plt.bar(x + width, recalls, width, label='Recall', color='#ff7f0e', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars_f1, bars_precision, bars_recall]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Customize the plot
    plt.xlabel('Land Cover Classes', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Per-Class Performance Metrics\n{architecture.upper()} | {data_processing.title()} Data | Date: {dates}', 
              fontsize=14)
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1.0)
    
    # Add overall metrics as text
    metrics_text = f"""Overall Metrics:
    Overall Accuracy: {overall_accuracy:.3f}
    Macro F1 Score: {macro_f1:.3f}"""
    
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Generate output filename
    run_name = run_dir.name.replace("run_", "")
    output_filename = f"f1_scores_{run_name}.png"
    output_path = output_dir / output_filename
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved F1 bar plot: {output_path}")
    plt.close()
    
    # Also create a simple F1-only plot
    plt.figure(figsize=(6, 6))  # 1:1 ratio, square image
    
    bars = plt.bar(classes, f1_scores, color='#1f77b4', 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Land Cover Classes', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title(f'F1 Scores by Class\n{architecture.upper()} | {data_processing.title()} Data | OA: {overall_accuracy:.3f} | Macro F1: {macro_f1:.3f}', 
              fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    
    # Save F1-only plot
    f1_only_filename = f"f1_only_{run_name}.png"
    f1_only_path = output_dir / f1_only_filename
    
    plt.savefig(f1_only_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved F1-only plot: {f1_only_path}")
    plt.close()
    
    # Print summary to console
    logger.info("\n" + "="*60)
    logger.info(f"PERFORMANCE SUMMARY: {run_dir.name}")
    logger.info("="*60)
    logger.info(f"Architecture: {architecture}")
    logger.info(f"Data Processing: {data_processing}")
    logger.info(f"Date(s): {dates}")
    logger.info(f"Overall Accuracy: {overall_accuracy:.3f}")
    logger.info(f"Macro F1 Score: {macro_f1:.3f}")
    logger.info("\nPer-Class Metrics:")
    for i, class_name in enumerate(classes):
        logger.info(f"  {class_name}: F1={f1_scores[i]:.3f}, Precision={precisions[i]:.3f}, Recall={recalls[i]:.3f}")

def main():
    """Main function."""
    
    # Specific training run to analyze - SHUFFLED LABELS RUN
    run_path = "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_shuffled_zeromean_single_20220611_conv2d_n2_dd182e25"
    
    # Output directory
    output_dir = Path(__file__).parent.parent / "results" / "individual_run_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        create_f1_barplot_for_run(run_path, output_dir)
        logger.info(f"\n✅ Plots created successfully in: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Error creating plots: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
