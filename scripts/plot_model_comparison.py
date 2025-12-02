#!/usr/bin/env python3
"""
Create comparison bar plots between shuffled and non-shuffled models.

This script creates:
1. Side-by-side comparison of F1 scores for both models
2. Delta bar plot showing the performance difference
"""

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_results(run_path: str) -> dict:
    """Load performance results from a training run."""
    run_dir = Path(run_path)
    
    # Find training summary JSON
    summary_files = list(run_dir.glob("training_summary_*.json"))
    if not summary_files:
        raise FileNotFoundError(f"No training summary found in {run_dir}")
    
    summary_file = summary_files[0]
    
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    # Extract metrics
    test_results = data.get("test_results", {})
    per_class_metrics = test_results.get("per_class_metrics", {})
    
    # Get per-class F1 scores
    f1_scores = {}
    precisions = {}
    recalls = {}
    
    for class_id in sorted(per_class_metrics.keys(), key=int):
        metrics = per_class_metrics[class_id]
        f1_scores[f"Class {class_id}"] = metrics.get("f1_score", 0.0)
        precisions[f"Class {class_id}"] = metrics.get("precision", 0.0)
        recalls[f"Class {class_id}"] = metrics.get("recall", 0.0)
    
    # Overall metrics
    overall_accuracy = test_results.get("test_accuracy", 0.0)
    macro_f1 = test_results.get("macro_avg", {}).get("f1-score", 0.0)
    
    return {
        'f1_scores': f1_scores,
        'precisions': precisions,
        'recalls': recalls,
        'overall_accuracy': overall_accuracy,
        'macro_f1': macro_f1,
        'run_name': run_dir.name
    }

def create_comparison_plot(normal_results: dict, shuffled_results: dict, output_dir: Path):
    """Create side-by-side comparison plot."""
    logger.info("Creating F1 comparison plot...")
    
    classes = list(normal_results['f1_scores'].keys())
    normal_f1 = list(normal_results['f1_scores'].values())
    shuffled_f1 = list(shuffled_results['f1_scores'].values())
    
    x = np.arange(len(classes))
    width = 0.35
    
    plt.figure(figsize=(8, 8))  # Square format
    
    # Create bars
    bars1 = plt.bar(x - width/2, normal_f1, width, label='Normal Labels', 
                    color='#1f77b4', alpha=0.8, edgecolor='black')
    bars2 = plt.bar(x + width/2, shuffled_f1, width, label='Shuffled Labels', 
                    color='#ff7f0e', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Customize plot
    plt.xlabel('Land Cover Classes', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score Comparison: Normal vs Shuffled Labels\nTestConv2D_N2 | Zero-mean | Single Date', 
              fontsize=14)
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 0.8)
    
    # Add overall metrics text
    metrics_text = f"""Overall Metrics:
Normal:   OA={normal_results['overall_accuracy']:.3f}, F1={normal_results['macro_f1']:.3f}
Shuffled: OA={shuffled_results['overall_accuracy']:.3f}, F1={shuffled_results['macro_f1']:.3f}"""
    
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = output_dir / "f1_comparison_normal_vs_shuffled.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison plot: {output_path}")
    plt.close()

def create_delta_plot(normal_results: dict, shuffled_results: dict, output_dir: Path):
    """Create delta bar plot showing performance differences."""
    logger.info("Creating delta plot...")
    
    classes = list(normal_results['f1_scores'].keys())
    normal_f1 = list(normal_results['f1_scores'].values())
    shuffled_f1 = list(shuffled_results['f1_scores'].values())
    
    # Calculate deltas (Normal - Shuffled)
    deltas = [normal - shuffled for normal, shuffled in zip(normal_f1, shuffled_f1)]
    
    plt.figure(figsize=(8, 8))  # Square format
    
    # Create bars with colors based on delta magnitude
    colors = ['#2ca02c' if delta > 0.3 else '#ff7f0e' if delta > 0.1 else '#d62728' 
              for delta in deltas]
    
    bars = plt.bar(classes, deltas, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.02,
                f'{delta:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=12)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Customize plot
    plt.xlabel('Land Cover Classes', fontsize=12)
    plt.ylabel('F1 Score Difference (Normal - Shuffled)', fontsize=12)
    plt.title('Performance Gain from Real vs Shuffled Labels\nTestConv2D_N2 | Zero-mean | Single Date', 
              fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Calculate overall delta
    overall_oa_delta = normal_results['overall_accuracy'] - shuffled_results['overall_accuracy']
    overall_f1_delta = normal_results['macro_f1'] - shuffled_results['macro_f1']
    
    # Add delta metrics text
    delta_text = f"""Performance Gains:
Overall Accuracy: +{overall_oa_delta:.3f} ({overall_oa_delta/shuffled_results['overall_accuracy']*100:.1f}% improvement)
Macro F1 Score: +{overall_f1_delta:.3f} ({overall_f1_delta/shuffled_results['macro_f1']*100:.1f}% improvement)

Color Legend:
🟢 Green: High gain (>0.3)
🟠 Orange: Medium gain (0.1-0.3)  
🔴 Red: Low gain (<0.1)"""
    
    plt.text(0.02, 0.98, delta_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = output_dir / "f1_delta_normal_minus_shuffled.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved delta plot: {output_path}")
    plt.close()

def create_comprehensive_metrics_plot(normal_results: dict, shuffled_results: dict, output_dir: Path):
    """Create comprehensive plot with F1, Precision, and Recall comparison."""
    logger.info("Creating comprehensive metrics plot...")
    
    classes = list(normal_results['f1_scores'].keys())
    
    # Extract all metrics
    normal_f1 = list(normal_results['f1_scores'].values())
    normal_precision = list(normal_results['precisions'].values())
    normal_recall = list(normal_results['recalls'].values())
    
    shuffled_f1 = list(shuffled_results['f1_scores'].values())
    shuffled_precision = list(shuffled_results['precisions'].values())
    shuffled_recall = list(shuffled_results['recalls'].values())
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    x = np.arange(len(classes))
    width = 0.35
    
    # F1 Score subplot
    bars1 = ax1.bar(x - width/2, normal_f1, width, label='Normal Labels', 
                    color='#1f77b4', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, shuffled_f1, width, label='Shuffled Labels', 
                    color='#ff7f0e', alpha=0.8, edgecolor='black')
    
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('F1 Score Comparison', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 0.8)
    
    # Precision subplot
    bars3 = ax2.bar(x - width/2, normal_precision, width, label='Normal Labels', 
                    color='#2ca02c', alpha=0.8, edgecolor='black')
    bars4 = ax2.bar(x + width/2, shuffled_precision, width, label='Shuffled Labels', 
                    color='#ff7f0e', alpha=0.8, edgecolor='black')
    
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision Comparison', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.0)
    
    # Recall subplot
    bars5 = ax3.bar(x - width/2, normal_recall, width, label='Normal Labels', 
                    color='#d62728', alpha=0.8, edgecolor='black')
    bars6 = ax3.bar(x + width/2, shuffled_recall, width, label='Shuffled Labels', 
                    color='#ff7f0e', alpha=0.8, edgecolor='black')
    
    ax3.set_xlabel('Land Cover Classes', fontsize=12)
    ax3.set_ylabel('Recall', fontsize=12)
    ax3.set_title('Recall Comparison', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1.0)
    
    # Add value labels for all subplots
    for ax, bars_list in [(ax1, [bars1, bars2]), (ax2, [bars3, bars4]), (ax3, [bars5, bars6])]:
        for bars in bars_list:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Comprehensive Metrics Comparison: Normal vs Shuffled Labels\nTestConv2D_N2 | Zero-mean | Single Date', 
                 fontsize=16)
    plt.tight_layout()
    
    output_path = output_dir / "comprehensive_metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comprehensive plot: {output_path}")
    plt.close()

def main():
    """Main function."""
    logger.info("Creating comparison plots between normal and shuffled models...")
    
    # Define the two models to compare
    normal_run_path = "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_normal_zeromean_single_20220611_conv2d_n2_a8e279a9"
    shuffled_run_path = "/home/davideidmann/code/lc_specific_speckle_analysis/data/training_output/run_shuffled_zeromean_single_20220611_conv2d_n2_dd182e25"
    
    # Output directory
    output_dir = Path(__file__).parent.parent / "results" / "comparison_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load results from both models
        logger.info("Loading normal model results...")
        normal_results = load_model_results(normal_run_path)
        
        logger.info("Loading shuffled model results...")
        shuffled_results = load_model_results(shuffled_run_path)
        
        # Create all comparison plots
        create_comparison_plot(normal_results, shuffled_results, output_dir)
        create_delta_plot(normal_results, shuffled_results, output_dir)
        create_comprehensive_metrics_plot(normal_results, shuffled_results, output_dir)
        
        logger.info("All comparison plots created successfully!")
        logger.info(f"Plots saved in: {output_dir}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*60)
        logger.info("Normal Model (Real Labels):")
        logger.info(f"  - Overall Accuracy: {normal_results['overall_accuracy']:.3f}")
        logger.info(f"  - Macro F1 Score: {normal_results['macro_f1']:.3f}")
        
        logger.info("Shuffled Model (Random Labels):")
        logger.info(f"  - Overall Accuracy: {shuffled_results['overall_accuracy']:.3f}")
        logger.info(f"  - Macro F1 Score: {shuffled_results['macro_f1']:.3f}")
        
        oa_gain = normal_results['overall_accuracy'] - shuffled_results['overall_accuracy']
        f1_gain = normal_results['macro_f1'] - shuffled_results['macro_f1']
        
        logger.info("Performance Gain from Real Learning:")
        logger.info(f"  - OA Improvement: +{oa_gain:.3f} ({oa_gain/shuffled_results['overall_accuracy']*100:.1f}%)")
        logger.info(f"  - F1 Improvement: +{f1_gain:.3f} ({f1_gain/shuffled_results['macro_f1']*100:.1f}%)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error creating comparison plots: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
