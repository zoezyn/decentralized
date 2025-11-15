#!/usr/bin/env python3
"""
Create comprehensive comparison plots for FL quantization study.

Generates:
1. Accuracy vs Model Size
2. Compression Ratio comparison
3. Communication Cost Analysis
4. Accuracy Drop visualization
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def find_latest_results():
    """Find the most recent quantization_comparison.json file."""
    results_files = list(Path("outputs").rglob("quantization_comparison.json"))
    if not results_files:
        raise FileNotFoundError("No quantization_comparison.json found!")

    # Sort by modification time, get latest
    latest = max(results_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest}")
    return latest

def load_results(json_path):
    """Load quantization comparison results."""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_comparison_plots(results):
    """Create comprehensive comparison plots."""

    # Extract data
    precisions = ['fp32', 'fp16', 'int8']
    labels = ['FP32\n(Baseline)', 'FP16\n(Half)', 'INT8\n(Quantized)']
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    accuracies = [results['precisions'][p]['accuracy_percent'] for p in precisions]
    sizes = [results['precisions'][p]['file_size_mb'] for p in precisions]
    compressions = [results['precisions'][p]['compression_ratio'] for p in precisions]
    accuracy_drops = [results['precisions'][p]['accuracy_drop_percent'] for p in precisions]

    # Calculate communication costs (10 clients, 10 rounds, $5/MB)
    num_clients = 10
    num_rounds = 10
    cost_per_mb = 5

    costs = []
    for size_mb in sizes:
        per_round = size_mb * 2 * num_clients  # down + up
        total = per_round * num_rounds * cost_per_mb
        costs.append(total)

    cost_savings = [costs[0] - cost for cost in costs]
    cost_savings_pct = [(saving / costs[0]) * 100 for saving in cost_savings]

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Federated Learning Quantization Comparison\nEuroSAT Satellite Imagery Classification',
                 fontsize=16, fontweight='bold')

    # 1. Accuracy vs Model Size (scatter)
    ax1 = plt.subplot(2, 3, 1)
    for i, (acc, size, label, color) in enumerate(zip(accuracies, sizes, labels, colors)):
        ax1.scatter(size, acc, s=300, c=color, alpha=0.6, edgecolors='black', linewidth=2)
        ax1.annotate(label, (size, acc), xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    ax1.set_xlabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs Model Size', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(accuracies) - 1, max(accuracies) + 1])

    # 2. Model Size Comparison (bar chart)
    ax2 = plt.subplot(2, 3, 2)
    bars = ax2.bar(labels, sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Size Comparison', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.2f} MB',
                ha='center', va='bottom', fontweight='bold')

    # 3. Compression Ratio (bar chart)
    ax3 = plt.subplot(2, 3, 3)
    bars = ax3.bar(labels, compressions, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Compression Ratio', fontsize=12, fontweight='bold')
    ax3.set_title('Compression Efficiency', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    # Add value labels
    for bar, comp in zip(bars, compressions):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{comp:.2f}x',
                ha='center', va='bottom', fontweight='bold')

    # 4. Accuracy Drop (bar chart)
    ax4 = plt.subplot(2, 3, 4)
    bars = ax4.bar(labels, accuracy_drops, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Accuracy Drop (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Accuracy Loss from FP32', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=2)

    # Add value labels
    for bar, drop in zip(bars, accuracy_drops):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{drop:+.2f}%',
                ha='center', va='bottom' if drop >= 0 else 'top', fontweight='bold')

    # 5. Communication Cost (stacked bar showing savings)
    ax5 = plt.subplot(2, 3, 5)
    bars = ax5.bar(labels, costs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax5.set_ylabel('Total Communication Cost ($)', fontsize=12, fontweight='bold')
    ax5.set_title('Communication Cost Analysis\n(10 clients, 10 rounds, $5/MB)',
                  fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, cost, saving_pct in zip(bars, costs, cost_savings_pct):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost:.0f}\n({saving_pct:.0f}% saved)',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 6. Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')

    table_data = [
        ['Metric', 'FP32', 'FP16', 'INT8'],
        ['Accuracy (%)', f'{accuracies[0]:.2f}', f'{accuracies[1]:.2f}', f'{accuracies[2]:.2f}'],
        ['Size (MB)', f'{sizes[0]:.2f}', f'{sizes[1]:.2f}', f'{sizes[2]:.2f}'],
        ['Compression', f'{compressions[0]:.2f}x', f'{compressions[1]:.2f}x', f'{compressions[2]:.2f}x'],
        ['Acc. Drop (%)', f'{accuracy_drops[0]:+.2f}', f'{accuracy_drops[1]:+.2f}', f'{accuracy_drops[2]:+.2f}'],
        ['Cost ($)', f'{costs[0]:.0f}', f'{costs[1]:.0f}', f'{costs[2]:.0f}'],
        ['Savings ($)', f'{cost_savings[0]:.0f}', f'{cost_savings[1]:.0f}', f'{cost_savings[2]:.0f}'],
    ]

    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.23, 0.23, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style metric column
    for i in range(1, 7):
        table[(i, 0)].set_facecolor('#ecf0f1')
        table[(i, 0)].set_text_props(weight='bold')

    # Color code best values
    table[(1, 0)].set_facecolor('#2ecc71')  # Best accuracy (FP32)
    table[(2, 3)].set_facecolor('#2ecc71')  # Smallest size (INT8)
    table[(3, 3)].set_facecolor('#2ecc71')  # Best compression (INT8)
    table[(6, 3)].set_facecolor('#2ecc71')  # Best savings (INT8)

    ax6.set_title('Summary Comparison Table', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()

    return fig

def main():
    """Main execution."""
    print("\n" + "="*60)
    print("QUANTIZATION COMPARISON PLOT GENERATOR")
    print("="*60 + "\n")

    # Find and load results
    results_path = find_latest_results()
    results = load_results(results_path)

    # Create plots
    print("\nGenerating comparison plots...")
    fig = create_comparison_plots(results)

    # Save to same directory as JSON
    output_dir = results_path.parent
    plot_path = output_dir / "comparison_plots.png"

    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {plot_path}")

    # Also save high-res version
    plot_path_hr = output_dir / "comparison_plots_highres.png"
    fig.savefig(plot_path_hr, dpi=600, bbox_inches='tight')
    print(f"✅ High-res plot saved to: {plot_path_hr}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Print key insights
    fp32_acc = results['precisions']['fp32']['accuracy_percent']
    int8_acc = results['precisions']['int8']['accuracy_percent']
    int8_size = results['precisions']['int8']['file_size_mb']
    fp32_size = results['precisions']['fp32']['file_size_mb']
    compression = results['precisions']['int8']['compression_ratio']

    print(f"\n📊 Key Insights:")
    print(f"  • INT8 achieves {int8_acc:.2f}% accuracy (only {fp32_acc - int8_acc:.2f}% drop)")
    print(f"  • {compression:.2f}x compression: {fp32_size:.2f}MB → {int8_size:.2f}MB")
    print(f"  • Cost reduction: ~{((fp32_size - int8_size) / fp32_size * 100):.0f}% communication savings")
    print(f"\n💡 Recommendation: Use INT8 quantization for satellite FL deployment!")
    print("\n")

if __name__ == "__main__":
    main()
