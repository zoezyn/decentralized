#!/usr/bin/env python3
"""
Run IID vs Non-IID comparison experiment and display results table.

Usage:
    python run_noniid_experiment.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
import time


def run_experiment(config_path: str, experiment_name: str) -> Dict[str, Any]:
    """Run a single experiment and return the results."""
    print(f"\n{'='*80}")
    print(f"Running {experiment_name} Experiment")
    print(f"{'='*80}\n")

    cmd = [
        "uv", "run", "flwr", "run", ".",
        "--run-config", config_path,
    ]

    print(f"Command: {' '.join(cmd)}\n")

    # Set WandB API key in environment
    env = os.environ.copy()
    env["WANDB_API_KEY"] = "62418e1723904400d80cf59e99e6e5989f862f47"

    result = subprocess.run(cmd, capture_output=False, text=True, env=env)

    if result.returncode != 0:
        print(f"❌ {experiment_name} experiment failed!")
        return None

    print(f"\n✅ {experiment_name} experiment completed!\n")

    # Find the most recent output directory
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("No outputs directory found")
        return None

    # Get all run directories sorted by modification time
    run_dirs = sorted(outputs_dir.rglob("per_client_eval_metrics.json"),
                     key=lambda p: p.stat().st_mtime, reverse=True)

    if not run_dirs:
        print("No results found")
        return None

    # Load the most recent results
    latest_results = run_dirs[0]
    with open(latest_results, 'r') as f:
        data = json.load(f)

    return data


def analyze_results(iid_data: Dict, noniid_data: Dict) -> None:
    """Analyze and display comparison table."""

    # Group satellites by region
    regions = {
        "Wilderness": [1, 2],  # Forest, Brushland
        "Agricultural": [0, 5, 6],  # Annual Crop, Pasture, Permanent Crop
        "Urban": [3, 4, 7],  # Highway, Industrial, Residential
        "Water/Coastal": [8, 9],  # River, Lake/Sea
    }

    def get_final_accuracy(data: Dict, partition_id: int) -> float:
        """Get final round accuracy for a partition."""
        partition_str = str(partition_id)
        if partition_str not in data:
            return 0.0
        records = data[partition_str]["records"]
        if not records:
            return 0.0
        # Get last round's accuracy
        return records[-1]["metrics"].get("eval_acc", 0.0) * 100

    def get_avg_accuracy_by_region(data: Dict, partition_ids: List[int]) -> float:
        """Get average accuracy across partitions."""
        accs = [get_final_accuracy(data, pid) for pid in partition_ids]
        return sum(accs) / len(accs) if accs else 0.0

    # Calculate metrics
    print("\n" + "="*100)
    print("📊 IID vs Non-IID COMPARISON RESULTS")
    print("="*100 + "\n")

    # Overall comparison
    print("┌─────────────────────────────────────────────────────────────────────────────────┐")
    print("│                          OVERALL ACCURACY COMPARISON                            │")
    print("├─────────────────────────┬──────────────────┬──────────────────┬─────────────────┤")
    print("│ Metric                  │  IID (Baseline)  │  Non-IID (α=0.1) │  Difference     │")
    print("├─────────────────────────┼──────────────────┼──────────────────┼─────────────────┤")

    all_partitions = list(range(10))
    iid_overall = get_avg_accuracy_by_region(iid_data, all_partitions)
    noniid_overall = get_avg_accuracy_by_region(noniid_data, all_partitions)
    diff_overall = noniid_overall - iid_overall

    print(f"│ Average Accuracy        │     {iid_overall:5.2f}%       │     {noniid_overall:5.2f}%       │    {diff_overall:+6.2f}%      │")
    print("└─────────────────────────┴──────────────────┴──────────────────┴─────────────────┘")

    # Per-region breakdown
    print("\n┌─────────────────────────────────────────────────────────────────────────────────┐")
    print("│                       PER-REGION ACCURACY BREAKDOWN                             │")
    print("├────────────────────────┬──────────────────┬──────────────────┬──────────────────┤")
    print("│ Region                 │  IID (Baseline)  │  Non-IID (α=0.1) │  Difference      │")
    print("├────────────────────────┼──────────────────┼──────────────────┼──────────────────┤")

    for region_name, partition_ids in regions.items():
        iid_acc = get_avg_accuracy_by_region(iid_data, partition_ids)
        noniid_acc = get_avg_accuracy_by_region(noniid_data, partition_ids)
        diff = noniid_acc - iid_acc

        # Pad region name to 22 chars
        region_display = f"{region_name:<22}"
        print(f"│ {region_display} │     {iid_acc:5.2f}%       │     {noniid_acc:5.2f}%       │    {diff:+6.2f}%       │")

    print("└────────────────────────┴──────────────────┴──────────────────┴──────────────────┘")

    # Individual satellite breakdown
    print("\n┌─────────────────────────────────────────────────────────────────────────────────┐")
    print("│                   INDIVIDUAL SATELLITE PERFORMANCE                              │")
    print("├────┬────────────────────────────────┬──────────────────┬──────────────────┬─────┤")
    print("│ ID │ Satellite Name                 │  IID (Baseline)  │  Non-IID (α=0.1) │ Δ   │")
    print("├────┼────────────────────────────────┼──────────────────┼──────────────────┼─────┤")

    satellite_names = {
        0: "AgriSat-AnnualCrop",
        1: "WildSat-Forest",
        2: "WildSat-Brushland",
        3: "UrbanSat-Highway",
        4: "UrbanSat-Industrial",
        5: "AgriSat-Pasture",
        6: "AgriSat-PermanentCrop",
        7: "UrbanSat-Residential",
        8: "CoastSat-River",
        9: "CoastSat-LakeSea",
    }

    for pid in range(10):
        name = satellite_names.get(pid, f"Client-{pid}")
        iid_acc = get_final_accuracy(iid_data, pid)
        noniid_acc = get_final_accuracy(noniid_data, pid)
        diff = noniid_acc - iid_acc

        # Pad name to 30 chars
        name_display = f"{name:<30}"
        print(f"│ {pid:2d} │ {name_display} │     {iid_acc:5.2f}%       │     {noniid_acc:5.2f}%       │{diff:+4.1f}%│")

    print("└────┴────────────────────────────────────┴──────────────────┴──────────────────┴─────┘")

    # Key insights
    print("\n" + "="*100)
    print("💡 KEY INSIGHTS")
    print("="*100)
    print(f"""
1. 📉 Impact of Non-IID Data:
   - IID baseline achieved {iid_overall:.2f}% average accuracy
   - Non-IID (realistic) achieved {noniid_overall:.2f}% average accuracy
   - Performance {'degraded' if diff_overall < 0 else 'improved'} by {abs(diff_overall):.2f}% under realistic conditions

2. 🎯 Regional Fairness:
   - Check which regions suffered most from data imbalance
   - Identify if certain terrain types are underrepresented

3. 🛰️  Next Steps to Improve Non-IID Performance:
   - Try FedProx: Adds proximal term to handle heterogeneity
   - Use personalization: Combine local + global model layers
   - Apply FedAvg with momentum: Smoother convergence
   - Implement class balancing or data augmentation strategies
""")
    print("="*100 + "\n")


def main():
    """Main experiment runner."""
    print("\n🚀 Starting IID vs Non-IID Comparison Experiment\n")

    # Run IID baseline
    print("Step 1/2: Running IID baseline...")
    iid_results = run_experiment("experiments/iid_baseline.toml", "IID Baseline")

    if iid_results is None:
        print("❌ IID experiment failed. Aborting.")
        sys.exit(1)

    # Small delay to ensure directory ordering
    time.sleep(2)

    # Run Non-IID
    print("\nStep 2/2: Running Non-IID experiment...")
    noniid_results = run_experiment("experiments/non_iid_realistic.toml", "Non-IID Realistic")

    if noniid_results is None:
        print("❌ Non-IID experiment failed. Aborting.")
        sys.exit(1)

    # Analyze and display results
    analyze_results(iid_results, noniid_results)

    print("✅ Experiment complete! Check outputs/ directory for detailed logs.")


if __name__ == "__main__":
    main()
