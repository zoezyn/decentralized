#!/usr/bin/env python3
"""
Compare different training approaches for satellite networks:
1. Local-only (each satellite trains independently - no collaboration)
2. Federated Learning (satellites collaborate via FedAvg)
3. Centralized (all data pooled - privacy violation but optimal baseline)

This demonstrates the VALUE of federated learning!
"""

import json
import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

from eurosat.task import Net, load_data, train as train_fn, test as test_fn
from datasets import load_dataset
from torch.utils.data import DataLoader, ConcatDataset
from eurosat.task import apply_transforms


def train_local_only(num_satellites=10, partitioning="dirichlet", alpha=0.1, max_samples=200, epochs=3):
    """Each satellite trains only on its own data (no collaboration)."""
    print(f"\n{'='*80}")
    print("🔴 LOCAL-ONLY TRAINING (No Collaboration)")
    print(f"{'='*80}")

    results = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for satellite_id in range(num_satellites):
        print(f"\nTraining satellite {satellite_id}...")

        # Each satellite gets its own model
        model = Net()
        model.to(device)

        # Load satellite's local data
        trainloader, testloader = load_data(
            partition_id=satellite_id,
            num_partitions=num_satellites,
            partitioning=partitioning,
            dirichlet_alpha=alpha,
            partition_seed=42,
            max_samples_per_client=max_samples,
        )

        # Train only on local data
        for epoch in range(epochs):
            train_fn(model, trainloader, epochs=1, lr=0.001, device=device)

        # Evaluate on local test set
        loss, acc = test_fn(model, testloader, device)

        results[satellite_id] = {
            "eval_acc": acc,
            "eval_loss": loss,
        }
        print(f"  Satellite {satellite_id} local accuracy: {acc*100:.2f}%")

    return results


def train_centralized(num_satellites=10, partitioning="dirichlet", alpha=0.1, max_samples=200, epochs=3):
    """Pool all data centrally (privacy violation but optimal performance)."""
    print(f"\n{'='*80}")
    print("🟢 CENTRALIZED TRAINING (All Data Pooled)")
    print(f"{'='*80}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Collect all training data
    all_train_datasets = []
    satellite_testloaders = {}

    print("\nCollecting data from all satellites...")
    for satellite_id in range(num_satellites):
        trainloader, testloader = load_data(
            partition_id=satellite_id,
            num_partitions=num_satellites,
            partitioning=partitioning,
            dirichlet_alpha=alpha,
            partition_seed=42,
            max_samples_per_client=max_samples,
        )
        all_train_datasets.append(trainloader.dataset)
        satellite_testloaders[satellite_id] = testloader

    # Create centralized training dataset
    from torch.utils.data import ConcatDataset
    centralized_dataset = ConcatDataset(all_train_datasets)
    centralized_loader = DataLoader(centralized_dataset, batch_size=32, shuffle=True)

    print(f"Total centralized training samples: {len(centralized_dataset)}")

    # Train single global model
    model = Net()
    model.to(device)

    print("\nTraining centralized model...")
    for epoch in range(epochs):
        train_fn(model, centralized_loader, epochs=1, lr=0.001, device=device)
        print(f"  Epoch {epoch+1}/{epochs} completed")

    # Evaluate on each satellite's test set
    results = {}
    print("\nEvaluating on each satellite's test set...")
    for satellite_id, testloader in satellite_testloaders.items():
        loss, acc = test_fn(model, testloader, device)
        results[satellite_id] = {
            "eval_acc": acc,
            "eval_loss": loss,
        }
        print(f"  Satellite {satellite_id} accuracy: {acc*100:.2f}%")

    return results


def load_federated_results():
    """Load results from the most recent Non-IID federated learning experiment."""
    outputs_dir = Path("outputs")
    result_files = sorted(
        outputs_dir.rglob("per_client_eval_metrics.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not result_files:
        return None

    # Load most recent (Non-IID)
    with open(result_files[0], 'r') as f:
        data = json.load(f)

    results = {}
    for satellite_id in range(10):
        sid = str(satellite_id)
        if sid in data and data[sid]["records"]:
            final_round = data[sid]["records"][-1]
            results[satellite_id] = {
                "eval_acc": final_round["metrics"]["eval_acc"],
                "eval_loss": final_round["metrics"]["eval_loss"],
            }

    return results


def print_comparison_table(local_results, federated_results, centralized_results):
    """Print comprehensive comparison table."""

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

    print("\n" + "="*120)
    print("📊 TRAINING APPROACH COMPARISON: LOCAL vs FEDERATED vs CENTRALIZED")
    print("="*120 + "\n")

    # Calculate averages
    local_avg = sum(r["eval_acc"] for r in local_results.values()) / len(local_results) * 100
    fed_avg = sum(r["eval_acc"] for r in federated_results.values()) / len(federated_results) * 100
    cent_avg = sum(r["eval_acc"] for r in centralized_results.values()) / len(centralized_results) * 100

    # Overall comparison
    print("┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    print("│                                    OVERALL PERFORMANCE COMPARISON                                        │")
    print("├─────────────────────────────┬────────────────────┬────────────────────┬────────────────────────────────┤")
    print("│ Training Approach           │  Avg Accuracy      │  Privacy Preserved │  Communication Cost            │")
    print("├─────────────────────────────┼────────────────────┼────────────────────┼────────────────────────────────┤")
    print(f"│ Local-Only (No Collab)      │      {local_avg:5.2f}%        │         ✅ Yes      │      ✅ None (Isolated)         │")
    print(f"│ Federated Learning          │      {fed_avg:5.2f}%        │         ✅ Yes      │      🟡 Low (Models Only)       │")
    print(f"│ Centralized (All Data)      │      {cent_avg:5.2f}%        │         ❌ No       │      🔴 High (Raw Data)         │")
    print("└─────────────────────────────┴────────────────────┴────────────────────┴────────────────────────────────┘")

    # Performance gains
    fed_vs_local = fed_avg - local_avg
    fed_vs_cent = fed_avg - cent_avg

    print("\n┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    print("│                                    FEDERATED LEARNING BENEFITS                                           │")
    print("├──────────────────────────────────────────────────────────────────────────────────────────────────────────┤")
    print(f"│  📈 Accuracy gain vs Local-only:     {fed_vs_local:+6.2f}%  ({abs(fed_vs_local/local_avg*100):.1f}% relative improvement)           │")
    print(f"│  🔒 Privacy-Utility gap vs Central:  {abs(fed_vs_cent):6.2f}%  (cost of keeping data private)                      │")
    print(f"│  ⚖️  Best of both worlds:            Collaboration + Privacy                                              │")
    print("└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘")

    # Per-satellite breakdown
    print("\n┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    print("│                              PER-SATELLITE PERFORMANCE BREAKDOWN                                         │")
    print("├────┬─────────────────────────────┬──────────────────┬──────────────────┬──────────────────┬─────────────┤")
    print("│ ID │ Satellite Name              │  Local-Only      │  Federated (FL)  │  Centralized     │  FL Gain    │")
    print("├────┼─────────────────────────────┼──────────────────┼──────────────────┼──────────────────┼─────────────┤")

    for sid in range(10):
        name = satellite_names[sid]
        local_acc = local_results[sid]["eval_acc"] * 100
        fed_acc = federated_results[sid]["eval_acc"] * 100
        cent_acc = centralized_results[sid]["eval_acc"] * 100
        fl_gain = fed_acc - local_acc

        name_display = f"{name:<27}"
        print(f"│ {sid:2d} │ {name_display} │     {local_acc:5.2f}%       │     {fed_acc:5.2f}%       │     {cent_acc:5.2f}%       │  {fl_gain:+6.2f}%  │")

    print("└────┴─────────────────────────────┴──────────────────┴──────────────────┴──────────────────┴─────────────┘")

    # Key insights
    satellites_improved = sum(1 for sid in range(10) if federated_results[sid]["eval_acc"] > local_results[sid]["eval_acc"])
    max_gain_sid = max(range(10), key=lambda s: federated_results[s]["eval_acc"] - local_results[s]["eval_acc"])
    max_gain = (federated_results[max_gain_sid]["eval_acc"] - local_results[max_gain_sid]["eval_acc"]) * 100

    print("\n" + "="*120)
    print("💡 KEY INSIGHTS - WHY FEDERATED LEARNING MATTERS")
    print("="*120)
    print(f"""
1. 🤝 Collaboration Benefits:
   - {satellites_improved}/10 satellites improved with federated learning vs local-only
   - Average improvement: {fed_vs_local:+.2f} percentage points
   - Best case: {satellite_names[max_gain_sid]} gained {max_gain:+.2f}% accuracy
   - Satellites learn from each other's data without sharing raw images

2. 🔒 Privacy-Utility Tradeoff:
   - Federated learning achieves {fed_avg:.2f}% accuracy while keeping data private
   - Centralized approach achieves {cent_avg:.2f}% but requires data sharing (privacy violation)
   - Privacy gap: only {abs(fed_vs_cent):.2f}% - acceptable tradeoff for satellite networks

3. 🛰️  Real-World Satellite Scenario:
   - Local-only: Each satellite is isolated, can't learn from global patterns
   - Federated: Satellites collaborate while images stay on-board
   - Centralized: Infeasible due to bandwidth constraints + privacy concerns

4. 📡 Communication Efficiency:
   - Local-only: 0 communication (but poor performance)
   - Federated: Only model weights transmitted (~1.7 MB/round)
   - Centralized: Must transmit all satellite images (TBs of data!)

🎯 CONCLUSION: Federated Learning enables satellite collaboration without compromising privacy
              or overwhelming communication links - the ONLY viable approach for space networks!
""")
    print("="*120 + "\n")


def main():
    """Run comparison experiment."""
    config = {
        "num_satellites": 10,
        "partitioning": "dirichlet",
        "alpha": 0.1,
        "max_samples": 200,
        "epochs": 3,
    }

    print("\n🚀 SATELLITE FEDERATED LEARNING COMPARISON EXPERIMENT")
    print("="*120)
    print("\nComparing three training approaches:")
    print("  1. 🔴 Local-Only: Each satellite trains independently (no collaboration)")
    print("  2. 🟡 Federated Learning: Satellites collaborate via FedAvg (privacy-preserving)")
    print("  3. 🟢 Centralized: Pool all data (optimal but violates privacy)")
    print("\nConfiguration:")
    print(f"  - Number of satellites: {config['num_satellites']}")
    print(f"  - Data distribution: Non-IID (Dirichlet α={config['alpha']})")
    print(f"  - Samples per satellite: {config['max_samples']}")
    print(f"  - Local epochs: {config['epochs']}")
    print("="*120)

    # 1. Train local-only models
    local_results = train_local_only(**config)

    # 2. Load federated learning results
    print(f"\n{'='*80}")
    print("🟡 FEDERATED LEARNING (Loading previous results)")
    print(f"{'='*80}")
    federated_results = load_federated_results()
    if federated_results is None:
        print("❌ No federated learning results found. Please run: uv run python run_noniid_experiment.py")
        return
    print("✅ Loaded federated learning results from previous experiment")

    # 3. Train centralized model
    centralized_results = train_centralized(**config)

    # 4. Print comparison
    print_comparison_table(local_results, federated_results, centralized_results)


if __name__ == "__main__":
    main()
