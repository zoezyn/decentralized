#!/usr/bin/env python3
"""Analyze data distribution across satellites for IID vs Non-IID."""

from collections import Counter
from eurosat.task import load_data
import numpy as np

EUROSAT_LABELS = {
    0: "Annual Crop",
    1: "Forest",
    2: "Brushland",
    3: "Highway",
    4: "Industrial",
    5: "Pasture",
    6: "Permanent Crop",
    7: "Residential",
    8: "River",
    9: "Lake/Sea",
}

def analyze_partition_distribution(partition_id, num_partitions, partitioning, alpha=None):
    """Analyze the class distribution for a single partition."""
    config = {
        "partitioning": partitioning,
        "partition_seed": 42,
        "max_samples_per_client": 200,
    }
    if alpha:
        config["dirichlet_alpha"] = alpha

    trainloader, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        **config
    )

    # Count labels
    all_labels = []
    for batch in trainloader:
        all_labels.extend(batch["label"].tolist())

    counter = Counter(all_labels)
    total = len(all_labels)

    return counter, total


def print_distribution_table(partitioning_name, partitioning_type, alpha=None):
    """Print distribution table for all satellites."""
    print(f"\n{'='*100}")
    print(f"📊 {partitioning_name} DATA DISTRIBUTION")
    print(f"{'='*100}\n")

    num_partitions = 10

    # Header
    print("┌────┬─────────────────────────┬───────┬─────────────────────────────────────────────┐")
    print("│ ID │ Satellite Name          │ Total │ Class Distribution (%)                      │")
    print("├────┼─────────────────────────┼───────┼─────────────────────────────────────────────┤")

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

    for pid in range(num_partitions):
        counter, total = analyze_partition_distribution(
            pid, num_partitions, partitioning_type, alpha
        )

        # Find top 3 classes
        top_classes = counter.most_common(3)
        dist_str = ", ".join([
            f"{EUROSAT_LABELS[cls]}:{cnt/total*100:.0f}%"
            for cls, cnt in top_classes
        ])

        name = satellite_names.get(pid, f"Client-{pid}")
        print(f"│ {pid:2d} │ {name:<23} │ {total:5d} │ {dist_str:<43} │")

    print("└────┴─────────────────────────┴───────┴─────────────────────────────────────────────┘")

    # Calculate overall balance metric
    print("\n💡 Key Observations:")
    all_distributions = []
    for pid in range(num_partitions):
        counter, total = analyze_partition_distribution(
            pid, num_partitions, partitioning_type, alpha
        )
        # Calculate entropy/balance
        probs = np.array([counter.get(i, 0) / total for i in range(10)])
        dominant_class_pct = max(probs) * 100
        all_distributions.append(dominant_class_pct)

    avg_dominance = np.mean(all_distributions)
    print(f"   - Average dominant class percentage: {avg_dominance:.1f}%")
    print(f"   - Perfect balance would be: 10.0% (each class equal)")
    print(f"   - Imbalance score: {avg_dominance - 10:.1f}% above baseline")


if __name__ == "__main__":
    print("\n🛰️  SATELLITE DATA DISTRIBUTION ANALYSIS\n")

    # IID
    print_distribution_table("IID (BALANCED)", "iid")

    # Non-IID
    print_distribution_table("NON-IID (REALISTIC - Dirichlet α=0.1)", "dirichlet", alpha=0.1)

    print("\n" + "="*100)
    print("📈 EXPECTED IMPACT ON FEDERATED LEARNING:")
    print("="*100)
    print("""
IID (Balanced):
  ✅ All satellites see roughly equal distribution of all terrain types
  ✅ Model learns all classes equally well
  ✅ Better convergence and higher accuracy

Non-IID (Realistic):
  ⚠️  Each satellite sees predominantly 1-2 terrain types
  ⚠️  Creates client drift - each local model specializes differently
  ⚠️  Harder to aggregate - conflicting gradients
  ⚠️  Lower global accuracy, potential fairness issues
    """)
    print("="*100 + "\n")
