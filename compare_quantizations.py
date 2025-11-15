#!/usr/bin/env python3
"""
Post-Training Quantization Comparison Script

This script takes a trained FP32 model and:
1. Measures its actual file size
2. Quantizes it to FP16 and INT8
3. Evaluates accuracy for each quantization
4. Calculates real model sizes
5. Compares accuracy drop vs compression

Usage:
    python compare_quantizations.py outputs/2025-11-15/14-30-00/final_model.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader

# Import from eurosat package
from eurosat.task import Net, apply_transforms, test


def get_model_size_mb(model_path: str) -> float:
    """Get actual file size in MB."""
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 ** 2)
    return size_mb


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_theoretical_size(num_params, bytes_per_param):
    """Calculate theoretical model size."""
    return (num_params * bytes_per_param) / (1024 ** 2)  # MB


def load_test_data():
    """Load global EuroSAT test set."""
    print("Loading EuroSAT test set...")
    global_test_set = load_dataset("tanganke/eurosat")["test"]
    testloader = DataLoader(
        global_test_set.with_transform(apply_transforms),
        batch_size=32,
    )
    print(f"Test set loaded: {len(global_test_set)} images\n")
    return testloader


def evaluate_model(model, testloader, device, precision_name):
    """Evaluate model and return metrics."""
    print(f"Evaluating {precision_name}...")
    model.to(device)
    model.eval()

    loss, accuracy = test(model, testloader, device)

    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "accuracy_percent": float(accuracy * 100)
    }


def create_fp16_model(fp32_model):
    """Convert FP32 model to FP16."""
    print("Converting to FP16...")
    model_fp16 = Net()
    model_fp16.load_state_dict(fp32_model.state_dict())
    model_fp16 = model_fp16.half()  # Convert to FP16
    return model_fp16


def create_int8_model(fp32_model):
    """
    Create INT8 quantized model using dynamic quantization.

    Note: Dynamic quantization quantizes weights to INT8 but keeps
    activations in FP32. This is the simplest form of quantization
    that works well for inference.
    """
    print("Quantizing to INT8 (dynamic quantization)...")

    model_int8 = Net()
    model_int8.load_state_dict(fp32_model.state_dict())
    model_int8.eval()

    # Apply dynamic quantization to Linear and Conv2d layers
    model_int8_quantized = torch.quantization.quantize_dynamic(
        model_int8,
        {nn.Linear, nn.Conv2d},  # Quantize these layer types
        dtype=torch.qint8  # Use INT8
    )

    return model_int8_quantized


def save_and_measure(model, save_path, precision_name):
    """Save model and measure actual file size."""
    model_path = f"{save_path}/model_{precision_name}.pt"

    # Save model
    torch.save(model.state_dict(), model_path)

    # Measure size
    size_mb = get_model_size_mb(model_path)

    print(f"  Saved: {model_path}")
    print(f"  File size: {size_mb:.2f} MB\n")

    return model_path, size_mb


def main():
    parser = argparse.ArgumentParser(description="Compare model quantizations")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained FP32 model (e.g., outputs/2025-11-15/14-30-00/final_model.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for quantized models (default: same as input model)"
    )

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"PyTorch version: {torch.__version__}\n")

    # Determine output directory
    model_dir = Path(args.model_path).parent
    output_dir = args.output if args.output else str(model_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("POST-TRAINING QUANTIZATION COMPARISON")
    print("="*60)
    print(f"Input model: {args.model_path}")
    print(f"Output directory: {output_dir}\n")

    # Load FP32 model
    print("Loading FP32 model...")
    model_fp32 = Net()
    model_fp32.load_state_dict(torch.load(args.model_path))
    model_fp32.eval()

    num_params = count_parameters(model_fp32)
    print(f"Model parameters: {num_params:,}")
    print(f"Theoretical FP32 size: {calculate_theoretical_size(num_params, 4):.2f} MB")
    print(f"Theoretical FP16 size: {calculate_theoretical_size(num_params, 2):.2f} MB")
    print(f"Theoretical INT8 size: {calculate_theoretical_size(num_params, 1):.2f} MB\n")

    # Load test data
    testloader = load_test_data()

    # Results storage
    results = {
        "model_path": args.model_path,
        "num_parameters": num_params,
        "precisions": {}
    }

    print("="*60)
    print("1. FP32 (Baseline)")
    print("="*60)

    # Evaluate FP32
    fp32_metrics = evaluate_model(model_fp32, testloader, device, "FP32")
    fp32_path, fp32_size = save_and_measure(model_fp32, output_dir, "fp32")

    results["precisions"]["fp32"] = {
        **fp32_metrics,
        "model_path": fp32_path,
        "file_size_mb": fp32_size,
        "compression_ratio": 1.0,
        "accuracy_drop_percent": 0.0
    }

    baseline_accuracy = fp32_metrics["accuracy"]

    print("="*60)
    print("2. FP16 (Half Precision)")
    print("="*60)

    # Create and evaluate FP16
    model_fp16 = create_fp16_model(model_fp32)

    # For FP16 evaluation, need to convert data to FP16 too
    fp16_metrics = evaluate_model(model_fp16, testloader, device, "FP16")
    fp16_path, fp16_size = save_and_measure(model_fp16, output_dir, "fp16")

    accuracy_drop_fp16 = (baseline_accuracy - fp16_metrics["accuracy"]) * 100
    compression_fp16 = fp32_size / fp16_size if fp16_size > 0 else 0

    results["precisions"]["fp16"] = {
        **fp16_metrics,
        "model_path": fp16_path,
        "file_size_mb": fp16_size,
        "compression_ratio": compression_fp16,
        "accuracy_drop_percent": accuracy_drop_fp16
    }

    print("="*60)
    print("3. INT8 (Dynamic Quantization)")
    print("="*60)

    # Create and evaluate INT8
    model_int8 = create_int8_model(model_fp32)
    int8_metrics = evaluate_model(model_int8, testloader, device, "INT8")
    int8_path, int8_size = save_and_measure(model_int8, output_dir, "int8")

    accuracy_drop_int8 = (baseline_accuracy - int8_metrics["accuracy"]) * 100
    compression_int8 = fp32_size / int8_size if int8_size > 0 else 0

    results["precisions"]["int8"] = {
        **int8_metrics,
        "model_path": int8_path,
        "file_size_mb": int8_size,
        "compression_ratio": compression_int8,
        "accuracy_drop_percent": accuracy_drop_int8
    }

    # Summary
    print("="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Precision':<10} {'Accuracy':<12} {'Drop':<12} {'Size (MB)':<12} {'Compression':<12}")
    print("-" * 60)

    for prec in ["fp32", "fp16", "int8"]:
        data = results["precisions"][prec]
        print(f"{prec.upper():<10} "
              f"{data['accuracy_percent']:>6.2f}%     "
              f"{data['accuracy_drop_percent']:>+6.2f}%     "
              f"{data['file_size_mb']:>7.2f}      "
              f"{data['compression_ratio']:>6.2f}x")

    print("\n" + "="*60)
    print("COMMUNICATION COST ANALYSIS (Theoretical)")
    print("="*60)

    # Calculate communication costs for FL
    num_clients = 10
    num_rounds_example = 10
    cost_per_mb = 5  # dollars

    print(f"\nAssumptions:")
    print(f"  - {num_clients} clients (satellites)")
    print(f"  - {num_rounds_example} FL rounds")
    print(f"  - Each client: downloads + uploads model")
    print(f"  - Cost: ${cost_per_mb}/MB")
    print()

    for prec in ["fp32", "fp16", "int8"]:
        size_mb = results["precisions"][prec]["file_size_mb"]

        # Per round: each client downloads and uploads
        bytes_per_round = size_mb * 2 * num_clients  # down + up, all clients
        total_bytes = bytes_per_round * num_rounds_example
        total_cost = total_bytes * cost_per_mb

        cost_saved = (results["precisions"]["fp32"]["file_size_mb"] * 2 * num_clients * num_rounds_example * cost_per_mb) - total_cost
        savings_percent = (cost_saved / (results["precisions"]["fp32"]["file_size_mb"] * 2 * num_clients * num_rounds_example * cost_per_mb)) * 100 if prec != "fp32" else 0

        print(f"{prec.upper()}:")
        print(f"  Per round: {bytes_per_round:.2f} MB")
        print(f"  Total ({num_rounds_example} rounds): {total_bytes:.2f} MB")
        print(f"  Total cost: ${total_cost:.2f}")
        if prec != "fp32":
            print(f"  Savings vs FP32: ${cost_saved:.2f} ({savings_percent:.1f}%)")
        print()

    # Save results to JSON
    results_path = f"{output_dir}/quantization_comparison.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("="*60)
    print(f"Results saved to: {results_path}")
    print("="*60)
    print("\nDone!")


if __name__ == "__main__":
    main()
