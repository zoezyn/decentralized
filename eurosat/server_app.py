"""eurosat: A Flower / PyTorch app."""

import io
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from torch.utils.data import DataLoader
import wandb
import threading
import requests
import json

from eurosat.task import build_model, test, apply_transforms, create_run_dir
from eurosat.battery_aware_strategy import BatteryAwareFedAvg, print_final_battery_report
from eurosat.cubesat_battery_reader import initialize_cubesat, read_cubesat_battery, cleanup_cubesat

# Webhook URL for sending battery data to frontend
WEBHOOK_URL = "http://localhost:8001/api/webhook/battery"

_server_test_dataset = None

def _get_server_test_dataset():
    global _server_test_dataset
    if _server_test_dataset is None:
        _server_test_dataset = load_dataset("tanganke/eurosat")["test"].with_transform(apply_transforms)
    return _server_test_dataset

def evaluate_model_arrays(arrays: ArrayRecord, model_variant: str):
    """Evaluate given arrays on central test set."""
    dataset = _get_server_test_dataset()
    testloader = DataLoader(dataset, batch_size=32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = build_model(model_variant)
    net.load_state_dict(arrays.to_torch_state_dict())
    net.to(device)
    loss, accuracy = test(net, testloader, device=device)
    return loss, accuracy

def send_battery_webhook(data, round_num):
    """Send battery and model transmission data to frontend via webhook"""
    try:
        if isinstance(data, dict) and "battery_data" in data:
            # New format with transmission data
            payload = {
                "type": "fl_update",
                "round": round_num,
                "battery_levels": data["battery_data"],
                "selected_satellites": data["selected_satellites"],
                "total_satellites": data["total_satellites"],
                "transmission_phase": data["transmission_phase"],
                "timestamp": torch.tensor(0).item()
            }
        else:
            # Legacy format for backwards compatibility
            payload = {
                "type": "battery_update",
                "round": round_num,
                "battery_levels": data,
                "timestamp": torch.tensor(0).item()
            }
        
        response = requests.post(WEBHOOK_URL, json=payload, timeout=1)
        if response.status_code == 200:
            print(f"🔋 Sent battery data for round {round_num}")
        else:
            print(f"⚠️ Webhook failed: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Webhook error: {e}")

# Custom strategy that sends webhooks
class WebhookBatteryAwareFedAvg(BatteryAwareFedAvg):
    def aggregate_train(self, server_round, replies):
        print(f"🔄 WebhookBatteryAwareFedAvg.aggregate_train called for round {server_round}")
        arrays, metrics = super().aggregate_train(server_round, replies)
        
        if arrays is not None:
            state_dict = arrays.to_torch_state_dict()
            param_count = sum(t.numel() for t in state_dict.values())
            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            payload_kb = buffer.tell() / 1024
            print(f"   📦 Round {server_round}: model payload ≈ {payload_kb:.1f} KB ({param_count:,} params)")
            wandb.log(
                {
                    "payload_kb": payload_kb,
                    "payload_param_count": param_count,
                },
                step=server_round,
            )
        
        # Send battery data via webhook
        if hasattr(self, 'selected_clients_this_round'):
            battery_stats = self.battery_sim.step(server_round, self.selected_clients_this_round)
            
            # Convert battery data to frontend format
            frontend_data = {}
            for sat_id, battery in battery_stats['batteries'].items():
                frontend_data[f"sat-{sat_id}"] = {
                    "battery": battery,
                    "in_sunlight": self.battery_sim.is_in_sunlight(server_round, sat_id),
                    "can_train": battery >= self.battery_sim.min_battery_threshold,
                    "status": "operational" if battery >= self.battery_sim.min_battery_threshold else "low_battery"
                }
            
            # Send webhook with model transmission info
            transmission_data = {
                "battery_data": frontend_data,
                "round": server_round,
                "selected_satellites": self.selected_clients_this_round,
                "total_satellites": self.battery_sim.num_satellites,
                "transmission_phase": "model_aggregation"
            }
            
            threading.Thread(
                target=send_battery_webhook, 
                args=(transmission_data, server_round),
                daemon=True
            ).start()
        
        return arrays, metrics

# Create ServerApp
app = ServerApp()

PROJECT_NAME = "Hackathon-Berlin25-Eurosat"

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Create run directory
    run_dir, save_path = create_run_dir()

    # Initialize Weights & Biases logging
    wandb.init(project=PROJECT_NAME, name=f"{str(run_dir)}-ServerApp")

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    
    # Battery simulation config (optional)
    battery_enabled: bool = context.run_config.get("battery-enabled", True)
    
    # CubeSat integration config
    cubesat_enabled: bool = context.run_config.get("cubesat-enabled", False)
    cubesat_port: str = context.run_config.get("cubesat-port", "/dev/cu.usbserial-1110")
    cubesat_baud: int = context.run_config.get("cubesat-baud", 9600)
    cubesat_id: int = context.run_config.get("cubesat-id", 0)  # Which satellite is the CubeSat

    model_variant: str = context.run_config.get("model-variant", "baseline")

    # Load global model
    global_model = build_model(model_variant)
    arrays = ArrayRecord(global_model.state_dict())
    total_params = sum(p.numel() for p in global_model.parameters())
    buf = io.BytesIO()
    torch.save(global_model.state_dict(), buf)
    initial_payload_kb = buf.tell() / 1024
    print(f"   Model variant '{model_variant}': {total_params:,} params (~{initial_payload_kb:.1f} KB payload)")
    wandb.config.update(
        {
            "model_variant": model_variant,
            "model_variant_params": total_params,
            "model_variant_payload_kb": initial_payload_kb,
            "battery_enabled": battery_enabled,
            "initial_battery": context.run_config.get("initial-battery", 80.0),
            "charge_rate": context.run_config.get("charge-rate", 3.0),
            "train_cost": context.run_config.get("train-cost", 15.0),
            "comm_cost": context.run_config.get("comm-cost", 5.0),
            "min_battery_threshold": context.run_config.get("min-battery-threshold", 30.0),
        }
    )
    wandb.log(
        {
            "model_variant_payload_kb": initial_payload_kb,
            "model_variant_params": total_params,
        },
        step=0,
    )
    wandb.run.summary.update(
        {
            "model_variant": model_variant,
            "model_variant_params": total_params,
            "model_variant_payload_kb": initial_payload_kb,
        }
    )

    # Initialize CubeSat connection if enabled
    cubesat_reader = None
    if cubesat_enabled and battery_enabled:
        print(f"\n🛰️  Initializing CubeSat on {cubesat_port}...")
        if initialize_cubesat(cubesat_port, cubesat_baud):
            cubesat_reader = read_cubesat_battery
            print(f"   CubeSat will be Satellite {cubesat_id} (real hardware)")
        else:
            print("   ⚠️  CubeSat not available, continuing with full simulation")

    # Initialize strategy based on configuration
    if battery_enabled:
        print("\n🔋 Battery-Aware Mode Enabled")
        battery_config = {
            'num_satellites': 10,
            'initial_battery': context.run_config.get("initial-battery", 80.0),
            'charge_rate': context.run_config.get("charge-rate", 3.0),
            'train_cost': context.run_config.get("train-cost", 15.0),
            'comm_cost': context.run_config.get("comm-cost", 5.0),
            'min_battery_threshold': context.run_config.get("min-battery-threshold", 30.0),
            'day_night_cycle': context.run_config.get("day-night-cycle", True),
            'orbit_period': context.run_config.get("orbit-period", 6),
            'cubesat_id': cubesat_id if cubesat_reader else None,
            'cubesat_battery_reader': cubesat_reader,
        }
        strategy = WebhookBatteryAwareFedAvg(
            fraction_train=fraction_train,
            battery_config=battery_config
        )
    else:
        print("\n⚙️  Standard FedAvg Mode")
        strategy = FedAvg(fraction_train=fraction_train)

    # Start strategy, run FedAvg for `num_rounds`
    print(f"\n{'='*70}")
    print(f"🚀 Starting Federated Learning")
    print(f"   Rounds: {num_rounds}")
    print(f"   Fraction train: {fraction_train}")
    print(f"   Learning rate: {lr}")
    print(f"{'='*70}\n")
    
    evaluate_fn = get_global_evaluate_fn(strategy, model_variant)
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,
    )

    # Print battery report if battery mode was enabled
    if battery_enabled and isinstance(strategy, BatteryAwareFedAvg):
        print_final_battery_report(strategy)

    # Cleanup CubeSat connection
    if cubesat_reader:
        cleanup_cubesat()

    # Save final model to disk
    print(f"\nSaving final model to disk at {save_path}...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, f"{save_path}/final_model.pt")
    
    # Run a final evaluation to log comparable metrics
    final_loss, final_accuracy = evaluate_model_arrays(result.arrays, model_variant)
    wandb.log(
        {
            "final_accuracy": final_accuracy,
            "final_loss": final_loss,
            "final_model_payload_kb": initial_payload_kb,
            "final_model_params": total_params,
        },
        step=num_rounds,
    )
    wandb.run.summary["final_accuracy"] = final_accuracy
    wandb.run.summary["final_loss"] = final_loss

    log_row = ",".join(
        [
            datetime.now().isoformat(),
            run_dir,
            model_variant,
            str(total_params),
            f"{initial_payload_kb:.2f}",
            f"{final_accuracy:.4f}" if final_accuracy is not None else "",
        ]
    )
    log_path = Path("outputs/model_metrics.csv")
    if not log_path.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("timestamp,run_dir,model_variant,total_params,payload_kb,final_accuracy\n")
    with log_path.open("a") as log_file:
        log_file.write(log_row + "\n")
    print(f"\n📁 Logged run summary to {log_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ Training Complete!")
    print(f"{'='*70}\n")


def get_global_evaluate_fn(strategy, model_variant: str):
    """Return an evaluation function for server-side evaluation."""

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data."""

        loss, accuracy = evaluate_model_arrays(arrays, model_variant)
        
        # Prepare logging dictionary
        log_dict = {
            "round": server_round,
            "Global Test Loss": loss,
            "Global Test Accuracy": accuracy,
            "model_variant": wandb.config.get("model_variant"),
            "model_variant_params": wandb.config.get("model_variant_params"),
            "model_variant_payload_kb": wandb.config.get("model_variant_payload_kb"),
        }
        
        # Add battery information if using battery-aware strategy
        if isinstance(strategy, BatteryAwareFedAvg):
            battery_stats = strategy.battery_sim.get_statistics()
            log_dict.update({
                "evaluation/battery_avg": battery_stats['avg_battery'],
                "evaluation/battery_min": min(battery_stats['current_batteries'].values()),
                "evaluation/battery_max": max(battery_stats['current_batteries'].values()),
                "evaluation/satellites_below_threshold": battery_stats['satellites_below_threshold'],
                "evaluation/total_skipped": battery_stats['total_skipped'],
            })
            
            print(f"   Battery status during evaluation: avg={battery_stats['avg_battery']:.1f}%, "
                  f"below threshold={battery_stats['satellites_below_threshold']}")
        
        wandb.log(log_dict)
        return MetricRecord({"accuracy": accuracy, "loss": loss})

    return global_evaluate
