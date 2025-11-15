"""eurosat: A Flower / PyTorch app."""

import torch
from datasets import load_dataset
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from torch.utils.data import DataLoader
import wandb

from eurosat.task import Net, test, apply_transforms, create_run_dir
from eurosat.battery_aware_strategy import BatteryAwareFedAvg, print_final_battery_report
from eurosat.cubesat_battery_reader import initialize_cubesat, read_cubesat_battery, cleanup_cubesat

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

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

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
        strategy = BatteryAwareFedAvg(
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
    
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=get_global_evaluate_fn(strategy),
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
    
    print(f"\n{'='*70}")
    print(f"✅ Training Complete!")
    print(f"{'='*70}\n")


def get_global_evaluate_fn(strategy):
    """Return an evaluation function for server-side evaluation."""

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data."""

        # This is the exact same dataset as the one downloaded by the clients via
        # FlowerDatasets. However, we don't use FlowerDatasets for the server since
        # partitioning is not needed.
        # We make use of the "test" split only
        global_test_set = load_dataset("tanganke/eurosat")["test"]

        testloader = DataLoader(
            global_test_set.with_transform(apply_transforms),
            batch_size=32,
        )
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Apply global model parameters
        net = Net()
        net.load_state_dict(arrays.to_torch_state_dict())
        net.to(device)
        # Evaluate global model on test set
        loss, accuracy = test(net, testloader, device=device)
        
        # Prepare logging dictionary
        log_dict = {
            "round": server_round,
            "Global Test Loss": loss,
            "Global Test Accuracy": accuracy,
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