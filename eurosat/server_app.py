"""eurosat: A Flower / PyTorch app."""

import time
import torch
from datasets import load_dataset
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from torch.utils.data import DataLoader

from eurosat.task import Net, test, apply_transforms, create_run_dir

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read precision first to create proper directory
    precision: str = context.run_config.get("precision", "fp32")

    # Create run directory with precision
    run_dir, save_path = create_run_dir(precision)

    print(f"\n{'='*60}")
    print(f"Starting Federated Learning Training")
    print(f"Run Directory: {run_dir}")
    print(f"Save Path: {save_path}")
    print(f"{'='*60}\n")

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    precision: str = context.run_config.get("precision", "fp32")
    time_limit_minutes: float = context.run_config.get("time-limit-minutes", None)

    # Initialize time tracking
    start_time = time.time()
    time_limit_seconds = time_limit_minutes * 60 if time_limit_minutes else None

    if time_limit_seconds:
        print(f"⏱️  Time limit: {time_limit_minutes} minutes ({time_limit_seconds:.0f} seconds)")
        print(f"⏱️  Training will stop after {time_limit_minutes} min OR {num_rounds} rounds (whichever comes first)")
    else:
        print(f"🔄 No time limit - will run for {num_rounds} rounds")
    print()

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train)

    # Create time-aware evaluate function
    evaluate_fn = get_global_evaluate_fn(start_time, time_limit_seconds)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,
    )

    # Calculate final statistics
    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Total time: {elapsed_minutes:.2f} minutes ({elapsed_time:.1f} seconds)")
    if time_limit_seconds and elapsed_time >= time_limit_seconds:
        print(f"⏱️  Stopped due to time limit ({time_limit_minutes} min)")
    print(f"{'='*60}\n")

    # Save final model to disk
    print(f"\nSaving final model to disk at {save_path}...")
    state_dict = result.arrays.to_torch_state_dict()
    model_path = f"{save_path}/final_model.pt"
    torch.save(state_dict, model_path)

    print(f"Model saved: {model_path}")
    print(f"\nTo compare quantizations, run:")
    print(f"  python compare_quantizations.py {model_path}")
    print()


def get_global_evaluate_fn(start_time: float, time_limit_seconds: float = None):
    """Return an evaluation function for server-side evaluation with time tracking.

    Args:
        start_time: Training start timestamp
        time_limit_seconds: Optional time limit in seconds. If set, training stops when exceeded.
    """

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data."""

        # Check if time limit exceeded
        elapsed_time = time.time() - start_time
        if time_limit_seconds and elapsed_time >= time_limit_seconds:
            elapsed_minutes = elapsed_time / 60
            time_limit_minutes = time_limit_seconds / 60
            print(f"\n⏱️  TIME LIMIT REACHED!")
            print(f"   Elapsed: {elapsed_minutes:.2f} min / Limit: {time_limit_minutes:.1f} min")
            print(f"   Stopping training after round {server_round}")
            # Return None to signal Flower to stop
            return None

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

        # Calculate elapsed time for this round
        round_elapsed = time.time() - start_time
        round_minutes = round_elapsed / 60

        # Log to console with time info
        time_info = f" | Elapsed: {round_minutes:.2f} min"
        if time_limit_seconds:
            remaining = (time_limit_seconds - round_elapsed) / 60
            time_info += f" | Remaining: {remaining:.2f} min"

        print(f"Round {server_round} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}{time_info}")

        return MetricRecord({"accuracy": accuracy, "loss": loss})

    return global_evaluate