"""eurosat: A Flower / PyTorch app."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from datasets import load_dataset
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from torch.utils.data import DataLoader
import wandb

from eurosat.task import Net, test, apply_transforms, create_run_dir

# Create ServerApp
app = ServerApp()

PROJECT_NAME = "Hackathon-Berlin25-Eurosat"

SATELLITE_NAMES = {
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


class FedAvgWithClientTracking(FedAvg):
    """FedAvg strategy extended to capture per-client evaluation metrics."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.client_eval_history: Dict[int, list[Dict[str, Any]]] = defaultdict(list)

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Optional[MetricRecord]:
        """Aggregate MetricRecords while logging per-client metrics."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)

        metrics = None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            # Track per-client metrics
            for message in valid_replies:
                metric_record = message.content.get("metrics")
                if metric_record is None:
                    continue
                metric_dict = dict(metric_record)
                partition_id = int(metric_dict.get("partition_id", -1))

                if partition_id >= 0:
                    per_client_metrics = {
                        key: float(value)
                        for key, value in metric_dict.items()
                        if key not in {"partition_id", "num-examples"}
                    }
                    if per_client_metrics:
                        entry: Dict[str, Any] = {
                            "round": server_round,
                            "metrics": per_client_metrics,
                            "num_examples": int(metric_dict.get("num-examples", 0)),
                        }
                        self.client_eval_history[partition_id].append(entry)

                # Remove partition_id before aggregation
                metric_record.pop("partition_id", None)

            # Aggregate MetricRecords
            metrics = self.evaluate_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )
        return metrics


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

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy with client tracking
    strategy = FedAvgWithClientTracking(fraction_train=fraction_train)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=get_global_evaluate_fn(),
    )

    # Save final model to disk
    print(f"\nSaving final model to disk at {save_path}...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, f"{save_path}/final_model.pt")

    # Save per-client evaluation metrics
    if strategy.client_eval_history:
        run_path = Path(save_path)
        fairness_path = run_path / "per_client_eval_metrics.json"
        serializable_history = {
            str(partition_id): {
                "satellite_name": SATELLITE_NAMES.get(partition_id, f"Client-{partition_id}"),
                "records": history,
            }
            for partition_id, history in strategy.client_eval_history.items()
        }
        with fairness_path.open("w", encoding="utf-8") as fairness_file:
            json.dump(serializable_history, fairness_file, indent=2)
        print(f"Saved per-client evaluation metrics to {fairness_path}")


def get_global_evaluate_fn():
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
        wandb.log(
            {
                "Global Test Loss": loss,
                "Global Test Accuracy": accuracy,
            }
        )
        return MetricRecord({"accuracy": accuracy, "loss": loss})

    return global_evaluate
