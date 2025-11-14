"""eurosat: A Flower / PyTorch app."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import torch
from datasets import load_dataset
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from torch.utils.data import DataLoader
import wandb

from eurosat.task import Net, apply_transforms, create_run_dir, test

# Create ServerApp
app = ServerApp()

PROJECT_NAME = "Hackathon-Berlin25-Eurosat"

EUROSAT_LABELS = {
    0: "Annual Crop Land",
    1: "Forest",
    2: "Brushland/Shrubland",
    3: "Highway/Road",
    4: "Industrial Buildings",
    5: "Pasture Land",
    6: "Permanent Crop Land",
    7: "Residential Buildings",
    8: "River",
    9: "Lake or Sea",
}

SATELLITE_ASSIGNMENTS: Dict[int, Dict[str, Any]] = {
    0: {
        "name": "AgriSat-AnnualCrop",
        "region": "Agricultural",
        "class_id": 0,
        "class_label": EUROSAT_LABELS[0],
    },
    1: {
        "name": "WildSat-Forest",
        "region": "Wilderness",
        "class_id": 1,
        "class_label": EUROSAT_LABELS[1],
    },
    2: {
        "name": "WildSat-Brushland",
        "region": "Wilderness",
        "class_id": 2,
        "class_label": EUROSAT_LABELS[2],
    },
    3: {
        "name": "UrbanSat-Highway",
        "region": "Urban",
        "class_id": 3,
        "class_label": EUROSAT_LABELS[3],
    },
    4: {
        "name": "UrbanSat-Industrial",
        "region": "Urban",
        "class_id": 4,
        "class_label": EUROSAT_LABELS[4],
    },
    5: {
        "name": "AgriSat-Pasture",
        "region": "Agricultural",
        "class_id": 5,
        "class_label": EUROSAT_LABELS[5],
    },
    6: {
        "name": "AgriSat-PermanentCrop",
        "region": "Agricultural",
        "class_id": 6,
        "class_label": EUROSAT_LABELS[6],
    },
    7: {
        "name": "UrbanSat-Residential",
        "region": "Urban",
        "class_id": 7,
        "class_label": EUROSAT_LABELS[7],
    },
    8: {
        "name": "CoastSat-River",
        "region": "Water/Coastal",
        "class_id": 8,
        "class_label": EUROSAT_LABELS[8],
    },
    9: {
        "name": "CoastSat-LakeSea",
        "region": "Water/Coastal",
        "class_id": 9,
        "class_label": EUROSAT_LABELS[9],
    },
}


class FedAvgWithClientTracking(FedAvg):
    """FedAvg strategy extended to capture per-client evaluation metrics."""

    def __init__(
        self,
        *args,
        per_client_eval_callback: Optional[
            Callable[[int, int, Dict[str, float], int], None]
        ] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.per_client_eval_callback = per_client_eval_callback
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

            for message in valid_replies:
                metric_record = message.content.get("metrics")
                if metric_record is None:
                    continue
                metric_dict = dict(metric_record)
                partition_id = int(
                    metric_dict.get("partition_id", message.metadata.src_node_id)
                )
                # Extract metrics except bookkeeping keys
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
                    satellite_meta = SATELLITE_ASSIGNMENTS.get(partition_id)
                    if satellite_meta:
                        entry["meta"] = satellite_meta
                    self.client_eval_history[partition_id].append(entry)
                    if self.per_client_eval_callback is not None:
                        self.per_client_eval_callback(
                            server_round,
                            partition_id,
                            per_client_metrics,
                            int(metric_dict.get("num-examples", 0)),
                        )
                # Remove helper field before aggregation to avoid meaningless averages
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

    # Initialize FedAvg strategy
    run_path = Path(save_path)

    def log_per_client_metrics(
        server_round: int,
        partition_id: int,
        metrics_map: Dict[str, float],
        num_examples: int,
    ) -> None:
        """Log per-client metrics to Weights & Biases."""
        meta = SATELLITE_ASSIGNMENTS.get(partition_id)
        if meta:
            prefix = f"{meta['region']}/{meta['name']}"
        else:
            prefix = f"client/{partition_id}"
        log_payload = {
            f"{prefix}/eval_{metric_name}": metric_value
            for metric_name, metric_value in metrics_map.items()
        }
        log_payload[f"{prefix}/num_examples"] = num_examples
        wandb.log(log_payload, step=server_round)

    strategy = FedAvgWithClientTracking(
        fraction_train=fraction_train,
        per_client_eval_callback=log_per_client_metrics,
    )

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

    # Persist per-client evaluation history for fairness analysis
    if strategy.client_eval_history:
        fairness_path = run_path / "per_client_eval_metrics.json"
        serializable_history = {
            str(partition_id): {
                "meta": SATELLITE_ASSIGNMENTS.get(
                    partition_id,
                    {
                        "name": f"Client-{partition_id}",
                        "region": "Unknown",
                        "class_id": partition_id,
                        "class_label": EUROSAT_LABELS.get(
                            partition_id, "Unknown Label"
                        ),
                    },
                ),
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
