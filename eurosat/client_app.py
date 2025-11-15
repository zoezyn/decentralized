"""eurosat: A Flower / PyTorch app."""

from typing import Any, Dict

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from eurosat.task import Net, load_data
from eurosat.task import test as test_fn
from eurosat.task import train as train_fn

# Flower ClientApp
app = ClientApp()


def _resolve_partitioning_config(context: Context) -> Dict[str, Any]:
    """Extract partitioning configuration from the run config."""
    run_config = context.run_config
    config: Dict[str, Any] = {}

    # Get partitioning strategy
    partitioning = run_config.get("partitioning", "iid")
    config["partitioning"] = str(partitioning).lower()

    # Get Dirichlet alpha if specified
    alpha_value = run_config.get("dirichlet-alpha")
    if alpha_value is not None:
        config["dirichlet_alpha"] = float(alpha_value)

    # Get partition seed
    seed_value = run_config.get("partition-seed", 42)
    config["partition_seed"] = int(seed_value)

    # Get max samples per client if specified
    max_samples = run_config.get("max-samples-per-client")
    if max_samples is not None:
        config["max_samples_per_client"] = int(max_samples)

    return config


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data with partitioning config
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    partition_config = _resolve_partitioning_config(context)
    trainloader, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        **partition_config,
    )

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data with partitioning config
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    partition_config = _resolve_partitioning_config(context)
    _, valloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        **partition_config,
    )

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
        "partition_id": partition_id,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
