"""eurosat: A Flower / PyTorch app."""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, cast

import torch
import torch.nn as nn
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    IidPartitioner,
    PathologicalPartitioner,
)
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torchvision.models import vit_b_16

warnings.filterwarnings(
    "ignore",
    message=r"The currently tested dataset are",
    category=UserWarning,
)

class Net(nn.Module):
    """Model (Vision Transformer B/16)"""

    def __init__(self):
        super(Net, self).__init__()
        self.vit = vit_b_16(pretrained=False)
        self.vit.heads = nn.Linear(self.vit.heads.head.in_features, 10)

    def forward(self, x):
        return self.vit(x)


fds_cache: Dict[
    Tuple[
        int,
        str,
        Optional[float],
        int,
        Optional[int],
        Optional[str],
    ],
    FederatedDataset,
] = {}

pytorch_transforms = Compose([
    Resize((224, 224)),
    ToTensor(), 
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch


def _get_federated_dataset(
    num_partitions: int,
    partitioning: str,
    dirichlet_alpha: Optional[float],
    seed: int,
    num_classes_per_partition: Optional[int],
    class_assignment_mode: Optional[str],
) -> FederatedDataset:
    """Return a cached FederatedDataset configured with the requested partitioner."""
    normalized_partitioning = partitioning.lower()
    normalized_assignment_mode = (
        class_assignment_mode.lower() if class_assignment_mode else None
    )
    cache_key = (
        num_partitions,
        normalized_partitioning,
        dirichlet_alpha,
        seed,
        num_classes_per_partition,
        normalized_assignment_mode,
    )
    if cache_key not in fds_cache:
        if normalized_partitioning == "iid":
            partitioner = IidPartitioner(num_partitions=num_partitions)
        elif normalized_partitioning in {"dirichlet", "non-iid"}:
            if dirichlet_alpha is None:
                raise ValueError(
                    "dirichlet_alpha must be provided when using the Dirichlet "
                    "partitioner."
                )
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=dirichlet_alpha,
                shuffle=True,
                seed=seed,
            )
        elif normalized_partitioning in {
            "pathological",
            "pathological-single-class",
            "single-class",
        }:
            classes_per_partition = num_classes_per_partition or 1
            assignment_mode = normalized_assignment_mode or "deterministic"
            valid_modes = {"random", "deterministic", "first-deterministic"}
            if assignment_mode not in valid_modes:
                raise ValueError(
                    "Invalid class_assignment_mode for pathological partitioning. "
                    f"Expected one of {valid_modes}, received '{assignment_mode}'."
                )
            cast_assignment_mode = cast(
                Literal["random", "deterministic", "first-deterministic"],
                assignment_mode,
            )
            partitioner = PathologicalPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                num_classes_per_partition=classes_per_partition,
                class_assignment_mode=cast_assignment_mode,
                shuffle=True,
                seed=seed,
            )
        else:
            raise ValueError(
                f"Unsupported partitioning strategy '{partitioning}'. "
                "Valid options are 'iid', 'dirichlet', or 'pathological-single-class'."
            )
        fds_cache[cache_key] = FederatedDataset(
            dataset="tanganke/eurosat",
            partitioners={"train": partitioner},
        )
    return fds_cache[cache_key]


def load_data(
    partition_id: int,
    num_partitions: int,
    partitioning: str = "iid",
    dirichlet_alpha: Optional[float] = None,
    partition_seed: int = 42,
    num_classes_per_partition: Optional[int] = None,
    class_assignment_mode: Optional[str] = None,
    max_samples_per_client: Optional[int] = None,
):
    """Load the EuroSAT partition specified by the node configuration."""
    fds = _get_federated_dataset(
        num_partitions=num_partitions,
        partitioning=partitioning,
        dirichlet_alpha=dirichlet_alpha,
        seed=partition_seed,
        num_classes_per_partition=num_classes_per_partition,
        class_assignment_mode=class_assignment_mode,
    )
    partition = fds.load_partition(partition_id)
    if max_samples_per_client is not None:
        subset_size = min(max_samples_per_client, len(partition))
        partition = partition.shuffle(seed=partition_seed).select(range(subset_size))
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def create_run_dir() -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    return run_dir, str(save_path)
