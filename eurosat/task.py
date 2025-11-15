"""eurosat: A Flower / PyTorch app."""

import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from typing import Optional
from torchvision.transforms import Compose, Normalize, ToTensor

warnings.filterwarnings(
    "ignore",
    message=r"The currently tested dataset are",
    category=UserWarning,
)

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 96, 3)
        self.bn3 = nn.BatchNorm2d(96)
        self.fc1 = nn.Linear(96 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 96 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds_cache = {}  # Cache FederatedDataset by config

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch


def load_data(
    partition_id: int,
    num_partitions: int,
    partitioning: str = "iid",
    dirichlet_alpha: Optional[float] = None,
    partition_seed: int = 42,
    max_samples_per_client: Optional[int] = None,
):
    """Load partition EuroSAT data with configurable partitioning."""
    global fds_cache

    # Create cache key based on partitioning config
    cache_key = (num_partitions, partitioning, dirichlet_alpha, partition_seed)

    if cache_key not in fds_cache:
        # Create partitioner based on config
        if partitioning.lower() == "iid":
            partitioner = IidPartitioner(num_partitions=num_partitions)
        elif partitioning.lower() in ["dirichlet", "non-iid"]:
            if dirichlet_alpha is None:
                raise ValueError("dirichlet_alpha must be provided for Dirichlet partitioning")
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=dirichlet_alpha,
                shuffle=True,
                seed=partition_seed,
            )
        else:
            raise ValueError(f"Unknown partitioning: {partitioning}")

        fds_cache[cache_key] = FederatedDataset(
            dataset="tanganke/eurosat",
            partitioners={"train": partitioner},
        )

    fds = fds_cache[cache_key]
    partition = fds.load_partition(partition_id)

    # Optionally limit samples per client
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