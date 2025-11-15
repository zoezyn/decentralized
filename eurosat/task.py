"""eurosat: A Flower / PyTorch app."""

import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, DistributionPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision import models

warnings.filterwarnings(
    "ignore",
    message=r"The currently tested dataset are",
    category=UserWarning,
)

try:
    RESNET18_WEIGHTS = models.ResNet18_Weights.DEFAULT
except AttributeError:
    RESNET18_WEIGHTS = None


class ResNet(nn.Module):
    """ResNet-18 model adapted for EuroSAT (10 classes)"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = models.resnet18(weights=RESNET18_WEIGHTS)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class SlimResidualBlock(nn.Module):
    """Lightweight residual block used in the compressed ResNet variant."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)


class CompressedResNet(nn.Module):
    """Shrinked residual network (~2x fewer params than ResNet-18)."""

    def __init__(self, num_classes=10):
        super().__init__()
        width = [32, 64, 96, 128]
        self.stem = nn.Sequential(
            nn.Conv2d(3, width[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width[0]),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(width[0], width[0], blocks=3, stride=1)
        self.layer2 = self._make_layer(width[0], width[1], blocks=3, stride=2)
        self.layer3 = self._make_layer(width[1], width[2], blocks=3, stride=2)
        self.layer4 = self._make_layer(width[2], width[3], blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width[3], num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [SlimResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(SlimResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


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


class CompressedNet(nn.Module):
    """Smaller CNN that trades accuracy for 3x fewer parameters and bandwidth."""

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # Depthwise + pointwise keep expressiveness with fewer parameters
        self.dwconv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        self.pwconv3 = nn.Conv2d(32, 48, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.dwconv4 = nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48)
        self.pwconv4 = nn.Conv2d(48, 64, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.pwconv3(self.dwconv3(x)))))
        x = self.pool(F.relu(self.bn4(self.pwconv4(self.dwconv4(x)))))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


fds = None  # Cache FederatedDataset

# Predefine transforms so we can switch per model variant (e.g., ImageNet vs. custom)
MODEL_TRANSFORMS = {
    "baseline": Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "compressed": Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "resnet": Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "resnet_compressed": Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}
pytorch_transforms = MODEL_TRANSFORMS["baseline"]

# partitioner = context.run_config["partitioner"]

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch


def load_data(partition_id: int, num_partitions: int, partitioner_method: str):
    """Load partition EuroSAT data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        if partitioner_method == 'iid':
            print(f"Loading IID partition with {num_partitions} partitions")
            partitioner = IidPartitioner(num_partitions=num_partitions)
        else:
            print(f"Loading Non-IID partition with {num_partitions} partitions")
            # partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by='label', alpha=0.1, seed=42)
            distribution_array = np.full((num_partitions, 1), [1], dtype=float)
            partitioner = DistributionPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                num_unique_labels_per_partition=1,
                preassigned_num_samples_per_label = 5,
                distribution_array=distribution_array,
            )
        fds = FederatedDataset(
            dataset="tanganke/eurosat",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
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


MODEL_VARIANTS = {
    "baseline": Net,
    "compressed": CompressedNet,
    "resnet": ResNet,
    "resnet_compressed": CompressedResNet,
}


def build_model(variant: str = "baseline") -> nn.Module:
    """Factory to create the requested model architecture."""
    variant_key = variant.lower()
    model_cls = MODEL_VARIANTS.get(variant_key)
    if model_cls is None:
        raise ValueError(
            f"Unknown model variant '{variant}'. "
            f"Available options: {', '.join(MODEL_VARIANTS.keys())}"
        )
    set_transform_for_variant(variant_key)
    return model_cls()


def set_transform_for_variant(variant_key: str) -> None:
    """Update the transforms used by loaders based on model selection."""
    global pytorch_transforms
    pytorch_transforms = MODEL_TRANSFORMS.get(variant_key, pytorch_transforms)
