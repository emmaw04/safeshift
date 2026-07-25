from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# Use exactly the same normalisation during training and evaluation.
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_train_transform() -> transforms.Compose:
    """Transform used when training on clean CIFAR-100."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def get_test_transform() -> transforms.Compose:
    """Transform used for clean and corrupted evaluation data."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def get_cifar100_loader(
    data_dir: str,
    train: bool,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """Load clean CIFAR-100."""
    dataset = datasets.CIFAR100(
        root=data_dir,
        train=train,
        download=True,
        transform=get_train_transform() if train else get_test_transform(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


class CIFAR100C(Dataset):
    """Loader for one CIFAR-100-C corruption at one severity."""

    def __init__(
        self,
        root: str,
        corruption: str,
        severity: int,
        transform: Callable | None = None,
    ) -> None:
        if not 1 <= severity <= 5:
            raise ValueError("severity must be between 1 and 5")

        root_path = Path(root)
        image_path = root_path / f"{corruption}.npy"
        label_path = root_path / "labels.npy"

        if not image_path.exists():
            raise FileNotFoundError(f"Missing corruption file: {image_path}")

        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")

        all_images = np.load(image_path, mmap_mode="r")
        all_labels = np.load(label_path, mmap_mode="r")

        # Each severity contains 10,000 images.
        start = (severity - 1) * 10_000
        end = severity * 10_000

        self.images = all_images[start:end]
        self.labels = all_labels[start:end]
        self.transform = transform or get_test_transform()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.fromarray(self.images[index])
        label = int(self.labels[index])

        return self.transform(image), label


def get_cifar100c_loader(
    data_dir: str,
    corruption: str,
    severity: int,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    dataset = CIFAR100C(
        root=data_dir,
        corruption=corruption,
        severity=severity,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )