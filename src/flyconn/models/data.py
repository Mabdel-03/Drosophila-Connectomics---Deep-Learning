"""MNIST data loaders (torchvision), cached to scratch.

Data lives under ``$FLYCONN_DATA_ROOT/v783/mnist`` (scratch — never /home). Images are
normalized and flattened to 784-vectors. A 10% slice of the train set is held out for
validation.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from ..paths import data_root

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def mnist_dir() -> Path:
    d = data_root() / "v783" / "mnist"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        transforms.Lambda(lambda t: t.view(-1)),   # flatten 1x28x28 -> 784
    ])


def get_loaders(batch_size: int = 128, val_frac: float = 0.1, num_workers: int = 4,
                download: bool = True, seed: int = 0):
    """Return (train_loader, val_loader, test_loader) of flattened MNIST."""
    from torchvision.datasets import MNIST

    root = str(mnist_dir())
    tfm = _transform()
    full_train = MNIST(root, train=True, download=download, transform=tfm)
    test = MNIST(root, train=False, download=download, transform=tfm)

    n_val = int(len(full_train) * val_frac)
    n_train = len(full_train) - n_val
    g = torch.Generator().manual_seed(seed)
    train, val = random_split(full_train, [n_train, n_val], generator=g)

    pin = torch.cuda.is_available()
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers, pin_memory=pin, drop_last=False)
    return mk(train, True), mk(val, False), mk(test, False)
