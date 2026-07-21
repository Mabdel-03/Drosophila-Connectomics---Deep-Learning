"""Image data loaders (torchvision), cached to scratch.

Data lives under ``$FLYCONN_DATA_ROOT/v783/<subdir>`` (scratch — never /home). Images are
normalized and flattened to a per-dataset pixel vector. A 10% slice of the train set is
held out for validation.

Two datasets are supported, selected by ``get_loaders(dataset=...)``:
  * ``mnist``   — 28x28 grayscale, flattened to 784. (Default; the original stage-3 data.)
  * ``cifar10`` — 32x32 color. The ``color`` arg picks the channel handling:
        ``luma`` (default) -> RGB collapsed to a single luminance channel (1024-d) so the
            biologically faithful rigid eye (one brightness per ommatidial column) stays
            honest; ``rgb`` -> full 3-channel (3072-d), a learned-encoder-only control that
            measures how much the colorblind constraint costs.

All new args default to the MNIST values, so existing callers are byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from ..paths import data_root

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
# CIFAR-10 luminance (Y = 0.299 R + 0.587 G + 0.114 B) mean/std, computed over the actual
# luma channel of the 50k train set (NOT a weighted collapse of the per-channel RGB stds —
# the channels are correlated, so that would be wrong). The rigid eye has no learned encoder
# to absorb a scale error, so these matter more than for the learned path.
CIFAR_LUMA_MEAN, CIFAR_LUMA_STD = 0.4809, 0.2392
# Standard CIFAR-10 per-channel stats for the RGB control path.
CIFAR_RGB_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_RGB_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of a dataset as the model sees it (post-transform)."""

    name: str            # 'mnist' | 'cifar10'
    tv_class: str        # torchvision.datasets class name
    H: int               # image height
    W: int               # image width
    C: int               # channels the MODEL sees (1 for mnist/luma, 3 for rgb)
    n_classes: int
    subdir: str          # cache subdir under v783/

    @property
    def n_pixels(self) -> int:
        """Flattened input dimension the encoder/eye consumes."""
        return self.H * self.W * self.C


def dataset_spec(name: str, color: str = "luma") -> DatasetSpec:
    """Resolve a (dataset, color) pair to a DatasetSpec.

    ``color`` only applies to cifar10 ('luma' single-channel vs 'rgb' three-channel); it is
    ignored for mnist (always single-channel).
    """
    name = name.lower()
    if name == "mnist":
        return DatasetSpec("mnist", "MNIST", 28, 28, 1, 10, "mnist")
    if name == "cifar10":
        if color == "rgb":
            return DatasetSpec("cifar10", "CIFAR10", 32, 32, 3, 10, "cifar10")
        if color == "luma":
            return DatasetSpec("cifar10", "CIFAR10", 32, 32, 1, 10, "cifar10")
        raise ValueError(f"unknown color {color!r} (expected 'luma' or 'rgb')")
    raise ValueError(f"unknown dataset {name!r} (expected 'mnist' or 'cifar10')")


def dataset_dir(spec: DatasetSpec) -> Path:
    d = data_root() / "v783" / spec.subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def mnist_dir() -> Path:
    """Backward-compatible alias for the MNIST cache dir."""
    return dataset_dir(dataset_spec("mnist"))


def _transform(spec: DatasetSpec):
    from torchvision import transforms

    if spec.name == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
            transforms.Lambda(lambda t: t.view(-1)),   # flatten 1x28x28 -> 784
        ])
    # cifar10
    if spec.C == 1:                                     # luma: collapse RGB -> 1 channel
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),   # PIL op, before ToTensor
            transforms.ToTensor(),
            transforms.Normalize((CIFAR_LUMA_MEAN,), (CIFAR_LUMA_STD,)),
            transforms.Lambda(lambda t: t.view(-1)),    # flatten 1x32x32 -> 1024
        ])
    # rgb control: 3-channel, channel-major flatten [R|G|B] (order is arbitrary to a dense
    # encoder, which is the only consumer of the rgb path; documented).
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_RGB_MEAN, CIFAR_RGB_STD),
        transforms.Lambda(lambda t: t.view(-1)),        # flatten 3x32x32 -> 3072
    ])


def get_loaders(batch_size: int = 128, val_frac: float = 0.1, num_workers: int = 4,
                download: bool = True, seed: int = 0,
                dataset: str = "mnist", color: str = "luma"):
    """Return (train_loader, val_loader, test_loader) of flattened images.

    ``dataset``/``color`` default to MNIST so existing callers are unchanged.
    """
    import torchvision.datasets as tvd

    spec = dataset_spec(dataset, color=color)
    cls = getattr(tvd, spec.tv_class)
    root = str(dataset_dir(spec))
    tfm = _transform(spec)
    full_train = cls(root, train=True, download=download, transform=tfm)
    test = cls(root, train=False, download=download, transform=tfm)

    n_val = int(len(full_train) * val_frac)
    n_train = len(full_train) - n_val
    g = torch.Generator().manual_seed(seed)
    train, val = random_split(full_train, [n_train, n_val], generator=g)

    pin = torch.cuda.is_available()
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers, pin_memory=pin, drop_last=False)
    return mk(train, True), mk(val, False), mk(test, False)
