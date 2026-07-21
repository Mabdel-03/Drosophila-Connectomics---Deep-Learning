"""Unit tests for the dataset registry (DatasetSpec) — no data files, no torchvision.

Guards that (dataset, color) resolves to the right image dims / n_pixels / n_classes, and
that MNIST stays the default so legacy configs are byte-identical.

Run:  /orcd/home/002/mabdel03/conda_envs/consortium/bin/python -m pytest tests/test_data.py -q
"""

from __future__ import annotations

import pytest

from flyconn.models.data import dataset_spec


def test_mnist_spec():
    s = dataset_spec("mnist")
    assert (s.H, s.W, s.C, s.n_classes, s.n_pixels) == (28, 28, 1, 10, 784)
    assert s.tv_class == "MNIST" and s.subdir == "mnist"


def test_mnist_is_default_when_unspecified():
    # color is ignored for mnist (always single channel).
    assert dataset_spec("mnist", color="rgb").n_pixels == 784


def test_cifar_luma_spec():
    s = dataset_spec("cifar10", color="luma")
    assert (s.H, s.W, s.C, s.n_classes, s.n_pixels) == (32, 32, 1, 10, 1024)
    assert s.tv_class == "CIFAR10" and s.subdir == "cifar10"


def test_cifar_luma_is_default_color():
    assert dataset_spec("cifar10").n_pixels == 1024


def test_cifar_rgb_spec():
    s = dataset_spec("cifar10", color="rgb")
    assert (s.H, s.W, s.C, s.n_pixels) == (32, 32, 3, 3072)


def test_unknown_dataset_and_color_raise():
    with pytest.raises(ValueError):
        dataset_spec("imagenet")
    with pytest.raises(ValueError):
        dataset_spec("cifar10", color="hsv")
