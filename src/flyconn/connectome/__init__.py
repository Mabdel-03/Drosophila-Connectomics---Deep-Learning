"""Live connectome access (FAFB + male-CNS) via CAVE, cached to scratch.

The repo's Stage-1..4 work is offline-only (FlyWire FAFB v783 parquet/feather). The
muscular-projection completion needs the *male* whole-CNS connectome (MCNS), whose
DN->motor-neuron->muscle wiring is only reachable through the live CAVE API. This
package wraps ``caveclient`` for both datasets and caches every query to scratch in the
same parquet pattern as the offline data, so a populated cache makes verification fully
reproducible and offline.

``caveclient`` is imported lazily; importing this package does not require the ``[cave]``
extra (only live queries do).
"""

from __future__ import annotations

from .cache import cached_or_call, cache_path, query_hash
from .client import ConnectomeClient
from .datasets import DATASETS, FAFB, MCNS, CaveDataset, get_dataset
from .secrets import have_cave_token, read_cave_token

__all__ = [
    "ConnectomeClient",
    "CaveDataset",
    "DATASETS",
    "FAFB",
    "MCNS",
    "get_dataset",
    "read_cave_token",
    "have_cave_token",
    "query_hash",
    "cache_path",
    "cached_or_call",
]
