"""Filesystem path resolution for flyconn.

All bulk data lives on scratch (the group disk is ~87% full), under a single
version-stamped root resolved from the ``FLYCONN_DATA_ROOT`` env var so nothing is
hardcoded and a collaborator can redirect it. Nothing here imports heavy deps.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Default scratch root (overridable via FLYCONN_DATA_ROOT). The version subdir
# (e.g. v783) is appended by DataPaths so multiple FlyWire releases coexist.
DEFAULT_DATA_ROOT = "/orcd/scratch/orcd/012/mabdel03/connectome_data"

# Repo root = three parents up from this file (src/flyconn/paths.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data_v783.yaml"


def data_root() -> Path:
    """Base scratch directory for all FlyWire data, from env or default."""
    return Path(os.environ.get("FLYCONN_DATA_ROOT", DEFAULT_DATA_ROOT))


# Live API caches (synapse pulls, skeletons) are SMALL and re-auditable but must NOT land on
# scratch, which is over quota. They default to the project tree (group volume, tractable),
# kept separate from data_root() so the large offline dumps still resolve from scratch.
DEFAULT_CACHE_ROOT = str(REPO_ROOT / ".flyconn_cache")


def cache_root() -> Path:
    """Base directory for live-API caches (synapses, skeletons), from env or the project tree."""
    return Path(os.environ.get("FLYCONN_CACHE_ROOT", DEFAULT_CACHE_ROOT))


@dataclass(frozen=True)
class DataPaths:
    """Resolved, version-stamped directories and artifact paths."""

    version: str
    root: Path  # <data_root>/v<version>

    @classmethod
    def for_version(cls, version: str, root: Path | None = None) -> "DataPaths":
        base = root if root is not None else (data_root() / f"v{version}")
        return cls(version=version, root=Path(base))

    # --- top-level dirs ---
    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> "DataPaths":
        """Create raw/processed/reports (idempotent). Returns self for chaining."""
        for d in (self.raw, self.processed, self.reports):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # --- raw inputs ---
    @property
    def download_manifest(self) -> Path:
        return self.raw / "_download_manifest.json"

    def raw_file(self, name: str) -> Path:
        return self.raw / name

    # --- processed artifacts ---
    @property
    def neurons(self) -> Path:
        return self.processed / "neurons.parquet"

    @property
    def node_index_map(self) -> Path:
        return self.processed / "node_index_map.parquet"

    @property
    def edges(self) -> Path:
        """Canonical thresholded edge list (>= synapse_threshold)."""
        return self.processed / "edges.parquet"

    @property
    def edges_full(self) -> Path:
        """No-threshold aggregated edge list (the full ~15.1M-pair graph)."""
        return self.processed / "edges_full.parquet"

    @property
    def adjacency_counts(self) -> Path:
        return self.processed / "adjacency_counts_csr.npz"

    @property
    def adjacency_counts_full(self) -> Path:
        """No-threshold unsigned raw-count adjacency."""
        return self.processed / "adjacency_counts_full_csr.npz"

    def adjacency_signed(self, policy: str) -> Path:
        return self.processed / f"adjacency_{policy}_csr.npz"

    @property
    def adjacency_pt(self) -> Path:
        return self.processed / "adjacency.pt"

    @property
    def schema(self) -> Path:
        return self.processed / "_schema.json"

    # --- reports ---
    @property
    def data_card(self) -> Path:
        return self.reports / f"data_card_v{self.version}.md"
