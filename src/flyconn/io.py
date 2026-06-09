"""I/O helpers: parquet tables, scipy sparse, torch sparse, JSON manifests/schema.

Torch is imported lazily so the package is usable for download/build of the parquet
and npz artifacts even in a torch-less environment; only the .pt writer needs it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ---------------- hashing ----------------
def md5_file(path: str | Path, chunk: int = 8 * 1024 * 1024) -> str:
    """Streaming MD5 of a file (handles the 9.5 GB feather without loading it)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------- json ----------------
def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def read_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ---------------- parquet ----------------
def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


# ---------------- sparse (scipy) ----------------
def save_csr(path: str | Path, mat: sp.csr_matrix) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(str(path), mat)


def load_csr(path: str | Path) -> sp.csr_matrix:
    return sp.load_npz(str(path)).tocsr()


# ---------------- sparse (torch) ----------------
def save_torch_sparse(path: str | Path, mat: sp.csr_matrix, meta: dict[str, Any]) -> None:
    """Write a CSR scipy matrix as a torch sparse_csr tensor + metadata dict.

    Stored on CPU; the modeling stage casts/normalizes/moves to GPU at build time.
    """
    import torch  # lazy

    mat = mat.tocsr()
    t = torch.sparse_csr_tensor(
        crow_indices=torch.from_numpy(mat.indptr.astype(np.int64)),
        col_indices=torch.from_numpy(mat.indices.astype(np.int64)),
        values=torch.from_numpy(mat.data.astype(np.float32)),
        size=tuple(mat.shape),
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adj": t, **meta}, str(path))
