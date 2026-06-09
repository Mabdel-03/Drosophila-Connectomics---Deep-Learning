"""Induced subgraph extraction for the six target networks.

Each subgraph is defined by a boolean selector over the neuron table. We induce the
signed adjacency submatrix and return EDGE BUFFERS in the orientation a recurrent
update needs:

    h_post = W @ h_pre        with   W[post, pre]

The stored artifact is source-major ``A[i, j] = i -> j`` (row = presynaptic), so we
TRANSPOSE when building buffers: ``row_idx = post``, ``col_idx = pre``.

Edge buffers (all length E = number of edges in the induced subgraph):
    row_idx : int64   local postsynaptic node id  (target of the edge)
    col_idx : int64   local presynaptic  node id  (source of the edge)
    sign    : float32 {+1, -1} from the presynaptic neuron's neurotransmitter
    count   : float32 raw synapse count (unsigned), for init-from-data magnitudes

No dense N x N matrix is ever materialized (139k^2 fp32 = 78 GB). Everything is sparse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..config import Config, load_config
from ..io import load_csr, read_parquet

# Photoreceptor cell types (the visual input cells; live under super_class 'sensory').
PHOTORECEPTOR_TYPES = ("R1-6", "R7", "R8")
# Visual super-classes (optic core + projections + centrifugal feedback).
VISUAL_SUPERCLASSES = ("optic", "visual_projection", "visual_centrifugal")


def visual_mask(neurons: pd.DataFrame) -> np.ndarray:
    """Boolean mask: optic/VP/VC neurons + photoreceptors (from sensory).

    Mirrors the stage-2 EDA definition so the 'optic' subgraphs include the retina.
    """
    in_vis_sc = neurons["super_class"].isin(VISUAL_SUPERCLASSES)
    is_photo = (neurons["super_class"] == "sensory") & neurons["cell_type"].isin(
        PHOTORECEPTOR_TYPES
    )
    return (in_vis_sc | is_photo).to_numpy()


# The six subgraph selectors. Each maps the neuron table -> bool mask over all nodes.
# 'optic*' use the visual_mask union (so photoreceptors are present as input cells).
SUBGRAPHS = {
    "whole": lambda n: np.ones(len(n), dtype=bool),
    "whole_right": lambda n: (n["side"] == "right").to_numpy(),
    "whole_left": lambda n: (n["side"] == "left").to_numpy(),
    "optic": lambda n: visual_mask(n),
    "optic_right": lambda n: visual_mask(n) & (n["side"] == "right").to_numpy(),
    "optic_left": lambda n: visual_mask(n) & (n["side"] == "left").to_numpy(),
}


@dataclass
class Subgraph:
    """An induced subgraph with transposed edge buffers (W[post, pre])."""

    subgraph_id: str
    policy: str
    N: int                       # number of nodes
    row_idx: np.ndarray          # int64[E] postsynaptic local id
    col_idx: np.ndarray          # int64[E] presynaptic  local id
    sign: np.ndarray             # float32[E] {+1,-1}
    count: np.ndarray            # float32[E] raw synapse counts (unsigned)
    local2global: np.ndarray     # int64[N] map local id -> global neuron idx
    neurons: pd.DataFrame        # the N rows of the neuron table (local order)

    @property
    def E(self) -> int:
        return len(self.row_idx)

    def node_local_ids(self, mask: np.ndarray) -> np.ndarray:
        """Local ids of nodes in this subgraph satisfying a boolean mask over its rows."""
        return np.flatnonzero(np.asarray(mask))


def build_subgraph(
    subgraph_id: str,
    *,
    policy: str = "flyvis_standard",
    cfg: Config | None = None,
) -> Subgraph:
    """Build the induced, transposed, signed edge buffers for one subgraph."""
    if subgraph_id not in SUBGRAPHS:
        raise KeyError(f"unknown subgraph {subgraph_id!r}; choices: {sorted(SUBGRAPHS)}")
    cfg = cfg or load_config()
    P = cfg.paths()
    neurons = read_parquet(P.neurons)

    keep = SUBGRAPHS[subgraph_id](neurons)
    idx = np.flatnonzero(keep)                       # global ids of kept nodes
    if idx.size == 0:
        raise ValueError(f"subgraph {subgraph_id!r} selected 0 nodes")

    A_signed = load_csr(P.adjacency_signed(policy))  # source-major A[i,j]=i->j, signed
    A_counts = load_csr(P.adjacency_counts)          # unsigned raw synapse counts

    sub_s = A_signed[idx][:, idx].tocoo()            # induced signed submatrix
    sub_c = A_counts[idx][:, idx].tocsr()            # induced counts (align via lookup)

    # Drop any structural zeros that survived slicing.
    nz = sub_s.data != 0
    pre = sub_s.row[nz]          # source-major: row = presynaptic
    post = sub_s.col[nz]         # source-major: col = postsynaptic
    signed_w = sub_s.data[nz]

    # TRANSPOSE for W[post, pre]: row_idx = post, col_idx = pre.
    row_idx = post.astype(np.int64)
    col_idx = pre.astype(np.int64)
    sign = np.sign(signed_w).astype(np.float32)

    # Raw unsigned counts for the same (pre, post) edges (init-from-data magnitudes).
    count = np.asarray(sub_c[pre, post]).ravel().astype(np.float32)
    count = np.abs(count)        # counts matrix is unsigned but guard anyway

    return Subgraph(
        subgraph_id=subgraph_id,
        policy=policy,
        N=int(idx.size),
        row_idx=row_idx,
        col_idx=col_idx,
        sign=sign,
        count=count,
        local2global=idx.astype(np.int64),
        neurons=neurons.iloc[idx].reset_index(drop=True),
    )


def subgraph_from_arrays(
    N: int, pre: np.ndarray, post: np.ndarray, sign: np.ndarray,
    count: np.ndarray | None = None, subgraph_id: str = "synthetic",
) -> Subgraph:
    """Construct a Subgraph directly from (pre, post, sign) arrays — for tests.

    ``pre``/``post`` are in source orientation (pre -> post); this transposes them
    into W[post, pre] buffers exactly like build_subgraph.
    """
    pre = np.asarray(pre, dtype=np.int64)
    post = np.asarray(post, dtype=np.int64)
    sign = np.asarray(sign, dtype=np.float32)
    count = (np.ones(len(pre), np.float32) if count is None
             else np.asarray(count, np.float32))
    return Subgraph(
        subgraph_id=subgraph_id, policy="synthetic", N=int(N),
        row_idx=post, col_idx=pre, sign=sign, count=count,
        local2global=np.arange(N, dtype=np.int64),
        neurons=pd.DataFrame({"idx": np.arange(N)}),
    )
