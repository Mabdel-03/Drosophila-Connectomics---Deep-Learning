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


def _normalize_photoreceptor_sign(value: str | int) -> int | None:
    """Map the config value to a numeric override sign, or None for 'inherit'."""
    if value in ("inherit", None):
        return None
    v = int(value)
    if v not in (-1, 1):
        raise ValueError(f"photoreceptor_sign must be 'inherit', -1, or 1; got {value!r}")
    return v


def _normalize_force_sign(value: str | int | None) -> int | None:
    """Map the config value to a GLOBAL sign override, or None for 'none'/'inherit'."""
    if value in ("none", "inherit", None):
        return None
    v = int(value)
    if v not in (-1, 1):
        raise ValueError(f"force_sign must be 'none', -1, or 1; got {value!r}")
    return v


def _restore_photoreceptor_edges(
    sub_s: "sp.coo_matrix", sub_c: "sp.csr_matrix", sub_neurons: pd.DataFrame
) -> tuple["sp.coo_matrix", "sp.csr_matrix"]:
    """Re-add photoreceptor presynaptic edges that the NT policy dropped to sign 0.

    The signed CSR omits edges whose presynaptic NT maps to sign 0 (the ~956 sero/octo/
    dopa/None-labelled photoreceptors). Those edges still exist in the unsigned counts
    matrix. We graft them back into ``sub_s`` with a placeholder +1 (the caller's sign
    override replaces it), so every real photoreceptor->target synapse is represented.
    Both matrices are source-major (row = presynaptic) over local subgraph ids.
    """
    is_photo = sub_neurons["cell_type"].isin(PHOTORECEPTOR_TYPES).to_numpy()
    photo_local = np.flatnonzero(is_photo)
    if photo_local.size == 0:
        return sub_s, sub_c
    # All count edges out of photoreceptors (these are the biologically real ones).
    cc = sub_c.tocoo()
    photo_set = set(photo_local.tolist())
    keep = np.fromiter((r in photo_set for r in cc.row), dtype=bool, count=cc.nnz)
    add_pre, add_post, add_cnt = cc.row[keep], cc.col[keep], cc.data[keep]
    # Which (pre, post) already carry a nonzero sign? Don't double-add those.
    ss = sub_s.tocoo()
    existing = set(zip(ss.row.tolist(), ss.col.tolist()))
    new = np.fromiter(
        (((p, q) not in existing) for p, q in zip(add_pre.tolist(), add_post.tolist())),
        dtype=bool, count=add_pre.size,
    )
    if not new.any():
        return sub_s, sub_c
    new_rows = np.concatenate([ss.row, add_pre[new]])
    new_cols = np.concatenate([ss.col, add_post[new]])
    new_data = np.concatenate([ss.data, np.ones(int(new.sum()), dtype=ss.data.dtype)])
    restored = sp.coo_matrix((new_data, (new_rows, new_cols)), shape=sub_s.shape)
    return restored, sub_c


def build_subgraph(
    subgraph_id: str,
    *,
    policy: str = "flyvis_standard",
    photoreceptor_sign: str | int = "inherit",
    sign_shuffle: str | int | None = "none",
    force_sign: str | int | None = "none",
    cfg: Config | None = None,
) -> Subgraph:
    """Build the induced, transposed, signed edge buffers for one subgraph.

    ``photoreceptor_sign`` corrects a known data artifact: the FlyWire predicted-NT
    classifier has NO histamine class, so photoreceptors (R1-6/R7/R8) carry wrong NT
    labels (ACh/Glut/GABA/...) and ~956 are even sign-0 (dropped -> inject no drive).
    Real photoreceptors are histaminergic and INHIBITORY/sign-inverting onto L1/L2
    (flyvis hand-sets these to -1). With ``photoreceptor_sign=-1`` every edge whose
    PREsynaptic neuron is a photoreceptor is forced to sign -1 (and the dropped edges
    are restored), as a per-edge override on top of the NT policy. ``"inherit"`` keeps
    the raw (artifactual) NT-derived signs — the original stage-3 behavior.

    Two further sign overrides support the magnitude-vs-direction init ablation. They are
    applied AFTER the photoreceptor override (so they compose predictably), in this order:
      * ``sign_shuffle`` (an int seed, or ``"none"``): randomly PERMUTE the ±1 signs across
        edges. Preserves the E:I ratio (the count of + vs -) but destroys WHICH edge gets
        which sign — the null control separating "amount of inhibition" from "correct
        inhibition placement".
      * ``force_sign`` (``"none"`` | ``+1`` | ``-1``): a GLOBAL override that sets EVERY
        edge's sign to the given value. ``force_sign=1`` is the "no-direction" ablation
        (all excitatory, an unsigned adjacency). It dominates both the photoreceptor
        override and ``sign_shuffle`` (use one or the other, not both, in practice).
    All overrides operate on the SAME restored edge set, so the connectivity mask is
    identical across the ablation's variants — only the per-edge sign/magnitude changes.
    """
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

    photoreceptor_sign = _normalize_photoreceptor_sign(photoreceptor_sign)
    if photoreceptor_sign is not None:
        # Edges dropped by the NT policy (sign 0) where the presynaptic neuron is a
        # photoreceptor must be RESTORED before we filter structural zeros, else they
        # never enter the edge buffers. Source-major sub_s: row = presynaptic.
        sub_s, sub_c = _restore_photoreceptor_edges(
            sub_s, sub_c, neurons.iloc[idx].reset_index(drop=True)
        )

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

    if photoreceptor_sign is not None:
        # Force the sign of every edge whose PRESYNAPTIC neuron (local col_idx) is a
        # photoreceptor. This colocates the biology fix with sign assembly (O(E)).
        sub_neurons = neurons.iloc[idx].reset_index(drop=True)
        is_photo = sub_neurons["cell_type"].isin(PHOTORECEPTOR_TYPES).to_numpy()
        pre_is_photo = is_photo[col_idx]
        sign[pre_is_photo] = float(photoreceptor_sign)

    # Init-ablation sign overrides, applied (in order) AFTER the photoreceptor fix so they
    # operate on the final restored edge set. Mask/edges are untouched — only signs change.
    if sign_shuffle not in ("none", None):
        # Permute the ±1 signs across edges: keeps the E:I ratio, destroys placement.
        rng = np.random.default_rng(int(sign_shuffle))
        sign = sign[rng.permutation(len(sign))].copy()
    force_sign = _normalize_force_sign(force_sign)
    if force_sign is not None:
        sign[:] = float(force_sign)        # global override; dominates everything above

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
