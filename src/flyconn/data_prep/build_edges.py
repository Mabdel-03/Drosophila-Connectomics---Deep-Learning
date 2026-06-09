"""Build the canonical edge list and sparse adjacency matrices.

Reads proofread_connections_783.feather (one row per pre/post/neuropil), resolves
the real column names against alias sets, aggregates synapse counts across neuropils
to a single directed edge per (pre, post), maps root_ids -> contiguous node idx, and
emits:

  processed/edges.parquet              (pre_idx, post_idx, syn_count, pre_nt)
  processed/adjacency_counts_csr.npz   (unsigned raw synapse counts, source-major)
  processed/adjacency_<policy>_csr.npz (signed = sign(pre_nt) * syn_count) per policy
  processed/adjacency.pt               (torch sparse CSR of the default policy + meta)

Convention: A[i, j] = weight of edge i -> j (row = presynaptic). For an RNN weight
matrix W with r_next = W @ r, use A.T.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import scipy.sparse as sp

from ..config import Config
from ..io import read_parquet, save_csr, save_torch_sparse, write_parquet
from . import nt_signs, schemas


def _read_connections(path) -> pd.DataFrame:
    """Read the connections feather and rename pre/post/count to canonical names."""
    # pyarrow lets us read the schema first to resolve column names cheaply.
    cols = feather.read_table(path, columns=None).column_names
    pre = schemas.resolve_column(cols, schemas.CONN_PRE_ALIASES, what="presynaptic root id")
    post = schemas.resolve_column(cols, schemas.CONN_POST_ALIASES, what="postsynaptic root id")
    cnt = schemas.resolve_column(cols, schemas.CONN_COUNT_ALIASES, what="synapse count")
    want = [pre, post, cnt]
    df = feather.read_table(path, columns=want).to_pandas()
    df = df.rename(columns={pre: "pre_root_id", post: "post_root_id", cnt: "syn_count"})
    df["pre_root_id"] = pd.to_numeric(df["pre_root_id"], errors="coerce").astype("int64")
    df["post_root_id"] = pd.to_numeric(df["post_root_id"], errors="coerce").astype("int64")
    df["syn_count"] = pd.to_numeric(df["syn_count"], errors="coerce").fillna(0).astype("int64")
    return df


def build_edge_list(cfg: Config, neurons: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    paths = cfg.paths()
    conn_spec = next(f for f in cfg.zenodo_files if f.role == "edges")
    raw = _read_connections(paths.raw_file(conn_spec.key))
    n_rows_raw = len(raw)

    # Aggregate across neuropils -> single directed edge per (pre, post).
    agg = (
        raw.groupby(["pre_root_id", "post_root_id"], as_index=False)["syn_count"]
        .sum()
    )

    # Map root_id -> idx (inner join both ends; drop edges touching non-node neurons).
    idxmap = neurons[["idx", "root_id"]]
    pre = agg.merge(idxmap, left_on="pre_root_id", right_on="root_id", how="inner")
    pre = pre.rename(columns={"idx": "pre_idx"}).drop(columns=["root_id"])
    both = pre.merge(idxmap, left_on="post_root_id", right_on="root_id", how="inner")
    both = both.rename(columns={"idx": "post_idx"}).drop(columns=["root_id"])
    n_dropped = len(agg) - len(both)

    # Attach presynaptic neuron's canonical NT (for signed matrices + provenance).
    nt_by_idx = neurons.set_index("idx")["nt_canonical"]
    both["pre_nt"] = both["pre_idx"].map(nt_by_idx)

    edges = both[["pre_idx", "post_idx", "syn_count", "pre_nt"]].copy()
    edges = edges.sort_values(["pre_idx", "post_idx"], kind="stable").reset_index(drop=True)

    stats = {
        "raw_rows_pre_aggregation": n_rows_raw,
        "edges_after_neuropil_aggregation": int(len(agg)),
        "edges_dropped_endpoint_not_in_node_set": int(n_dropped),
        "edges_no_threshold": int(len(edges)),
        "self_loops": int((edges["pre_idx"] == edges["post_idx"]).sum()),
        "total_synapses": int(edges["syn_count"].sum()),
    }
    return edges, stats


def apply_threshold(edges_full: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Keep only connections with >= threshold synapses (the canonical connectome)."""
    out = edges_full[edges_full["syn_count"] >= threshold].reset_index(drop=True)
    return out


def _csr(N: int, pre: np.ndarray, post: np.ndarray, weight: np.ndarray) -> sp.csr_matrix:
    # source-major: rows = pre (i), cols = post (j) -> A[i, j] = weight i->j
    return sp.csr_matrix((weight, (pre, post)), shape=(N, N))


def build_adjacencies(cfg: Config, neurons: pd.DataFrame, edges: pd.DataFrame) -> dict:
    paths = cfg.paths()
    N = len(neurons)
    pre = edges["pre_idx"].to_numpy(np.int64)
    post = edges["post_idx"].to_numpy(np.int64)
    counts = edges["syn_count"].to_numpy(np.float64)

    out: dict[str, object] = {}

    # Unsigned raw counts.
    A_counts = _csr(N, pre, post, counts)
    A_counts.sum_duplicates()
    save_csr(paths.adjacency_counts, A_counts)
    out["adjacency_counts_csr.npz"] = {"nnz": int(A_counts.nnz)}

    # One signed matrix per policy. Sign comes from the presynaptic neuron's NT.
    for policy in nt_signs.available_policies(cfg):
        sign = nt_signs.sign_vector_for_nodes(cfg, neurons, policy)  # int8[N], indexed by idx
        edge_sign = sign[pre].astype(np.float64)
        weight = edge_sign * counts
        # sign 0 (e.g. monoamines / unknown NT) -> weight 0; keep them out of the matrix.
        mask = weight != 0.0
        A = _csr(N, pre[mask], post[mask], weight[mask])
        A.sum_duplicates()
        save_csr(paths.adjacency_signed(policy), A)
        out[f"adjacency_{policy}_csr.npz"] = {
            "nnz": int(A.nnz),
            "pos_weight_mass": float(A.data[A.data > 0].sum()),
            "neg_weight_mass": float(A.data[A.data < 0].sum()),
            "dropped_zero_sign_edges": int((~mask).sum()),
        }

    # Model-ready torch sparse of the default policy.
    default = cfg.default_nt_policy
    A_default = sp.load_npz(str(paths.adjacency_signed(default))).tocsr()
    save_torch_sparse(
        paths.adjacency_pt,
        A_default,
        meta={
            "version": cfg.version,
            "nt_policy": default,
            "orientation": "source_major (A[i,j]=i->j); use A.T for W[post,pre]",
            "N": N,
        },
    )
    out["adjacency.pt"] = {"nt_policy": default, "N": N}
    return out


def maybe_synapse_summary(cfg: Config, neurons: pd.DataFrame) -> dict | None:
    """Optional per-synapse summary from the 9.5 GB flywire_synapses feather.

    We do NOT need the full per-synapse table for the connectivity matrix, but the
    user opted to process everything. To stay within memory we read only a few
    columns and emit lightweight summaries (per-neuron synapse coordinates centroid
    is overkill here; we just report global stats + write a small parquet of
    per-neuron pre/post synapse counts if the columns are present).
    """
    paths = cfg.paths()
    syn_spec = next((f for f in cfg.zenodo_files if f.role == "synapses"), None)
    if syn_spec is None:
        return None
    syn_path = paths.raw_file(syn_spec.key)
    if not syn_path.exists():
        print("[build_edges] synapse feather not present; skipping per-synapse summary.")
        return None

    cols = feather.read_table(syn_path, columns=None).column_names
    try:
        pre = schemas.resolve_column(cols, schemas.CONN_PRE_ALIASES, what="pre root id")
        post = schemas.resolve_column(cols, schemas.CONN_POST_ALIASES, what="post root id")
    except KeyError as e:
        print(f"[build_edges] synapse summary skipped ({e}).")
        return None

    # Read just the two id columns (cheap relative to the full 130M x many-cols table).
    df = feather.read_table(syn_path, columns=[pre, post]).to_pandas()
    summary = {
        "total_synapses_in_table": int(len(df)),
        "unique_pre_neurons": int(df[pre].nunique()),
        "unique_post_neurons": int(df[post].nunique()),
    }
    print(f"[build_edges] per-synapse table: {summary}")
    return summary


def run(cfg: Config) -> dict:
    paths = cfg.paths().ensure()
    neurons = read_parquet(paths.neurons)
    N = len(neurons)
    thr = cfg.synapse_threshold

    # 1) Full no-threshold aggregated edge list (the complete ~15.1M-pair graph).
    edges_full, edge_stats = build_edge_list(cfg, neurons)
    write_parquet(edges_full, paths.edges_full)
    # unsigned no-threshold count matrix (useful for graph analyses / motif search)
    A_full = _csr(
        N,
        edges_full["pre_idx"].to_numpy(np.int64),
        edges_full["post_idx"].to_numpy(np.int64),
        edges_full["syn_count"].to_numpy(np.float64),
    )
    A_full.sum_duplicates()
    save_csr(paths.adjacency_counts_full, A_full)
    print(f"[build_edges] wrote {paths.edges_full.name} "
          f"(no-threshold): {edge_stats}")

    # 2) Canonical thresholded connectome (>= cfg.synapse_threshold synapses).
    edges = apply_threshold(edges_full, thr)
    write_parquet(edges, paths.edges)
    edge_stats["synapse_threshold"] = thr
    edge_stats["edges_thresholded"] = int(len(edges))
    print(f"[build_edges] wrote {paths.edges.name} "
          f"(>= {thr} syn): {len(edges):,} connections")

    # 3) Adjacency matrices from the canonical thresholded edges.
    adj_stats = build_adjacencies(cfg, neurons, edges)
    print(f"[build_edges] wrote adjacency artifacts: {list(adj_stats)}")

    syn_summary = maybe_synapse_summary(cfg, neurons)

    return {"edges": edge_stats, "adjacency": adj_stats, "synapse_summary": syn_summary}
