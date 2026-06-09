"""Single source of truth for column schemas.

The raw FlyWire feather files have evolved column names across releases/exports
(e.g. ``pre_root_id`` vs ``pre_pt_root_id``). Rather than hardcode one spelling,
we declare *alias sets* and resolve the actual column at load time, failing loudly
if none is present. This also documents the canonical output schema that all
downstream stages rely on.
"""

from __future__ import annotations

# --- Raw connectivity (proofread_connections_783.feather) column aliases ---
# Resolved case-insensitively against the file's real columns.
CONN_PRE_ALIASES = ("pre_root_id", "pre_pt_root_id", "pre", "presynaptic_root_id")
CONN_POST_ALIASES = ("post_root_id", "post_pt_root_id", "post", "postsynaptic_root_id")
CONN_COUNT_ALIASES = ("syn_count", "synapse_count", "count", "n_syn", "weight")
CONN_NEUROPIL_ALIASES = ("neuropil", "neuropil_region", "roi")
CONN_NT_ALIASES = ("nt_type", "neurotransmitter", "top_nt", "predicted_nt")

# --- Raw annotation TSV (Supplemental_file1) columns we keep, with the
# canonical (output) name on the left. Source columns are taken verbatim. ---
ANNOTATION_KEEP = [
    "root_id",
    "super_class",
    "cell_class",
    "cell_sub_class",
    "cell_type",
    "hemibrain_type",
    "side",
    "flow",
    "top_nt",
    "top_nt_conf",
    "ito_lee_hemilineage",
    "hartenstein_hemilineage",
    "nucleus_id",
    "fbbt_id",
    "soma_x",
    "soma_y",
    "soma_z",
    "pos_x",
    "pos_y",
    "pos_z",
]

# --- Canonical neurotransmitter vocabulary (lower-cased, normalized) ---
# Maps many observed spellings/abbreviations onto canonical keys used by nt_policies.
NT_CANONICAL = {
    "ach": "acetylcholine",
    "acetylcholine": "acetylcholine",
    "cholinergic": "acetylcholine",
    "gaba": "gaba",
    "gabaergic": "gaba",
    "glut": "glutamate",
    "glutamate": "glutamate",
    "glutamatergic": "glutamate",
    "oct": "octopamine",
    "octopamine": "octopamine",
    "octopaminergic": "octopamine",
    "ser": "serotonin",
    "serotonin": "serotonin",
    "serotonergic": "serotonin",
    "5ht": "serotonin",
    "da": "dopamine",
    "dop": "dopamine",
    "dopamine": "dopamine",
    "dopaminergic": "dopamine",
}

# --- Canonical OUTPUT schemas (documented in _schema.json) ---
NEURONS_SCHEMA = {
    "idx": "int64  — contiguous node index 0..N-1 (== row number); the ANN node id",
    "root_id": "int64  — FlyWire v783 root id (stable join key)",
    "super_class": "str  — highest-level class (e.g. central, optic, sensory, ...)",
    "cell_class": "str  — cell class annotation",
    "cell_sub_class": "str  — finer class",
    "cell_type": "str  — cell type",
    "hemibrain_type": "str  — matched hemibrain type (may be null)",
    "side": "str  — soma/nerve-entry side (left/right/center)",
    "flow": "str  — afferent/intrinsic/efferent",
    "top_nt": "str  — predicted top neurotransmitter (raw label)",
    "top_nt_conf": "float  — confidence of top_nt",
    "nt_canonical": "str  — top_nt normalized to {acetylcholine,gaba,glutamate,octopamine,serotonin,dopamine} or null",
    "ito_lee_hemilineage": "str",
    "hartenstein_hemilineage": "str",
    "nucleus_id": "Int64 (nullable)",
    "fbbt_id": "str  — VirtualFlyBrain ontology id",
    "soma_x": "float (4x4x40nm voxels)",
    "soma_y": "float",
    "soma_z": "float",
    "pos_x": "float  — anchor coord (4x4x40nm voxels)",
    "pos_y": "float",
    "pos_z": "float",
}

EDGES_SCHEMA = {
    "pre_idx": "int64  — presynaptic node index (row in neurons.parquet)",
    "post_idx": "int64  — postsynaptic node index",
    "syn_count": "int64  — total synapses pre->post (summed across neuropils)",
    "pre_nt": "str  — presynaptic neuron's canonical neurotransmitter",
}

EDGES_NOTE = (
    "edges.parquet is the CANONICAL connectome: connections with >= synapse_threshold "
    "synapses (default 5), summed across neuropils per pair (2,700,513 for v783, "
    "matching FlyWire's documented ~2.70M thresholded connections). edges_full.parquet "
    "is the NO-THRESHOLD aggregated graph (~15.1M pairs); "
    "adjacency_counts_full_csr.npz is its unsigned matrix. All adjacency_*.npz signed "
    "matrices and adjacency.pt are built from the thresholded canonical edges."
)

ADJACENCY_NOTE = (
    "scipy.sparse CSR, shape (N, N), source-major: A[i, j] = weight of edge i->j "
    "(row = presynaptic). adjacency_counts_csr.npz holds raw syn_count (unsigned). "
    "adjacency_<policy>_csr.npz holds sign(pre_nt under policy) * syn_count. "
    "For an RNN recurrent weight matrix W where r_next = W @ r, use A.T (W[post, pre])."
)


def resolve_column(columns, aliases, *, what: str) -> str:
    """Return the actual column name matching one of ``aliases`` (case-insensitive).

    Raises a clear error listing the available columns if none match.
    """
    lower = {c.lower(): c for c in columns}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    raise KeyError(
        f"Could not find a column for {what!r}. Tried aliases {aliases}. "
        f"Available columns: {sorted(columns)}"
    )


def canonical_nt(value) -> str | None:
    """Normalize a raw neurotransmitter label to the canonical vocabulary, or None."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s or s in {"nan", "none", "unknown", "na"}:
        return None
    return NT_CANONICAL.get(s)
