"""Unit tests for stage-1 pure logic (no network, no big files).

Run:  python -m pytest tests/ -q   (inside the consortium env)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flyconn.config import load_config
from flyconn.data_prep import build_edges, nt_signs, schemas


# ---------------- schema resolution ----------------
def test_resolve_column_case_insensitive():
    cols = ["Pre_Root_Id", "POST_ROOT_ID", "syn_count"]
    assert schemas.resolve_column(cols, schemas.CONN_PRE_ALIASES, what="pre") == "Pre_Root_Id"
    assert schemas.resolve_column(cols, schemas.CONN_POST_ALIASES, what="post") == "POST_ROOT_ID"
    assert schemas.resolve_column(cols, schemas.CONN_COUNT_ALIASES, what="cnt") == "syn_count"


def test_resolve_column_missing_raises():
    with pytest.raises(KeyError):
        schemas.resolve_column(["foo", "bar"], schemas.CONN_PRE_ALIASES, what="pre")


def test_canonical_nt_normalizes_aliases():
    assert schemas.canonical_nt("ACH") == "acetylcholine"
    assert schemas.canonical_nt("gabaergic") == "gaba"
    assert schemas.canonical_nt("Glut") == "glutamate"
    assert schemas.canonical_nt("5HT") == "serotonin"
    assert schemas.canonical_nt("") is None
    assert schemas.canonical_nt("nan") is None
    assert schemas.canonical_nt(None) is None


# ---------------- config ----------------
def test_config_loads_and_has_expected_counts():
    cfg = load_config()
    assert cfg.version == "783"
    assert cfg.expected_neurons == 139255
    assert cfg.expected_connections == 2700513   # >=5 syn, summed across neuropils
    assert cfg.synapse_threshold == 5
    # the 5 zenodo + 3 github files are all declared
    assert len(cfg.zenodo_files) == 5
    assert len(cfg.github_files) == 3
    # primary edge + node-set roles present
    roles = {f.role for f in cfg.zenodo_files}
    assert {"edges", "node_set", "synapses"} <= roles


def test_zenodo_urls_resolved():
    cfg = load_config()
    es = cfg.file("proofread_connections_783.feather")
    assert es.url.endswith("proofread_connections_783.feather/content")
    assert es.md5 == "f48f972d262323a102aed49af1396b8a"


# ---------------- nt sign policy ----------------
def _toy_neurons():
    # idx 0..4 with assorted NTs
    return pd.DataFrame({
        "idx": [0, 1, 2, 3, 4],
        "root_id": [10, 20, 30, 40, 50],
        "nt_canonical": ["acetylcholine", "gaba", "glutamate", "dopamine", None],
        "super_class": ["central"] * 5,
    })


def test_sign_vector_flyvis_standard():
    cfg = load_config()
    neurons = _toy_neurons()
    sign = nt_signs.sign_vector_for_nodes(cfg, neurons, "flyvis_standard")
    # ACh +1, GABA -1, Glut -1 (inhibitory in fly!), dopamine 0, None 0
    assert list(sign) == [1, -1, -1, 0, 0]


def test_sign_vector_glut_excitatory_differs_only_on_glut():
    cfg = load_config()
    neurons = _toy_neurons()
    sign = nt_signs.sign_vector_for_nodes(cfg, neurons, "glut_excitatory")
    assert list(sign) == [1, -1, 1, 0, 0]


def test_unknown_policy_raises():
    cfg = load_config()
    with pytest.raises(KeyError):
        nt_signs.sign_vector_for_nodes(cfg, _toy_neurons(), "nope")


# ---------------- edge build logic (toy, in-memory) ----------------
def _patch_read_connections(monkeypatch, df):
    monkeypatch.setattr(build_edges, "_read_connections", lambda path: df)


def test_build_edge_list_aggregates_and_maps(monkeypatch):
    cfg = load_config()
    neurons = _toy_neurons()
    # raw connections: pre/post by root_id, split across two "neuropils" (rows)
    raw = pd.DataFrame({
        "pre_root_id":  [10, 10, 20, 30, 999],   # 999 not a node -> dropped
        "post_root_id": [20, 20, 30, 40, 10],
        "syn_count":    [3,  2,  5,  7,  4],
    })
    _patch_read_connections(monkeypatch, raw)
    edges, stats = build_edges.build_edge_list(cfg, neurons)

    # 10->20 aggregates 3+2=5; 20->30=5; 30->40=7; 999->10 dropped
    em = {(int(r.pre_idx), int(r.post_idx)): int(r.syn_count) for r in edges.itertuples()}
    assert em == {(0, 1): 5, (1, 2): 5, (2, 3): 7}
    assert stats["edges_dropped_endpoint_not_in_node_set"] == 1
    assert stats["edges_no_threshold"] == 3
    # pre_nt carried from presynaptic neuron
    pre_nt = {int(r.pre_idx): r.pre_nt for r in edges.itertuples()}
    assert pre_nt[0] == "acetylcholine" and pre_nt[1] == "gaba"


def test_apply_threshold_keeps_only_strong_connections():
    edges_full = pd.DataFrame({
        "pre_idx":  [0, 1, 2, 3],
        "post_idx": [1, 2, 3, 4],
        "syn_count": [4, 5, 6, 1],   # >=5 keeps the middle two
        "pre_nt": ["acetylcholine"] * 4,
    })
    out = build_edges.apply_threshold(edges_full, 5)
    assert len(out) == 2
    assert set(out["syn_count"]) == {5, 6}


class _TmpPaths:
    """Minimal paths shim writing all artifacts under one tmp dir."""

    def __init__(self, root, version):
        self._root = root
        self.version = version

    def ensure(self):
        return self

    @property
    def neurons(self):
        return self._root / "neurons.parquet"

    @property
    def edges(self):
        return self._root / "edges.parquet"

    @property
    def adjacency_counts(self):
        return self._root / "adjacency_counts_csr.npz"

    def adjacency_signed(self, policy):
        return self._root / f"adjacency_{policy}_csr.npz"

    @property
    def adjacency_pt(self):
        return self._root / "adjacency.pt"


def test_build_adjacency_signs_and_no_dangling(monkeypatch, tmp_path):
    cfg = load_config()
    neurons = _toy_neurons()
    edges = pd.DataFrame({
        "pre_idx":  [0, 1, 2],          # ACh, GABA, Glut
        "post_idx": [1, 2, 3],
        "syn_count": [5, 5, 7],
        "pre_nt": ["acetylcholine", "gaba", "glutamate"],
    })
    # redirect all artifact writes into tmp by patching cfg.paths() -> shim
    monkeypatch.setattr(cfg.__class__, "paths",
                        lambda self: _TmpPaths(tmp_path, cfg.version))

    build_edges.build_adjacencies(cfg, neurons, edges)
    import scipy.sparse as sp
    A = sp.load_npz(str(tmp_path / "adjacency_flyvis_standard_csr.npz")).tocsr()
    # ACh edge +5 (0->1), GABA -5 (1->2), Glut -7 (2->3, inhibitory in fly)
    assert A[0, 1] == 5
    assert A[1, 2] == -5
    assert A[2, 3] == -7
    # shape spans the full node set -> no dangling indices possible
    assert A.shape == (len(neurons), len(neurons))
    # torch .pt of the default policy was written
    assert (tmp_path / "adjacency.pt").exists()
