"""Unit tests for CAVE column normalisation - no network.

We exercise the client's private normalisers directly on fixture DataFrames with
CAVE-style column names, asserting they map to the repo-canonical vocabulary so the
existing motif/circuit derivations work on CAVE results unchanged.
"""

from __future__ import annotations

import pandas as pd

from flyconn.connectome import ConnectomeClient


def _client():
    # allow_network=False is fine; we only call the pure normalisers.
    return ConnectomeClient("mcns", allow_network=False)


def test_synapse_columns_normalized_and_thresholded():
    c = _client()
    raw = pd.DataFrame({
        "pre_pt_root_id": [1, 2, 3],
        "post_pt_root_id": [10, 20, 30],
        "cleftscore": [40, 80, 120],     # alias for cleft_score
        "neuropil_region": ["LOP_R", "LOP_R", "LOP_R"],
    })
    out = c._normalize_synapses(raw, cleft_threshold=50)
    assert "cleft_score" in out.columns and "neuropil" in out.columns
    assert list(out["cleft_score"]) == [80, 120]   # 40 dropped by threshold


def test_synapse_no_threshold_keeps_all():
    c = _client()
    raw = pd.DataFrame({"pre_root_id": [1], "post_root_id": [2], "cleft_score": [10]})
    out = c._normalize_synapses(raw, cleft_threshold=None)
    assert len(out) == 1 and "pre_pt_root_id" in out.columns and "post_pt_root_id" in out.columns


def test_annotation_columns_normalized_with_nt_canonical():
    c = _client()
    raw = pd.DataFrame({
        "pt_root_id": [100, 200],
        "type": ["DNp26", "hg1"],          # -> cell_type
        "soma_side": ["RHS", "LHS"],       # -> somaSide
        "super_class": ["descending", "motor"],   # -> cell_sub_class
        "top_nt": ["acetylcholine", "gaba"],
    })
    out = c._normalize_annotations(raw)
    assert {"root_id", "cell_type", "somaSide", "cell_sub_class", "nt_canonical"} <= set(out.columns)
    assert list(out["nt_canonical"]) == ["acetylcholine", "gaba"]
    assert list(out["somaSide"]) == ["RHS", "LHS"]


def test_normalizer_idempotent_on_canonical_names():
    c = _client()
    raw = pd.DataFrame({"pre_pt_root_id": [1], "post_pt_root_id": [2], "cleft_score": [99]})
    out = c._normalize_synapses(raw, cleft_threshold=None)
    assert list(out.columns) == list(raw.columns)
