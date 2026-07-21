"""Unit tests for the Stage-4 motif verification logic — pure functions on a tiny
synthetic graph, no big files or network.

Synthetic world (VCH root 999, all T4/T5 on the right, LPi14 a downstream sink):
  neurons: 999 VCH(left,gaba), 1 T4a(right,ach), 2 T4a(right,ach), 3 T5a(right,ach),
           4 LPi14(right,gaba), 5 LT52(right,ach) [substring contaminant]
  synapses:
    1->999, 2->999, 3->999   (T4/T5 inputs to VCH)
    999->1, 999->2           (VCH outputs; 1 & 2 reciprocal, 3 input-only)
    1->4, 2->4, 3->4         (reciprocal T4/T5 -> LPi14 downstream)

Run:  /orcd/home/002/mabdel03/conda_envs/consortium/bin/python -m pytest tests/test_motif_verify.py -q
"""

from __future__ import annotations

import io

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from flyconn.motif import compare as K
from flyconn.motif import downstream as D
from flyconn.motif import vch_verify as V
from flyconn.motif.synapse_extract import _filter_batch, _roots_array

VCH = 999


@pytest.fixture
def neurons():
    return pd.DataFrame({
        "idx": [0, 1, 2, 3, 4, 5],
        "root_id": [999, 1, 2, 3, 4, 5],
        "cell_type": ["VCH", "T4a", "T4a", "T5a", "LPi14", "LT52"],
        "side": ["left", "right", "right", "right", "right", "right"],
        "nt_canonical": ["gaba", "acetylcholine", "acetylcholine", "acetylcholine",
                         "gaba", "acetylcholine"],
        "super_class": ["visual_centrifugal", "optic", "optic", "optic", "optic", "optic"],
    })


def _syn(pre, post, neuropil="LOP_R", **nt):
    base = {"gaba": 0.1, "ach": 0.1, "glut": 0.1, "oct": 0.0, "ser": 0.0, "da": 0.0}
    base.update(nt)
    return {
        "pre_pt_root_id": pre, "post_pt_root_id": post, "neuropil": neuropil,
        "connection_score": 1.0, "cleft_score": 100,
        "pre_pt_position_x": 700, "pre_pt_position_y": 0, "pre_pt_position_z": 0,
        "post_pt_position_x": 720, "post_pt_position_y": 0, "post_pt_position_z": 0,
        **base,
    }


@pytest.fixture
def vch_syn():
    rows = [
        _syn(1, VCH, ach=0.9), _syn(2, VCH, ach=0.9), _syn(3, VCH, ach=0.9),  # inputs
        _syn(VCH, 1, gaba=0.9), _syn(VCH, 2, gaba=0.9),                       # outputs
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def downstream_syn():
    rows = [_syn(1, 4), _syn(2, 4), _syn(3, 4)]  # reciprocal T4/T5 -> LPi14
    return pd.DataFrame(rows)


# --- classification ---
def test_classify_t4t5_tiers():
    # The report's rule is a CASE-SENSITIVE substring of "T4"/"T5", so uppercase
    # types like LT52 are contaminants, but lowercase "MLt4" is not matched at all.
    s = pd.Series(["T4a", "T5d", "T4", "LT52", "LPT51", "MLt4", "VCH", None])
    tiers = V.classify_t4t5(s).tolist()
    assert tiers == ["T4a", "T5d", "Other/unclear", "contamination",
                     "contamination", "non_t4t5", "non_t4t5", "non_t4t5"]


def test_loose_vs_canonical_membership():
    tiers = V.classify_t4t5(pd.Series(["T4a", "T4", "LT52", "VCH"]))
    assert V.is_canonical_t4t5(tiers).tolist() == [True, False, False, False]
    assert V.is_loose_t4t5(tiers).tolist() == [True, True, True, False]


# --- overview ---
def test_derive_overview_counts(vch_syn, neurons):
    ins, outs = V.split_in_out(vch_syn, VCH)
    ov = V.derive_overview(ins, outs, "gaba")
    assert ov["total_input_syn"] == 3
    assert ov["total_output_syn"] == 2
    assert ov["upstream_partners"] == 3
    assert ov["downstream_partners"] == 2
    assert ov["nt"] == "gaba"


# --- subtype tables ---
def test_t4t5_input_subtype_table(vch_syn, neurons):
    ins, _ = V.split_in_out(vch_syn, VCH)
    lk = V.load_lookup(neurons)
    summary, table, in_roots = V.derive_t4t5_inputs(ins, lk, total_input_syn=3)
    assert summary["total_neurons_loose"] == 3          # roots 1,2,3
    assert summary["total_syn_loose"] == 3
    assert in_roots == {1, 2, 3}
    # T4a has 2 neurons / 2 syn; T5a 1 neuron / 1 syn.
    assert int(table.loc["T4a", "neurons"]) == 2
    assert int(table.loc["T4a", "syn"]) == 2
    assert int(table.loc["T5a", "neurons"]) == 1


# --- reciprocal ---
def test_derive_reciprocal_intersection(vch_syn, neurons):
    ins, outs = V.split_in_out(vch_syn, VCH)
    lk = V.load_lookup(neurons)
    _, _, in_roots = V.derive_t4t5_inputs(ins, lk, 3)
    _, _, out_roots = V.derive_t4t5_outputs(outs, lk, 2)
    summary, roots = V.derive_reciprocal(in_roots, out_roots)
    assert summary["n"] == 2                       # roots 1 & 2
    assert set(roots.tolist()) == {1, 2}
    assert summary["pct_of_inputs"] == pytest.approx(100 * 2 / 3, abs=0.1)
    assert summary["pct_of_outputs"] == 100.0      # both outputs are reciprocal


# --- hemisphere ---
def test_hemisphere_same_side(vch_syn, neurons):
    ins, outs = V.split_in_out(vch_syn, VCH)
    lk = V.load_lookup(neurons)
    hemi = V.derive_hemisphere(ins, outs, lk)
    assert hemi["input_dominant_hemi"] == "R"
    assert hemi["output_dominant_hemi"] == "R"
    assert hemi["input_output_differ"] is False    # refutes the report's claim


# --- NT synapse-level ---
def test_nt_synapse_level(vch_syn, neurons):
    ins, outs = V.split_in_out(vch_syn, VCH)
    lk = V.load_lookup(neurons)
    nt = V.derive_nt(outs, ins, lk)
    assert nt["vch_output_frac_gaba"] == 1.0       # both outputs argmax gaba
    assert nt["vch_output_plurality_nt"] == "gaba"
    assert nt["t4t5_input_frac_ach"] == 1.0        # all three inputs argmax ach
    assert nt["t4t5_input_plurality_nt"] == "ach"
    assert nt["t4t5_input_neuron_frac_ach"] == 1.0  # all annotated cholinergic


# --- downstream aggregation ---
def test_aggregate_downstream(downstream_syn, neurons):
    lk = V.load_lookup(neurons)
    targets = D.aggregate_targets(downstream_syn, lk)
    assert len(targets) == 1                        # only LPi14 (root 4)
    row = targets.iloc[0]
    assert row["root_id"] == 4
    assert row["syn"] == 3
    assert row["n_drivers"] == 3
    assert row["cell_type"] == "LPi14"


def test_celltype_populations_copies(neurons):
    # two distinct roots sharing cell_type "Foo" -> copies == 2.
    df = pd.DataFrame({
        "root_id": [10, 11], "syn": [5, 7], "n_drivers": [2, 3],
        "cell_type": ["Foo", "Foo"], "side": ["right", "right"],
        "nt_canonical": ["gaba", "gaba"], "super_class": ["optic", "optic"],
    })
    pops = D.celltype_populations(df)
    assert int(pops.loc[pops["cell_type"] == "Foo", "copies"].iloc[0]) == 2
    assert int(pops.loc[pops["cell_type"] == "Foo", "total_syn"].iloc[0]) == 12


def test_nodulus_filter():
    df = pd.DataFrame({
        "root_id": [1, 2, 3], "syn": [5, 6, 7], "n_drivers": [1, 1, 1],
        "cell_type": ["Nod1", "Nod12", "Nodulus"],  # only Nod1 matches ^Nod[0-9]$
        "side": ["right"] * 3, "nt_canonical": ["ach"] * 3, "super_class": ["x"] * 3,
    })
    nod = D.nodulus_subtable(df)
    assert nod["cell_type"].tolist() == ["Nod1"]


# --- extract filter (in-memory Arrow IPC) ---
def _build_ipc(rows):
    tbl = pa.table({
        "pre_pt_root_id": pa.array([r[0] for r in rows], pa.int64()),
        "post_pt_root_id": pa.array([r[1] for r in rows], pa.int64()),
    })
    buf = io.BytesIO()
    w = ipc.new_file(buf, tbl.schema)
    for b in tbl.to_batches():
        w.write_batch(b)
    w.close()
    buf.seek(0)
    return ipc.open_file(buf)


def test_filter_batch_or_semantics():
    reader = _build_ipc([(1, 999), (2, 3), (999, 5), (7, 8)])
    batch = reader.get_batch(0)
    pre = _roots_array({999})
    post = _roots_array({999})
    kept = _filter_batch(batch, pre, post)
    # rows touching 999 on either side: (1,999), (999,5) -> 2 rows.
    assert kept.num_rows == 2


def test_filter_batch_pre_only():
    reader = _build_ipc([(1, 999), (2, 3), (999, 5)])
    batch = reader.get_batch(0)
    kept = _filter_batch(batch, _roots_array({1, 999}), None)
    # pre in {1,999}: (1,999) and (999,5) -> 2.
    assert kept.num_rows == 2


# --- verdict engine ---
def test_compare_count_verdicts():
    assert K.compare_count("x", "x", 1000, 1010).verdict == K.CONFIRMED         # +1%
    # large downward gap, drift explains -> CONFIRMED_WITH_CAVEAT
    assert K.compare_count("x", "x", 1000, 600).verdict == K.CONFIRMED_WITH_CAVEAT
    # upward gap that drift can't explain -> REFUTED
    assert K.compare_count("x", "x", 1000, 2000).verdict == K.REFUTED


def test_compare_pct_and_categorical():
    assert K.compare_pct("x", "x", 47.2, 49.0).verdict == K.CONFIRMED           # 1.8 pp
    assert K.compare_pct("x", "x", 47.2, 60.0).verdict == K.REFUTED
    assert K.compare_categorical("x", "x", True, True).verdict == K.CONFIRMED
    assert K.compare_categorical("x", "x", True, False).verdict == K.REFUTED


def test_check_table_sum():
    assert K.check_table_sum("x", "x", 13018, 13018).verdict == K.CONFIRMED
    assert K.check_table_sum("x", "x", 9000, 13018).verdict == K.REFUTED
