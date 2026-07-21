"""Unit tests for Family L (FD3 -> descending neurons -> motor) — pure functions on synthetic
inputs, no live CAVE and no big files.

Run: /orcd/home/002/mabdel03/conda_envs/consortium/bin/python -m pytest tests/test_l_fd3_descending.py -q
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "src")

from flyconn.motif import compare as K
from flyconn.paper.derive import l_fd3_descending as LD
from flyconn.paper.oracle import l_fd3_descending as LO


# --- derive: _dn_block summarises descending rows of a partner table ---
def _partner_table(rows):
    """rows: list of (cell_type, syn, super_class, nt). One row per (synthetic) partner cell."""
    return pd.DataFrame([
        {"root_id": i, "syn": s, "cell_type": ct, "super_class": sc, "nt_canonical": nt}
        for i, (ct, s, sc, nt) in enumerate(rows)
    ])


def test_dn_block_ranks_and_flags_steering():
    pt = _partner_table([
        ("DNp26", 68, "descending", "acetylcholine"),
        ("DNp12", 18, "descending", "acetylcholine"),
        ("DNge094", 10, "descending", "acetylcholine"),
        ("SomeCentral", 500, "central", "acetylcholine"),   # not descending -> excluded
    ])
    b = LD._dn_block(pt)
    assert b["dn_syn"] == 68 + 18 + 10
    assert b["n_dns"] == 3
    assert b["top_dn"] == "DNp26"
    # DNp26 + DNge094 are steering DNs; DNp12 is not.
    assert b["steering_syn"] == 68 + 10
    assert b["steering_frac"] == pytest.approx(100.0 * 78 / 96, abs=0.1)
    assert b["ranking"][0]["cell_type"] == "DNp26" and b["ranking"][0]["is_steering"] is True


def test_dn_block_empty():
    pt = _partner_table([("SomeCentral", 500, "central", "acetylcholine")])
    b = LD._dn_block(pt)
    assert b["dn_syn"] == 0 and b["n_dns"] == 0 and b["top_dn"] is None


# --- derive: relay seed selection respects the synapse floor ---
def test_relay_seeds_threshold():
    pt = _partner_table([
        ("PLP230", 687, "central", "acetylcholine"),
        ("WED075", 25, "central", "acetylcholine"),
        ("DNp26", 68, "descending", "acetylcholine"),   # descending: never a relay seed
    ])
    seeds = LD._relay_seeds(pt, min_syn=30)
    assert set(seeds["cell_type"]) == {"PLP230"}        # WED075 below floor, DNp26 excluded
    seeds2 = LD._relay_seeds(pt, min_syn=10)
    assert set(seeds2["cell_type"]) == {"PLP230", "WED075"}


# --- derive: _relay_at filters cached second-hop rows by threshold ---
def _hop_rows(rows):
    """rows: list of (pre_root, post_root, syn, post_super_class, post_cell_type, post_nt)."""
    return pd.DataFrame([
        {"pre_pt_root_id": pr, "post_pt_root_id": po, "syn": s,
         "post_super_class": psc, "post_cell_type": pct, "post_nt": nt}
        for pr, po, s, psc, pct, nt in rows
    ])


def test_relay_at_counts_dns_above_threshold():
    seeds_all = pd.DataFrame({"root_id": [10, 11], "cell_type": ["PLP230", "WED075"],
                              "syn": [687, 25], "super_class": ["central", "central"]})
    hop = _hop_rows([
        (10, 100, 200, "descending", "DNae002", "acetylcholine"),
        (10, 101, 50, "descending", "DNbe001", "acetylcholine"),
        (11, 102, 30, "descending", "DNp18", "glutamate"),   # only present at low threshold
        (10, 103, 999, "central", "SomethingCentral", "gaba"),  # non-DN -> excluded
    ])
    hi = LD._relay_at(seeds_all, hop, min_syn=30)   # only PLP230 (687) qualifies
    assert hi["n_seed_cells"] == 1
    assert hi["top_dn"] == "DNae002"
    assert hi["dn_syn"] == 250                      # DNae002 200 + DNbe001 50
    lo = LD._relay_at(seeds_all, hop, min_syn=10)   # both seeds qualify
    assert lo["n_seed_cells"] == 2
    assert lo["dn_syn"] == 280                      # adds DNp18 30


def test_relay_at_empty_when_no_seed_clears_threshold():
    seeds_all = pd.DataFrame({"root_id": [10], "cell_type": ["PLP230"], "syn": [25],
                              "super_class": ["central"]})
    hop = _hop_rows([(10, 100, 200, "descending", "DNae002", "acetylcholine")])
    r = LD._relay_at(seeds_all, hop, min_syn=30)    # 25 < 30 -> nothing
    assert r["n_seed_cells"] == 0 and r["dn_syn"] == 0 and r["top_dn"] is None


# --- derive: motor-system classification from MaleCNS subclass codes ---
def _motor_edges(rows):
    """rows: list of (subclass, muscle, is_wing_steering, syn)."""
    return pd.DataFrame([
        {"body_pre": 1, "body_post": 1000 + i, "subclass": sc, "muscle": m,
         "is_wing_steering": ws, "syn": s, "somaSide": "R"}
        for i, (sc, m, ws, s) in enumerate(rows)
    ])


def test_system_breakdown_subclass_codes():
    e = _motor_edges([
        ("wm", "hg1", True, 100),     # wing steering
        ("wm", "DLM", False, 40),     # wing power (DLM excluded from steering)
        ("nm", "CvN6", False, 30),    # neck/gaze
        ("hm", "MNhm42", False, 20),  # haltere
        ("fl", "Ti extensor", False, 10),  # leg
    ])
    b = LD._system_breakdown(e)
    assert b["wing_steering"] == 100
    assert b["wing_power"] == 40
    assert b["neck_gaze"] == 30
    assert b["haltere"] == 20
    assert b["leg"] == 10


def test_wing_category_thresholds():
    assert LD._wing_category(0.99) == "ipsilateral"
    assert LD._wing_category(0.23) == "contralateral"
    assert LD._wing_category(0.50) == "bilateral"


# --- oracle: claim construction over a synthetic derive payload ---
def _derive_payload(motor_available=True):
    direct = {
        "dn_syn": 138, "n_dns": 24, "steering_frac": 58.0, "top_dn": "DNp26",
        "ranking": [
            {"cell_type": "DNp26", "syn": 68, "n_cells": 2, "nt": "acetylcholine", "is_steering": True},
            {"cell_type": "DNge094", "syn": 10, "n_cells": 5, "nt": "acetylcholine", "is_steering": True},
            {"cell_type": "DNp12", "syn": 18, "n_cells": 2, "nt": "acetylcholine", "is_steering": False},
        ],
        "per_side": {"right": {"dn_syn": 74}, "left": {"dn_syn": 64}},
    }
    relay = {"dn_syn": 10029, "n_dns": 263, "top_dn": "DNae002", "n_seed_types": 16,
             "ranking": [{"cell_type": "DNbe001", "syn": 644, "is_steering": True}],
             "seed_types": [{"cell_type": "PLP230", "fd3_syn": 687}]}
    overlap = {"n_overlap": 16, "overlap": ["DNp26", "DNbe001", "DNge094", "DNg82"]}
    motor = {"available": motor_available}
    if motor_available:
        motor.update({
            "dominant_motor_system": "wing_steering",
            "motor_system_pct": {"wing_steering": 43.5, "neck_gaze": 17.3},
            "per_dn": {
                "DNp26": {"n_bodies": 2, "ipsi_frac": 0.23, "wing": "contralateral",
                          "top_muscles": {"hg1": 122, "hg2": 117, "i1": 105}},
                "DNa04": {"n_bodies": 2, "ipsi_frac": 0.99, "wing": "ipsilateral",
                          "top_muscles": {"hg1": 138}},
            },
        })
    else:
        motor["reason"] = "MaleCNS annotations not found"
    return {"direct": direct, "relay": relay, "overlap": overlap, "motor": motor,
            "relay_min_syn": 30, "track": "offline"}


def test_oracle_all_confirmed_no_refutes():
    claims = LO.build_claims(_derive_payload())
    vc = K.verdict_counts(claims)
    assert vc.get("REFUTED", 0) == 0
    assert vc.get("CONFIRMED", 0) >= 8
    ids = {c.id for c in claims}
    assert "L.converges_on_dnp26" in ids
    assert "L.DNp26.wing" in ids and "L.DNa04.wing" in ids
    assert "L.dnp26_muscles" in ids
    # the behavioural reading is recorded as a (proxy) prediction, not a hard verdict
    assert any(c.id == "L.steering_dominant" and c.verdict == K.UNVERIFIABLE for c in claims)


def test_oracle_dnp26_convergence_refutes_if_top_changes():
    d = _derive_payload()
    d["direct"]["top_dn"] = "DNp12"          # not a steering DN, not DNp26
    claims = {c.id: c for c in LO.build_claims(d)}
    assert claims["L.converges_on_dnp26"].verdict == K.REFUTED
    assert claims["L.top_direct_is_steering"].verdict == K.REFUTED


def test_oracle_wing_laterality_refutes_on_mismatch():
    d = _derive_payload()
    d["motor"]["per_dn"]["DNp26"]["ipsi_frac"] = 0.95   # would read ipsilateral, paper says contra
    claims = {c.id: c for c in LO.build_claims(d)}
    assert claims["L.DNp26.wing"].verdict == K.REFUTED


def test_oracle_graceful_without_malecns():
    claims = LO.build_claims(_derive_payload(motor_available=False))
    ids = {c.id: c for c in claims}
    assert ids["L.motor_unavailable"].verdict == K.UNVERIFIABLE
    # the brain-side claims still resolve.
    assert ids["L.converges_on_dnp26"].verdict == K.CONFIRMED
    # no per-DN wing claims when MaleCNS is absent.
    assert not any(cid.endswith(".wing") for cid in ids)


def test_descending_summary_shape():
    s = LO.descending_summary(_derive_payload())
    assert s["top_direct_dn"] == "DNp26"
    assert s["dominant_motor_system"] == "wing_steering"
    assert s["n_relay_dns"] == 263


# --- report: reader-facing prose carries no internal verdict/claims jargon ---
def test_descending_report_tex_is_reader_facing():
    from flyconn.fd3_report import descending as RD
    results = {"meta": {"flywire_track": "offline"},
               "families": {"L": {"derived": _derive_payload()}}}
    tex = RD.build_tex(results)
    low = tex.lower()
    for banned in ["confirmed", "refuted", "unverifiable", "claim", "verdict",
                   "core", "discriminating", "corroborating", "oracle",
                   "pre-registered", "falsifier"]:
        assert banned not in low, f"jargon leaked into reader-facing report: {banned!r}"
    assert "anatomical proxy" in low
    # the headline biology IS present.
    assert "DNp26" in tex and "wing" in low and "descending" in low
