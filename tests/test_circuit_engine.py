"""Unit tests for the generic circuit engine - claim dispatch, oracle cross-check, nulls.

No network: claims are evaluated on pre-derived values, and the oracle cross-check runs on
plain dicts. Confirms the engine reuses motif.compare verdicts correctly (CONFIRMED on a
matching fixture, REFUTED on a deliberately-wrong oracle row).
"""

from __future__ import annotations

import numpy as np

from flyconn.circuit import engine as E
from flyconn.circuit import report as R
from flyconn.circuit.spec import CircuitSpec, EdgeClaim
from flyconn.motif import compare as K


def _claim(report_value, compare="count", **kw):
    return EdgeClaim(id="c", description="d", src={"seed": "X"}, dst={"cell_type": "Y"},
                     dataset="mcns", metric="syn_count", report_value=report_value,
                     compare=compare, compare_kwargs=kw)


def test_evaluate_claim_count_confirmed():
    r = E.evaluate_claim(_claim(100, rel=0.1, abs_floor=5, drift_dir="down"),
                         {"primary": 98, "secondary": 90})
    assert r.verdict == K.CONFIRMED


def test_evaluate_claim_count_caveat_when_drift_down():
    r = E.evaluate_claim(_claim(100, rel=0.1, abs_floor=5, drift_dir="down"),
                         {"primary": 50, "secondary": 40})
    assert r.verdict == K.CONFIRMED_WITH_CAVEAT


def test_evaluate_claim_count_refuted_when_wrong_direction():
    r = E.evaluate_claim(_claim(100, rel=0.1, abs_floor=5, drift_dir="down"),
                         {"primary": 300, "secondary": 280})
    assert r.verdict == K.REFUTED


def test_evaluate_claim_categorical():
    r = E.evaluate_claim(_claim("contralateral", compare="categorical"),
                         {"primary": "contralateral"})
    assert r.verdict == K.CONFIRMED
    r2 = E.evaluate_claim(_claim("ipsilateral", compare="categorical"),
                          {"primary": "contralateral"})
    assert r2.verdict == K.REFUTED


def test_crosscheck_oracle_confirmed_and_refuted():
    spec = CircuitSpec(name="t", datasets=("mcns",), seeds=(),
                       oracle_tables={"S14": {"DNp26": {"ipsi_frac": 0.23, "wing": "contralateral"}}})
    good = E.crosscheck_oracle(spec, {"S14": {"DNp26": {"ipsi_frac": 0.25, "wing": "contralateral"}}})
    assert all(c.verdict == K.CONFIRMED for c in good)

    bad = E.crosscheck_oracle(spec, {"S14": {"DNp26": {"ipsi_frac": 0.9, "wing": "ipsilateral"}}})
    assert all(c.verdict == K.REFUTED for c in bad)


def test_label_permutation_null_detects_real_bias():
    # 4 ipsi-heavy MNs (large syn) on RHS, contra MNs (small syn) on LHS, DN on RHS.
    syn = np.array([100, 100, 100, 100, 5, 5], dtype=float)
    sides = np.array(["RHS", "RHS", "RHS", "RHS", "LHS", "LHS"])

    def ipsi_frac(v, lab):
        return v[lab == "RHS"].sum() / v.sum()

    res = E.label_permutation_null(syn, sides, statistic=ipsi_frac, n_perm=300, rng_seed=0)
    assert res["observed"] > 0.9          # strongly ipsilateral
    assert res["p"] < 0.2 or abs(res["z"]) > 1.0   # separable from chance


def test_report_surfaces_flat_refuted_list():
    spec = CircuitSpec(name="t", datasets=("mcns",), seeds=(),
                       oracle_tables={"S14": {"DNp26": {"wing": "ipsilateral"}}})
    claims = E.crosscheck_oracle(spec, {"S14": {"DNp26": {"wing": "contralateral"}}})
    payload = R.build_results({"name": "t"}, claims)
    assert payload["refuted_claims"] == ["S14.DNp26.wing"]
    assert payload["meta"]["verdict_counts"].get(K.REFUTED) == 1
