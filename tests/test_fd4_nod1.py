"""Unit tests for the FD4 search logic: candidate elimination, Nod1 split, phenotype crosswalk,
null-aware confidence, and the oracle claims. Pure functions on synthetic input; no live data.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from flyconn.paper.derive import fd4_nod1 as F4  # noqa: E402
from flyconn.paper.oracle import fd4_nod1 as F4O  # noqa: E402
from flyconn.paper.derive import k_fd3_lpt42 as K  # noqa: E402
from flyconn.motif import compare as CMP  # noqa: E402


# ---------------------------------------------------------------------------
# Corrected FD signatures: FD4 has NO frontal gap (that is FD3-unique).
# ---------------------------------------------------------------------------
def test_fd4_signature_has_no_frontal_gap():
    fd4 = K.FD_SIGNATURES["FD4"]
    assert fd4["layer"] == "a"                      # progressive
    assert fd4["lateral_gap"] is False             # FD4 has no frontal gap
    assert fd4.get("whole_eye_lateralwt") is True  # its own whole-eye lateral signature
    assert fd4["heterolateral"] is True
    # FD3 keeps the frontal gap; FD1 differs from FD4 only in the RF feature.
    assert K.FD_SIGNATURES["FD3"]["lateral_gap"] is True
    assert K.FD_SIGNATURES["FD1"]["layer"] == "a" and K.FD_SIGNATURES["FD1"]["lateral_gap"] is False


# ---------------------------------------------------------------------------
# Null-aware confidence.
# ---------------------------------------------------------------------------
def _phenotype(match_discriminating=0):
    """Synthetic phenotype: 4 shared-class properties always match; `match_discriminating` of the
    3 discriminating properties (RF laterality, DV dendrite, and the RF-width proxy) match."""
    props = [
        {"property": "progressive (layer-a) preferred direction", "match": True},
        {"property": "heterolateral noduli-group axon (contra POF)", "match": True},
        {"property": "cholinergic output", "match": True},
        {"property": "no lateral-protocerebrum second arbor", "match": True},
        {"property": "whole-eye, laterally-weighted RF (no frontal gap)",
         "match": match_discriminating >= 1},
        {"property": "restricted dorso-ventral dendrite", "match": match_discriminating >= 2},
        {"property": "bidirectional contralateral inhibition", "match": match_discriminating >= 3},
    ]
    return {"properties": props, "n_properties": len(props),
            "n_matched": sum(1 for p in props if p["match"]),
            "discriminating_properties": 3, "shared_class_properties": 4,
            "fd3_ruled_out": {"excluded": True, "fd3_dominant_layer": "b", "fd3_layer_b_pct": 98.6}}


def test_confidence_null_when_no_discriminating_match():
    """The empirical case: no discriminating property matches and the split is not separable."""
    conf = F4._confidence(_phenotype(0), {"only_survivor_is_prog_type": True},
                          {"separable": False, "fd4_pair": None})
    assert conf["identity_verdict"] == "UNRESOLVED_HONEST_NULL"
    assert conf["point_estimate"] <= 0.1           # collapses toward zero
    lo, hi = conf["interval"]
    assert lo <= conf["point_estimate"] <= hi
    assert conf["ceiling"] <= 0.85                 # FD1-collinearity caps it below FD2's 0.95
    assert "fd1_collinearity_shared_class" in conf["discounts"]


def test_confidence_ceiling_below_fd2():
    """Even with a separable pair and all discriminating properties, FD4's ceiling stays below 0.95."""
    conf = F4._confidence(_phenotype(3), {"only_survivor_is_prog_type": False},
                          {"separable": True, "fd4_pair": [[1, 2], [3, 4]]})
    assert conf["ceiling"] < 0.95                  # capped below FD2 by the collinearity discount
    assert conf["uniqueness_factor"] == 0.5        # never 1.0: FD4 shares FD1's output class


def test_confidence_interval_is_wide():
    conf = F4._confidence(_phenotype(0), {"only_survivor_is_prog_type": True},
                          {"separable": False, "fd4_pair": None})
    lo, hi = conf["interval"]
    assert (hi - lo) >= 0.20                        # wide by construction (least-resolved FD cell)


# ---------------------------------------------------------------------------
# Oracle claims: elimination CONFIRMED, identity UNVERIFIABLE.
# ---------------------------------------------------------------------------
def _derived_null():
    return {
        "candidate_screen": {"universe_size": 8806, "n_layer_a_lowcopy_screened": 28,
                             "survivors": ["Nod1"], "n_survivors": 1,
                             "only_survivor_is_prog_type": True},
        "nod1_split": {"separable": False, "homogeneous": True,
                       "same_side_input_jaccard": 0.33, "cross_side_input_jaccard": 0.007,
                       "pools_two_populations": True,
                       "best_pairing": {"silhouette": -4.9}, "per_cell": {}},
        "phenotype": _phenotype(0),
        "confidence": {"point_estimate": 0.0, "interval": [0.0, 0.27], "ceiling": 0.85,
                       "identity_verdict": "UNRESOLVED_HONEST_NULL",
                       "discounts": {"fd1_collinearity_shared_class": 0.10},
                       "statement": "..."},
    }


def test_oracle_identity_is_unverifiable():
    claims = F4O.build_claims(_derived_null())
    by_id = {c.id: c for c in claims}
    assert by_id["FD4.identity_verdict"].verdict == CMP.UNVERIFIABLE


def test_oracle_elimination_confirmed():
    claims = F4O.build_claims(_derived_null())
    by_id = {c.id: c for c in claims}
    # The strong POSITIVE findings are CONFIRMED.
    assert by_id["FD4.no_other_progressive_output_cell"].verdict == CMP.CONFIRMED
    assert by_id["FD4.nod1_homogeneous_not_fd1_fd4_split"].verdict == CMP.CONFIRMED
    assert by_id["FD4.fd3_excluded"].verdict == CMP.CONFIRMED
    assert by_id["FD4.nod1_pools_two_populations"].verdict == CMP.CONFIRMED


def test_oracle_discriminating_phenotype_unverifiable_when_unmatched():
    claims = F4O.build_claims(_derived_null())
    # The FD4-discriminating phenotype claims (RF laterality, DV dendrite) are UNVERIFIABLE, not
    # forced to CONFIRMED, when the population does not realize them.
    disc = [c for c in claims if c.id.startswith("FD4.phenotype_")
            and c.verdict == CMP.UNVERIFIABLE]
    assert len(disc) >= 2


def test_oracle_no_refuted():
    """A negative result is reported honestly (UNVERIFIABLE), never as REFUTED of a claim we did
    not test, and never crashes."""
    claims = F4O.build_claims(_derived_null())
    assert all(c.verdict != CMP.REFUTED for c in claims)
    assert len(claims) >= 10
