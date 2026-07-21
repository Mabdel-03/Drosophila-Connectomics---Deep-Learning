"""Guard tests for the FD2 = LPT21 full-circuit report builder (reader-facing, 4 parts, no em dash)."""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from flyconn.fd2_report import combined  # noqa: E402

_FORBIDDEN_JARGON = ("Full claim ledger", "Verdict tally", "Claim ID", "CONFIRMED_WITH_CAVEAT",
                     "CONFIRMED WITH CAVEAT", "pre-registered", "CORE", "DISCRIMINATING",
                     "Corroborating only", "REFUTED", "UNVERIFIABLE", "proxy jargon")


def _synthetic_results() -> dict:
    D = {
        "track": "live",
        "identity": {"layer_b_pct": 99.2, "contra_output_pct": 1.5, "mean_rf_offset": 0.03,
                     "frontal_colocated": True, "homolateral": True, "smallfield_bounded": True,
                     "nt": "acetylcholine", "nt_conf": 0.81, "smallfield_null": {"p_value": 0.002}},
        "dual_output": {"both_bimodal": True, "cells": [
            {"side": "right", "available": True, "ml_gap_um": 23.0, "minor_lobe_frac": 0.33,
             "lobe_sizes": [5511, 2659]},
            {"side": "left", "available": True, "ml_gap_um": 25.0, "minor_lobe_frac": 0.33,
             "lobe_sizes": [5161, 2568]}]},
        "uniqueness": {"unique": True, "n_fd2_hits": 1, "competitors": [{"type": "LPT23"}],
                       "core_hits": [{"type": "LPT21", "layer_b_pct": 99.2, "contra_pct": 1.5,
                                      "core_match": True, "single_pair": True, "n_cells": 2}],
                       "full_hits": []},
        "confidence": {"point_estimate": 0.9, "interval": [0.85, 0.95], "n_matched": 6,
                       "n_properties": 6, "properties": [
                           {"property": "regressive", "match": True, "support": "99% layer-b"}],
                       "uniqueness_factor": 1.0, "discounts": {"no_independent_anchor": 0.05}},
        "afferent": {"census": {"total_input_syn": 11528, "n_partners": 2855,
                                "t4t5_frac_of_total": 41.7, "layer_b_frac_of_t4t5": 99.2,
                                "t4b_syn": 2375, "t5b_syn": 2388,
                                "on_off_split": {"t4_frac": 49.9}, "t4t5_layer_frac": {"b": 99.2},
                                "top_input_types": [{"cell_type": "T5b", "syn": 2388, "nt": "acetylcholine"}]},
                     "upstream_cascade": {"t4_on_medulla_present": ["Mi1", "Tm3"],
                                          "t5_off_medulla_present": ["Tm9", "Tm2"],
                                          "lamina_present": ["L1", "L2"],
                                          "photoreceptor_present": ["R1-6"]}},
        "sheet": {"named_sheet": "LPC1"},
        "gate": {"named_inhibitor": "LPi14"},
        "efferent": {"direct": {"total_syn": 88, "ranking": [
                         {"cell_type": "DNp26", "syn": 60, "is_steering": True},
                         {"cell_type": "DNge094", "syn": 24, "is_steering": True}]},
                     "motor": {"dominant_motor_system": "wing_steering",
                               "motor_system_pct": {"wing_steering": 43.5, "neck_gaze": 17.3},
                               "per_dn": {"DNp26": {"wing": "contralateral",
                                                    "top_muscles": {"hg1": 122, "hg2": 117}}}}},
        "census": {"input": {"total_syn": 11528, "n_partners": 2855, "ranked_types": [
                       {"cell_type": "T5b", "syn": 2388}]},
                   "output": {"total_syn": 15899, "n_partners": 9733, "ranked_types": [
                       {"cell_type": "Tlp1", "syn": 524}]}},
        "compartment": {720575940627341736: {"dendrite_frac": 0.99,
                                              "class_by_compartment": {"dendrite": {"motion": 4000},
                                                                       "axon": {"motion": 30}}}},
    }
    return {"meta": {"flywire_track": "live"}, "families": {"FD2": {"derived": D, "claims": []}}}


def test_build_tex_compiles_to_four_parts():
    tex = combined.build_tex(_synthetic_results())
    assert tex.count(r"\part{") == 4
    assert tex.count(r"\begin{document}") == 1 and tex.count(r"\end{document}") == 1
    assert tex.count(r"\maketitle") == 1


def test_build_tex_no_em_dash():
    tex = combined.build_tex(_synthetic_results())
    assert "—" not in tex          # unicode em dash
    assert "---" not in tex             # LaTeX em dash


def test_build_tex_reader_facing_no_jargon():
    tex = combined.build_tex(_synthetic_results())
    for j in _FORBIDDEN_JARGON:
        assert j not in tex, f"internal jargon leaked into the FD2 report: {j!r}"
    assert "_esc(" not in tex and "_g(" not in tex


def test_build_tex_braces_balanced():
    tex = combined.build_tex(_synthetic_results())
    assert tex.count("{") == tex.count("}")


def test_build_tex_single_bibliography():
    tex = combined.build_tex(_synthetic_results())
    assert tex.count(r"\begin{thebibliography}") == 1


def test_build_tex_has_confidence_and_dual_output():
    tex = combined.build_tex(_synthetic_results())
    assert "confidence" in tex.lower()
    assert "dual" in tex.lower() or "second" in tex.lower()   # the FD2-unique dual output
    assert "LPT21" in tex and "FD2" in tex
