"""Smoke test for the FD3-Identification report builder: build_tex on a synthetic Family-K
JSON contains every figure, the TikZ schematic, the bibliography and the claim ledger.

Run: /orcd/home/002/mabdel03/conda_envs/consortium/bin/python -m pytest tests/test_fd3_report.py -q
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from flyconn.fd3_report import builder


def _synthetic_results() -> dict:
    derived = {
        "candidate": "LPT42_Nod4", "track": "live",
        "cand_n_cells": 2, "cand_nt": "acetylcholine", "cand_mean_nt_conf": 0.894,
        "cand_super_class": "visual_projection",
        "cand_layer_frac": {"a": 0.1, "b": 98.4, "c": 1.2, "d": 0.3},
        "cand_dominant_direction": "back_to_front", "cand_contra_output_pct": 80.5,
        "cand_top_targets": {"PLP230": 687, "WED075": 295},
        "profiles": {"LPT42_Nod4": {"n_cells": 2, "dominant_layer": "b", "layer_frac": {"b": 98.4}}},
        "rf": {"per_side": {"left": {"centroid_offset_p": 13.4, "offset_ci95": [12.4, 14.3],
                                     "cand_frontal_occ": 0.02, "ref_frontal_occ": 0.63,
                                     "more_lateral": True, "has_frontal_gap": True}},
               "calibration_error_budget": {"combined_deg": 16.0}},
        "smallfield_null": {"obs_radius": 6.6, "null_radius": 10.9, "z_score": -24.5,
                            "p_value": 0.002, "n_perms": 500, "bounded": True},
        "soma": {"bilateral_split": True, "post_z_percentile": 0.78, "dz_to_anchor": 135.2,
                 "centroid_xyz": {"soma_x": 134000, "soma_z": 5415}, "midline_x": 133292},
        "fd_family_screen": {"best_match_for_FD3": "LPT42_Nod4", "best_FD_for_LPT42": "FD3",
                             "fd3_margin": 2, "reciprocal_best_hit": True, "n_candidates": 5,
                             "score_matrix": {"FD3": {"LPT42_Nod4": 3, "Nod1": 1}}},
        "cross_version": {"available": True, "v630": {"dominant_layer": "b", "layer_b": 98.4,
                          "contra_output_pct": 82.5}, "agree": {"rf_lateral_gap": True}},
        "robustness": {"grid_pass_fraction": 1.0,
                       "per_knob": {"gap_drop": {"grid": [0.3, 0.5, 0.7], "pass": [True, True, True]}},
                       "seed_stability": {"offset_ci_lower_min": 9.5, "offset_ci_lower_max": 9.8}},
    }
    claims = [
        {"id": "K.layer_b", "report_value": 100.0, "computed_primary": 98.4, "verdict": "CONFIRMED"},
        {"id": "K.identity_verdict", "report_value": "FD3 == LPT42_Nod4",
         "computed_primary": "CONFIRMED_WITH_CAVEAT", "verdict": "CONFIRMED_WITH_CAVEAT"},
    ]
    return {"meta": {"flywire_track": "live"},
            "families": {"K": {"claims": claims, "derived": derived}}}


def test_build_tex_has_all_anchors():
    tex = builder.build_tex(_synthetic_results())
    # every figure is included
    for fig in ("fd3_crosswalk", "fd3_layer_composition", "fd3_rf_differential",
                "fd3_family_heatmap", "fd3_replication_robustness", "fd3_circuit_placement"):
        assert f"figures/{fig}.png" in tex, f"missing figure {fig}"
    # the TikZ schematic and the bibliography
    assert r"\begin{tikzpicture}" in tex
    for key in ("egelhaaf1985b", "rp1979", "schlegel2024", "fd1989"):
        assert rf"\bibitem{{{key}}}" in tex, f"missing bibitem {key}"
    # the reader-facing evidence table + key narrative anchors
    assert r"\section{The evidence at a glance}" in tex
    assert "frontal gap" in tex
    assert "crosses to the other side" in tex


def test_build_tex_is_reader_facing_no_internal_jargon():
    # The report must read as a self-contained argument, not an internal claim ledger:
    # none of the verification scaffolding vocabulary should reach the reader.
    tex = builder.build_tex(_synthetic_results())
    for jargon in ("Full claim ledger", "Verdict tally", "Claim ID", "CONFIRMED WITH CAVEAT",
                   "pre-registered", "claim by claim", "Corroborating only", "caveat-capped"):
        assert jargon not in tex, f"internal jargon leaked into the report: {jargon!r}"


def test_build_tex_escapes_underscores_in_ids():
    tex = builder.build_tex(_synthetic_results())
    # LaTeX-unsafe cell-type underscores must be escaped
    assert r"LPT42\_Nod4" in tex


def test_build_tex_no_raw_python_braces_leak():
    # a crude guard against f-string leakage of helper calls into the tex
    tex = builder.build_tex(_synthetic_results())
    assert "_esc(" not in tex and "_g(" not in tex


def test_identity_report_calibrates_unavailable_cross_version():
    r = _synthetic_results()
    r["meta"]["flywire_track"] = "offline"
    r["families"]["K"]["derived"]["cross_version"] = {
        "available": False,
        "reason": "v630 cross-check is live-only",
    }
    tex = builder.build_tex(r)
    assert "The v630 cross-version check is not claimed here" in tex
    assert "reproduced on a frozen copy and on a second independent release" not in tex


def test_identity_report_has_no_report_facing_em_dash():
    tex = builder.build_tex(_synthetic_results())
    assert "—" not in tex
    assert "---" not in tex


# ---------------------------------------------------------------------------
# Family P — the FD3 input-pathway report + the combined 3-part report.
# ---------------------------------------------------------------------------
from flyconn.fd3_report import input as input_report  # noqa: E402
from flyconn.fd3_report import combined  # noqa: E402
from flyconn.fd3_report import descending  # noqa: E402
from flyconn.fd3_report import audit  # noqa: E402

_FORBIDDEN_JARGON = ("Full claim ledger", "Verdict tally", "Claim ID", "CONFIRMED_WITH_CAVEAT",
                     "CONFIRMED WITH CAVEAT", "pre-registered", "CORE", "DISCRIMINATING",
                     "Corroborating only", "REFUTED", "UNVERIFIABLE")


def _synthetic_p_results() -> dict:
    derived = {
        "track": "live",
        "census": {"total_input_syn": 13157, "n_partners": 2991, "t4t5_syn": 3299,
                   "t4t5_frac_of_total": 25.1, "layer_b_frac_of_t4t5": 98.6,
                   "t4t5_dominant_direction": "back_to_front",
                   "on_off_split": {"T4b_ON": 1793, "T5b_OFF": 1459, "t4_frac": 55.1},
                   "t4t5_layer_frac": {"a": 0.09, "b": 98.6, "c": 1.2, "d": 0.15},
                   "input_syn_by_super_class": {"optic": 6238, "visual_projection": 4775},
                   "top_input_types": [{"cell_type": "T4b", "syn": 1793, "nt": "acetylcholine"},
                                       {"cell_type": "LPC1", "syn": 1120, "nt": "acetylcholine"},
                                       {"cell_type": "LPi14", "syn": 961, "nt": "gaba"}]},
        "upstream_cascade": {
            "on_limb_t4b": {"n_expected_present": 3, "expected_medulla_present": ["Mi1", "Tm3", "C3"],
                            "expected_frac_of_input": 61.0},
            "off_limb_t5b": {"n_expected_present": 3, "expected_medulla_present": ["Tm1", "Tm2", "Tm9"],
                             "expected_frac_of_input": 70.0},
            "lamina_present": ["L1", "L2", "L3"], "photoreceptor_present": ["R1-6", "R7", "R8"]},
        "central_inputs": {"any_layerb_carrier": True, "n_layerb_carriers": 1, "total_central_syn": 5000,
                           "top_types": [{"cell_type": "LPC1", "dominant_layer": "b", "reads_layer_b": True}]},
        "contra_inhibition": {"available": True, "both_present": True, "sufficient": False},
        "negative_controls": {"no_vch_gate": True, "no_layer_a_drive": True,
                              "vch_input_syn": 6, "vch_input_frac": 0.05, "layer_a_frac_of_t4t5": 0.09},
    }
    return {"meta": {"flywire_track": "live"}, "families": {"P": {"derived": derived}}}


def _synthetic_l_results() -> dict:
    derived = {
        "track": "live",
        "direct": {"top_dn": "DNp26", "n_dns": 24, "steering_frac": 56.0,
                   "ranking": [{"cell_type": "DNp26", "syn": 68, "is_steering": True}]},
        "relay": {"n_dns": 263},
        "motor": {"available": True, "dominant_motor_system": "wing_steering",
                  "motor_system_pct": {"wing_steering": 42.7},
                  "per_dn": {"DNp26": {"steering_syn": 100, "wing": "contralateral",
                                       "top_muscles": {"hg1": 122, "i1": 105}}}},
    }
    return {"meta": {"flywire_track": "live"}, "families": {"L": {"derived": derived}}}


def test_input_report_has_figure_anchors_and_bibliography():
    tex = input_report.build_tex(_synthetic_p_results())
    for fig in ("fd3_input_census", "fd3_afferent_cascade", "fd3_input_hexmap",
                "fd3_arbor_inputs", "fd3_circuit_3d"):
        assert f"figures/{fig}.png" in tex, f"missing figure {fig}"
    assert r"\begin{thebibliography}" in tex
    assert r"LPT42\_Nod4" in tex


def test_input_report_is_reader_facing_no_jargon():
    tex = input_report.build_tex(_synthetic_p_results())
    for j in _FORBIDDEN_JARGON:
        assert j not in tex, f"internal jargon leaked into the input report: {j!r}"
    assert "_esc(" not in tex and "_g(" not in tex


def test_combined_report_stitches_three_parts():
    tex = combined.build_tex(_synthetic_results(), _synthetic_p_results(), _synthetic_l_results())
    # three \part dividers, one document
    assert tex.count(r"\part{") == 3
    assert tex.count(r"\begin{document}") == 1 and tex.count(r"\end{document}") == 1
    # only the outer title survives (inner \maketitle blocks stripped)
    assert tex.count(r"\maketitle") == 1
    # content from each family present
    assert "figures/fd3_crosswalk.png" in tex          # K identity
    assert "figures/fd3_input_census.png" in tex        # P inputs
    assert "figures/fd3_motor_systems.png" in tex       # L outputs


def test_combined_report_has_single_unique_bibliography():
    tex = combined.build_tex(_synthetic_results(), _synthetic_p_results(), _synthetic_l_results())
    assert tex.count(r"\begin{thebibliography}") == 1
    keys = []
    for line in tex.splitlines():
        if line.startswith(r"\bibitem{"):
            keys.append(line.split("{", 1)[1].split("}", 1)[0])
    assert len(keys) == len(set(keys))


def test_combined_report_has_no_report_facing_em_dash():
    tex = combined.build_tex(_synthetic_results(), _synthetic_p_results(), _synthetic_l_results())
    assert "—" not in tex
    assert "---" not in tex


def test_combined_report_no_jargon():
    tex = combined.build_tex(_synthetic_results(), _synthetic_p_results(), _synthetic_l_results())
    for j in _FORBIDDEN_JARGON:
        assert j not in tex, f"internal jargon leaked into the combined report: {j!r}"


def test_combined_report_braces_balanced_no_title_leak():
    """The per-report \\title{...} blocks (which contain nested braces) must be stripped
    cleanly -- a naive non-greedy strip leaves dangling braces and a 'Too many }'s' compile."""
    tex = combined.build_tex(_synthetic_results(), _synthetic_p_results(), _synthetic_l_results())
    assert tex.count("{") == tex.count("}"), "unbalanced braces (title block stripped naively?)"
    # the tail of each report's title must not survive as body text
    assert "is the modern connectomic correlate" not in tex


def test_full_circuit_audit_records_proxy_and_unavailable_cross_version():
    k = _synthetic_results()
    k["meta"]["flywire_track"] = "offline"
    k["families"]["K"]["derived"]["cross_version"] = {
        "available": False,
        "reason": "v630 cross-check is live-only",
    }
    a = audit.build_audit(k, _synthetic_p_results(), _synthetic_l_results())
    by_id = {e["claim_id"]: e for e in a["entries"]}
    assert by_id["K.cross_version"]["status"] == "unavailable"
    assert by_id["L.motor_proxy"]["evidence_class"] == "cross_connectome_proxy"


# ---------------------------------------------------------------------------
# Family Q/R/S functional-circuit report + combined 4-part report.
# ---------------------------------------------------------------------------
from flyconn.fd3_report import functional  # noqa: E402


def _synthetic_func_results() -> dict:
    return {"meta": {"flywire_track": "live"}, "families": {
        "Q": {"derived": {"named_sheet": "LPC1", "sheet_set": ["LPC1", "LLPC3", "LLPC2"],
                          "profiles": {"LPC1": {"to_fd3_syn": 1120, "layer_b_input_syn": 28848}},
                          "retinotopy_null": {"available": True, "z_score": -231.0,
                                              "obs_radius_um": 10.8, "null_radius_um": 207.6}}},
        "R": {"derived": {"winner": {"cell_type": "LPi14", "direction": "opponent",
                                     "layer_a_pct": 94.8, "t4t5_in_frac": 72.6, "to_fd3_syn": 961,
                                     "to_sheet_syn": 6631, "to_detectors_syn": 7004},
                          "same_direction_gates": ["LPi15"]}},
        "S": {"derived": {"circuit": "named"}},
        "census": {"derived": {"input": {"total_syn": 20961, "n_partners": 3760, "ranked_types": []},
                               "output": {"total_syn": 15122, "n_partners": 8435, "ranked_types": []}}},
    }}


def test_functional_report_names_circuit_and_figures():
    tex = functional.build_tex(_synthetic_func_results())
    assert r"LPC1" in tex and r"LPi14" in tex
    for fig in ("fd3_connectivity_wheel", "fd3_compartment_split",
                "fd3_functional_circuit_schematic"):
        assert f"figures/{fig}.png" in tex, f"missing figure {fig}"
    assert r"\begin{thebibliography}" in tex


def test_functional_report_is_reader_facing_no_jargon():
    tex = functional.build_tex(_synthetic_func_results())
    for j in _FORBIDDEN_JARGON:
        assert j not in tex, f"internal jargon leaked into the functional report: {j!r}"
    assert "_esc(" not in tex and "_g(" not in tex


def test_combined_includes_functional_part_when_present():
    tex = combined.build_tex(_synthetic_results(), _synthetic_p_results(), _synthetic_l_results(),
                             _synthetic_func_results())
    assert tex.count(r"\part{") == 4
    assert "functional figure-ground circuit within FD3" in tex
    assert tex.count("{") == tex.count("}")   # brace balance holds with the 4th part


def test_combined_without_functional_is_three_parts():
    tex = combined.build_tex(_synthetic_results(), _synthetic_p_results(), _synthetic_l_results())
    assert tex.count(r"\part{") == 3
