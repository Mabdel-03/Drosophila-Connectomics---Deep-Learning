"""Smoke tests for the comprehensive-census + compartment scalar figures — synthetic run dicts.

Run: /orcd/home/002/mabdel03/conda_envs/consortium/bin/python -m pytest tests/test_figures_census.py -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "src")

from flyconn.paper import figures as PF


def _assert_png(path):
    assert os.path.exists(path) and os.path.getsize(path) > 1000, f"empty/missing PNG: {path}"


def _census_run():
    ranked_in = [{"cell_type": "T4b", "syn": 1793, "super_class": "optic", "nt": "acetylcholine"},
                 {"cell_type": "T5b", "syn": 1459, "super_class": "optic", "nt": "acetylcholine"},
                 {"cell_type": "LPC1", "syn": 1120, "super_class": "visual_projection", "nt": "acetylcholine"},
                 {"cell_type": "LPi14", "syn": 961, "super_class": "optic", "nt": "gaba"},
                 {"cell_type": "PLP230", "syn": 200, "super_class": "central", "nt": "gaba"}]
    ranked_out = [{"cell_type": "PLP230", "syn": 687, "super_class": "central", "nt": "gaba"},
                  {"cell_type": "DNp26", "syn": 106, "super_class": "descending", "nt": "acetylcholine"}]
    return {"derived": {"census": {
        "input": {"total_syn": 13157, "n_partners": 2991, "ranked_types": ranked_in},
        "output": {"total_syn": 15122, "n_partners": 8435, "ranked_types": ranked_out},
    }}}


def test_connectivity_wheel(tmp_path):
    _assert_png(PF.fd3_connectivity_wheel(_census_run(), tmp_path / "wheel.png"))


def test_connectivity_matrix(tmp_path):
    _assert_png(PF.fd3_connectivity_matrix(_census_run(), tmp_path / "matrix.png"))


def test_compartment_split(tmp_path):
    run = {"derived": {"compartment": [
        {"side": "right", "source": "skeleton", "dendrite_frac": 0.995,
         "dendrite": {"by_class": {"sheet": 2345, "inhibitor": 2024, "motion": 1891, "other": 670}},
         "axon": {"by_class": {"other": 22, "contra_inhibitor": 3}}},
        {"side": "left", "source": "skeleton", "dendrite_frac": 0.994,
         "dendrite": {"by_class": {"sheet": 2438, "motion": 1408, "inhibitor": 1397}},
         "axon": {"by_class": {"other": 19}}},
    ]}}
    _assert_png(PF.fd3_compartment_split(run, tmp_path / "compartment.png"))


def test_figures_handle_missing_data(tmp_path):
    # graceful placeholders when the blocks are absent
    _assert_png(PF.fd3_connectivity_matrix({"derived": {}}, tmp_path / "m2.png"))
    _assert_png(PF.fd3_compartment_split({"derived": {}}, tmp_path / "c2.png"))
