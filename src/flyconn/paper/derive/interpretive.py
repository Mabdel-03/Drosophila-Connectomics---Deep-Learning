"""Derive family Z (interpretive): gather the anatomical proxy values that the
interpretive oracle reports alongside its UNVERIFIABLE verdicts.

These are not new computations — they reuse families A and E so the proxy statements
(graded reciprocal loop; dual-field motion+form integrator) carry concrete numbers.
"""

from __future__ import annotations

from . import a_vch_loop as DA
from . import e_cable_distance as DE


def run(src, meta) -> dict:
    a = DA.run(src, meta)
    # E is expensive; reuse cached source queries. Only the composition % is needed here.
    try:
        e = DE.run(src, meta)
        motion_pct, form_pct = e["motion_pct"], e["form_pct"]
    except Exception:
        motion_pct = form_pct = None
    return {
        "gain_loop_proxy": {"reciprocal_n": a["reciprocal_n"], "ratio": a["exc_inhib_ratio"]},
        "motion_pct": motion_pct,
        "form_pct": form_pct,
        "track": src.track,
    }
