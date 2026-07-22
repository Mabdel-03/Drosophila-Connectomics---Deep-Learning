"""The paper's claims, encoded as data — one module per claim family (A-J) plus the
shared constants and the interpretive/untestable set.

Each family module exposes:
  * the claims as plain Python constants/dicts (every number cites its page/table); and
  * ``build_claims(derived) -> list[ClaimResult]`` which compares the paper value to the
    computed value via ``flyconn.motif.compare``.

``family(id)`` / ``all_families()`` import family modules lazily so importing ``consts``
(used widely) never forces every family to import — and a half-built family module
during development doesn't break the package.
"""

from __future__ import annotations

import importlib

from . import consts  # noqa: F401  (cheap, dependency-free; safe to import eagerly)

_FAMILY_MODULES = {
    "A": "a_vch_loop",
    "B": "b_inhibitor_screen",
    "C": "c_sheet_elimination",
    "D": "d_retinotopy_null",
    "E": "e_cable_distance",
    "F": "f_sheet_regulation",
    "G": "g_output_census",
    "H": "h_nod_to_dnp26",
    "I": "i_descending_channels",
    "J": "j_motor_mcns",
    "Y": "internal_consistency",
    "Z": "interpretive",
}


def family(fid: str):
    """Import and return the oracle module for family id (e.g. 'A')."""
    return importlib.import_module(f".{_FAMILY_MODULES[fid]}", __name__)


def all_families() -> dict:
    """{family_id: module} for every family that imports cleanly."""
    out = {}
    for fid in _FAMILY_MODULES:
        try:
            out[fid] = family(fid)
        except Exception as e:  # a not-yet-written family shouldn't break the rest
            print(f"[oracle] family {fid} unavailable: {e}")
    return out
