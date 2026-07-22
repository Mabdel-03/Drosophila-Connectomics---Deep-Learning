"""Regression guard for the Stage-8 bilateral parametrization.

The scientific value of running left and right through the SAME code path depends on the
right path staying the paper's positive control — byte-identical to the published numbers.
These tests assert that:

  1. ``SideConfig`` invariants hold (RIGHT == the old hardcoded scope; LEFT is its mirror).
  2. ``get_sheet(cfg=C.RIGHT)`` still yields the paper's 454 VCH-gated T4a -> 100 LLPC1 ->
     9,223 synapses, and is NOT silently the left sheet (the previous cache-key bug).
  3. The ``_CACHE`` is keyed by (track, side) so a left run after a right run cannot return
     the right sheet.

Tests (2) require the cached right-side live pulls under .flyconn_cache (no network); they
skip cleanly if neither the live cache nor the offline dump is available, so the suite still
runs in a bare checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from flyconn.paper.oracle import consts as C  # noqa: E402

# Paper headline (Methods / Table S2), right-hemisphere exemplar.
RIGHT_N_T4A = 454
RIGHT_N_LLPC1 = 100
RIGHT_N_T4A_LLPC1_SYN = 9223
LEFT_GATER_VCH = 720_575_940_627_502_338  # right-soma VCH (gates the left sheet)


def test_sideconfig_invariants():
    # RIGHT reproduces the old hardcoded scope exactly.
    assert C.RIGHT.vch_root == C.VCH_ROOT
    assert C.RIGHT.sheet_side == C.SHEET_SIDE == "right"
    assert C.RIGHT.gating_soma_side == "left"
    assert C.RIGHT.exemplar_t4 == "T4a"
    assert C.RIGHT.opponent_layer_letter == "b"
    # LEFT is the mirror: left sheet, gated by the right-soma VCH.
    assert C.LEFT.sheet_side == "left"
    assert C.LEFT.gating_soma_side == "right"
    assert C.LEFT.vch_root == LEFT_GATER_VCH
    # The two scopes are genuinely different cells.
    assert C.LEFT.vch_root != C.RIGHT.vch_root
    assert C.GATING_SOMA_FOR_SHEET == {"right": "left", "left": "right"}


def _make_cached_source():
    """A live source restricted to its parquet cache (no network), or None if unavailable."""
    from flyconn.paper.fw_access import NeuronMeta, LiveCaveFlyWire, caveclient_available
    if not caveclient_available():
        return None, None
    try:
        src = LiveCaveFlyWire(mat_version=783)
    except Exception:
        return None, None
    meta = NeuronMeta.load("783")
    return src, meta


def _right_pulls_cached(src) -> bool:
    """True iff the right-side VCH->T4a and T4a->LLPC1 pulls are already on disk."""
    return (src.cache_dir / "pre-80c33a0249732b88-n1__post-none.parquet").exists()


@pytest.mark.skipif(
    _make_cached_source()[0] is None, reason="no live CAVE client / cache available"
)
def test_right_sheet_reproduces_paper():
    from flyconn.paper.derive import sheet as SH
    src, meta = _make_cached_source()
    if not _right_pulls_cached(src):
        pytest.skip("right-side live pulls not cached; needs a credentialed warm run")
    SH._CACHE.clear()
    r = SH.get_sheet(src, meta, C.RIGHT)
    assert r.n_t4a == RIGHT_N_T4A, r.n_t4a
    assert r.n_llpc1 == RIGHT_N_LLPC1, r.n_llpc1
    assert r.n_t4a_llpc1_syn == RIGHT_N_T4A_LLPC1_SYN, r.n_t4a_llpc1_syn


@pytest.mark.skipif(
    _make_cached_source()[0] is None, reason="no live CAVE client / cache available"
)
def test_cache_keyed_by_side_no_collision():
    """Left sheet must not return the right sheet after a right run (the old bug)."""
    from flyconn.paper.derive import sheet as SH
    src, meta = _make_cached_source()
    if not _right_pulls_cached(src):
        pytest.skip("right-side live pulls not cached")
    SH._CACHE.clear()
    r = SH.get_sheet(src, meta, C.RIGHT)
    # the right entry is cached under (track, "right"), never reused for "left"
    assert (src.track, "right") in SH._CACHE
    assert (src.track, "left") not in SH._CACHE
    # if the left pulls are cached too, assert it is a DIFFERENT sheet
    left_pull = src.cache_dir / "pre-c86edcb2b1dbbd50-n1__post-none.parquet"  # right-soma VCH out
    if left_pull.exists():
        ll = SH.get_sheet(src, meta, C.LEFT)
        assert set(ll.llpc1_roots.tolist()) != set(r.llpc1_roots.tolist())
        assert (src.track, "left") in SH._CACHE
