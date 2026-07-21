"""Unit tests for skeleton_fetch: cache round-trip (vertices + edges) and graceful fallback
when fafbseg / a live client is unavailable. No network, no fafbseg required.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from flyconn.paper import skeleton_fetch as SK


@pytest.fixture(autouse=True)
def _cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYCONN_CACHE_ROOT", str(tmp_path))
    yield


def test_cache_roundtrip_with_edges():
    verts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    edges = np.array([[0, 1], [1, 2]], dtype=int)
    skel = SK.Skeleton(root_id=123, vertices_um=verts, edges=edges, source="fafbseg")
    SK._to_cache(skel)
    back = SK._from_cache(123)
    assert back is not None and back.ok
    assert np.allclose(back.vertices_um, verts)
    assert np.array_equal(back.edges, edges)        # edges persisted + reloaded
    assert back.source == "fafbseg"


def test_fetch_skeleton_uses_cache_first():
    verts = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    SK._to_cache(SK.Skeleton(456, verts, np.empty((0, 2), int), "fafbseg"))

    class _NoClient:
        track = "offline"
        client = None
    got = SK.fetch_skeleton(_NoClient(), 456)
    assert got is not None and np.allclose(got.vertices_um, verts)


def test_fetch_skeleton_graceful_when_nothing_available(monkeypatch):
    # No cache, no fafbseg, no live client -> returns None with a reason, never raises.
    monkeypatch.setattr(SK, "_from_fafbseg", lambda root_id: None)

    class _NoClient:
        track = "offline"
        client = None
    assert SK.fetch_skeleton(_NoClient(), 999) is None
    assert SK.fetch_skeleton.last_skip_reason
