"""Unit tests for the CAVE cache + secrets + cache-only client - no network, no token.

The network thunk is injected as a plain function so nothing imports caveclient. We
assert the cache writes-then-reads (the thunk runs once), respects refresh, hashes
deterministically, and that an offline (allow_network=False) client raises on a miss
instead of touching the network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from flyconn.connectome import ConnectomeClient, FAFB, query_hash
from flyconn.connectome import cache as Cache
from flyconn.connectome import secrets as Secrets


@pytest.fixture(autouse=True)
def _scratch(tmp_path, monkeypatch):
    # The CAVE cache resolves under FLYCONN_CACHE_ROOT (live caches live in the project tree,
    # off scratch); isolate it to a tmp dir so each test starts with an empty cache.
    monkeypatch.setenv("FLYCONN_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("FLYCONN_DATA_ROOT", str(tmp_path))
    yield


def test_query_hash_order_insensitive_and_deterministic():
    a = query_hash("fafb", 783, "syn", "synapses", {"pre": [1, 2], "post": None})
    b = query_hash("fafb", 783, "syn", "synapses", {"post": None, "pre": [1, 2]})
    c = query_hash("fafb", 783, "syn", "synapses", {"pre": [2, 1], "post": None})
    assert a == b           # key order doesn't matter
    assert a != c           # list order is meaningful (caller sorts before hashing)
    assert query_hash("mcns", "v1.0", "syn", "synapses", {}) != a


def test_materialization_changes_hash():
    p = {"pre": [1]}
    assert query_hash("fafb", 783, "t", "synapses", p) != query_hash("fafb", 784, "t", "synapses", p)


def test_cached_or_call_writes_then_reads():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return pd.DataFrame({"pre_pt_root_id": [1, 2], "post_pt_root_id": [3, 4]})

    df1 = Cache.cached_or_call(FAFB, "synapses", {"pre": [1]}, fn)
    assert calls["n"] == 1 and len(df1) == 2
    df2 = Cache.cached_or_call(FAFB, "synapses", {"pre": [1]}, fn)
    assert calls["n"] == 1           # served from cache, fn NOT called again
    pd.testing.assert_frame_equal(df1, df2)


def test_cached_or_call_refresh_reinvokes():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return pd.DataFrame({"a": [calls["n"]]})

    Cache.cached_or_call(FAFB, "synapses", {"k": 1}, fn)
    Cache.cached_or_call(FAFB, "synapses", {"k": 1}, fn, refresh=True)
    assert calls["n"] == 2


def test_cached_or_call_writes_meta_sidecar():
    def fn():
        return pd.DataFrame({"x": [1, 2, 3]})

    Cache.cached_or_call(FAFB, "synapses", {"pre": [9]}, fn)
    qh = query_hash(FAFB.key, FAFB.materialization, FAFB.synapse_table, "synapses", {"pre": [9]})
    path = Cache.cache_path(FAFB, "synapses", qh)
    meta = Cache.meta_path(path)
    assert path.is_file() and meta.is_file()
    import json
    m = json.loads(meta.read_text())
    assert m["n_rows"] == 3 and m["dataset"] == "fafb" and m["query_hash"] == qh


def test_offline_client_raises_on_miss_without_network():
    """allow_network=False must raise on a cache miss, never invoke the thunk."""
    c = ConnectomeClient("fafb", allow_network=False)
    with pytest.raises(RuntimeError, match="Cache miss"):
        c.synapse_query(pre_ids=[12345])


def test_offline_client_serves_cached_query():
    """A populated cache lets the offline client return data with no network."""
    def fn():
        return pd.DataFrame({"pre_pt_root_id": [7], "post_pt_root_id": [8], "cleft_score": [120]})

    # populate via the cache layer using the same key the client will compute
    params = {"pre": [7], "post": None, "cleft_threshold": None, "columns": None}
    Cache.cached_or_call(FAFB, "synapses", params, fn)
    c = ConnectomeClient("fafb", allow_network=False)
    df = c.synapse_query(pre_ids=[7])
    assert list(df["post_pt_root_id"]) == [8]


def test_token_precedence_env_over_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CAVE_TOKEN", "from-env")
    assert Secrets.read_cave_token() == "from-env"
    assert Secrets.read_cave_token(explicit="explicit-wins") == "explicit-wins"


def test_token_missing_raises_listing_locations(monkeypatch):
    monkeypatch.delenv("CAVE_TOKEN", raising=False)
    monkeypatch.delenv("CAVECLIENT_TOKEN", raising=False)
    monkeypatch.setattr(Secrets, "_CLOUDVOLUME_SECRET", tmp_path_nonexist())
    monkeypatch.setattr(Secrets, "_REPO_SECRET", tmp_path_nonexist())
    with pytest.raises(RuntimeError, match="No CAVE token found"):
        Secrets.read_cave_token()


def tmp_path_nonexist():
    from pathlib import Path
    return Path("/nonexistent/definitely/not/here.json")
