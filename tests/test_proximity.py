from __future__ import annotations

import json
import sys
import types

import numpy as np
import pandas as pd

from flyconn.experiments import proximity


def _toy_neurons(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame({
        "idx": np.arange(n, dtype=np.int64),
        "root_id": np.arange(1000, 1000 + n, dtype=np.int64),
        "side": ["left"] * n,
        "super_class": ["optic"] * n,
        "cell_type": ["T4"] * n,
    })


def test_select_neuron_sample_is_deterministic():
    neurons = _toy_neurons(20)
    a = proximity.select_neuron_sample(neurons, "optic_left", n=5, seed=7)
    b = proximity.select_neuron_sample(neurons, "optic_left", n=5, seed=7)

    assert list(a["idx"]) == list(b["idx"])
    assert list(a["idx"]) == sorted(a["idx"])
    assert len(a) == 5


def test_downsample_points_nm_keeps_one_point_per_grid_cell():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
        [300.0, 0.0, 0.0],
        [301.0, 1.0, 1.0],
    ])
    out = proximity.downsample_points_nm(pts, spacing_nm=250)

    assert out.shape == (2, 3)
    assert [tuple(x) for x in out] == [(0.0, 0.0, 0.0), (300.0, 0.0, 0.0)]


def test_dendrite_samples_from_mesh_restricts_to_postsynaptic_neighborhood():
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [50.0, 0.0, 0.0],
        [5000.0, 0.0, 0.0],
    ])
    faces = np.array([[0, 1, 2]])
    post_sites = np.array([[0.0, 0.0, 0.0]])

    samples = proximity.dendrite_samples_from_mesh(
        vertices,
        faces,
        post_sites,
        site_radius_nm=100.0,
        sample_spacing_nm=25.0,
    )

    assert samples.shape == (2, 3)
    assert np.all(samples[:, 0] < 100.0)


def test_find_near_pairs_uses_unordered_neuron_pairs_and_min_distance():
    points = {
        1: np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]]),
        2: np.array([[1500.0, 0.0, 0.0]]),
        3: np.array([[10000.0, 0.0, 0.0]]),
    }
    roots = {1: 101, 2: 102, 3: 103}

    pairs = proximity.find_near_pairs(points, roots, threshold_nm=600.0)

    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert (int(row.idx_a), int(row.idx_b)) == (1, 2)
    assert int(row.root_id_a) == 101
    assert int(row.root_id_b) == 102
    assert row.min_distance_nm == 500.0


def test_annotate_connections_counts_either_direction():
    near = pd.DataFrame({
        "idx_a": [1, 3],
        "idx_b": [2, 4],
        "root_id_a": [101, 103],
        "root_id_b": [102, 104],
        "min_distance_nm": [500.0, 400.0],
        "n_close_sample_pairs": [1, 1],
    })
    edges = pd.DataFrame({
        "pre_idx": [2, 8],
        "post_idx": [1, 9],
        "syn_count": [7, 2],
    })

    out = proximity.annotate_connections(near, edges)

    assert list(out["syn_count_a_to_b"]) == [0, 0]
    assert list(out["syn_count_b_to_a"]) == [7, 0]
    assert list(out["connected_any"]) == [True, False]


def test_summary_handles_zero_near_pairs():
    params = proximity.ProximityParams(n=2)
    sample = pd.DataFrame({"idx": [1, 2]})
    failures = pd.DataFrame(columns=["idx", "root_id", "error"])
    near = proximity.annotate_connections(proximity._empty_near_pairs(), pd.DataFrame())

    summary = proximity.summarize(params, near, sample, failures)

    assert summary["near_pair_count"] == 0
    assert summary["fraction_connected"] is None
    assert summary["n_processed"] == 2


def test_load_postsynaptic_sites_nm_filters_roots_and_converts_voxels(tmp_path):
    df = pd.DataFrame({
        "post_pt_root_id": [10, 11, 10],
        "post_pt_position_x": [1, 2, 3],
        "post_pt_position_y": [4, 5, 6],
        "post_pt_position_z": [7, 8, 9],
    })
    path = tmp_path / "synapses.feather"
    df.to_feather(path)

    out = proximity.load_postsynaptic_sites_nm(path, [10, 12], coordinate_units="voxel")

    assert set(out) == {10, 12}
    assert np.allclose(out[10], np.array([[4.0, 16.0, 280.0], [12.0, 24.0, 360.0]]))
    assert out[12].shape == (0, 3)


def test_load_postsynaptic_sites_nm_keeps_nm_coordinates_by_default(tmp_path):
    df = pd.DataFrame({
        "post_pt_root_id": [10, 11, 10],
        "post_pt_position_x": [1, 2, 3],
        "post_pt_position_y": [4, 5, 6],
        "post_pt_position_z": [7, 8, 9],
    })
    path = tmp_path / "synapses.feather"
    df.to_feather(path)

    out = proximity.load_postsynaptic_sites_nm(path, [10, 12])

    assert set(out) == {10, 12}
    assert np.allclose(out[10], np.array([[1.0, 4.0, 7.0], [3.0, 6.0, 9.0]]))
    assert out[12].shape == (0, 3)


def test_fetch_cloudvolume_mesh_arrays_falls_back_to_available_lod():
    class FakeMesh:
        vertices = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        faces = np.array([[0, 1, 1]])

    class FakeMeshSource:
        def __init__(self):
            self.calls = []

        def get(self, root_id, lod=0, allow_missing=False):
            self.calls.append((root_id, lod, allow_missing))
            if lod == 1:
                raise RuntimeError("LOD value out of range")
            return {root_id: FakeMesh()}

    class FakeVolume:
        def __init__(self):
            self.mesh = FakeMeshSource()

    volume = FakeVolume()

    out = proximity.fetch_cloudvolume_mesh_arrays(
        42,
        volume=volume,
        mesh_path=proximity.PUBLIC_FLYWIRE_MESH_PATH,
        lod=1,
        lod_fallback=0,
        mesh_units="nm",
    )

    assert out.mesh_source == proximity.MESH_SOURCE_CLOUDVOLUME_PUBLIC
    assert out.lod_requested == 1
    assert out.lod_used == 0
    assert np.allclose(out.vertices_nm, FakeMesh.vertices)
    assert volume.mesh.calls == [(42, 1, True), (42, 0, True)]


def test_load_or_fetch_mesh_arrays_writes_and_reads_metadata_cache(tmp_path):
    def fetcher(root_id):
        return proximity.MeshFetchResult(
            vertices_nm=np.array([[1.0, 2.0, 3.0]]),
            faces=np.array([[0, 0, 0]]),
            mesh_source=proximity.MESH_SOURCE_CLOUDVOLUME_PUBLIC,
            mesh_path=proximity.PUBLIC_FLYWIRE_MESH_PATH,
            lod_requested=1,
            lod_used=0,
        )

    first = proximity.load_or_fetch_mesh_arrays(tmp_path, 42, fetcher)
    second = proximity.load_or_fetch_mesh_arrays(tmp_path, 42, fetcher)

    assert first.cache_status == "fetched"
    assert second.cache_status == "cached"
    assert second.mesh_source == proximity.MESH_SOURCE_CLOUDVOLUME_PUBLIC
    assert second.mesh_path == proximity.PUBLIC_FLYWIRE_MESH_PATH
    assert second.lod_requested == 1
    assert second.lod_used == 0
    assert np.allclose(second.vertices_nm, np.array([[1.0, 2.0, 3.0]]))


def test_configure_flywire_auth_from_env_persists_token(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CAVE_TOKEN", "test-token")

    def set_chunkedgraph_secret(token, overwrite=False):
        secret = tmp_path / ".cloudvolume" / "secrets" / "global.daf-apis.com-cave-secret.json"
        secret.parent.mkdir(parents=True)
        secret.write_text(json.dumps({"token": token, "overwrite": overwrite}))
        print("should be suppressed")

    fake_fafbseg = types.SimpleNamespace(
        flywire=types.SimpleNamespace(set_chunkedgraph_secret=set_chunkedgraph_secret)
    )
    monkeypatch.setitem(sys.modules, "fafbseg", fake_fafbseg)

    assert proximity.configure_flywire_auth_from_env()
    assert proximity.has_saved_flywire_auth()
    assert capsys.readouterr().out == ""
