from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flyconn.experiments import proximity_full


class _DummyPaths:
    def __init__(self, root: Path):
        self.root = root

    def ensure(self):
        self.root.mkdir(parents=True, exist_ok=True)
        return self


class _DummyCfg:
    def __init__(self, root: Path):
        self._paths = _DummyPaths(root)

    def paths(self):
        return self._paths


def test_build_shard_manifest_assigns_stable_chunks():
    neurons = pd.DataFrame({
        "idx": [10, 11, 12, 13, 14],
        "root_id": [100, 101, 102, 103, 104],
    })

    out, shards = proximity_full.build_shard_manifest(neurons, shard_size=2)

    assert list(out["shard_id"]) == [0, 0, 1, 1, 2]
    assert list(shards["n_neurons"]) == [2, 2, 1]
    assert list(shards["min_idx"]) == [10, 12, 14]


def test_compute_neuropil_stats_reports_dominant_and_top_regions():
    counts = pd.DataFrame({
        "post_pt_root_id": [10, 10, 10, 11],
        "neuropil": ["ME_L", "LO_L", "AL_R", "AL_R"],
        "count": [5, 20, 10, 7],
    })

    out = proximity_full.compute_neuropil_stats(
        counts,
        root_col="post_pt_root_id",
        prefix="post",
        root_ids=[10, 11, 12],
        top_k=2,
    ).sort_values("root_id")

    row10 = out[out["root_id"] == 10].iloc[0]
    row12 = out[out["root_id"] == 12].iloc[0]
    assert int(row10["post_synapse_count"]) == 35
    assert row10["dominant_post_neuropil"] == "LO_L"
    assert np.isclose(row10["dominant_post_neuropil_fraction"], 20 / 35)
    assert row10["top_post_neuropils"] == "LO_L:20;AL_R:10"
    assert int(row12["post_synapse_count"]) == 0
    assert row12["top_post_neuropils"] == ""


def test_partition_postsynaptic_sites_writes_shard_parquets(tmp_path):
    syn = pd.DataFrame({
        "post_pt_root_id": [10, 11, 12, 99],
        "post_pt_position_x": [1, 2, 3, 4],
        "post_pt_position_y": [5, 6, 7, 8],
        "post_pt_position_z": [9, 10, 11, 12],
        "neuropil": ["ME_L", "LO_L", "AL_R", "GNG"],
    })
    syn_path = tmp_path / "synapses.feather"
    syn.to_feather(syn_path)
    neurons = pd.DataFrame({
        "root_id": [10, 11, 12],
        "idx": [0, 1, 2],
        "shard_id": [0, 0, 1],
    })

    proximity_full.partition_postsynaptic_sites(
        syn_path,
        neurons,
        tmp_path / "postsites",
        coordinate_units="nm",
    )

    shard0 = pd.read_parquet(tmp_path / "postsites" / "shard_00000.parquet")
    shard1 = pd.read_parquet(tmp_path / "postsites" / "shard_00001.parquet")
    assert list(shard0["root_id"]) == [10, 11]
    assert list(shard0["idx"]) == [0, 1]
    assert list(shard1["root_id"]) == [12]
    assert (tmp_path / "postsites" / "_SUCCESS.json").exists()


def test_tiled_pair_search_finds_cross_tile_pair_once(tmp_path):
    cfg = _DummyCfg(tmp_path)
    params = proximity_full.FullRunParams(run_name="toy", threshold_um=0.5, tile_nm=1000.0)
    out = proximity_full.run_dir(cfg, params)
    (out / "samples").mkdir(parents=True)
    np.savez_compressed(
        out / "samples" / "shard_00000.npz",
        points_nm=np.array([[900.0, 0.0, 0.0], [1100.0, 0.0, 0.0]], dtype=np.float32),
        idx=np.array([1, 2], dtype=np.int64),
        root_id=np.array([101, 102], dtype=np.int64),
    )

    built = proximity_full.build_tiles(cfg, params)
    assert built["n_tiles"] == 2
    manifest = pd.read_parquet(out / "tiles" / "tile_manifest.parquet")
    for tile_id in manifest["tile_id"]:
        proximity_full.pair_tile(cfg, params, int(tile_id))
    near = proximity_full.reduce_near_pair_tiles((out / "near_pairs_tiles").glob("tile_*.parquet"))

    assert len(near) == 1
    row = near.iloc[0]
    assert (int(row.idx_a), int(row.idx_b)) == (1, 2)
    assert row.min_distance_nm == 200.0
    assert int(row.n_close_sample_pairs) == 1


def test_streaming_pair_tile_counts_and_spills_exactly(tmp_path):
    cfg = _DummyCfg(tmp_path)
    params = proximity_full.FullRunParams(
        run_name="toy",
        threshold_um=0.5,
        tile_nm=1000.0,
        pair_core_chunk_size=1,
        pair_spill_rows=1,
    )
    out = proximity_full.run_dir(cfg, params)
    (out / "samples").mkdir(parents=True)
    np.savez_compressed(
        out / "samples" / "shard_00000.npz",
        points_nm=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 100.0, 0.0],
                [100.0, 0.0, 0.0],
                [300.0, 0.0, 0.0],
                [10_000.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        idx=np.array([1, 1, 2, 2, 3], dtype=np.int64),
        root_id=np.array([101, 101, 102, 102, 103], dtype=np.int64),
    )

    proximity_full.build_tiles(cfg, params)
    manifest = pd.read_parquet(out / "tiles" / "tile_manifest.parquet")
    for tile_id in manifest["tile_id"]:
        proximity_full.pair_tile(cfg, params, int(tile_id))
    near = proximity_full.reduce_near_pair_tiles((out / "near_pairs_tiles").glob("tile_*.parquet"))

    assert len(near) == 1
    row = near.iloc[0]
    assert (int(row.idx_a), int(row.idx_b)) == (1, 2)
    assert int(row.n_close_sample_pairs) == 4
    assert row.min_distance_nm == 100.0
    assert row.closest_a_x_nm == 0.0
    assert row.closest_b_x_nm == 100.0
    assert not list((out / "near_pairs_tile_spills").glob("*"))


def test_reduce_annotation_and_enrichment_adds_metadata():
    near = pd.DataFrame({
        "idx_a": [1],
        "idx_b": [2],
        "root_id_a": [101],
        "root_id_b": [102],
        "min_distance_nm": [200.0],
        "n_close_sample_pairs": [1],
    })
    for name in [
        "closest_a_x_nm",
        "closest_a_y_nm",
        "closest_a_z_nm",
        "closest_b_x_nm",
        "closest_b_y_nm",
        "closest_b_z_nm",
        "mid_x_nm",
        "mid_y_nm",
        "mid_z_nm",
    ]:
        near[name] = 0.0
    edges = pd.DataFrame({"pre_idx": [2], "post_idx": [1], "syn_count": [7]})
    manifest = pd.DataFrame({
        "idx": [1, 2],
        "root_id": [101, 102],
        "cell_type": ["T4", "T5"],
        "super_class": ["optic", "optic"],
        "nt_canonical": ["acetylcholine", "gaba"],
        "dominant_post_neuropil": ["ME_L", "LO_L"],
    })

    annotated = proximity_full.annotate_connection_counts(near, edges, prefix="any")
    annotated["connected_any"] = annotated["any_connected_any"]
    enriched = proximity_full.enrich_pairs(annotated, manifest)

    assert int(enriched.iloc[0]["any_syn_count_b_to_a"]) == 7
    assert bool(enriched.iloc[0]["connected_any"])
    assert enriched.iloc[0]["cell_type_a"] == "T4"
    assert enriched.iloc[0]["cell_type_b"] == "T5"
    assert enriched.iloc[0]["dominant_post_neuropil_a"] == "ME_L"


def test_reduce_bucket_matches_full_aggregation(tmp_path):
    cfg = _DummyCfg(tmp_path)
    params = proximity_full.FullRunParams(run_name="toy", reduce_buckets=2)
    out = proximity_full.run_dir(cfg, params)
    (out / "near_pairs_tiles").mkdir(parents=True)
    (out / "tiles").mkdir(parents=True)
    pd.DataFrame({"tile_id": [0, 1], "tile_key": [0, 1]}).to_parquet(out / "tiles" / "tile_manifest.parquet", index=False)

    def pairs_frame(rows):
        df = pd.DataFrame(rows)
        for name in [
            "closest_a_x_nm",
            "closest_a_y_nm",
            "closest_a_z_nm",
            "closest_b_x_nm",
            "closest_b_y_nm",
            "closest_b_z_nm",
            "mid_x_nm",
            "mid_y_nm",
            "mid_z_nm",
        ]:
            df[name] = df.get(name, 0.0)
        return df[proximity_full._empty_tile_pairs().columns]

    pairs_frame([
        {
            "idx_a": 1,
            "idx_b": 2,
            "root_id_a": 101,
            "root_id_b": 102,
            "min_distance_nm": 200.0,
            "n_close_sample_pairs": 3,
        },
        {
            "idx_a": 3,
            "idx_b": 4,
            "root_id_a": 103,
            "root_id_b": 104,
            "min_distance_nm": 150.0,
            "n_close_sample_pairs": 1,
        },
    ]).to_parquet(out / "near_pairs_tiles" / "tile_000000.parquet", index=False)
    pairs_frame([
        {
            "idx_a": 1,
            "idx_b": 2,
            "root_id_a": 101,
            "root_id_b": 102,
            "min_distance_nm": 100.0,
            "n_close_sample_pairs": 5,
            "closest_a_x_nm": 1.0,
        },
    ]).to_parquet(out / "near_pairs_tiles" / "tile_000001.parquet", index=False)

    for bucket_id in range(params.reduce_buckets):
        proximity_full.reduce_bucket(cfg, params, bucket_id)
    bucketed = proximity_full.reduce_near_pair_tiles((out / "near_pairs_reduce_buckets").glob("bucket_*.parquet"))
    full = proximity_full.reduce_near_pair_tiles((out / "near_pairs_tiles").glob("tile_*.parquet"))
    bucketed = bucketed.sort_values(["idx_a", "idx_b"]).reset_index(drop=True)
    full = full.sort_values(["idx_a", "idx_b"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(bucketed, full)
