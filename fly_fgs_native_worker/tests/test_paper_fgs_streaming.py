import json
from pathlib import Path

import numpy as np
import pytest

from fly_sensor2behavior.paper_fgs_streaming import (
    MemmapTrialSequence,
    PaperStreamingError,
    StreamingPaperRunStore,
)
from fly_sensor2behavior.paper_fgs_release_cli import _stage_web_replays


def _trial(trial_id):
    return {
        "time_s": np.arange(12, dtype=float) * 0.001,
        "raw_internal_padded_reported_yaw_torque_Nm": (
            np.arange(9000, dtype=float) + trial_id * 10000
        ),
        "wing_reported_yaw_torque_Nm": np.full((12, 2), trial_id + 0.25),
    }


def _receipt(trial_id):
    return {"diagnostics": {"trial_id": trial_id, "passed": True}}


def _complete(root, *, interrupt=False):
    identity = {"schema_version": "test.v1", "seed": 73}
    store = StreamingPaperRunStore(
        root, repetitions=2, identity=identity, resume=False
    )
    store.begin_trial(0)
    store.commit_trial(0, _trial(0), _receipt(0))
    if interrupt:
        store = StreamingPaperRunStore(
            root, repetitions=2, identity=identity, resume=True
        )
    store.begin_trial(1)
    store.commit_trial(1, _trial(1), _receipt(1))
    return store


def test_streaming_store_resume_matches_uninterrupted(tmp_path):
    uninterrupted = _complete(tmp_path / "uninterrupted")
    resumed = _complete(tmp_path / "resumed", interrupt=True)

    left = MemmapTrialSequence(uninterrupted.root, 2)
    right = MemmapTrialSequence(resumed.root, 2)
    assert set(left.arrays) == set(right.arrays)
    for name in left.arrays:
        np.testing.assert_array_equal(left.stacked_array(name), right.stacked_array(name))
    assert (
        uninterrupted.state["completed_trials"]
        == resumed.state["completed_trials"]
    )
    assert uninterrupted.state["block_samples"] == 4096


def test_completed_trials_are_immutable(tmp_path):
    store = StreamingPaperRunStore(
        tmp_path / "run",
        repetitions=1,
        identity={"schema_version": "test.v1"},
        resume=False,
    )
    store.begin_trial(0)
    store.commit_trial(0, _trial(0), _receipt(0))
    with pytest.raises(PaperStreamingError, match="immutable"):
        store.begin_trial(0)


def test_resume_rejects_identity_and_array_corruption(tmp_path):
    root = tmp_path / "run"
    store = StreamingPaperRunStore(
        root,
        repetitions=1,
        identity={"schema_version": "test.v1", "seed": 1},
        resume=False,
    )
    store.begin_trial(0)
    store.commit_trial(0, _trial(0), _receipt(0))

    with pytest.raises(PaperStreamingError, match="identity mismatch"):
        StreamingPaperRunStore(
            root,
            repetitions=1,
            identity={"schema_version": "test.v1", "seed": 2},
            resume=True,
        )

    specs = json.loads((root / "array_specs.json").read_text("utf-8"))["arrays"]
    torque_path = root / "arrays" / specs[
        "raw_internal_padded_reported_yaw_torque_Nm"
    ]["file"]
    torque = np.load(torque_path, mmap_mode="r+")
    torque[0, 17] += 1.0
    torque.flush()
    del torque
    with pytest.raises(PaperStreamingError, match="array checksum mismatch"):
        StreamingPaperRunStore(
            root,
            repetitions=1,
            identity={"schema_version": "test.v1", "seed": 1},
            resume=True,
        )


def test_incomplete_trial_is_restarted_and_overwritten(tmp_path):
    root = tmp_path / "run"
    identity = {"schema_version": "test.v1"}
    store = StreamingPaperRunStore(
        root, repetitions=1, identity=identity, resume=False
    )
    store.begin_trial(0)
    resumed = StreamingPaperRunStore(
        root, repetitions=1, identity=identity, resume=True
    )
    assert resumed.completed_trial_ids == ()
    resumed.begin_trial(0)
    resumed.commit_trial(0, _trial(0), _receipt(0))
    np.testing.assert_array_equal(
        resumed.trial_views()[0]["time_s"], _trial(0)["time_s"]
    )


def test_replay_publication_is_atomic_and_idempotent(tmp_path):
    release = tmp_path / "release_manifest.json"
    release.write_text('{"schema_version":"2.0.0"}\n', encoding="utf-8")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "web_replay.json").write_text(
        '{"schema_version":"paper_fgs_web_replay.v3"}\n', encoding="utf-8"
    )
    (artifact / "manifest.json").write_text(
        '{"scientific_status":"authoritative_native_torque"}\n',
        encoding="utf-8",
    )
    source = {
        "R83_Fig3a_0_to_90": {
            "artifact": artifact,
            "run_id": "run-a",
            "scientific_status": "authoritative_native_torque",
        }
    }
    web = tmp_path / "web"
    _stage_web_replays(
        web_root=web, release_path=release, canonical_sources=source
    )
    first_index = json.loads((web / "paper_replay_index.json").read_text("utf-8"))
    _stage_web_replays(
        web_root=web, release_path=release, canonical_sources=source
    )
    second_index = json.loads((web / "paper_replay_index.json").read_text("utf-8"))
    assert first_index == second_index
    assert not list(web.glob("paper_replay_index.*.backup.json"))
