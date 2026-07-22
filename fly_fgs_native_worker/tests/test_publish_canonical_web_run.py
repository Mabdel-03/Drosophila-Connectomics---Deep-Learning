import hashlib
import json
import threading
import time
from copy import deepcopy
from pathlib import Path

import pytest

from fly_sensor2behavior.canonical_artifacts import (
    CANONICAL_ARTIFACT_SCHEMA_VERSION,
    CANONICAL_WEB_PROJECTION_CANONICALIZATION,
    canonical_web_projection_sha256,
)
from scripts.publish_canonical_web_run import PublicationError, publish_canonical_run
import scripts.publish_canonical_web_run as publisher_module


def _dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, *, episode_id: str = "online-baseline", run_id: str = "online-baseline-abc123"):
    replay = {
        "schema_version": "1.0.0",
        "id": episode_id,
        "label": "Online baseline",
        "status": "exploratory",
        "source_kind": "canonical_online_fly_fgs_streaming_flybody",
        "frames": [{"t": 0.0}],
        "source_run_id": run_id,
    }
    artifact_dir = tmp_path / "candidate"
    artifact_dir.mkdir(parents=True)
    manifest = {
        "schema_version": CANONICAL_ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "web_replay_projection": {
            "schema_version": "1.0.0",
            "sha256": canonical_web_projection_sha256(replay),
            "canonicalization": CANONICAL_WEB_PROJECTION_CANONICALIZATION,
            "excluded_top_level_fields": [
                "source_artifact_manifest_sha256",
                "source_artifact_schema_version",
            ],
        },
    }
    _dump(artifact_dir / "manifest.json", manifest)
    manifest_digest = hashlib.sha256((artifact_dir / "manifest.json").read_bytes()).hexdigest()
    (artifact_dir / "manifest.sha256").write_text(
        f"{manifest_digest}  manifest.json\n", encoding="ascii"
    )
    replay["source_artifact_manifest_sha256"] = manifest_digest
    replay["source_artifact_schema_version"] = CANONICAL_ARTIFACT_SCHEMA_VERSION
    replay_path = tmp_path / "candidate-replay.json"
    _dump(replay_path, replay)
    public_root = tmp_path / "public"
    _dump(
        public_root / "data" / "manifest.json",
        {"schema_version": "1.0.0", "created_at": "old", "episodes": []},
    )
    return artifact_dir, replay_path, public_root


def _publish(artifact_dir: Path, replay_path: Path, public_root: Path, **overrides):
    values = {
        "artifact_dir": artifact_dir,
        "replay_path": replay_path,
        "web_public_root": public_root,
        "episode_id": "online-baseline",
        "condition": "moving figure",
        "description": "Exact causal integration test.",
        "color": "#12ABef",
        "created_at": "2026-07-18T00:00:00Z",
    }
    values.update(overrides)
    return publish_canonical_run(**values)


def test_publisher_installs_content_bound_run_and_updates_index_last(tmp_path):
    artifact_dir, replay_path, public_root = _fixture(tmp_path)
    summary = _publish(artifact_dir, replay_path, public_root)
    manifest = json.loads((public_root / "data" / "manifest.json").read_text())
    assert manifest["episodes"] == [summary]
    assert summary["color"] == "#12abef"
    assert summary["replay_sha256"] == hashlib.sha256(replay_path.read_bytes()).hexdigest()
    target_run = public_root / "data" / "runs" / "online-baseline-abc123"
    assert (target_run / "manifest.json").read_bytes() == (artifact_dir / "manifest.json").read_bytes()
    assert summary["artifact_manifest_sha256"] == hashlib.sha256(
        (target_run / "manifest.json").read_bytes()
    ).hexdigest()
    assert (public_root / "data" / "episodes" / "online-baseline.json").read_bytes() == replay_path.read_bytes()
    projection_path = public_root / summary["artifact_projection_url"]
    artifact_manifest = json.loads((target_run / "manifest.json").read_text())
    assert hashlib.sha256(projection_path.read_bytes()).hexdigest() == artifact_manifest[
        "web_replay_projection"
    ]["sha256"]
    projected_replay = json.loads(projection_path.read_text())
    assert projected_replay == {
        key: value
        for key, value in json.loads(replay_path.read_text()).items()
        if key
        not in (
            "source_artifact_manifest_sha256",
            "source_artifact_schema_version",
        )
    }
    # An exact retry is idempotent and does not require replacement authority.
    assert _publish(artifact_dir, replay_path, public_root) == summary


def test_publisher_can_place_the_canonical_episode_first(tmp_path):
    artifact_dir, replay_path, public_root = _fixture(tmp_path)
    historical_dir, historical_replay, _ = _fixture(
        tmp_path / "historical",
        episode_id="historical",
        run_id="historical-abc123",
    )
    _publish(
        historical_dir,
        historical_replay,
        public_root,
        episode_id="historical",
    )
    public_manifest_path = public_root / "data" / "manifest.json"
    # Simulate the existing exporter contract: exact files and artifact binding
    # are present, but summary-level browser byte receipts are not yet attached.
    exported = json.loads(public_manifest_path.read_text())
    for field in (
        "replay_sha256",
        "artifact_manifest_sha256",
        "artifact_projection_url",
    ):
        exported["episodes"][0].pop(field)
    _dump(public_manifest_path, exported)
    (public_root / "data" / "episodes" / "historical.artifact-projection.json").unlink()
    summary = _publish(artifact_dir, replay_path, public_root, position=0)
    published = json.loads(public_manifest_path.read_text())
    assert published["episodes"][0] == summary
    assert published["episodes"][1]["id"] == "historical"
    assert published["episodes"][1]["replay_sha256"] == hashlib.sha256(
        historical_replay.read_bytes()
    ).hexdigest()
    assert (
        public_root
        / published["episodes"][1]["artifact_projection_url"]
    ).is_file()


def test_concurrent_distinct_publications_cannot_lose_an_index_update(
    tmp_path, monkeypatch
):
    first_dir, first_replay, public_root = _fixture(
        tmp_path / "first",
        episode_id="online-first",
        run_id="online-first-abc123",
    )
    second_dir, second_replay, _ = _fixture(
        tmp_path / "second",
        episode_id="online-second",
        run_id="online-second-def456",
    )
    # Both candidates target the same public tree; the second fixture's private
    # public directory is intentionally unused.
    first_read = threading.Event()
    release_first = threading.Event()
    second_read = threading.Event()
    original_load_json = publisher_module._load_json

    def synchronized_load(path, label):
        if label == "public replay manifest":
            if threading.current_thread().name == "first-publisher":
                first_read.set()
                assert release_first.wait(timeout=5.0)
            elif threading.current_thread().name == "second-publisher":
                second_read.set()
        return original_load_json(path, label)

    monkeypatch.setattr(publisher_module, "_load_json", synchronized_load)
    errors = []

    def publish_first():
        try:
            _publish(
                first_dir,
                first_replay,
                public_root,
                episode_id="online-first",
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def publish_second():
        try:
            _publish(
                second_dir,
                second_replay,
                public_root,
                episode_id="online-second",
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first = threading.Thread(target=publish_first, name="first-publisher")
    second = threading.Thread(target=publish_second, name="second-publisher")
    first.start()
    assert first_read.wait(timeout=5.0)
    second.start()
    time.sleep(0.1)
    assert not second_read.is_set(), "second publisher entered the manifest critical section"
    release_first.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert second_read.is_set()
    published = json.loads(
        (public_root / "data" / "manifest.json").read_text(encoding="utf-8")
    )
    assert {episode["id"] for episode in published["episodes"]} == {
        "online-first",
        "online-second",
    }


def test_publisher_rejects_projection_or_manifest_receipt_tampering(tmp_path):
    artifact_dir, replay_path, public_root = _fixture(tmp_path)
    replay = json.loads(replay_path.read_text())
    replay["frames"][0]["t"] = 0.1
    _dump(replay_path, replay)
    with pytest.raises(PublicationError, match="projection receipt"):
        _publish(artifact_dir, replay_path, public_root)

    artifact_dir, replay_path, public_root = _fixture(tmp_path / "second")
    (artifact_dir / "manifest.sha256").write_text(
        f"{'0' * 64}  manifest.json\n", encoding="ascii"
    )
    with pytest.raises(PublicationError, match="manifest.sha256"):
        _publish(artifact_dir, replay_path, public_root)


def test_publisher_requires_explicit_episode_replacement(tmp_path):
    artifact_dir, replay_path, public_root = _fixture(tmp_path)
    _publish(artifact_dir, replay_path, public_root)

    original = json.loads(replay_path.read_text())
    replacement_replay = deepcopy(original)
    replacement_replay["source_run_id"] = "online-baseline-def456"
    replacement_replay.pop("source_artifact_manifest_sha256")
    replacement_replay.pop("source_artifact_schema_version")
    replacement_dir = tmp_path / "replacement"
    replacement_dir.mkdir()
    replacement_manifest = {
        "schema_version": CANONICAL_ARTIFACT_SCHEMA_VERSION,
        "run_id": replacement_replay["source_run_id"],
        "web_replay_projection": {
            "schema_version": "1.0.0",
            "sha256": canonical_web_projection_sha256(replacement_replay),
            "canonicalization": CANONICAL_WEB_PROJECTION_CANONICALIZATION,
            "excluded_top_level_fields": [
                "source_artifact_manifest_sha256",
                "source_artifact_schema_version",
            ],
        },
    }
    _dump(replacement_dir / "manifest.json", replacement_manifest)
    digest = hashlib.sha256((replacement_dir / "manifest.json").read_bytes()).hexdigest()
    (replacement_dir / "manifest.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )
    replacement_replay["source_artifact_manifest_sha256"] = digest
    replacement_replay["source_artifact_schema_version"] = CANONICAL_ARTIFACT_SCHEMA_VERSION
    replacement_path = tmp_path / "replacement-replay.json"
    _dump(replacement_path, replacement_replay)

    with pytest.raises(PublicationError, match="--replace-episode"):
        _publish(replacement_dir, replacement_path, public_root)
    summary = _publish(
        replacement_dir,
        replacement_path,
        public_root,
        replace_episode=True,
    )
    assert summary["artifact_manifest_url"].endswith(
        "/online-baseline-def456/manifest.json"
    )
