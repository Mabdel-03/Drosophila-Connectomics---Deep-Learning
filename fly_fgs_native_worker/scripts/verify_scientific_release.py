#!/usr/bin/env python3
"""Deep-verify all journals, trial arrays, manifests, and authority gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fly_sensor2behavior.paper_fgs import _sha256_file, default_release_matrix_path
from fly_sensor2behavior.paper_fgs_streaming import (
    PAPER_STREAMING_MAX_RSS_BYTES,
    StreamingPaperRunStore,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=default_release_matrix_path())
    arguments = parser.parse_args()
    root = arguments.release_root.resolve()
    matrix_bytes = arguments.matrix.read_bytes()
    matrix = json.loads(matrix_bytes)
    release_path = root / "release_manifest.json"
    release = json.loads(release_path.read_text("utf-8"))
    if release.get("schema_version") != "2.0.0":
        raise RuntimeError("scientific release manifest schema mismatch")
    if release.get("matrix_sha256") != hashlib.sha256(matrix_bytes).hexdigest():
        raise RuntimeError("scientific release matrix checksum mismatch")
    declared = {item["run_name"]: item for item in release["runs"]}
    expected_names = [item["run_name"] for item in matrix["runs"]]
    if set(declared) != set(expected_names) or len(expected_names) != 12:
        raise RuntimeError("scientific release run inventory mismatch")
    verified_trials = 0
    maximum_rss = 0
    canonical_authority = []
    run_results = []
    for run_name in expected_names:
        run_root = root / run_name
        state = json.loads((run_root / "release_state.json").read_text("utf-8"))
        store = StreamingPaperRunStore(
            run_root,
            repetitions=100,
            identity=state["identity"],
            resume=True,
        )
        if state.get("status") != "complete" or state.get("in_progress_trial_id") is not None:
            raise RuntimeError("run is not complete: {}".format(run_name))
        if len(store.completed_trial_ids) != 100:
            raise RuntimeError("run does not have 100 verified trials: {}".format(run_name))
        maximum_rss = max(maximum_rss, int(state.get("peak_rss_bytes", 0)))
        if int(state.get("peak_rss_bytes", 0)) > PAPER_STREAMING_MAX_RSS_BYTES:
            raise RuntimeError("run exceeded the 16 GiB RSS gate: {}".format(run_name))
        manifest_path = run_root / "artifact/manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        record = declared[run_name]
        if _sha256_file(manifest_path) != record["manifest_sha256"]:
            raise RuntimeError("artifact manifest checksum mismatch: {}".format(run_name))
        if manifest["run_id"] != record["run_id"]:
            raise RuntimeError("artifact run ID mismatch: {}".format(run_name))
        replay_path = run_root / "artifact/web_replay.json"
        if replay_path.stat().st_size > 90 * 1024**2:
            raise RuntimeError("browser replay exceeds 90 MiB: {}".format(run_name))
        if run_name.endswith("-canonical"):
            canonical_authority.append(
                manifest.get("scientific_status") == "authoritative_native_torque"
            )
        verified_trials += 100
        run_results.append(
            {
                "run_name": run_name,
                "run_id": manifest["run_id"],
                "scientific_status": manifest["scientific_status"],
                "verified_trials": 100,
            }
        )
    if verified_trials != 1200 or not canonical_authority or not all(canonical_authority):
        raise RuntimeError("release authority or verified-trial gate failed")
    result = {
        "ok": True,
        "release_manifest_sha256": _sha256_file(release_path),
        "verified_runs": 12,
        "verified_trials": verified_trials,
        "all_canonical_runs_authoritative": True,
        "maximum_peak_rss_bytes": maximum_rss,
        "runs": run_results,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
