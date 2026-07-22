"""Execute and resume the declared 1,200-trial paper-assay release matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .paper_fgs import _sha256_file, default_release_matrix_path
from .paper_fgs_streaming import _atomic_write_json


RELEASE_JOURNAL_SCHEMA_VERSION = "paper_fgs_matrix_release_state.v1"
MAXIMUM_GIT_BOUND_FILE_BYTES = 90 * 1024**2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fly-s2b-paper-fgs-release")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-node-version", default="v22.22.1")
    parser.add_argument("--worker-manifest", type=Path, default=None)
    parser.add_argument("--matrix", type=Path, default=default_release_matrix_path())
    parser.add_argument(
        "--web-replay-root",
        type=Path,
        default=None,
        help="optional fly_fgs_source/data directory for authoritative replay registration",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--publish-only",
        action="store_true",
        help="register already-verified canonical replays without mutating artifacts",
    )
    parser.add_argument(
        "--minimum-free-gib",
        type=float,
        default=250.0,
        help="fail before execution unless the external artifact volume has this capacity",
    )
    return parser


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _regular_file_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _load_matrix(path: Path) -> tuple[bytes, Mapping[str, Any]]:
    payload = path.read_bytes()
    matrix = json.loads(payload)
    if matrix.get("schema_version") != "1.0.0":
        raise ValueError("paper release matrix schema mismatch")
    if int(matrix.get("repetitions_per_run", -1)) != 100:
        raise ValueError("scientific release requires exactly 100 repetitions per run")
    runs = matrix.get("runs")
    if not isinstance(runs, list) or len(runs) != 12:
        raise ValueError("scientific release requires the declared 12-run matrix")
    names = [item.get("run_name") for item in runs]
    if len(set(names)) != 12 or any(not isinstance(name, str) for name in names):
        raise ValueError("release run names must be unique strings")
    return payload, matrix


def _journal_identity(matrix_bytes: bytes, source_root: Path) -> Mapping[str, Any]:
    return {
        "matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
        "source_root": str(source_root.resolve()),
        "repetitions_per_run": 100,
        "expected_total_trials": 1200,
    }


def _read_status(output_root: Path) -> Mapping[str, Any]:
    state_path = output_root / "release_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = json.loads(state_path.read_text("utf-8"))
    run_status = {}
    for run_name, value in state.get("runs", {}).items():
        run_state_path = output_root / run_name / "release_state.json"
        if run_state_path.is_file():
            run_state = json.loads(run_state_path.read_text("utf-8"))
            run_status[run_name] = {
                **value,
                "completed_trials": len(run_state.get("completed_trials", {})),
                "in_progress_trial_id": run_state.get("in_progress_trial_id"),
                "peak_rss_bytes": run_state.get("peak_rss_bytes"),
            }
        else:
            run_status[run_name] = value
    return {**state, "runs": run_status}


def _streaming_command(
    run: Mapping[str, Any],
    matrix: Mapping[str, Any],
    *,
    output_root: Path,
    source_root: Path,
    expected_node_version: str,
    worker_manifest: Optional[Path],
    resume: bool,
) -> list[str]:
    output = output_root / run["run_name"]
    command = [
        sys.executable,
        "-m",
        "fly_sensor2behavior.paper_fgs_streaming",
        "--protocol",
        run["protocol"],
        "--repetitions",
        str(matrix["repetitions_per_run"]),
        "--run-seed",
        str(matrix["run_seed"]),
        "--texture-mode",
        run["texture_mode"],
        "--texture-seed",
        str(run["texture_seed"]),
        "--circuit-rate-hz",
        str(run["circuit_rate_hz"]),
        "--physics-rate-hz",
        str(run["physics_rate_hz"]),
        "--torque-meter",
        run.get("torque_meter", "fixed-load-cell"),
        "--wingbeat-phase-mode",
        run.get("wingbeat_phase_mode", "uniformly_stratified"),
        "--output",
        str(output),
        "--source-root",
        str(source_root),
        "--expected-node-version",
        expected_node_version,
    ]
    if worker_manifest is not None:
        command.extend(["--worker-manifest", str(worker_manifest.resolve())])
    references = run.get("convergence_references")
    if references is None:
        legacy = run.get("convergence_reference")
        references = [] if legacy is None else [legacy]
    for name in references:
        command.extend(["--convergence-reference", str(output_root / name)])
    if resume:
        command.append("--resume")
    return command


def _stage_web_replays(
    *,
    web_root: Path,
    release_path: Path,
    canonical_sources: Mapping[str, Mapping[str, Any]],
) -> None:
    web_root.mkdir(parents=True, exist_ok=True)
    release_digest = _sha256_file(release_path)[:16]
    bundle_name = "paper_replays_v3_{}".format(release_digest)
    final_bundle = web_root / bundle_name
    web_replays = {}
    sources = {}
    for protocol_id, source in sorted(canonical_sources.items()):
        if source["scientific_status"] != "authoritative_native_torque":
            raise RuntimeError("canonical replay lost authoritative status")
        replay_name = "{}.json".format(protocol_id)
        replay_source = source["artifact"] / "web_replay.json"
        manifest_source = source["artifact"] / "manifest.json"
        for candidate in (replay_source, manifest_source):
            if candidate.stat().st_size > MAXIMUM_GIT_BOUND_FILE_BYTES:
                raise RuntimeError(
                    "Git-bound replay exceeds the 90 MiB release limit: {}".format(
                        candidate
                    )
                )
        sources[replay_name] = replay_source
        sources[replay_name + ".manifest.json"] = manifest_source
        web_replays[protocol_id] = {
            "replay": "data/{}/{}".format(bundle_name, replay_name),
            "manifest": "data/{}/{}.manifest.json".format(bundle_name, replay_name),
            "run_id": source["run_id"],
        }
    if final_bundle.exists():
        for name, source in sources.items():
            installed = final_bundle / name
            if not installed.is_file() or _sha256_file(installed) != _sha256_file(source):
                raise RuntimeError("existing immutable replay bundle is incomplete or corrupt")
    else:
        stage_bundle = Path(
            tempfile.mkdtemp(prefix=".paper_replays_v3_stage-", dir=str(web_root))
        )
        try:
            for name, source in sources.items():
                shutil.copyfile(source, stage_bundle / name)
            os.replace(stage_bundle, final_bundle)
        except Exception:
            if stage_bundle.exists():
                shutil.rmtree(stage_bundle)
            raise
    index_path = web_root / "paper_replay_index.json"
    desired_index = {
        "schema_version": "paper_replay_index.v3",
        "release_manifest_sha256": _sha256_file(release_path),
        "immutable_bundle": bundle_name,
        "replays": web_replays,
    }
    if index_path.exists():
        current_index = json.loads(index_path.read_text("utf-8"))
        if current_index == desired_index:
            return
        backup = web_root / "paper_replay_index.{}.backup.json".format(
            time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        )
        shutil.copyfile(index_path, backup)
    _atomic_write_json(
        web_root / ".paper_replay_index.json.tmp",
        desired_index,
    )
    os.replace(web_root / ".paper_replay_index.json.tmp", index_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    output_root = arguments.output_root.resolve()
    source_root = arguments.source_root.resolve()
    if arguments.status:
        print(json.dumps(_read_status(output_root), indent=2, sort_keys=True))
        return 0
    matrix_bytes, matrix = _load_matrix(arguments.matrix.resolve())
    if arguments.publish_only:
        if arguments.web_replay_root is None:
            raise ValueError("--publish-only requires --web-replay-root")
        release_path = output_root / "release_manifest.json"
        release = json.loads(release_path.read_text("utf-8"))
        if release.get("matrix_sha256") != hashlib.sha256(matrix_bytes).hexdigest():
            raise RuntimeError("release and matrix checksums disagree")
        if release.get("verified_trial_count") != 1200 or len(release.get("runs", [])) != 12:
            raise RuntimeError("publication requires all 12 runs and 1,200 trials")
        if release.get("all_canonical_runs_authoritative") is not True:
            raise RuntimeError("canonical runs are not authoritative")
        matrix_state = json.loads((output_root / "release_state.json").read_text("utf-8"))
        if (
            matrix_state.get("status") != "complete"
            or matrix_state.get("release_manifest_sha256") != _sha256_file(release_path)
        ):
            raise RuntimeError("matrix journal does not authenticate the release manifest")
        release_runs = {item["run_name"]: item for item in release["runs"]}
        canonical_sources = {}
        for run in matrix["runs"]:
            if not run["run_name"].endswith("-canonical"):
                continue
            record = release_runs[run["run_name"]]
            artifact = output_root / run["run_name"] / "artifact"
            manifest_path = artifact / "manifest.json"
            if _sha256_file(manifest_path) != record["manifest_sha256"]:
                raise RuntimeError("canonical artifact manifest checksum mismatch")
            canonical_sources[run["protocol"]] = {
                "artifact": artifact,
                "run_id": record["run_id"],
                "scientific_status": record["scientific_status"],
            }
        _stage_web_replays(
            web_root=arguments.web_replay_root.resolve(),
            release_path=release_path,
            canonical_sources=canonical_sources,
        )
        print(
            json.dumps(
                {
                    "published": True,
                    "release_manifest_sha256": _sha256_file(release_path),
                    "web_replay_root": str(arguments.web_replay_root.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    worker_root = Path(__file__).resolve().parents[2]
    repository_root = worker_root.parent
    if _is_within(output_root, repository_root) or _is_within(output_root, source_root):
        raise ValueError("PAPER_FGS_ARTIFACT_ROOT must be outside the Git/source trees")
    output_parent = output_root.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_parent).free
    retained_release_bytes = _regular_file_bytes(output_root)
    available_capacity_bytes = free_bytes + retained_release_bytes
    minimum_bytes = int(arguments.minimum_free_gib * 1024**3)
    if available_capacity_bytes < minimum_bytes:
        raise RuntimeError(
            "artifact volume has {:.1f} GiB available including the resumable "
            "release; {:.1f} GiB is required".format(
                available_capacity_bytes / 1024**3, arguments.minimum_free_gib
            )
        )
    commands = [
        _streaming_command(
            run,
            matrix,
            output_root=output_root,
            source_root=source_root,
            expected_node_version=arguments.expected_node_version,
            worker_manifest=arguments.worker_manifest,
            resume=(output_root / run["run_name"]).exists(),
        )
        for run in matrix["runs"]
    ]
    if arguments.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
                    "run_count": 12,
                    "trial_count": 1200,
                    "available_bytes": free_bytes,
                    "retained_release_bytes": retained_release_bytes,
                    "available_capacity_bytes": available_capacity_bytes,
                    "minimum_required_bytes": minimum_bytes,
                    "commands": commands,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    identity = _journal_identity(matrix_bytes, source_root)
    state_path = output_root / "release_state.json"
    if output_root.exists():
        if not arguments.resume:
            raise FileExistsError(
                "release root exists; pass --resume after inspecting status"
            )
        state = json.loads(state_path.read_text("utf-8"))
        if state.get("schema_version") != RELEASE_JOURNAL_SCHEMA_VERSION:
            raise RuntimeError("matrix release journal schema mismatch")
        if state.get("identity") != identity:
            raise RuntimeError("release journal identity mismatch")
    else:
        output_root.mkdir()
        state = {
            "schema_version": RELEASE_JOURNAL_SCHEMA_VERSION,
            "identity": identity,
            "status": "running",
            "runs": {},
        }
        _atomic_write_json(state_path, state)
    completed = []
    canonical_sources = {}
    for run in matrix["runs"]:
        run_name = run["run_name"]
        run_root = output_root / run_name
        artifact = run_root / "artifact"
        manifest_path = artifact / "manifest.json"
        existing = state["runs"].get(run_name, {})
        if existing.get("status") == "complete":
            if (
                not manifest_path.is_file()
                or _sha256_file(manifest_path) != existing["manifest_sha256"]
            ):
                raise RuntimeError("completed matrix run checksum mismatch")
        else:
            state["runs"][run_name] = {"status": "running"}
            _atomic_write_json(state_path, state)
            command = _streaming_command(
                run,
                matrix,
                output_root=output_root,
                source_root=source_root,
                expected_node_version=arguments.expected_node_version,
                worker_manifest=arguments.worker_manifest,
                resume=run_root.exists(),
            )
            subprocess.run(command, check=True)
            manifest = json.loads(manifest_path.read_text("utf-8"))
            state["runs"][run_name] = {
                "status": "complete",
                "run_id": manifest["run_id"],
                "manifest_sha256": _sha256_file(manifest_path),
                "scientific_status": manifest["scientific_status"],
            }
            _atomic_write_json(state_path, state)
        manifest = json.loads(manifest_path.read_text("utf-8"))
        record = {
            "run_name": run_name,
            "run_id": manifest["run_id"],
            "manifest_sha256": _sha256_file(manifest_path),
            "scientific_status": manifest["scientific_status"],
        }
        completed.append(record)
        if run_name.endswith("-canonical"):
            canonical_sources[run["protocol"]] = {
                "artifact": artifact,
                "run_id": manifest["run_id"],
                "scientific_status": manifest["scientific_status"],
            }
    release = {
        "schema_version": "2.0.0",
        "matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
        "runs": completed,
        "verified_trial_count": 1200,
        "all_canonical_runs_authoritative": all(
            item["scientific_status"] == "authoritative_native_torque"
            for item in completed
            if item["run_name"].endswith("-canonical")
        ),
        "behavioral_targets_were_release_conditions": False,
    }
    release_path = output_root / "release_manifest.json"
    if release_path.exists():
        observed = json.loads(release_path.read_text("utf-8"))
        if observed != release:
            raise RuntimeError("immutable release manifest differs from completed runs")
    else:
        _atomic_write_json(release_path, release)
    state["status"] = "complete"
    state["release_manifest_sha256"] = _sha256_file(release_path)
    _atomic_write_json(state_path, state)
    if arguments.web_replay_root is not None and release[
        "all_canonical_runs_authoritative"
    ]:
        _stage_web_replays(
            web_root=arguments.web_replay_root.resolve(),
            release_path=release_path,
            canonical_sources=canonical_sources,
        )
    print(json.dumps(release, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
