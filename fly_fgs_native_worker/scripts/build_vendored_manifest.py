#!/usr/bin/env python3
"""Regenerate the native-worker inventory after a deliberate source update."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    destination = root / "VENDORED_SOURCE_MANIFEST.json"
    files = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path == destination
            or "__pycache__" in path.parts
            or ".pytest_cache" in path.parts
            or any(part.endswith(".egg-info") for part in path.parts)
            or path.suffix == ".pyc"
        ):
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    inventory_digest = hashlib.sha256(
        json.dumps(files, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "fly_fgs_native_worker_bundle.v1",
        "bundle_id": "fly-fgs-native-worker-{}".format(inventory_digest[:20]),
        "inventory_sha256": inventory_digest,
        "origin": {
            "repository": "https://github.com/Mabdel-03/fly-sensor2behavior.git",
            "revision": "78bfb26c7a4137f183bf864c6215c41dab72eb85",
            "snapshot_note": (
                "The paper-metrology implementation originated in a dirty local "
                "worktree. This per-file inventory, not the origin revision alone, "
                "is the reproducible source authority."
            ),
        },
        "runtime": {
            "platform": "linux/amd64",
            "python": "3.12.11",
            "node": "22.22.1",
            "flygym": "2.1.0",
            "mujoco": "3.9.0",
            "python_base_image": (
                "python:3.12.11-slim-bookworm@sha256:"
                "519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7"
            ),
            "node_base_image": (
                "node:22.22.1-bookworm-slim@sha256:"
                "4f77a690f2f8946ab16fe1e791a3ac0667ae1c3575c3e4d0d4589e9ed5bfaf3d"
            ),
            "dependency_lock": "requirements.lock",
        },
        "reference_inputs": {
            "reference/BF00595226.pdf": (
                "9159ec24548e1ee0eaf4edca0d4790d7656890d2fb611dd0b7d27cc85a6f05d7"
            ),
            "reference/figure_ground_relative_motion_simulation_spec.md": (
                "c43b9ec417a52d5fac5333c1df01ca0765961f5c88779a3769fc9b3a2968647c"
            ),
        },
        "excluded": [
            "web/ (unrelated 429 MB development build and generated run assets)",
            "artifacts/ and every generated scientific array",
            "Python caches and local environments",
        ],
        "files": files,
    }
    temporary = destination.with_name(".VENDORED_SOURCE_MANIFEST.json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    print(manifest["bundle_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
