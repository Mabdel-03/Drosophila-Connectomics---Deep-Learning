#!/usr/bin/env python3
"""Verify every file in the content-addressed native worker bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    arguments = parser.parse_args()
    root = arguments.worker_root.resolve()
    manifest_path = root / "VENDORED_SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("schema_version") != "fly_fgs_native_worker_bundle.v1":
        raise RuntimeError("vendored worker manifest schema mismatch")
    expected = {item["path"]: item for item in manifest["files"]}
    observed_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != manifest_path
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and not any(part.endswith(".egg-info") for part in path.parts)
        and path.suffix != ".pyc"
    }
    if observed_paths != set(expected):
        missing = sorted(set(expected) - observed_paths)
        extra = sorted(observed_paths - set(expected))
        raise RuntimeError(
            "vendored file inventory mismatch; missing={!r}, extra={!r}".format(
                missing, extra
            )
        )
    for relative, record in sorted(expected.items()):
        path = root / relative
        if path.is_symlink():
            raise RuntimeError("vendored worker may not contain symlinks")
        if path.stat().st_size != int(record["bytes"]):
            raise RuntimeError("vendored file size mismatch: {}".format(relative))
        if _sha256(path) != record["sha256"]:
            raise RuntimeError("vendored file checksum mismatch: {}".format(relative))
    print(
        json.dumps(
            {
                "ok": True,
                "bundle_id": manifest["bundle_id"],
                "file_count": len(expected),
                "worker_root": str(root),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
