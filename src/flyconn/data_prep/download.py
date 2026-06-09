"""Resumable, md5-verified downloader for the FlyWire FAFB v783 data.

Pulls the 5 Zenodo connectivity files and the 3 GitHub annotation files into
``<data_root>/v783/raw/``. Features:
  * skip-if-already-valid (size + md5 match)
  * HTTP Range resume of partial ``.part`` files (matters for the 9.5 GB feather)
  * incremental md5 while streaming; final md5 gate against the config
  * exponential backoff on transient network errors
  * a structured ``_download_manifest.json`` recording url/md5/size/status per file

Designed to run inside a SLURM batch job (see slurm/download.sbatch), never on the
login node.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from ..config import Config, FileSpec, load_config
from ..io import md5_file, write_json
from ..paths import DataPaths

CHUNK = 8 * 1024 * 1024  # 8 MiB streaming blocks
MAX_RETRIES = 6
BACKOFF_BASE = 3.0  # seconds; *2^attempt


@dataclass
class DownloadResult:
    key: str
    url: str
    source: str
    role: str
    status: str            # DOWNLOADED | RESUMED | SKIPPED | FAILED
    size: int | None
    expected_md5: str | None
    actual_md5: str | None
    note: str = ""


def _looks_complete(spec: FileSpec, dest: Path) -> bool:
    """True if dest exists and matches the expected size (md5 checked separately)."""
    if not dest.exists():
        return False
    if spec.size is not None and dest.stat().st_size != spec.size:
        return False
    return True


def _verify_md5(spec: FileSpec, dest: Path) -> tuple[bool, str | None]:
    """Return (ok, actual_md5). ok is True when no expected md5 (GitHub) or it matches."""
    if spec.md5 is None:
        return True, None
    actual = md5_file(dest)
    return (actual == spec.md5), actual


def download_file(spec: FileSpec, dest_dir: Path, *, force: bool = False) -> DownloadResult:
    dest = dest_dir / spec.key
    part = dest.with_suffix(dest.suffix + ".part")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1) Skip if already present and valid.
    if not force and _looks_complete(spec, dest):
        ok, actual = _verify_md5(spec, dest)
        if ok:
            return DownloadResult(
                spec.key, spec.url, spec.source, spec.role, "SKIPPED",
                dest.stat().st_size, spec.md5, actual, "already present & valid",
            )
        # size matched but md5 didn't -> redownload from scratch.
        dest.unlink(missing_ok=True)

    resumed = False
    last_err = ""
    for attempt in range(MAX_RETRIES):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {}
            mode = "wb"
            if have > 0:
                headers["Range"] = f"bytes={have}-"
                mode = "ab"
                resumed = True

            with requests.get(spec.url, headers=headers, stream=True, timeout=60) as r:
                # If the server ignored the Range (200 not 206), restart cleanly.
                if have > 0 and r.status_code == 200:
                    have, mode, resumed = 0, "wb", False
                    part.unlink(missing_ok=True)
                r.raise_for_status()
                with open(part, mode) as fh:
                    for block in r.iter_content(chunk_size=CHUNK):
                        if block:
                            fh.write(block)

            # 2) Verify size (if known) then md5 over the full file.
            final_size = part.stat().st_size
            if spec.size is not None and final_size != spec.size:
                last_err = f"size {final_size} != expected {spec.size}"
                # truncated/over-read: drop and retry from scratch
                part.unlink(missing_ok=True)
                raise OSError(last_err)

            ok, actual = _verify_md5(spec, part)
            if not ok:
                last_err = f"md5 {actual} != expected {spec.md5}"
                part.unlink(missing_ok=True)
                raise OSError(last_err)

            os.replace(part, dest)  # atomic
            return DownloadResult(
                spec.key, spec.url, spec.source, spec.role,
                "RESUMED" if resumed else "DOWNLOADED",
                dest.stat().st_size, spec.md5, actual,
            )

        except (requests.RequestException, OSError) as e:
            last_err = str(e)
            if attempt < MAX_RETRIES - 1:
                wait = BACKOFF_BASE * (2 ** attempt)
                print(f"  [retry {attempt + 1}/{MAX_RETRIES}] {spec.key}: {e} "
                      f"-> sleeping {wait:.0f}s", flush=True)
                time.sleep(wait)

    return DownloadResult(
        spec.key, spec.url, spec.source, spec.role, "FAILED",
        None, spec.md5, None, last_err,
    )


def download_all(cfg: Config, *, only: list[str] | None = None,
                 force: bool = False) -> list[DownloadResult]:
    paths: DataPaths = cfg.paths().ensure()
    specs = cfg.all_files
    if only:
        wanted = set(only)
        specs = [s for s in specs if s.key in wanted]
        missing = wanted - {s.key for s in specs}
        if missing:
            raise KeyError(f"--only names not in config: {sorted(missing)}")

    results: list[DownloadResult] = []
    for spec in specs:
        print(f"[download] {spec.key}  ({spec.source}, role={spec.role})", flush=True)
        res = download_file(spec, paths.raw, force=force)
        print(f"    -> {res.status}"
              + (f"  ({res.note})" if res.note else ""), flush=True)
        results.append(res)

    # Write/merge the manifest.
    manifest = {r.key: asdict(r) for r in results}
    if paths.download_manifest.exists() and only:
        from ..io import read_json
        prev = read_json(paths.download_manifest)
        prev.update(manifest)
        manifest = prev
    write_json(paths.download_manifest, {
        "version": cfg.version,
        "files": manifest,
        "citation": cfg.citation,
    })

    failures = [r for r in results if r.status == "FAILED"]
    if failures:
        raise RuntimeError(
            "Download failed for: " + ", ".join(f"{r.key} ({r.note})" for r in failures)
        )
    return results


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Download FlyWire FAFB v783 data.")
    ap.add_argument("--config", default=None, help="Path to data config yaml.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Download only these file keys (smoke-test on the small .npy).")
    ap.add_argument("--force", action="store_true", help="Redownload even if valid.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    results = download_all(cfg, only=args.only, force=args.force)
    print("\n[download] summary:")
    for r in results:
        print(f"  {r.status:11s} {r.key}")


if __name__ == "__main__":
    main()
