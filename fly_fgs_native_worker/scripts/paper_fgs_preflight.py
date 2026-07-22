#!/usr/bin/env python3
"""Fail-closed host, source, storage, and worker preflight for a paper release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


EXPECTED_PDF_SHA256 = "9159ec24548e1ee0eaf4edca0d4790d7656890d2fb611dd0b7d27cc85a6f05d7"
EXPECTED_SPEC_SHA256 = "c43b9ec417a52d5fac5333c1df01ca0765961f5c88779a3769fc9b3a2968647c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _memory_bytes() -> int:
    values = Path("/proc/meminfo").read_text("utf-8").splitlines()
    total = next(line for line in values if line.startswith("MemTotal:"))
    return int(total.split()[1]) * 1024


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--minimum-cpus", type=int, default=16)
    parser.add_argument("--minimum-memory-gib", type=float, default=128.0)
    parser.add_argument("--minimum-free-gib", type=float, default=250.0)
    arguments = parser.parse_args()
    worker = Path(__file__).resolve().parents[1]
    repository = worker.parent
    source = arguments.source_root.resolve()
    artifact = arguments.artifact_root.resolve()
    failures = []
    if platform.system() != "Linux":
        failures.append("host must be Linux")
    if platform.machine() not in ("x86_64", "amd64"):
        failures.append("host must be x86-64")
    cpus = os.cpu_count() or 0
    if cpus < arguments.minimum_cpus:
        failures.append("fewer than {} CPUs".format(arguments.minimum_cpus))
    memory_bytes = _memory_bytes()
    if memory_bytes < int(arguments.minimum_memory_gib * 1024**3):
        failures.append("less than {:.1f} GiB RAM".format(arguments.minimum_memory_gib))
    if not source.is_dir():
        failures.append("fly_fgs_source is missing")
    if not artifact.is_absolute() or _within(artifact, repository) or _within(artifact, source):
        failures.append("artifact root must be absolute and outside the repository")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(artifact.parent).free
    if free_bytes < int(arguments.minimum_free_gib * 1024**3):
        failures.append("less than {:.1f} GiB artifact storage".format(arguments.minimum_free_gib))
    for relative, expected in (
        ("reference/BF00595226.pdf", EXPECTED_PDF_SHA256),
        (
            "reference/figure_ground_relative_motion_simulation_spec.md",
            EXPECTED_SPEC_SHA256,
        ),
    ):
        path = worker / relative
        if not path.is_file() or _sha256(path) != expected:
            failures.append("reference checksum mismatch: {}".format(relative))
    verify = subprocess.run(
        [sys.executable, str(worker / "scripts/verify_vendored_bundle.py")],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if verify.returncode:
        failures.append("vendored worker verification failed: {}".format(verify.stderr.strip()))
    docker = shutil.which("docker")
    docker_version = None
    if docker is None:
        failures.append("Docker is unavailable")
    else:
        probe = subprocess.run(
            [docker, "version", "--format", "{{.Server.Version}}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if probe.returncode:
            failures.append("Docker daemon is unavailable")
        else:
            docker_version = probe.stdout.strip()
    result = {
        "ok": not failures,
        "failures": failures,
        "host": {"system": platform.system(), "machine": platform.machine()},
        "cpu_count": cpus,
        "memory_bytes": memory_bytes,
        "artifact_free_bytes": free_bytes,
        "artifact_root": str(artifact),
        "source_root": str(source),
        "docker_server_version": docker_version,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
