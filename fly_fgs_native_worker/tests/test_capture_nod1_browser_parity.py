from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "capture_nod1_browser_parity.py"
SPEC = importlib.util.spec_from_file_location("capture_nod1_browser_parity", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CAPTURE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CAPTURE
SPEC.loader.exec_module(CAPTURE)

NOD1_ROOT_IDS = CAPTURE.NOD1_ROOT_IDS
canonical_json_bytes = CAPTURE.canonical_json_bytes
compare_readouts = CAPTURE.compare_readouts
logical_sha256 = CAPTURE.logical_sha256
verify_legacy_source = CAPTURE.verify_legacy_source


def test_canonical_json_and_logical_hash_ignore_mapping_order() -> None:
    first = {"root_id": NOD1_ROOT_IDS[0], "nested": {"b": 2, "a": 1}}
    second = {"nested": {"a": 1, "b": 2}, "root_id": NOD1_ROOT_IDS[0]}
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert logical_sha256(first) == logical_sha256(second)


def test_compare_readouts_uses_si_units_and_declared_roots() -> None:
    browser = {
        root_id: (-0.060, -0.059 + index * 1.0e-5)
        for index, root_id in enumerate(NOD1_ROOT_IDS)
    }
    python = {
        root_id: (-0.060, -0.059 + index * 1.0e-5 + 1.0e-7)
        for index, root_id in enumerate(NOD1_ROOT_IDS)
    }
    metrics, by_root = compare_readouts(browser, python)
    assert metrics.sample_count == 8
    assert metrics.max_readout_error_v == pytest.approx(1.0e-7)
    assert set(by_root) == set(NOD1_ROOT_IDS)


def test_compare_readouts_rejects_missing_declared_root() -> None:
    values = {root_id: (-0.060, -0.059) for root_id in NOD1_ROOT_IDS}
    incomplete = dict(values)
    incomplete.pop(NOD1_ROOT_IDS[-1])
    with pytest.raises(ValueError, match="exactly the four"):
        compare_readouts(incomplete, values)


def test_verify_legacy_source_rejects_source_drift(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    source.mkdir()
    tracked = source / "loop.ts"
    tracked.write_text("expected", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "snapshot_id": "test",
                "source": {"source_commit_status": "unavailable"},
                "dataset": {"materialization": 783},
                "files": {"loop.ts": "sha256:" + "0" * 64},
                "morphology_sha256": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="loop.ts"):
        verify_legacy_source(source, manifest)
    receipt = verify_legacy_source(source, manifest, allow_source_drift=True)
    assert receipt["strict_match"] is False
    assert "loop.ts" in receipt["mismatches"]
