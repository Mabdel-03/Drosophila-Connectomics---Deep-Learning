#!/usr/bin/env python3
"""Capture a content-addressed NOD1 browser/Python parity fixture.

This utility drives the deployed legacy UI through Chromium, so the browser
trace is produced by the real Web Worker rather than by Node pretending to be
a browser.  It then reruns the exact configuration saved by the UI through the
legacy Python backend.  The output establishes only cross-runtime numerical
parity; it is not evidence that the circuit or its parameters are biological.

The legacy service is queried only for its circuit and static preparation
endpoints.  The time-advancing Python comparison runs locally, never through a
public per-request simulation route.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence, Tuple


NOD1_ROOT_IDS: Tuple[str, ...] = (
    "720575940628438427",
    "720575940625528556",
    "720575940623997949",
    "720575940629456860",
)
EXPECTED_CELL_COUNT = 1208
EXPECTED_EDGE_COUNT = 5188
EXPECTED_MATERIALIZATION = 783
READOUT_TOLERANCE_V = 5.0e-5
SUMMARY_RELATIVE_TOLERANCE = 1.0e-2
SUMMARY_DENOMINATOR_FLOOR_V = READOUT_TOLERANCE_V


def canonical_json_bytes(value: Any) -> bytes:
    """Return the deterministic JSON representation used for logical hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def logical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _finite_float_sequence(values: Iterable[Any], *, field: str) -> Tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{field} must be a non-empty finite sequence")
    return result


def _summary(values_v: Sequence[float]) -> Dict[str, float]:
    values = _finite_float_sequence(values_v, field="voltage readout")
    return {
        "mean_voltage_v": float(sum(values) / len(values)),
        "minimum_voltage_v": float(min(values)),
        "maximum_voltage_v": float(max(values)),
        "final_voltage_v": float(values[-1]),
        "peak_to_peak_voltage_v": float(max(values) - min(values)),
    }


@dataclass(frozen=True)
class ParityMetrics:
    max_readout_error_v: float
    max_summary_relative_error: float
    sample_count: int


def compare_readouts(
    browser_v: Mapping[str, Sequence[float]],
    python_v: Mapping[str, Sequence[float]],
) -> Tuple[ParityMetrics, Dict[str, Dict[str, Any]]]:
    """Compare SI-unit readouts and return preregistered parity metrics."""

    if set(browser_v) != set(NOD1_ROOT_IDS) or set(python_v) != set(NOD1_ROOT_IDS):
        raise ValueError("parity comparison requires exactly the four declared NOD1 roots")
    max_sample_error = 0.0
    max_summary_error = 0.0
    total_samples = 0
    by_root: Dict[str, Dict[str, Any]] = {}
    for root_id in NOD1_ROOT_IDS:
        browser_values = _finite_float_sequence(
            browser_v[root_id], field=f"browser voltage {root_id}"
        )
        python_values = _finite_float_sequence(
            python_v[root_id], field=f"python voltage {root_id}"
        )
        if len(browser_values) != len(python_values):
            raise ValueError(f"readout length mismatch for NOD1 root {root_id}")
        sample_error = max(
            abs(browser_value - python_value)
            for browser_value, python_value in zip(browser_values, python_values)
        )
        browser_summary = _summary(browser_values)
        python_summary = _summary(python_values)
        summary_errors = {
            name: abs(browser_summary[name] - python_summary[name])
            / max(abs(python_summary[name]), SUMMARY_DENOMINATOR_FLOOR_V)
            for name in browser_summary
        }
        root_summary_error = max(summary_errors.values())
        max_sample_error = max(max_sample_error, sample_error)
        max_summary_error = max(max_summary_error, root_summary_error)
        total_samples += len(browser_values)
        by_root[root_id] = {
            "browser_summary": browser_summary,
            "python_summary": python_summary,
            "summary_relative_error": summary_errors,
            "max_readout_error_v": sample_error,
        }
    return (
        ParityMetrics(
            max_readout_error_v=max_sample_error,
            max_summary_relative_error=max_summary_error,
            sample_count=total_samples,
        ),
        by_root,
    )


def verify_legacy_source(
    source_root: Path, manifest_path: Path, *, allow_source_drift: bool = False
) -> Dict[str, Any]:
    """Verify every source and morphology digest declared by the frozen manifest."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = manifest.get("dataset", {})
    if dataset.get("materialization") != EXPECTED_MATERIALIZATION:
        raise ValueError("legacy manifest is not FlyWire FAFB materialization 783")
    observed: Dict[str, str] = {}
    mismatches: Dict[str, Dict[str, Any]] = {}
    for relative, expected in manifest.get("files", {}).items():
        path = source_root / relative
        actual = sha256_path(path) if path.is_file() else None
        observed[relative] = actual or "missing"
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    for root_id, expected in manifest.get("morphology_sha256", {}).items():
        candidates = (
            source_root / "data" / "swc" / f"{root_id}.swc",
            source_root / "data" / "drosophila" / "skeletons" / f"{root_id}.swc",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        key = f"morphology:{root_id}"
        actual = sha256_path(path) if path is not None else None
        observed[key] = actual or "missing"
        if actual != expected:
            mismatches[key] = {"expected": expected, "actual": actual}
    if mismatches and not allow_source_drift:
        names = ", ".join(sorted(mismatches))
        raise RuntimeError(f"legacy source does not match the frozen manifest: {names}")
    return {
        "manifest_sha256": sha256_path(manifest_path),
        "manifest_snapshot_id": manifest.get("snapshot_id"),
        "source_status": manifest.get("source", {}).get("source_commit_status"),
        "observed_sha256": observed,
        "mismatches": mismatches,
        "strict_match": not mismatches,
        "expected_deployed_capture": manifest.get("deployed_capture", {}),
        "expected_circuit_inventory": manifest.get("circuit_inventory", {}),
    }


@contextlib.contextmanager
def _legacy_import_context(source_root: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    old_cache = os.environ.get("FG_SWC_CACHE_DIR")
    try:
        os.chdir(source_root)
        sys.path.insert(0, str(source_root))
        os.environ["FG_SWC_CACHE_DIR"] = str(source_root / "data" / "swc")
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        if old_cache is None:
            os.environ.pop("FG_SWC_CACHE_DIR", None)
        else:
            os.environ["FG_SWC_CACHE_DIR"] = old_cache


def run_python_backend(source_root: Path, saved_config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Advance the saved browser configuration with the local legacy backend."""

    payload = {
        "cells": saved_config["runCells"],
        "edges": saved_config["runEdges"],
        "stimulus": saved_config["stimulus"],
        "manualStimulus": saved_config["runManual"],
        "morphologies": saved_config.get("morphologies", {}),
        "synapses": saved_config.get("synapses", {}),
        "modelParams": saved_config["modelParams"],
        "normalize": saved_config.get("normalize", True),
        "dt": saved_config["dt"],
    }
    with _legacy_import_context(source_root):
        for name in tuple(sys.modules):
            if name == "backend" or name.startswith("backend."):
                del sys.modules[name]
        simulation = importlib.import_module("backend.simulation")
        return simulation.simulate(payload)


def _idb_latest_run_script() -> str:
    return """
    async () => {
      const db = await new Promise((resolve, reject) => {
        const req = indexedDB.open('vch-runs-db-nod1fix', 1);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
      const records = await new Promise((resolve, reject) => {
        const tx = db.transaction('runs', 'readonly');
        const req = tx.objectStore('runs').getAll();
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
      db.close();
      if (!records.length) throw new Error('browser run was not persisted');
      records.sort((a, b) => Number(b.savedAt || 0) - Number(a.savedAt || 0));
      return records[0];
    }
    """


def capture_browser_run(
    app_url: str,
    *,
    duration_s: float,
    timeout_s: float,
) -> Dict[str, Any]:
    """Run the actual deployed Web Worker and return trace plus response receipts."""

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - capability-specific path
        raise RuntimeError(
            "Playwright is required: install its Python package and run "
            "`playwright install chromium`"
        ) from exc

    timeout_ms = int(timeout_s * 1000)
    worker_urls = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        browser_version = browser.version
        context = browser.new_context()
        page = context.new_page()
        page.set_default_timeout(timeout_ms)
        page.on("worker", lambda worker: worker_urls.append(worker.url))
        with page.expect_response(
            lambda response: response.request.method == "POST"
            and response.url.rstrip("/").endswith("/api/drosophila/nod1-circuit"),
            timeout=timeout_ms,
        ) as circuit_response_info:
            navigation = page.goto(app_url, wait_until="domcontentloaded")
        if navigation is None or not navigation.ok:
            raise RuntimeError(f"failed to load NOD1 app: {app_url}")
        index_html_body = navigation.body()
        page_asset_urls = page.evaluate(
            """() => Array.from(
              document.querySelectorAll('script[src],link[rel="stylesheet"][href]')
            ).map(element => ({
              kind: element.tagName === 'SCRIPT' ? 'script' : 'stylesheet',
              url: element.src || element.href,
            }))"""
        )
        page_asset_receipts = []
        for asset in sorted(page_asset_urls, key=lambda item: item["url"]):
            response = context.request.get(asset["url"], timeout=timeout_ms)
            if not response.ok:
                raise RuntimeError(f"could not retrieve deployed app asset: {asset['url']}")
            body = response.body()
            page_asset_receipts.append(
                {
                    "kind": asset["kind"],
                    "url": asset["url"],
                    "sha256": sha256_bytes(body),
                    "bytes": len(body),
                }
            )
        if not page_asset_receipts:
            raise RuntimeError("the deployed app has no captured main assets")
        circuit_response = circuit_response_info.value
        circuit_body = circuit_response.body()
        circuit = json.loads(circuit_body)
        if len(circuit.get("cells", [])) != EXPECTED_CELL_COUNT:
            raise RuntimeError("deployed app did not return the frozen 1,208-cell circuit")
        if len(circuit.get("edges", [])) != EXPECTED_EDGE_COUNT:
            raise RuntimeError("deployed app did not return the frozen 5,188-edge circuit")

        run_button = page.locator("button.run")
        run_button.wait_for(state="visible", timeout=timeout_ms)
        page.wait_for_function(
            "() => { const b = document.querySelector('button.run'); return b && !b.disabled; }",
            timeout=timeout_ms,
        )
        badge = page.locator(".mode-badge")
        if "nod1_sim_fix" not in badge.inner_text().lower():
            raise RuntimeError("capture target is not the nod1_sim_fix deployment")

        duration_input = page.locator("label:has-text('Duration') input[type='range']").first
        duration_input.evaluate(
            """(element, value) => {
              const setter = Object.getOwnPropertyDescriptor(
                HTMLInputElement.prototype, 'value'
              ).set;
              setter.call(element, String(value));
              element.dispatchEvent(new Event('input', {bubbles: true}));
              element.dispatchEvent(new Event('change', {bubbles: true}));
            }""",
            duration_s,
        )
        page.wait_for_function(
            """duration => {
              const input = document.querySelector("label input[type='range'][min='.5'][max='8']");
              return input && Math.abs(Number(input.value) - Number(duration)) < 1e-12;
            }""",
            arg=duration_s,
            timeout=timeout_ms,
        )

        with page.expect_response(
            lambda response: response.request.method == "POST"
            and response.url.rstrip("/").endswith("/api/prepare"),
            timeout=timeout_ms,
        ) as prepared_response_info:
            run_button.click()
        prepared_response = prepared_response_info.value
        prepared_body = prepared_response.body()
        prepared = json.loads(prepared_body)
        prepared_request_body = prepared_response.request.post_data_buffer or b""

        page.wait_for_function(
            """() => {
              const button = document.querySelector('button.run');
              const index = JSON.parse(localStorage.getItem('vch-run-index:nod1fix') || '[]');
              return button && !button.disabled && index.length === 1;
            }""",
            timeout=timeout_ms,
        )
        saved_run = page.evaluate(_idb_latest_run_script())
        if saved_run.get("experimentKind") is not None or saved_run.get("result") is None:
            raise RuntimeError("latest browser record is not a standard simulation run")

        worker_receipts = []
        for worker_url in sorted(set(worker_urls)):
            response = context.request.get(worker_url, timeout=timeout_ms)
            if not response.ok:
                raise RuntimeError(f"could not retrieve deployed worker asset: {worker_url}")
            body = response.body()
            worker_receipts.append(
                {"url": worker_url, "sha256": sha256_bytes(body), "bytes": len(body)}
            )
        if not worker_receipts:
            raise RuntimeError("the simulation completed without a captured Web Worker asset")
        context.close()
        browser.close()
    try:
        prepared_request = json.loads(prepared_request_body)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("the /api/prepare request was not canonical JSON") from exc
    return {
        "browser_version": browser_version,
        "index_html": {
            "url": app_url,
            "sha256": sha256_bytes(index_html_body),
            "bytes": len(index_html_body),
        },
        "main_assets": page_asset_receipts,
        "circuit": circuit,
        "circuit_response_sha256": sha256_bytes(circuit_body),
        "prepared": prepared,
        "prepared_request": prepared_request,
        "prepared_bundle_sha256": sha256_bytes(prepared_body),
        "prepared_request_sha256": sha256_bytes(prepared_request_body),
        "saved_run": saved_run,
        "worker_assets": worker_receipts,
    }


def _extract_si_readouts(result: Mapping[str, Any]) -> Dict[str, Tuple[float, ...]]:
    voltage = result.get("voltage")
    if not isinstance(voltage, Mapping):
        raise ValueError("runtime result has no voltage mapping")
    readouts: Dict[str, Tuple[float, ...]] = {}
    for root_id in NOD1_ROOT_IDS:
        source_id = "r" + root_id
        values_mv = _finite_float_sequence(
            voltage.get(source_id, ()), field=f"voltage {source_id}"
        )
        readouts[root_id] = tuple(value * 1.0e-3 for value in values_mv)
    return readouts


def build_fixture(
    browser_capture: Mapping[str, Any],
    python_result: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    *,
    app_url: str,
) -> Dict[str, Any]:
    saved_run = browser_capture["saved_run"]
    saved_config = saved_run["config"]
    browser_result = saved_run["result"]
    browser_readouts = _extract_si_readouts(browser_result)
    python_readouts = _extract_si_readouts(python_result)
    browser_time = _finite_float_sequence(browser_result["time"], field="browser time")
    python_time = _finite_float_sequence(python_result["time"], field="python time")
    if len(browser_time) != len(python_time) or any(
        abs(browser - python) > 1.0e-12
        for browser, python in zip(browser_time, python_time)
    ):
        raise ValueError("browser and Python time bases differ")
    for root_id in NOD1_ROOT_IDS:
        if len(browser_readouts[root_id]) != len(browser_time):
            raise ValueError(
                f"browser readout length for NOD1 {root_id} differs from time base"
            )
        if len(python_readouts[root_id]) != len(python_time):
            raise ValueError(
                f"Python readout length for NOD1 {root_id} differs from time base"
            )
    metrics, per_root = compare_readouts(browser_readouts, python_readouts)
    circuit = browser_capture["circuit"]
    prepared = browser_capture["prepared"]
    prepared_meta = prepared.get("meta", {})
    mapping = prepared_meta.get("synapseMapping", {})
    event_count = len(prepared.get("events", {}).get("pre", []))
    mapped_event_count = int(mapping.get("events", -1))
    synthetic_event_count = int(mapping.get("syntheticFallbackSynapses", -1))
    real_event_count = mapped_event_count - synthetic_event_count
    if (
        mapped_event_count != event_count
        or synthetic_event_count < 0
        or real_event_count < 0
    ):
        raise ValueError("prepared synapse-event accounting is inconsistent")

    expected_deployment = source_receipt.get("expected_deployed_capture", {})
    expected_circuit_sha = expected_deployment.get("circuit_response_sha256")
    if expected_circuit_sha != browser_capture["circuit_response_sha256"]:
        raise RuntimeError(
            "deployed circuit response does not match the frozen legacy snapshot"
        )
    expected_worker_hashes = set(expected_deployment.get("worker_asset_sha256", ()))
    observed_worker_hashes = {
        item["sha256"] for item in browser_capture["worker_assets"]
    }
    if not expected_worker_hashes or expected_worker_hashes != observed_worker_hashes:
        raise RuntimeError(
            "deployed Web Worker asset does not match the frozen legacy snapshot"
        )
    expected_index = expected_deployment.get("index_html", {})
    observed_index = browser_capture["index_html"]
    if (
        expected_index.get("sha256") != observed_index.get("sha256")
        or int(expected_index.get("bytes", -1)) != int(observed_index.get("bytes", -2))
    ):
        raise RuntimeError("deployed index HTML does not match the frozen legacy snapshot")
    expected_main_assets = {
        (item.get("kind"), item.get("sha256"), int(item.get("bytes", -1)))
        for item in expected_deployment.get("main_assets", ())
    }
    observed_main_assets = {
        (item.get("kind"), item.get("sha256"), int(item.get("bytes", -2)))
        for item in browser_capture["main_assets"]
    }
    if not expected_main_assets or expected_main_assets != observed_main_assets:
        raise RuntimeError("deployed main app assets do not match the frozen snapshot")
    expected_inventory = source_receipt.get("expected_circuit_inventory", {})
    observed_inventory = {
        "cells": len(circuit.get("cells", [])),
        "directed_aggregated_edges": len(circuit.get("edges", [])),
        "prepared_events": event_count,
        "prepared_real_coordinate_events": real_event_count,
        "prepared_synthetic_fallback_events": synthetic_event_count,
    }
    for field, observed in observed_inventory.items():
        if int(expected_inventory.get(field, -1)) != observed:
            raise RuntimeError(
                f"deployed circuit inventory field {field} differs from snapshot"
            )
    config_receipt = {
        "app_version": saved_config.get("appVersion"),
        "logic_version": saved_config.get("logicVersion"),
        "schema_version": saved_config.get("schemaVersion"),
        "dt_s": saved_config.get("dt"),
        "duration_s": saved_config.get("stimulus", {}).get("duration"),
        "stimulus": saved_config.get("stimulus"),
        "manual": saved_config.get("runManual"),
        "model_params": saved_config.get("modelParams"),
        "normalize": saved_config.get("normalize"),
        "silenced": saved_config.get("silenced"),
        "cell_count": len(saved_config.get("runCells", [])),
        "edge_count": len(saved_config.get("runEdges", [])),
        "saved_config_sha256": logical_sha256(saved_config),
    }
    numerical_thresholds_satisfied = (
        metrics.max_readout_error_v <= READOUT_TOLERANCE_V
        and metrics.max_summary_relative_error <= SUMMARY_RELATIVE_TOLERANCE
    )
    passed = numerical_thresholds_satisfied and bool(source_receipt.get("strict_match"))
    return {
        "schema_version": "1.0.0",
        "fixture_kind": "nod1_browser_python_numerical_parity",
        "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "claim_scope": {
            "establishes": "cross-runtime numerical agreement for four declared NOD1 cable readouts",
            "does_not_establish": [
                "biological validity",
                "physiological synaptic weights",
                "correct anatomical laterality of legacy labels",
                "motor-output validity",
            ],
            "promotion_label": "numerically_converged" if passed else "failed",
        },
        "dataset": {
            "family": "FlyWire FAFB",
            "materialization": EXPECTED_MATERIALIZATION,
            "root_ids": list(NOD1_ROOT_IDS),
            "root_id_encoding": "decimal strings",
            "neuron_universe": "proofread_139255",
            "autapses_in_source_table": True,
            "cleft_score_threshold": None,
            "structural_counts_are_physiological_weights": False,
            "anatomical_side_rule": "higher soma x is fly-left; lower soma x is fly-right",
        },
        "source_receipt": dict(source_receipt),
        "browser_runtime": {
            "app_url": app_url,
            "engine": "Chromium via Playwright",
            "browser_version": browser_capture["browser_version"],
            "playwright_python_version": importlib.metadata.version("playwright"),
            "worker_assets": browser_capture["worker_assets"],
            "index_html": browser_capture["index_html"],
            "main_assets": browser_capture["main_assets"],
            "circuit_response_sha256": browser_capture["circuit_response_sha256"],
            "prepared_request_sha256": browser_capture["prepared_request_sha256"],
            "prepared_bundle_sha256": browser_capture["prepared_bundle_sha256"],
            "prepared_solver": prepared_meta.get("solver"),
        },
        "python_runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "solver": python_result.get("meta", {}).get("solver"),
        },
        "circuit_inventory": {
            "cells": len(circuit.get("cells", [])),
            "directed_aggregated_edges": len(circuit.get("edges", [])),
            "prepared_events": event_count,
            "prepared_real_coordinate_events": real_event_count,
            "prepared_synthetic_fallback_events": synthetic_event_count,
            "fast_circuit_warning": (circuit.get("warnings") or [None])[0],
        },
        "configuration": config_receipt,
        "replay_input": {
            "circuit_response": circuit,
            "saved_config": saved_config,
            "prepared_request": browser_capture["prepared_request"],
            "prepared_bundle": prepared,
        },
        "time_s": list(browser_time),
        "readouts": {
            root_id: {
                "browser_voltage_v": list(browser_readouts[root_id]),
                "python_voltage_v": list(python_readouts[root_id]),
                **per_root[root_id],
            }
            for root_id in NOD1_ROOT_IDS
        },
        "metrics": {
            "max_readout_error_v": metrics.max_readout_error_v,
            "max_summary_relative_error": metrics.max_summary_relative_error,
            "sample_count": metrics.sample_count,
            "readout_tolerance_v": READOUT_TOLERANCE_V,
            "summary_relative_tolerance": SUMMARY_RELATIVE_TOLERANCE,
            "summary_relative_denominator_floor_v": SUMMARY_DENOMINATOR_FLOOR_V,
            "numerical_thresholds_satisfied": numerical_thresholds_satisfied,
            "passed": passed,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--app-url",
        default="http://54.160.228.98/nod1_sim_fix/",
        help="deployed nod1_sim_fix URL (including its reverse-proxy base path)",
    )
    parser.add_argument(
        "--legacy-source",
        type=Path,
        default=Path("/home/ec2-user/figure-ground-lab-nod1-sim-fix"),
        help="legacy source tree used for the Python comparison",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/legacy_nod1_v0.5.0.json"),
        help="frozen legacy-source manifest",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=0.5)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--allow-source-drift",
        action="store_true",
        help="capture a diagnostic fixture with source mismatches recorded",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.duration_s < 0.5 or args.duration_s > 8.0:
        raise SystemExit("--duration-s must match the UI range 0.5..8.0 s")
    source_root = args.legacy_source.resolve()
    manifest_path = args.manifest.resolve()
    source_receipt = verify_legacy_source(
        source_root,
        manifest_path,
        allow_source_drift=args.allow_source_drift,
    )
    capture = capture_browser_run(
        args.app_url,
        duration_s=args.duration_s,
        timeout_s=args.timeout_s,
    )
    python_result = run_python_backend(source_root, capture["saved_run"]["config"])
    fixture = build_fixture(
        capture,
        python_result,
        source_receipt,
        app_url=args.app_url,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(fixture) + b"\n")
    metrics = fixture["metrics"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "max_readout_error_v": metrics["max_readout_error_v"],
                "max_summary_relative_error": metrics["max_summary_relative_error"],
                "passed": metrics["passed"],
            },
            sort_keys=True,
        )
    )
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
