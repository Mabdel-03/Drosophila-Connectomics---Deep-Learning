"""Strict import of the registered real-Chromium NOD1 parity fixture.

This boundary imports only the four browser-produced SI voltage readouts.  The
Python comparison arrays prove numerical parity but are never eligible circuit
inputs, and deprecated legacy motor proxies are rejected recursively.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import sysconfig
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

import numpy as np

from .schema import (
    AnatomicalSide,
    CircuitOutputTrace,
    CircuitSignal,
    CircuitSignalKind,
    Confidence,
    ConfidenceLevel,
    DatasetNamespace,
    DatasetRef,
    EntityKind,
    EntityRef,
    EvidenceTier,
    EyeSide,
    Provenance,
    SideContext,
    SideMappingMethod,
    SignalOrigin,
)
from .validation import (
    BenchmarkRegistry,
    default_benchmark_registry_path,
    load_benchmark_registry,
)


NOD1_BROWSER_PARITY_CASE_ID = "nod1.browser_python_parity"
NOD1_BROWSER_PARITY_FIXTURE_KIND = "nod1_browser_python_numerical_parity"
NOD1_BROWSER_PARITY_SCHEMA_VERSION = "1.0.0"
NOD1_BROWSER_PARITY_APP_SIDES = {
    "720575940628438427": AnatomicalSide.LEFT,
    "720575940625528556": AnatomicalSide.LEFT,
    "720575940623997949": AnatomicalSide.RIGHT,
    "720575940629456860": AnatomicalSide.RIGHT,
}
# Backwards-compatible public alias.  These are frozen browser/app labels, not
# anatomical FlyWire sides; all importer logic uses the explicit name above.
NOD1_BROWSER_PARITY_ROOT_SIDES = NOD1_BROWSER_PARITY_APP_SIDES
FORBIDDEN_MOTOR_FIELDS = ("steering", "forward_looking_phase_T")
_EXPECTED_TOP_LEVEL_KEYS = {
    "browser_runtime",
    "captured_at_utc",
    "circuit_inventory",
    "claim_scope",
    "configuration",
    "dataset",
    "fixture_kind",
    "metrics",
    "python_runtime",
    "readouts",
    "replay_input",
    "schema_version",
    "source_receipt",
    "time_s",
}
_EXPECTED_READOUT_KEYS = {
    "browser_summary",
    "browser_voltage_v",
    "max_readout_error_v",
    "python_summary",
    "python_voltage_v",
    "summary_relative_error",
}
_SUMMARY_KEYS = {
    "final_voltage_v",
    "maximum_voltage_v",
    "mean_voltage_v",
    "minimum_voltage_v",
    "peak_to_peak_voltage_v",
}
_METRIC_KEYS = {
    "max_readout_error_v",
    "max_summary_relative_error",
    "numerical_thresholds_satisfied",
    "passed",
    "readout_tolerance_v",
    "sample_count",
    "summary_relative_denominator_floor_v",
    "summary_relative_tolerance",
}


@dataclass(frozen=True)
class RegisteredNOD1BrowserFixture:
    """Validated registered fixture plus its browser-only circuit trace."""

    circuit_trace: CircuitOutputTrace
    fixture_sha256: str
    fixture_uri: str
    benchmark_case_version: str
    captured_at_utc: str
    duration_s: float
    dt_s: float
    dataset_metadata: Mapping[str, Any]
    configuration: Mapping[str, Any]
    stimulus: Mapping[str, Any]
    circuit_inventory: Mapping[str, Any]
    browser_runtime: Mapping[str, Any]
    claim_scope: Mapping[str, Any]

    def pipeline_source_metadata(self) -> Mapping[str, Any]:
        return {
            "input_mode": "registered_frozen_browser_nod1_parity",
            "visual_source": "frozen_browser_visual_circuit_output",
            "retinal_frames_present": False,
            "retinal_frame_count": 0,
            "fixture_uri": self.fixture_uri,
            "fixture_sha256": self.fixture_sha256,
            "benchmark_case_id": NOD1_BROWSER_PARITY_CASE_ID,
            "benchmark_case_version": self.benchmark_case_version,
            "captured_at_utc": self.captured_at_utc,
            "source_duration_s": self.duration_s,
            "source_sample_dt_s": self.dt_s,
            "source_sample_count": len(self.circuit_trace.sample_times_s),
            "source_sample_interval": "half-open [0, duration_s)",
            "source_sample_interval_end_s": self.duration_s,
            "source_availability_semantics": (
                "inferred_offline_playback_at_sample_time; the capture contains "
                "simulation timestamps but no streaming-availability or physiological-"
                "latency measurement"
            ),
            "latency_tail_semantics": (
                "the bridge consumes every source sample through the half-open "
                "interval end; downstream values retain their causal availability "
                "times even when latency places availability at or after that end"
            ),
            "circuit_model_scope": "frozen_real_chromium_1208_cell_circuit_four_nod1_readouts",
            "circuit_cell_count": int(self.circuit_inventory["cells"]),
            "exported_circuit_channels": len(self.circuit_trace.signals),
            "dataset": deepcopy(dict(self.dataset_metadata)),
            "configuration": deepcopy(dict(self.configuration)),
            "stimulus": deepcopy(dict(self.stimulus)),
            "circuit_inventory": deepcopy(dict(self.circuit_inventory)),
            "browser_runtime": deepcopy(dict(self.browser_runtime)),
            "claim_scope": deepcopy(dict(self.claim_scope)),
            "scientific_confidence": "low_exploratory",
            "retinal_boundary_notice": (
                "The fixture preserves frozen Chromium visual-circuit voltage output and "
                "stimulus configuration, but contains no source retinal frames and does "
                "not establish a calibrated visual model."
            ),
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), "%s must be an object" % label)
    return value


def _require_exact_fields(
    value: Mapping[str, Any], expected: Sequence[str], label: str
) -> None:
    _require(
        set(value) == set(expected),
        "%s fields do not match the registered schema" % label,
    )


def _sequence(value: Any, label: str) -> Sequence[Any]:
    _require(isinstance(value, list), "%s must be an array" % label)
    return value


def _finite(value: Any, label: str) -> float:
    _require(not isinstance(value, bool), "%s must be numeric" % label)
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric" % label) from exc
    _require(math.isfinite(converted), "%s must be finite" % label)
    return converted


def _integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        "%s must be an integer" % label,
    )
    return int(value)


def _logical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _finite_series(value: Any, count: int, label: str) -> Tuple[float, ...]:
    raw = _sequence(value, label)
    _require(len(raw) == count, "%s must contain %d samples" % (label, count))
    return tuple(_finite(item, "%s[%d]" % (label, index)) for index, item in enumerate(raw))


def _strict_json_object(raw: bytes) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError("fixture contains forbidden non-finite JSON constant %s" % value)

    def reject_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("fixture contains duplicate JSON key %r" % key)
            result[key] = value
        return result

    try:
        decoded = raw.decode("utf-8", errors="strict")
        payload = json.loads(
            decoded,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("fixture is not strict UTF-8 JSON: %s" % exc) from exc
    _require(isinstance(payload, Mapping), "fixture top level must be an object")
    return payload


def _contains_forbidden_motor_key(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN_MOTOR_FIELDS:
                return key
            nested = _contains_forbidden_motor_key(item)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _contains_forbidden_motor_key(item)
            if nested is not None:
                return nested
    return None


def _summary(values: Sequence[float]) -> Mapping[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "final_voltage_v": float(array[-1]),
        "maximum_voltage_v": float(np.max(array)),
        "mean_voltage_v": float(np.mean(array)),
        "minimum_voltage_v": float(np.min(array)),
        "peak_to_peak_voltage_v": float(np.ptp(array)),
    }


def _validate_summary(
    supplied_value: Any,
    values: Sequence[float],
    label: str,
) -> None:
    supplied = _mapping(supplied_value, label)
    _require(set(supplied) == _SUMMARY_KEYS, "%s has unexpected fields" % label)
    expected = _summary(values)
    for key, expected_value in expected.items():
        observed = _finite(supplied.get(key), "%s.%s" % (label, key))
        _require(
            math.isclose(observed, expected_value, rel_tol=1.0e-13, abs_tol=1.0e-15),
            "%s.%s does not match its voltage series" % (label, key),
        )


def _validate_fixture_payload(
    payload: Mapping[str, Any],
    *,
    fixture_sha256: str,
    fixture_uri: str,
    benchmark_case_version: str,
) -> RegisteredNOD1BrowserFixture:
    _require(set(payload) == _EXPECTED_TOP_LEVEL_KEYS, "fixture top-level fields do not match schema 1.0.0")
    _require(payload.get("fixture_kind") == NOD1_BROWSER_PARITY_FIXTURE_KIND, "unsupported fixture kind")
    _require(payload.get("schema_version") == NOD1_BROWSER_PARITY_SCHEMA_VERSION, "unsupported fixture schema")
    forbidden = _contains_forbidden_motor_key(payload)
    _require(forbidden is None, "deprecated motor field %r is forbidden in the parity importer" % forbidden)

    dataset_metadata = _mapping(payload.get("dataset"), "dataset")
    _require_exact_fields(
        dataset_metadata,
        (
            "anatomical_side_rule",
            "autapses_in_source_table",
            "cleft_score_threshold",
            "family",
            "materialization",
            "neuron_universe",
            "root_id_encoding",
            "root_ids",
            "structural_counts_are_physiological_weights",
        ),
        "dataset",
    )
    expected_root_ids = tuple(NOD1_BROWSER_PARITY_APP_SIDES)
    _require(dataset_metadata.get("family") == "FlyWire FAFB", "fixture dataset must be FlyWire FAFB")
    _require(dataset_metadata.get("materialization") == 783, "fixture must use FAFB materialization 783")
    _require(dataset_metadata.get("neuron_universe") == "proofread_139255", "unexpected neuron universe")
    _require(dataset_metadata.get("root_id_encoding") == "decimal strings", "root IDs must use decimal strings")
    dataset_root_ids = tuple(_sequence(dataset_metadata.get("root_ids"), "dataset.root_ids"))
    _require(dataset_root_ids == expected_root_ids, "fixture root ID inventory/order is not registered")
    _require(all(isinstance(root_id, str) and root_id.isdigit() for root_id in dataset_root_ids), "root IDs must be decimal strings")
    _require(
        dataset_metadata.get("anatomical_side_rule")
        == "higher soma x is fly-left; lower soma x is fly-right",
        "fixture anatomical side rule is invalid",
    )
    _require(dataset_metadata.get("structural_counts_are_physiological_weights") is False, "structural counts cannot be physiological weights")

    configuration = _mapping(payload.get("configuration"), "configuration")
    _require(
        set(configuration)
        == {
            "app_version",
            "cell_count",
            "dt_s",
            "duration_s",
            "edge_count",
            "logic_version",
            "manual",
            "model_params",
            "normalize",
            "saved_config_sha256",
            "schema_version",
            "silenced",
            "stimulus",
        },
        "configuration fields do not match the registered schema",
    )
    _require(configuration.get("app_version") == "0.5.0", "unsupported browser app version")
    _require(configuration.get("schema_version") == 2, "unsupported saved browser configuration")
    _require(configuration.get("cell_count") == 1208, "configuration must contain 1,208 cells")
    _require(configuration.get("edge_count") == 5188, "configuration must contain 5,188 edges")
    _require(configuration.get("normalize") is True, "registered fixture must preserve normalization")
    _require(configuration.get("silenced") is False, "registered fixture must not be globally silenced")
    dt_s = _finite(configuration.get("dt_s"), "configuration.dt_s")
    duration_s = _finite(configuration.get("duration_s"), "configuration.duration_s")
    _require(dt_s == 0.005 and duration_s == 0.5, "registered fixture timing must be 0.005 s over 0.5 s")
    stimulus = _mapping(configuration.get("stimulus"), "configuration.stimulus")
    _require(stimulus.get("inputModel") == "opticflow", "fixture stimulus must use opticflow input")
    _require(stimulus.get("preset") == "FG", "fixture stimulus must be the registered figure-ground preset")
    _require(_finite(stimulus.get("duration"), "stimulus.duration") == duration_s, "stimulus duration mismatch")

    inventory = _mapping(payload.get("circuit_inventory"), "circuit_inventory")
    _require_exact_fields(
        inventory,
        (
            "cells",
            "directed_aggregated_edges",
            "fast_circuit_warning",
            "prepared_events",
            "prepared_real_coordinate_events",
            "prepared_synthetic_fallback_events",
        ),
        "circuit_inventory",
    )
    expected_inventory = {
        "cells": 1208,
        "directed_aggregated_edges": 5188,
        "prepared_events": 53715,
        "prepared_real_coordinate_events": 8920,
        "prepared_synthetic_fallback_events": 44795,
    }
    for key, expected in expected_inventory.items():
        _require(inventory.get(key) == expected, "circuit inventory %s mismatch" % key)
    _require(
        inventory["prepared_real_coordinate_events"]
        + inventory["prepared_synthetic_fallback_events"]
        == inventory["prepared_events"],
        "prepared event inventory does not close",
    )

    replay_input = _mapping(payload.get("replay_input"), "replay_input")
    _require(
        set(replay_input)
        == {"circuit_response", "prepared_bundle", "prepared_request", "saved_config"},
        "replay_input fields do not match the registered schema",
    )
    saved_config = _mapping(replay_input.get("saved_config"), "replay_input.saved_config")
    prepared_request = _mapping(replay_input.get("prepared_request"), "replay_input.prepared_request")
    prepared_bundle = _mapping(replay_input.get("prepared_bundle"), "replay_input.prepared_bundle")
    circuit_response = _mapping(replay_input.get("circuit_response"), "replay_input.circuit_response")
    _require_exact_fields(
        saved_config,
        (
            "appVersion",
            "cells",
            "configHash",
            "dt",
            "edges",
            "excludedT4Ids",
            "logicVersion",
            "manual",
            "modelParams",
            "morphologies",
            "normalize",
            "part",
            "probes",
            "runCells",
            "runEdges",
            "runManual",
            "schemaVersion",
            "scopeCellIds",
            "silenced",
            "stimulus",
            "synapses",
            "t4ConnectionThreshold",
            "vchMode",
        ),
        "replay_input.saved_config",
    )
    _require_exact_fields(
        prepared_request,
        (
            "cells",
            "chunkSteps",
            "dt",
            "edges",
            "manualStimulus",
            "modelParams",
            "morphologies",
            "normalize",
            "stimulus",
            "synapses",
        ),
        "replay_input.prepared_request",
    )
    _require_exact_fields(
        prepared_bundle,
        ("cableIndices", "cables", "dt", "events", "ids", "meta", "steps"),
        "replay_input.prepared_bundle",
    )
    _require_exact_fields(
        circuit_response,
        (
            "cells",
            "colors",
            "counts",
            "countsBySide",
            "dchRootIds",
            "edges",
            "mainNeuropil",
            "motionCellType",
            "nod1Candidates",
            "nod1RootId",
            "nod1RootIds",
            "positionRule",
            "probes",
            "side",
            "vchRootId",
            "vchRootIds",
            "warnings",
        ),
        "replay_input.circuit_response",
    )
    _require(
        _logical_sha256(saved_config) == configuration.get("saved_config_sha256"),
        "saved browser configuration digest mismatch",
    )
    _require(saved_config.get("stimulus") == stimulus, "saved stimulus differs from registered configuration")
    _require(prepared_request.get("stimulus") == stimulus, "prepared stimulus differs from registered configuration")
    _require(saved_config.get("runManual") == configuration.get("manual"), "saved manual stimulus mismatch")
    _require(saved_config.get("modelParams") == configuration.get("model_params"), "saved model parameters mismatch")
    _require(saved_config.get("normalize") == configuration.get("normalize"), "saved normalization mismatch")
    _require(saved_config.get("silenced") == configuration.get("silenced"), "saved silencing state mismatch")
    for label, value in (
        ("saved_config.dt", saved_config.get("dt")),
        ("prepared_request.dt", prepared_request.get("dt")),
        ("prepared_bundle.dt", prepared_bundle.get("dt")),
    ):
        _require(_finite(value, label) == dt_s, "%s timing mismatch" % label)
    _require(prepared_bundle.get("steps") == 100, "prepared bundle must contain 100 steps")
    for label, values, expected_count in (
        ("saved_config.cells", saved_config.get("cells"), 1208),
        ("saved_config.runCells", saved_config.get("runCells"), 1208),
        ("prepared_request.cells", prepared_request.get("cells"), 1208),
        ("prepared_bundle.ids", prepared_bundle.get("ids"), 1208),
        ("saved_config.edges", saved_config.get("edges"), 5188),
        ("saved_config.runEdges", saved_config.get("runEdges"), 5188),
        ("prepared_request.edges", prepared_request.get("edges"), 5188),
    ):
        _require(len(_sequence(values, label)) == expected_count, "%s inventory mismatch" % label)
    _require(
        saved_config.get("runCells") == prepared_request.get("cells")
        and saved_config.get("runCells") == circuit_response.get("cells"),
        "replayed cell data are not identical across capture stages",
    )
    _require(
        saved_config.get("runEdges") == prepared_request.get("edges")
        and saved_config.get("runEdges") == circuit_response.get("edges"),
        "replayed edge data are not identical across capture stages",
    )
    event_vectors = _mapping(prepared_bundle.get("events"), "prepared_bundle.events")
    _require(
        set(event_vectors) == {"gunit", "post", "postc", "pre", "prec", "sign"},
        "prepared event representation must contain the six registered vectors",
    )
    _require(
        all(
            len(_sequence(vector, "prepared_bundle.events vector"))
            == inventory["prepared_events"]
            for vector in event_vectors.values()
        ),
        "prepared event vector length mismatch",
    )
    pre_events = _sequence(event_vectors["pre"], "prepared_bundle.events.pre")
    post_events = _sequence(event_vectors["post"], "prepared_bundle.events.post")
    pre_compartments = _sequence(event_vectors["prec"], "prepared_bundle.events.prec")
    post_compartments = _sequence(event_vectors["postc"], "prepared_bundle.events.postc")
    conductances = _sequence(event_vectors["gunit"], "prepared_bundle.events.gunit")
    signs = _sequence(event_vectors["sign"], "prepared_bundle.events.sign")
    for index, (pre, post, prec, postc, conductance, sign) in enumerate(
        zip(
            pre_events,
            post_events,
            pre_compartments,
            post_compartments,
            conductances,
            signs,
        )
    ):
        _require(
            0 <= _integer(pre, "prepared event pre[%d]" % index) < 1208,
            "prepared event presynaptic index is out of range",
        )
        _require(
            0 <= _integer(post, "prepared event post[%d]" % index) < 1208,
            "prepared event postsynaptic index is out of range",
        )
        _require(
            _integer(prec, "prepared event prec[%d]" % index) >= -1,
            "prepared event presynaptic compartment is invalid",
        )
        _require(
            _integer(postc, "prepared event postc[%d]" % index) >= -1,
            "prepared event postsynaptic compartment is invalid",
        )
        _require(
            _finite(conductance, "prepared event gunit[%d]" % index) > 0.0,
            "prepared event conductance must be positive",
        )
        _require(
            _finite(sign, "prepared event sign[%d]" % index) in (-1.0, 1.0),
            "prepared event sign must be -1 or 1",
        )
    prepared_meta = _mapping(prepared_bundle.get("meta"), "prepared_bundle.meta")
    synapse_mapping = _mapping(
        prepared_meta.get("synapseMapping"),
        "prepared_bundle.meta.synapseMapping",
    )
    _require(
        synapse_mapping.get("events") == inventory["prepared_events"]
        and synapse_mapping.get("syntheticFallbackSynapses")
        == inventory["prepared_synthetic_fallback_events"],
        "prepared bundle synapse accounting mismatch",
    )

    source_receipt = _mapping(payload.get("source_receipt"), "source_receipt")
    _require_exact_fields(
        source_receipt,
        (
            "expected_circuit_inventory",
            "expected_deployed_capture",
            "manifest_sha256",
            "manifest_snapshot_id",
            "mismatches",
            "observed_sha256",
            "source_status",
            "strict_match",
        ),
        "source_receipt",
    )
    _require(source_receipt.get("strict_match") is True, "source receipt is not a strict match")
    _require(source_receipt.get("mismatches") == {}, "source receipt records mismatches")
    receipt_inventory = _mapping(source_receipt.get("expected_circuit_inventory"), "source_receipt.expected_circuit_inventory")
    for key, expected in expected_inventory.items():
        _require(receipt_inventory.get(key) == expected, "source receipt inventory %s mismatch" % key)

    claim_scope = _mapping(payload.get("claim_scope"), "claim_scope")
    _require_exact_fields(
        claim_scope,
        ("does_not_establish", "establishes", "promotion_label"),
        "claim_scope",
    )
    does_not_establish = _sequence(claim_scope.get("does_not_establish"), "claim_scope.does_not_establish")
    _require("motor-output validity" in does_not_establish, "fixture must disclaim motor-output validity")
    _require("biological validity" in does_not_establish, "fixture must disclaim biological validity")
    browser_runtime = _mapping(payload.get("browser_runtime"), "browser_runtime")
    _require_exact_fields(
        browser_runtime,
        (
            "app_url",
            "browser_version",
            "circuit_response_sha256",
            "engine",
            "index_html",
            "main_assets",
            "playwright_python_version",
            "prepared_bundle_sha256",
            "prepared_request_sha256",
            "prepared_solver",
            "worker_assets",
        ),
        "browser_runtime",
    )
    _require("Chromium" in str(browser_runtime.get("engine")), "fixture must be a real Chromium capture")
    python_runtime = _mapping(payload.get("python_runtime"), "python_runtime")
    _require_exact_fields(
        python_runtime,
        ("platform", "python", "solver"),
        "python_runtime",
    )
    captured_at_utc = payload.get("captured_at_utc")
    _require(isinstance(captured_at_utc, str) and captured_at_utc.endswith("Z"), "captured_at_utc is invalid")

    raw_times = _finite_series(payload.get("time_s"), 100, "time_s")
    for index, value in enumerate(raw_times):
        _require(
            math.isclose(value, index * dt_s, rel_tol=0.0, abs_tol=1.0e-15),
            "time_s is not the exact registered 5 ms grid",
        )
    _require(raw_times[-1] + dt_s == duration_s, "time grid must cover half-open [0, 0.5 s)")

    readouts = _mapping(payload.get("readouts"), "readouts")
    _require(set(readouts) == set(expected_root_ids), "fixture must contain exactly four registered readouts")
    metrics = _mapping(payload.get("metrics"), "metrics")
    _require(set(metrics) == _METRIC_KEYS, "fixture metric fields do not match schema")
    readout_tolerance_v = _finite(metrics.get("readout_tolerance_v"), "metrics.readout_tolerance_v")
    summary_tolerance = _finite(
        metrics.get("summary_relative_tolerance"),
        "metrics.summary_relative_tolerance",
    )
    summary_denominator_floor_v = _finite(
        metrics.get("summary_relative_denominator_floor_v"),
        "metrics.summary_relative_denominator_floor_v",
    )
    _require(
        readout_tolerance_v == 5.0e-5
        and summary_tolerance == 1.0e-2
        and summary_denominator_floor_v == 5.0e-5,
        "fixture numerical tolerances are not the registered values",
    )
    _require(
        metrics.get("sample_count") == 400,
        "fixture metrics must account for all 400 readout samples",
    )
    _require(
        metrics.get("passed") is True
        and metrics.get("numerical_thresholds_satisfied") is True,
        "registered parity fixture did not pass its numerical gates",
    )
    saved_cells_by_root: Dict[str, Mapping[str, Any]] = {}
    for cell_value in _sequence(saved_config.get("cells"), "saved_config.cells"):
        cell = _mapping(cell_value, "saved_config cell")
        root_id = cell.get("rootId")
        if root_id in NOD1_BROWSER_PARITY_APP_SIDES:
            _require(root_id not in saved_cells_by_root, "duplicate registered NOD1 cell")
            saved_cells_by_root[str(root_id)] = cell
    _require(set(saved_cells_by_root) == set(expected_root_ids), "registered NOD1 cell metadata is incomplete")

    dataset = DatasetRef(
        namespace=DatasetNamespace.FLYWIRE_FAFB,
        release="FAFB",
        materialization=783,
        source_uri="http://54.160.228.98/drosophila/api/manifest",
        neuron_universe=str(dataset_metadata["neuron_universe"]),
        coordinate_units="nm",
    )
    confidence = Confidence(
        tier=EvidenceTier.MODEL_INFERENCE,
        level=ConfidenceLevel.LOW,
        score=0.1,
        basis=(
            "registered real-Chromium numerical fixture; passive circuit and "
            "synthetic-contact physiology remain exploratory and uncalibrated"
        ),
    )
    provenance = Provenance(
        source_uri=fixture_uri,
        method=(
            "compressed fixture SHA-256 verification before gzip; strict schema import "
            "of browser_voltage_v SI values only"
        ),
        accessed_at_utc=str(captured_at_utc),
        dataset_identity=dataset.identity_space,
        artifact_hash="sha256:%s" % fixture_sha256,
        source_run_id="%s@%s" % (NOD1_BROWSER_PARITY_CASE_ID, benchmark_case_version),
        filters={
            "fixture_kind": NOD1_BROWSER_PARITY_FIXTURE_KIND,
            "fixture_schema_version": NOD1_BROWSER_PARITY_SCHEMA_VERSION,
            "included_voltage_field": "browser_voltage_v",
            "excluded_comparison_fields": [
                "python_voltage_v",
                "python_summary",
                "steering",
                "forward_looking_phase_T",
            ],
            "readout_root_ids": list(expected_root_ids),
            "sample_interval": "half-open [0, 0.5 s)",
            "availability_schedule": "inferred_offline_playback_at_sample_time",
            "captured_streaming_availability": False,
            "stimulus": deepcopy(dict(stimulus)),
        },
        notes=(
            "Frozen-browser visual-circuit output only. No retinal frames, calibrated "
            "visual model, motor validity, physiological weights, or exact anatomical "
            "laterality are claimed. Fixture L/R labels are retained only as a low-confidence "
            "simulation convention required by the exploratory bridge. Per-sample "
            "availability equals simulation time only as an inferred offline playback "
            "schedule; the fixture did not capture streaming availability or physiological "
            "latency."
        ),
    )

    signals = []
    maximum_readout_error_v = 0.0
    maximum_summary_relative_error = 0.0
    for root_id in expected_root_ids:
        app_side = NOD1_BROWSER_PARITY_APP_SIDES[root_id]
        cell = saved_cells_by_root[root_id]
        expected_legacy_side = "L" if app_side is AnatomicalSide.LEFT else "R"
        _require(cell.get("rootId") == root_id, "saved cell root ID mismatch")
        _require(cell.get("type") == "Nod1" and cell.get("role") == "nod1", "saved cell is not NOD1")
        _require(cell.get("side") == expected_legacy_side, "registered NOD1 compatibility side mismatch")

        readout = _mapping(readouts[root_id], "readouts[%s]" % root_id)
        _require(set(readout) == _EXPECTED_READOUT_KEYS, "readout %s fields do not match schema" % root_id)
        browser_values = _finite_series(
            readout.get("browser_voltage_v"), 100, "readouts[%s].browser_voltage_v" % root_id
        )
        python_values = _finite_series(
            readout.get("python_voltage_v"), 100, "readouts[%s].python_voltage_v" % root_id
        )
        _require(all(-1.0 < value < 1.0 for value in browser_values), "browser voltage is not in SI volts")
        _validate_summary(readout.get("browser_summary"), browser_values, "readouts[%s].browser_summary" % root_id)
        _validate_summary(readout.get("python_summary"), python_values, "readouts[%s].python_summary" % root_id)
        observed_error = _finite(readout.get("max_readout_error_v"), "readout max_readout_error_v")
        expected_error = max(abs(left - right) for left, right in zip(browser_values, python_values))
        _require(
            math.isclose(observed_error, expected_error, rel_tol=1.0e-12, abs_tol=1.0e-15),
            "readout %s parity error is inconsistent" % root_id,
        )
        maximum_readout_error_v = max(maximum_readout_error_v, expected_error)
        supplied_summary_errors = _mapping(
            readout.get("summary_relative_error"),
            "readouts[%s].summary_relative_error" % root_id,
        )
        _require(
            set(supplied_summary_errors) == _SUMMARY_KEYS,
            "readout %s summary-relative-error fields do not match schema" % root_id,
        )
        browser_summary = _summary(browser_values)
        python_summary = _summary(python_values)
        for name in _SUMMARY_KEYS:
            expected_relative_error = abs(
                browser_summary[name] - python_summary[name]
            ) / max(abs(python_summary[name]), summary_denominator_floor_v)
            supplied_relative_error = _finite(
                supplied_summary_errors.get(name),
                "readouts[%s].summary_relative_error.%s" % (root_id, name),
            )
            _require(
                math.isclose(
                    supplied_relative_error,
                    expected_relative_error,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                ),
                "readout %s summary-relative error is inconsistent" % root_id,
            )
            maximum_summary_relative_error = max(
                maximum_summary_relative_error,
                expected_relative_error,
            )

        side_context = SideContext(
            dataset=dataset,
            raw_dataset_side=expected_legacy_side,
            anatomical_side=AnatomicalSide.UNKNOWN,
            visual_field_side=AnatomicalSide.UNKNOWN,
            app_rendering_side=app_side,
            eye_side=EyeSide.UNKNOWN,
            effector_side=AnatomicalSide.UNKNOWN,
            mapping_method=SideMappingMethod.SIMULATION_CONVENTION,
            confidence=confidence,
            notes=(
                "Frozen compatibility label only; no soma-x coordinate is present in the "
                "fixture, so this must not be promoted as anatomical laterality."
            ),
        )
        signals.append(
            CircuitSignal(
                neuron=EntityRef(
                    dataset=dataset,
                    entity_id=root_id,
                    kind=EntityKind.NEURON,
                    cell_type="NOD1",
                    anatomical_side=AnatomicalSide.UNKNOWN,
                    side_context=side_context,
                ),
                signal_kind=CircuitSignalKind.VOLTAGE,
                unit="V",
                values=browser_values,
                availability_times_s=raw_times,
                provenance=provenance,
                confidence=confidence,
                origin=SignalOrigin.SIMULATED,
            )
        )

    _require(
        math.isclose(
            _finite(metrics.get("max_readout_error_v"), "metrics.max_readout_error_v"),
            maximum_readout_error_v,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "fixture maximum readout error does not recompute",
    )
    _require(
        math.isclose(
            _finite(
                metrics.get("max_summary_relative_error"),
                "metrics.max_summary_relative_error",
            ),
            maximum_summary_relative_error,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "fixture maximum summary-relative error does not recompute",
    )
    _require(
        maximum_readout_error_v <= readout_tolerance_v
        and maximum_summary_relative_error <= summary_tolerance,
        "fixture readouts exceed the registered numerical tolerances",
    )

    trace = CircuitOutputTrace(
        dataset=dataset,
        sample_times_s=raw_times,
        signals=tuple(signals),
        provenance=provenance,
        confidence=confidence,
        exact_timebase=True,
        sample_interval_end_s=duration_s,
    )
    return RegisteredNOD1BrowserFixture(
        circuit_trace=trace,
        fixture_sha256=fixture_sha256,
        fixture_uri=fixture_uri,
        benchmark_case_version=benchmark_case_version,
        captured_at_utc=str(captured_at_utc),
        duration_s=duration_s,
        dt_s=dt_s,
        dataset_metadata=deepcopy(dict(dataset_metadata)),
        configuration=deepcopy(dict(configuration)),
        stimulus=deepcopy(dict(stimulus)),
        circuit_inventory=deepcopy(dict(inventory)),
        browser_runtime=deepcopy(dict(browser_runtime)),
        claim_scope=deepcopy(dict(claim_scope)),
    )


def parse_registered_nod1_browser_fixture_bytes(
    compressed_bytes: bytes,
    *,
    expected_sha256: str,
    fixture_uri: str,
    benchmark_case_version: str,
) -> RegisteredNOD1BrowserFixture:
    """Hash compressed bytes first, then strictly decompress, parse, and import."""

    _require(isinstance(compressed_bytes, bytes), "compressed fixture must be bytes")
    _require(
        isinstance(expected_sha256, str)
        and len(expected_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_sha256),
        "expected fixture SHA-256 is invalid",
    )
    observed_sha256 = hashlib.sha256(compressed_bytes).hexdigest()
    _require(observed_sha256 == expected_sha256, "compressed fixture SHA-256 mismatch")
    try:
        decompressed = gzip.decompress(compressed_bytes)
    except (OSError, EOFError) as exc:
        raise ValueError("registered fixture is not valid gzip data") from exc
    payload = _strict_json_object(decompressed)
    return _validate_fixture_payload(
        payload,
        fixture_sha256=observed_sha256,
        fixture_uri=fixture_uri,
        benchmark_case_version=benchmark_case_version,
    )


def _resolve_fixture_path(
    registry_path: Path,
    fixture_uri: str,
    fixture_path: Optional[Path],
) -> Path:
    if fixture_path is not None:
        return Path(fixture_path)
    parsed = urlsplit(fixture_uri)
    _require(not parsed.scheme and not parsed.netloc and not parsed.query and not parsed.fragment, "registered fixture URI must be local")
    pure = PurePosixPath(parsed.path)
    _require(not pure.is_absolute() and ".." not in pure.parts, "registered fixture URI is unsafe")
    source_root = registry_path.resolve().parents[2]
    candidates = (
        source_root / Path(*pure.parts),
        Path(sysconfig.get_path("data"))
        / "share"
        / "fly-sensor2behavior"
        / "reference"
        / pure.name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("registered NOD1 browser parity fixture is unavailable")


def load_registered_nod1_browser_fixture(
    *,
    registry_path: Optional[Path] = None,
    fixture_path: Optional[Path] = None,
) -> RegisteredNOD1BrowserFixture:
    """Locate the fixture only through its registered case URI and digest."""

    resolved_registry_path = (
        default_benchmark_registry_path()
        if registry_path is None
        else Path(registry_path)
    )
    registry: BenchmarkRegistry = load_benchmark_registry(resolved_registry_path)
    case = registry.case(NOD1_BROWSER_PARITY_CASE_ID)
    _require(case.fixture_uri is not None, "registered parity case has no fixture URI")
    _require(case.input_sha256 is not None, "registered parity case has no fixture SHA-256")
    resolved_fixture_path = _resolve_fixture_path(
        resolved_registry_path,
        case.fixture_uri,
        fixture_path,
    )
    compressed_bytes = resolved_fixture_path.read_bytes()
    return parse_registered_nod1_browser_fixture_bytes(
        compressed_bytes,
        expected_sha256=case.input_sha256,
        fixture_uri=case.fixture_uri,
        benchmark_case_version=case.version,
    )


def load_registered_nod1_browser_circuit_trace(
    *,
    registry_path: Optional[Path] = None,
    fixture_path: Optional[Path] = None,
) -> CircuitOutputTrace:
    return load_registered_nod1_browser_fixture(
        registry_path=registry_path,
        fixture_path=fixture_path,
    ).circuit_trace


__all__ = [
    "FORBIDDEN_MOTOR_FIELDS",
    "NOD1_BROWSER_PARITY_CASE_ID",
    "NOD1_BROWSER_PARITY_FIXTURE_KIND",
    "NOD1_BROWSER_PARITY_APP_SIDES",
    "NOD1_BROWSER_PARITY_ROOT_SIDES",
    "RegisteredNOD1BrowserFixture",
    "load_registered_nod1_browser_circuit_trace",
    "load_registered_nod1_browser_fixture",
    "parse_registered_nod1_browser_fixture_bytes",
]
