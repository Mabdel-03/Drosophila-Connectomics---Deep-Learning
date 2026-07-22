"""Strict import boundary for the frozen browser NOD1 simulation.

The deployed worker reports voltages in millivolts and a field named
``steering`` that is only the mean activity of all cells.  This module converts
only declared cable-neuron voltages into the public SI trace.  The legacy
population mean is deliberately never exposed as a motor signal.
"""

from __future__ import annotations

import json
import math
import sysconfig
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

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


LEGACY_NOD1_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "manifests"
    / "legacy_nod1_v0.5.0.json"
)


def load_legacy_nod1_manifest(path: Optional[Path] = None) -> Mapping[str, Any]:
    if path is None:
        manifest_path = LEGACY_NOD1_MANIFEST
        if not manifest_path.is_file():
            manifest_path = (
                Path(sysconfig.get_path("data"))
                / "share"
                / "fly-sensor2behavior"
                / "manifests"
                / "legacy_nod1_v0.5.0.json"
            )
    else:
        manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("snapshot_id") != "legacy-nod1-v0.5.0":
        raise ValueError("unsupported legacy NOD1 snapshot")
    if payload.get("status") != "frozen_reference":
        raise ValueError("legacy NOD1 manifest must be frozen_reference")
    return payload


def _finite_series(values: Any, count: int, label: str) -> Tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError("%s must contain %d samples" % (label, count))
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("%s must contain only finite values" % label)
    return converted


def legacy_result_to_circuit_trace(
    result: Mapping[str, Any],
    *,
    manifest_path: Optional[Path] = None,
    source_run_id: Optional[str] = None,
    source_artifact_hash: Optional[str] = None,
) -> CircuitOutputTrace:
    """Convert one frozen worker result to an SI, identity-qualified trace."""

    if not isinstance(result, Mapping):
        raise TypeError("legacy NOD1 result must be a mapping")
    manifest = load_legacy_nod1_manifest(manifest_path)
    raw_times = result.get("time")
    if not isinstance(raw_times, (list, tuple)) or len(raw_times) < 2:
        raise ValueError("legacy NOD1 result requires at least two time samples")
    times = tuple(float(value) for value in raw_times)
    if any(not math.isfinite(value) or value < 0.0 for value in times):
        raise ValueError("legacy NOD1 times must be finite and non-negative")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("legacy NOD1 times must be strictly increasing")
    voltage = result.get("voltage")
    if not isinstance(voltage, Mapping):
        raise ValueError("legacy NOD1 result requires a voltage mapping")
    if source_artifact_hash is not None:
        if not source_artifact_hash.startswith("sha256:") or len(source_artifact_hash) != 71:
            raise ValueError("source_artifact_hash must be a sha256 digest")

    dataset = DatasetRef(
        namespace=DatasetNamespace.FLYWIRE_FAFB,
        release="FAFB",
        materialization=int(manifest["dataset"]["materialization"]),
        source_uri="http://54.160.228.98/drosophila/api/manifest",
        neuron_universe=str(manifest["dataset"]["neuron_universe"]),
        # DatasetRef preserves the external source coordinate unit. Numeric
        # simulation arrays cross into SI at their owning process boundary.
        coordinate_units=str(manifest["dataset"]["coordinate_unit"]),
    )
    confidence = Confidence(
        tier=EvidenceTier.MODEL_INFERENCE,
        level=ConfidenceLevel.LOW,
        score=0.2,
        basis=(
            "legacy passive/graded circuit with mostly synthetic contact fallbacks; "
            "side labels remain a compatibility convention"
        ),
    )
    provenance = Provenance(
        source_uri=str(manifest["source"]["public_route"]),
        method="strict legacy-nod1-v0.5.0 voltage import and mV-to-V conversion",
        accessed_at_utc=str(manifest["captured_at_utc"]),
        dataset_identity=dataset.identity_space,
        artifact_hash=source_artifact_hash,
        source_run_id=source_run_id,
        filters={
            "included_cell_classes": ["vCH", "DCH", "NOD1"],
            "excluded_motor_fields": list(
                manifest["compatibility_contract"]["motor_forbidden_fields"]
            ),
        },
        notes=(
            "The legacy field named steering is population_mean_activity and was ignored. "
            "No sub-wingbeat timing was interpolated from this trace."
        ),
    )

    signals = []
    required_nod1 = {
        item["root_id"]
        for item in manifest["cable_neurons"]
        if item["type"] == "NOD1"
    }
    for cable in manifest["cable_neurons"]:
        root_id = str(cable["root_id"])
        if root_id not in voltage:
            if root_id in required_nod1:
                raise ValueError("legacy NOD1 voltage is missing required root %s" % root_id)
            continue
        source_mv = _finite_series(
            voltage[root_id], len(times), "voltage[%s]" % root_id
        )
        side = AnatomicalSide(str(cable["legacy_side"]))
        side_context = SideContext(
            dataset=dataset,
            raw_dataset_side=str(cable["legacy_side"]),
            anatomical_side=side,
            visual_field_side=AnatomicalSide.UNKNOWN,
            app_rendering_side=side,
            eye_side=EyeSide.UNKNOWN,
            effector_side=AnatomicalSide.UNKNOWN,
            mapping_method=SideMappingMethod.SIMULATION_CONVENTION,
            confidence=confidence,
            notes=(
                "Compatibility side from the frozen endpoint; anatomical soma-x and "
                "downstream effector reconciliation are still required."
            ),
        )
        signals.append(
            CircuitSignal(
                neuron=EntityRef(
                    dataset=dataset,
                    entity_id=root_id,
                    kind=EntityKind.NEURON,
                    cell_type=str(cable["type"]),
                    anatomical_side=side,
                    side_context=side_context,
                ),
                signal_kind=CircuitSignalKind.VOLTAGE,
                unit="V",
                values=tuple(value * 1.0e-3 for value in source_mv),
                availability_times_s=times,
                provenance=provenance,
                confidence=confidence,
                origin=SignalOrigin.SIMULATED,
            )
        )

    if len([signal for signal in signals if signal.neuron.cell_type == "NOD1"]) != 4:
        raise ValueError("legacy trace must contain all four individual NOD1 voltages")
    return CircuitOutputTrace(
        dataset=dataset,
        sample_times_s=times,
        signals=tuple(signals),
        provenance=provenance,
        confidence=confidence,
        exact_timebase=True,
    )


__all__ = [
    "LEGACY_NOD1_MANIFEST",
    "legacy_result_to_circuit_trace",
    "load_legacy_nod1_manifest",
]
