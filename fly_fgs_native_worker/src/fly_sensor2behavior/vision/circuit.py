"""Exploratory causal retinal-motion-to-NOD1 surrogate.

This is deliberately not the frozen 1,208-cell browser circuit.  It exists so
the complete software graph can be exercised from image samples without
granting downstream code access to analytic ground-truth velocity.  Outputs
retain the identities of the four audited NOD1 readouts but are labelled as a
low-confidence reduced model and cannot satisfy the browser-parity or neural
validation gates.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from ..nod1 import load_legacy_nod1_manifest
from ..schema import (
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
    RetinalFrame,
    SideContext,
    SideMappingMethod,
    SignalOrigin,
)
from .motion import ReichardtMotionDetector


class ReducedNOD1MotionCircuit:
    """Leaky NOD1 readout driven by causal local motion correlations."""

    def __init__(
        self,
        *,
        membrane_tau_s: float = 0.020,
        resting_voltage_v: float = -0.060,
        voltage_gain_v_per_motion_unit: float = 0.50,
        detector: Optional[ReichardtMotionDetector] = None,
    ) -> None:
        if not math.isfinite(membrane_tau_s) or membrane_tau_s <= 0.0:
            raise ValueError("membrane_tau_s must be finite and positive")
        if not math.isfinite(resting_voltage_v):
            raise ValueError("resting_voltage_v must be finite")
        if (
            not math.isfinite(voltage_gain_v_per_motion_unit)
            or voltage_gain_v_per_motion_unit < 0.0
        ):
            raise ValueError("voltage gain must be finite and non-negative")
        self.membrane_tau_s = float(membrane_tau_s)
        self.resting_voltage_v = float(resting_voltage_v)
        self.voltage_gain_v_per_motion_unit = float(
            voltage_gain_v_per_motion_unit
        )
        self.detector = detector if detector is not None else ReichardtMotionDetector()

    def run(self, frames: Sequence[RetinalFrame]) -> CircuitOutputTrace:
        frames = tuple(frames)
        if len(frames) < 2:
            raise ValueError("reduced circuit requires at least two retinal frames")
        self.detector.reset()
        origin_s = frames[0].measurement_time_s
        times = tuple(frame.measurement_time_s - origin_s for frame in frames)
        if abs(times[0]) > 1e-15 or any(
            current <= previous for previous, current in zip(times, times[1:])
        ):
            raise ValueError("retinal frames must have increasing measurement times")

        side_drives = {
            AnatomicalSide.LEFT: [],
            AnatomicalSide.RIGHT: [],
        }
        availability_times = []
        for frame in frames:
            motion = self.detector.update(
                frame, current_time_s=frame.availability_time_s
            )
            local = np.asarray(motion.local_direction_response, dtype=float)
            if frame.ommatidial_directions_body:
                lateral = np.asarray(
                    [direction[1] for direction in frame.ommatidial_directions_body]
                )
                left_mask = lateral >= 0.0
            else:
                left_mask = np.arange(len(local)) >= len(local) // 2
            right_mask = ~left_mask
            for side, mask in (
                (AnatomicalSide.LEFT, left_mask),
                (AnatomicalSide.RIGHT, right_mask),
            ):
                response = float(np.mean(local[mask])) if np.any(mask) else 0.0
                # T4a compatibility path is preferred-direction rectified.  A
                # causal T5/null pathway is not claimed by this surrogate.
                side_drives[side].append(max(0.0, response))
            availability_times.append(motion.availability_time_s - origin_s)

        states = {}
        for side, drives in side_drives.items():
            voltage = self.resting_voltage_v
            values = [voltage]
            for index in range(1, len(times)):
                dt_s = times[index] - times[index - 1]
                alpha = 1.0 - math.exp(-dt_s / self.membrane_tau_s)
                target = self.resting_voltage_v + (
                    self.voltage_gain_v_per_motion_unit * drives[index]
                )
                voltage += alpha * (target - voltage)
                values.append(voltage)
            states[side] = tuple(values)

        manifest = load_legacy_nod1_manifest()
        dataset = DatasetRef(
            namespace=DatasetNamespace.FLYWIRE_FAFB,
            release="FAFB",
            materialization=int(manifest["dataset"]["materialization"]),
            source_uri="http://54.160.228.98/drosophila/api/manifest",
            neuron_universe=str(manifest["dataset"]["neuron_universe"]),
            coordinate_units=str(manifest["dataset"]["coordinate_unit"]),
        )
        confidence = Confidence(
            tier=EvidenceTier.MODEL_INFERENCE,
            level=ConfidenceLevel.LOW,
            score=0.1,
            basis=(
                "causal image-derived reduced NOD1 software surrogate; not fitted to "
                "the frozen browser circuit or held-out neural recordings"
            ),
        )
        provenance = Provenance(
            source_uri="urn:fly-sensor2behavior:vision:reduced-nod1-motion:v1",
            method="causal Reichardt correlation plus leaky preferred-direction NOD1 readout",
            dataset_identity=dataset.identity_space,
            filters={
                "motion_estimation": "image_samples_only",
                "ground_truth_velocity_access": False,
                "photoreceptor_model": "finite_exposure_only",
                "lobula_plate_model": "reduced_surrogate",
            },
            notes=(
                "Times are rebased to the first retinal measurement. Compatibility side "
                "labels are retained; FAFB soma-x reconciliation remains unresolved."
            ),
        )
        signals = []
        side_seen = {AnatomicalSide.LEFT: 0, AnatomicalSide.RIGHT: 0}
        for cable in manifest["cable_neurons"]:
            if cable["type"] != "NOD1":
                continue
            side = AnatomicalSide(str(cable["legacy_side"]))
            replicate = side_seen[side]
            side_seen[side] += 1
            scale = 1.0 if replicate == 0 else 0.95
            side_context = SideContext(
                dataset=dataset,
                raw_dataset_side=side.value,
                anatomical_side=side,
                visual_field_side=side,
                app_rendering_side=side,
                eye_side=(EyeSide.LEFT if side is AnatomicalSide.LEFT else EyeSide.RIGHT),
                effector_side=AnatomicalSide.UNKNOWN,
                mapping_method=SideMappingMethod.SIMULATION_CONVENTION,
                confidence=confidence,
                notes="Reduced-model compatibility side; not soma-x-derived anatomical truth.",
            )
            values = tuple(
                self.resting_voltage_v + scale * (value - self.resting_voltage_v)
                for value in states[side]
            )
            signals.append(
                CircuitSignal(
                    neuron=EntityRef(
                        dataset=dataset,
                        entity_id=str(cable["root_id"]),
                        kind=EntityKind.NEURON,
                        cell_type="NOD1",
                        anatomical_side=side,
                        side_context=side_context,
                    ),
                    signal_kind=CircuitSignalKind.VOLTAGE,
                    unit="V",
                    values=values,
                    availability_times_s=tuple(availability_times),
                    provenance=provenance,
                    confidence=confidence,
                    origin=SignalOrigin.SIMULATED,
                )
            )
        return CircuitOutputTrace(
            dataset=dataset,
            sample_times_s=times,
            signals=tuple(signals),
            provenance=provenance,
            confidence=confidence,
            exact_timebase=True,
        )


__all__ = ["ReducedNOD1MotionCircuit"]
