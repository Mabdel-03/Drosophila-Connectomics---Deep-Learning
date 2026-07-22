"""Built-in, dependency-light evaluators for the scientific gate registry.

These evaluators intentionally cover only claims that can be established from
the checked-in source tree.  Worker-only, cross-runtime, and held-out
biological claims return :class:`~fly_sensor2behavior.validation.EvaluatorResult`
with ``blocked_reason`` instead of manufacturing a passing scalar.
"""

from __future__ import annotations

import dataclasses
import hashlib
import gzip
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from .evidence import EdgeRelation, default_seed_graph_path, load_evidence_graph
from .flight.bridge import AvailabilityQueue, AvailableValue
from .nod1_solver import dense_tree_matrix, hines_solve
from .schema import EntityKind
from .validation import (
    BenchmarkCase,
    BenchmarkEvaluator,
    BenchmarkRegistry,
    EvaluatorResult,
    MetricValue,
)


class FixtureIntegrityError(RuntimeError):
    """Raised when a registered immutable fixture no longer has its digest."""


_COMPARISON_SHAPE_MISMATCH = 1.0e300

# Updated together with the immutable non-dead fly-FGS capture.  This is the
# digest of the public four-channel CircuitOutputTrace serialization, not a
# claim of Chromium parity or an independently repeated runtime execution.
_FLY_FGS_REGISTERED_TRACE_SHA256 = (
    "0db62b35fdcd5fd5ad615e8e52e8e390459c1d0d75ec90dfdfb3c8de5e5c1f75"
)

# The scheduled canonical native gate is an immutable cross-runtime receipt,
# not merely a compatibility check against whichever ``node`` happens to be
# first on PATH.  Keep the exact release here beside the gate implementation;
# the worker image and CI install the same version.
_CANONICAL_ONLINE_PINNED_NODE_VERSION = "v22.22.1"


def _scaled_array_max_abs_delta(
    first: Any,
    second: Any,
    *,
    scale: float = 1.0,
) -> float:
    """Return a finite, fail-closed maximum delta for two numeric arrays."""

    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if (
        left.shape != right.shape
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        return _COMPARISON_SHAPE_MISMATCH
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("comparison scale must be finite and positive")
    if left.size == 0:
        return 0.0
    delta = np.abs(left - right) / scale
    if not np.all(np.isfinite(delta)):
        return _COMPARISON_SHAPE_MISMATCH
    return float(np.max(delta))


def _optional_scaled_array_max_abs_delta(
    first: Any,
    second: Any,
    *,
    scale: float = 1.0,
) -> float:
    if first is None and second is None:
        return 0.0
    if first is None or second is None:
        return _COMPARISON_SHAPE_MISMATCH
    return _scaled_array_max_abs_delta(first, second, scale=scale)


def _mapping_scaled_array_max_abs_delta(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    scale: float = 1.0,
) -> float:
    if set(first) != set(second):
        return _COMPARISON_SHAPE_MISMATCH
    return max(
        (
            _scaled_array_max_abs_delta(first[key], second[key], scale=scale)
            for key in sorted(first)
        ),
        default=0.0,
    )


def _paired_initial_state_standardized_delta(
    flights: Sequence[Any],
    expected_root_state: Any,
    expected_wing_angle_rad: Any,
    expected_wing_velocity_rad_s: Any,
    expected_wing_order: Sequence[str],
    expected_com_position_world_m: Any,
) -> float:
    """Compare every paired rollout's complete initial physical state."""

    deltas = []
    for flight in flights:
        initial_specs = (
            (
                flight.position_world_m,
                expected_root_state.position_world_m,
                1.0e-3,
            ),
            (
                flight.velocity_world_m_s,
                expected_root_state.velocity_world_m_s,
                1.0,
            ),
            (
                flight.quaternion_body_to_world,
                expected_root_state.quaternion_body_to_world,
                1.0,
            ),
            (
                flight.angular_velocity_body_rad_s,
                expected_root_state.angular_velocity_body_rad_s,
                100.0,
            ),
            (flight.measured_wing_joint_angle_rad, expected_wing_angle_rad, 1.0),
            (
                flight.measured_wing_joint_velocity_rad_s,
                expected_wing_velocity_rad_s,
                1000.0,
            ),
            (
                flight.whole_fly_com_position_world_m,
                expected_com_position_world_m,
                1.0e-3,
            ),
        )
        for series, expected, scale in initial_specs:
            if series is None:
                deltas.append(_COMPARISON_SHAPE_MISMATCH)
                continue
            values = np.asarray(series, dtype=float)
            expected_values = np.asarray(expected, dtype=float)
            if values.ndim != expected_values.ndim + 1 or values.shape[0] == 0:
                deltas.append(_COMPARISON_SHAPE_MISMATCH)
                continue
            deltas.append(
                _scaled_array_max_abs_delta(values[0], expected_values, scale=scale)
            )
        if tuple(flight.measured_wing_joint_order) != tuple(expected_wing_order):
            deltas.append(_COMPARISON_SHAPE_MISMATCH)
    return max(deltas, default=_COMPARISON_SHAPE_MISMATCH)


def _paired_timebase_max_abs_delta_s(
    reference_run: Any,
    compared_runs: Sequence[Any],
) -> float:
    """Compare retinal, circuit, and physics clocks across paired conditions."""

    def retinal_timebase(run: Any) -> np.ndarray:
        return np.asarray(
            [
                (
                    frame.exposure_start_s,
                    frame.exposure_end_s,
                    frame.measurement_time_s,
                    frame.availability_time_s,
                )
                for frame in run.retinal_frames
            ],
            dtype=float,
        )

    reference_clocks = (
        reference_run.flight.time_s,
        reference_run.circuit.sample_times_s,
        retinal_timebase(reference_run),
    )
    deltas = []
    for candidate in compared_runs:
        candidate_clocks = (
            candidate.flight.time_s,
            candidate.circuit.sample_times_s,
            retinal_timebase(candidate),
        )
        deltas.extend(
            _scaled_array_max_abs_delta(reference, observed)
            for reference, observed in zip(reference_clocks, candidate_clocks)
        )
    return max(deltas, default=_COMPARISON_SHAPE_MISMATCH)


def _visual_repeat_max_standardized_delta(first: Any, second: Any) -> float:
    """Compare complete moving-condition physical and muscle output traces."""

    array_specs = (
        (first.position_world_m, second.position_world_m, 1.0e-3),
        (first.velocity_world_m_s, second.velocity_world_m_s, 1.0),
        (first.quaternion_body_to_world, second.quaternion_body_to_world, 1.0),
        (
            first.angular_velocity_body_rad_s,
            second.angular_velocity_body_rad_s,
            100.0,
        ),
        (first.wing_stroke_rad, second.wing_stroke_rad, 1.0),
        (first.wing_angle_of_attack_rad, second.wing_angle_of_attack_rad, 1.0),
        (first.aerodynamic_force_body_n, second.aerodynamic_force_body_n, 1.0e-6),
        (
            first.aerodynamic_torque_body_n_m,
            second.aerodynamic_torque_body_n_m,
            1.0e-9,
        ),
    )
    deltas = [
        _scaled_array_max_abs_delta(left, right, scale=scale)
        for left, right, scale in array_specs
    ]
    deltas.extend(
        (
            _optional_scaled_array_max_abs_delta(
                first.measured_wing_joint_angle_rad,
                second.measured_wing_joint_angle_rad,
                scale=1.0,
            ),
            _optional_scaled_array_max_abs_delta(
                first.measured_wing_joint_velocity_rad_s,
                second.measured_wing_joint_velocity_rad_s,
                scale=1000.0,
            ),
            _optional_scaled_array_max_abs_delta(
                first.whole_fly_com_position_world_m,
                second.whole_fly_com_position_world_m,
                scale=1.0e-3,
            ),
            _optional_scaled_array_max_abs_delta(
                getattr(first, "ground_contact_count", None),
                getattr(second, "ground_contact_count", None),
            ),
            _optional_scaled_array_max_abs_delta(
                getattr(first, "external_actuator_torque_n_m", None),
                getattr(second, "external_actuator_torque_n_m", None),
                scale=1.0e-9,
            ),
            _optional_scaled_array_max_abs_delta(
                getattr(first, "physics_time_s", None),
                getattr(second, "physics_time_s", None),
                scale=1.0,
            ),
            _optional_scaled_array_max_abs_delta(
                getattr(first, "ground_contact_transition_point_count", None),
                getattr(second, "ground_contact_transition_point_count", None),
            ),
            _optional_scaled_array_max_abs_delta(
                getattr(first, "external_actuator_torque_physics_n_m", None),
                getattr(second, "external_actuator_torque_physics_n_m", None),
                scale=1.0e-9,
            ),
            _optional_scaled_array_max_abs_delta(
                getattr(first, "measured_wing_joint_angle_physics_rad", None),
                getattr(second, "measured_wing_joint_angle_physics_rad", None),
                scale=1.0,
            ),
            _optional_scaled_array_max_abs_delta(
                getattr(
                    first, "measured_wing_joint_velocity_physics_rad_s", None
                ),
                getattr(
                    second, "measured_wing_joint_velocity_physics_rad_s", None
                ),
                scale=1000.0,
            ),
            _optional_scaled_array_max_abs_delta(
                getattr(first, "aerodynamic_force_body_physics_n", None),
                getattr(second, "aerodynamic_force_body_physics_n", None),
                scale=1.0e-6,
            ),
            _optional_scaled_array_max_abs_delta(
                getattr(first, "aerodynamic_torque_body_physics_n_m", None),
                getattr(second, "aerodynamic_torque_body_physics_n_m", None),
                scale=1.0e-9,
            ),
            _mapping_scaled_array_max_abs_delta(
                first.motor_event_times_s, second.motor_event_times_s
            ),
            _mapping_scaled_array_max_abs_delta(
                first.motor_event_phases_rad, second.motor_event_phases_rad
            ),
            _mapping_scaled_array_max_abs_delta(
                first.muscle_activation, second.muscle_activation
            ),
            _mapping_scaled_array_max_abs_delta(
                first.muscle_force_n, second.muscle_force_n, scale=1.0e-4
            ),
            _mapping_scaled_array_max_abs_delta(
                first.muscle_phase_effect, second.muscle_phase_effect
            ),
            _mapping_scaled_array_max_abs_delta(
                first.muscle_work_j, second.muscle_work_j, scale=1.0e-7
            ),
        )
    )
    if tuple(first.measured_wing_joint_order) != tuple(
        second.measured_wing_joint_order
    ):
        deltas.append(_COMPARISON_SHAPE_MISMATCH)
    first_diagnostics = getattr(first, "diagnostics", None)
    second_diagnostics = getattr(second, "diagnostics", None)
    if first_diagnostics is None and second_diagnostics is None:
        diagnostics_equal = True
    elif first_diagnostics is None or second_diagnostics is None:
        diagnostics_equal = False
    else:
        try:
            raw_equality = first_diagnostics == second_diagnostics
            diagnostics_equal = isinstance(raw_equality, (bool, np.bool_)) and bool(
                raw_equality
            )
        except (TypeError, ValueError):
            diagnostics_equal = False
    if not diagnostics_equal:
        # Diagnostics contain the full-step contact summary, integrated
        # impulse, maxima, warnings, backend identity, and physics provenance.
        # A deterministic repeat must reproduce those records exactly, even
        # when a transient is absent from the decimated 1 ms output arrays.
        deltas.append(_COMPARISON_SHAPE_MISMATCH)
    return max(deltas, default=_COMPARISON_SHAPE_MISMATCH)


def _visual_pipeline_repeat_max_standardized_delta(
    first_run: Any, second_run: Any
) -> float:
    """Require exact upstream records and compare every downstream trace.

    Retinal frames, circuit traces, bridge rates/events, configuration, and
    source metadata are immutable value records.  Any difference is therefore
    a deterministic-repeat failure; physical arrays retain their explicit
    engineering scales through ``_visual_repeat_max_standardized_delta``.
    """

    upstream_pairs = (
        (first_run.retinal_frames, second_run.retinal_frames),
        (first_run.circuit, second_run.circuit),
        (first_run.bridge, second_run.bridge),
        (first_run.flight_config, second_run.flight_config),
        (dict(first_run.source_metadata), dict(second_run.source_metadata)),
    )
    if any(left != right for left, right in upstream_pairs):
        return _COMPARISON_SHAPE_MISMATCH
    return _visual_repeat_max_standardized_delta(
        first_run.flight, second_run.flight
    )


def _nonsteering_activation_max_abs_delta(
    moving: Any,
    neutral: Any,
    steering_keys: Sequence[str],
) -> float:
    moving_keys = set(moving.muscle_activation)
    neutral_keys = set(neutral.muscle_activation)
    declared_steering = set(steering_keys)
    if moving_keys != neutral_keys or not declared_steering.issubset(moving_keys):
        return _COMPARISON_SHAPE_MISMATCH
    nonsteering_keys = sorted(moving_keys - declared_steering)
    if not nonsteering_keys:
        return _COMPARISON_SHAPE_MISMATCH
    return max(
        _scaled_array_max_abs_delta(
            moving.muscle_activation[key], neutral.muscle_activation[key]
        )
        for key in nonsteering_keys
    )


def _intervention_upstream_mismatch_count(
    reference_run: Any, intervention_run: Any
) -> float:
    """Count exact record mismatches upstream of an MN intervention.

    A named-MN intervention is applied by :class:`CausalNeuralBridge` only
    after the retinal, reduced-circuit, and descending-neuron records have
    already been constructed.  Those value records must therefore be exactly
    identical.  This is a software-causality invariant, not a claim about the
    biological effect of silencing an MN.
    """

    pairs = (
        (reference_run.retinal_frames, intervention_run.retinal_frames),
        (reference_run.circuit, intervention_run.circuit),
        (reference_run.bridge.descending, intervention_run.bridge.descending),
        (dict(reference_run.source_metadata), dict(intervention_run.source_metadata)),
    )
    return float(sum(left != right for left, right in pairs))


def _intervention_nontarget_event_mismatch_count(
    reference_run: Any,
    intervention_run: Any,
    *,
    target_motor_neuron: str,
    target_side: str,
) -> float:
    """Count event-inventory changes outside one declared MN/side cut."""

    mismatches = 0
    reference_trace = reference_run.bridge.wing_motor
    intervention_trace = intervention_run.bridge.wing_motor
    reference_channels = {
        (channel.motor_neuron, channel.side.value): channel
        for channel in reference_trace.channels
    }
    intervention_channels = {
        (channel.motor_neuron, channel.side.value): channel
        for channel in intervention_trace.channels
    }
    if set(reference_channels) != set(intervention_channels):
        return _COMPARISON_SHAPE_MISMATCH
    target = (target_motor_neuron, target_side)
    for key in sorted(reference_channels):
        if key != target and reference_channels[key] != intervention_channels[key]:
            mismatches += 1

    target_runtime_key = "%s-%s" % target
    for attribute in ("motor_event_times_s", "motor_event_phases_rad"):
        reference_events = getattr(reference_run.flight, attribute)
        intervention_events = getattr(intervention_run.flight, attribute)
        if set(reference_events) != set(intervention_events):
            return _COMPARISON_SHAPE_MISMATCH
        for key in sorted(reference_events):
            if key == target_runtime_key:
                continue
            if not np.array_equal(reference_events[key], intervention_events[key]):
                mismatches += 1
    return float(mismatches)


def _intervention_nontarget_muscle_state_mismatch_count(
    reference: Any,
    intervention: Any,
    *,
    target_muscle_key: str,
) -> float:
    """Count exact intrinsic non-target muscle-state mismatches.

    Activation, force, and phase memory are independent state variables in the
    current actuator and are valid exact locality invariants.  Cumulative work
    is intentionally excluded: after the target changes the shared hinge,
    unchanged non-target force can do different work against changed wing
    velocity.  Shape, dtype, and values must all be identical for each
    non-target key/attribute pair; counting mismatches avoids combining
    dimensionless activation/phase and force values into one pseudo-unit.
    """

    mismatches = 0
    observed_nontarget = False
    for attribute in (
        "muscle_activation",
        "muscle_force_n",
        "muscle_phase_effect",
    ):
        reference_map = getattr(reference, attribute)
        intervention_map = getattr(intervention, attribute)
        reference_keys = set(reference_map) - {target_muscle_key}
        intervention_keys = set(intervention_map) - {target_muscle_key}
        observed_nontarget = observed_nontarget or bool(reference_keys)
        mismatches += len(reference_keys.symmetric_difference(intervention_keys))
        for key in sorted(reference_keys & intervention_keys):
            reference_values = np.asarray(reference_map[key])
            intervention_values = np.asarray(intervention_map[key])
            if (
                reference_values.shape != intervention_values.shape
                or reference_values.dtype != intervention_values.dtype
                or not np.array_equal(reference_values, intervention_values)
            ):
                mismatches += 1
    if not observed_nontarget:
        return _COMPARISON_SHAPE_MISMATCH
    return float(mismatches)


def _motor_channel_max_abs_rate_hz(channel: Any) -> float:
    """Return the largest absolute inferred-rate sample for one MN channel."""

    return max(
        (abs(float(sample.rate_hz)) for sample in channel.rate_samples),
        default=0.0,
    )


def _target_event_availability_violation_count(
    reference_run: Any,
    *,
    target_muscle: str,
    target_side: Any,
) -> float:
    """Check that target events reach the runtime no earlier than availability."""

    channel = reference_run.bridge.wing_motor.channel(target_muscle, target_side)
    actionable = tuple(
        event
        for event in channel.events
        if event.availability_time_s < reference_run.bridge.wing_motor.duration_s
    )
    runtime_key = "%s-%s" % (channel.motor_neuron, channel.side.value)
    runtime_times = np.asarray(
        reference_run.flight.motor_event_times_s.get(runtime_key, ()), dtype=float
    )
    violations = abs(len(actionable) - len(runtime_times))
    for event, runtime_time_s in zip(actionable, runtime_times):
        if event.availability_time_s + 1.0e-12 < event.event_time_s:
            violations += 1
        if float(runtime_time_s) + 1.0e-12 < event.availability_time_s:
            violations += 1
    return float(violations)


def _final_body_state_standardized_delta(first: Any, second: Any) -> float:
    """Return a finite standardized terminal root-state difference."""

    values = np.concatenate(
        (
            (first.position_world_m[-1] - second.position_world_m[-1]) / 1.0e-3,
            (first.velocity_world_m_s[-1] - second.velocity_world_m_s[-1]) / 1.0,
            first.quaternion_body_to_world[-1]
            - second.quaternion_body_to_world[-1],
            (
                first.angular_velocity_body_rad_s[-1]
                - second.angular_velocity_body_rad_s[-1]
            )
            / 100.0,
        )
    )
    if not np.all(np.isfinite(values)):
        return _COMPARISON_SHAPE_MISMATCH
    return float(np.linalg.norm(values))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_fixture(case: BenchmarkCase) -> Path:
    if case.fixture_uri is None:
        raise FileNotFoundError("benchmark does not declare a fixture_uri")
    relative = Path(case.fixture_uri)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("fixture_uri must be a repository-relative path")

    source_tree = Path(__file__).resolve().parents[2] / relative
    installed_relative = relative
    if installed_relative.parts and installed_relative.parts[0] == "data":
        installed_relative = Path(*installed_relative.parts[1:])
    installed = (
        Path(sysconfig.get_path("data"))
        / "share"
        / "fly-sensor2behavior"
        / installed_relative
    )
    for candidate in (source_tree, installed):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("registered fixture was not found: %s" % case.fixture_uri)


def _verify_fixture(case: BenchmarkCase, path: Optional[Path] = None) -> Path:
    target = _resolve_fixture(case) if path is None else Path(path)
    if case.input_sha256 is None:
        raise FixtureIntegrityError(
            "registered fixture %s has no input_sha256" % case.fixture_uri
        )
    actual = sha256_file(target)
    if actual != case.input_sha256:
        raise FixtureIntegrityError(
            "fixture digest mismatch for %s: expected %s, observed %s"
            % (case.fixture_uri, case.input_sha256, actual)
        )
    return target


def _forbidden_downstream_field_count(
    payload: Any,
    forbidden_fields: Sequence[str],
) -> int:
    """Count forbidden fly-FGS downstream keys without matching receipt text.

    The registered capture deliberately carries the rejected field *names* as
    string values.  Only object keys are executable/data-bearing fields, so the
    traversal checks keys and dotted key paths while ignoring scalar values.
    """

    forbidden = tuple(str(value) for value in forbidden_fields)
    count = 0

    def visit(value: Any, path: tuple = ()) -> None:
        nonlocal count
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                child_path = path + (key,)
                dotted = ".".join(child_path)
                for rejected in forbidden:
                    if (
                        ("." not in rejected and key == rejected)
                        or dotted == rejected
                        or dotted.endswith("." + rejected)
                    ):
                        count += 1
                        break
                visit(child, child_path)
        elif isinstance(value, list):
            for child in value:
                visit(child, path)

    visit(payload)
    return count


def evaluate_fly_fgs_fixed_step_integrity(
    case: BenchmarkCase,
) -> EvaluatorResult:
    """Verify the immutable fly-FGS circuit boundary and non-dead trace.

    This is a frozen-fixture integrity gate.  It establishes that the exact
    captured engine/bundle inventory, fixed-step clock, full-state projection,
    and four NOD1 motor-bound channels remain unchanged.  It does not establish
    a second runtime execution, Chromium parity, physiology, or behavior.
    """

    from .fly_fgs import (
        FLY_FGS_NOD1_RAW_APP_SIDES,
        FLY_FGS_NOD1_ROOT_IDS,
        FLY_FGS_REGISTERED_CAPTURE_SHA256,
        FLY_FGS_SOURCE_MANIFEST_SHA256,
        load_registered_fly_fgs_fixture,
    )

    fixture_path = _verify_fixture(case)
    fixture = load_registered_fly_fgs_fixture(capture_path=fixture_path)
    inventory = dict(fixture.circuit_inventory)
    receipts = {receipt.asset_id: receipt for receipt in fixture.asset_receipts}
    expected_receipt_ids = {
        "page_snapshot",
        "circuit_engine",
        "circuit_bundle",
        "wing_dns_excluded",
    }
    source_inventory_match = float(
        case.input_sha256 == FLY_FGS_REGISTERED_CAPTURE_SHA256
        and fixture.capture_sha256 == FLY_FGS_REGISTERED_CAPTURE_SHA256
        and fixture.source_manifest_sha256 == FLY_FGS_SOURCE_MANIFEST_SHA256
        and set(receipts) == expected_receipt_ids
        and {
            asset_id
            for asset_id, receipt in receipts.items()
            if receipt.eligible_circuit_input
        }
        == {"circuit_engine", "circuit_bundle"}
        and inventory.get("cell_count") == 1684
        and inventory.get("event_count") == 61789
        and inventory.get("cable_count") == 8
        and inventory.get("cable_indices") == [0, 1, 2, 3, 108, 109, 110, 111]
        and inventory.get("cell_type_counts")
        == {"DCH": 2, "LLPC1": 219, "Nod1": 4, "T4a": 1457, "VCH": 2}
        and inventory.get("t5_cell_count") == 0
        and tuple(inventory.get("nod1_root_ids", ())) == FLY_FGS_NOD1_ROOT_IDS
        and dict(inventory.get("nod1_raw_app_sides", {}))
        == dict(FLY_FGS_NOD1_RAW_APP_SIDES)
    )

    fixed_step = dict(fixture.fixed_step)
    time_s = np.asarray(fixed_step.get("time_s", ()), dtype=float)
    expected_time_s = np.arange(100, dtype=float) * 0.005
    fixed_step_contract_match = float(
        fixed_step.get("dt_s") == 0.005
        and fixed_step.get("duration_s") == 0.5
        and fixed_step.get("sample_count") == 100
        and fixed_step.get("pre_roll_steps") == 48
        and fixed_step.get("pre_roll_duration_s") == 0.24
        and fixed_step.get("sample_interval") == "half-open [0, 0.5 s)"
        and time_s.shape == expected_time_s.shape
        and np.array_equal(time_s, expected_time_s)
        and fixture.circuit_trace.exact_timebase is True
        and fixture.circuit_trace.sample_times_s == tuple(expected_time_s)
        and fixture.circuit_trace.sample_interval_end_s == 0.5
    )

    trace_digest = hashlib.sha256(
        fixture.circuit_trace.to_json().encode("utf-8")
    ).hexdigest()
    registered_trace_digest_match = float(
        trace_digest == _FLY_FGS_REGISTERED_TRACE_SHA256
    )

    full_state = fixture.full_cell_state
    full_state_readout_match = False
    if full_state is not None:
        cell_ids = tuple(full_state.get("cell_ids", ()))
        voltage_v = np.asarray(full_state.get("voltage_v", ()), dtype=float)
        by_root = {
            signal.neuron.entity_id: signal for signal in fixture.circuit_trace.signals
        }
        root_columns = {
            root_id: cell_ids.index("r" + root_id)
            for root_id in FLY_FGS_NOD1_ROOT_IDS
            if "r" + root_id in cell_ids
        }
        exact_root_projection = (
            voltage_v.shape == (100, 1684)
            and set(root_columns) == set(FLY_FGS_NOD1_ROOT_IDS)
            and set(by_root) == set(FLY_FGS_NOD1_ROOT_IDS)
            and all(
                np.array_equal(
                    voltage_v[:, root_columns[root_id]],
                    np.asarray(by_root[root_id].values, dtype=float),
                )
                for root_id in FLY_FGS_NOD1_ROOT_IDS
            )
        )
        pooled = fixture.pooled_readout_traces.get("nod1_voltage_v", {})
        exact_pooled_projection = True
        for raw_side in ("L", "R"):
            roots = tuple(
                root_id
                for root_id in FLY_FGS_NOD1_ROOT_IDS
                if FLY_FGS_NOD1_RAW_APP_SIDES[root_id] == raw_side
            )
            expected = np.asarray(
                [
                    sum(
                        float(voltage_v[row, root_columns[root_id]]) * 1000.0
                        for root_id in roots
                    )
                    / len(roots)
                    / 1000.0
                    for row in range(voltage_v.shape[0])
                ],
                dtype=float,
            )
            observed = np.asarray(pooled.get("raw_app_" + raw_side, ()), dtype=float)
            exact_pooled_projection = (
                exact_pooled_projection and np.array_equal(expected, observed)
            )
        full_state_readout_match = (
            exact_root_projection and exact_pooled_projection
        )

    t4_values = np.concatenate(
        tuple(
            np.asarray(values, dtype=float)
            for values in fixture.pooled_readout_traces["t4_activity"].values()
        )
    )
    t4_activity_peak_to_peak = float(np.ptp(t4_values))
    nod1_values = np.concatenate(
        tuple(
            np.asarray(signal.values, dtype=float)
            for signal in fixture.circuit_trace.signals
        )
    )
    nod1_voltage_peak_to_peak_v = float(np.ptp(nod1_values))

    with gzip.open(fixture_path, "rt", encoding="utf-8") as handle:
        raw_capture = json.load(handle)
    forbidden_count = _forbidden_downstream_field_count(
        raw_capture,
        fixture.rejected_downstream_fields,
    )
    return EvaluatorResult(
        values=(
            MetricValue(
                "registered_source_inventory_match", source_inventory_match
            ),
            MetricValue(
                "fixed_step_contract_match", fixed_step_contract_match
            ),
            MetricValue(
                "registered_trace_digest_match", registered_trace_digest_match
            ),
            MetricValue(
                "full_state_readout_match", float(full_state_readout_match)
            ),
            MetricValue(
                "t4_activity_peak_to_peak", t4_activity_peak_to_peak
            ),
            MetricValue(
                "nod1_voltage_peak_to_peak_v", nod1_voltage_peak_to_peak_v
            ),
            MetricValue(
                "forbidden_downstream_field_count", float(forbidden_count)
            ),
        )
    )


def _registered_fly_fgs_runtime_control(fixture: Any, sample_index: int) -> Any:
    """Build one SI control from the immutable source schedule."""

    from .fly_fgs_runtime import FlyFGSSceneBodyInput

    schedule = fixture.stimulus["schedule"]
    return FlyFGSSceneBodyInput(
        heading_rad=math.radians(schedule["heading_deg"][sample_index]),
        heading_velocity_rad_s=math.radians(
            schedule["heading_velocity_deg_s"][sample_index]
        ),
        figure_world_azimuth_rad=math.radians(
            schedule["figure_world_azimuth_deg"][sample_index]
        ),
        figure_velocity_rad_s=math.radians(
            schedule["figure_velocity_deg_s"][sample_index]
        ),
        ground_velocity_rad_s=math.radians(
            schedule["grating_velocity_deg_s"][sample_index]
        ),
    )


def _fly_fgs_runtime_sample_deltas(
    sample: Any, fixture: Any, sample_index: int
) -> Mapping[str, float]:
    """Compare every runtime-visible numeric class to the frozen capture."""

    full_state = fixture.full_cell_state
    if full_state is None:
        return {
            "time": _COMPARISON_SHAPE_MISMATCH,
            "nod1": _COMPARISON_SHAPE_MISMATCH,
            "voltage": _COMPARISON_SHAPE_MISMATCH,
            "activity": _COMPARISON_SHAPE_MISMATCH,
            "retinal": _COMPARISON_SHAPE_MISMATCH,
            "pooled": _COMPARISON_SHAPE_MISMATCH,
        }
    time_delta = max(
        abs(float(sample.measurement_time_s) - sample_index * fixture.dt_s),
        abs(float(sample.availability_time_s) - sample_index * fixture.dt_s),
    )
    by_root = {
        signal.neuron.entity_id: signal for signal in fixture.circuit_trace.signals
    }
    nod1_delta = max(
        abs(
            float(sample.nod1_voltage_v[root_id])
            - float(by_root[root_id].values[sample_index])
        )
        for root_id in by_root
    )
    voltage_delta = _scaled_array_max_abs_delta(
        sample.full_cell_voltage_v,
        full_state["voltage_v"][sample_index],
    )
    activity_delta = _scaled_array_max_abs_delta(
        sample.full_cell_activity,
        full_state["activity"][sample_index],
    )
    retinal_delta = _scaled_array_max_abs_delta(
        sample.retinal_input_luminance,
        fixture.stimulus["retinal_input"]["luminance"][sample_index],
    )
    readout = sample.pooled_readout
    pooled = fixture.pooled_readout_traces
    pooled_deltas = []
    for trace_name, runtime_left, runtime_right in (
        ("nod1_activity", "nod1L", "nod1R"),
        ("vch_activity", "vchL", "vchR"),
        ("dch_activity", "dchL", "dchR"),
        ("t4_activity", "t4L", "t4R"),
        ("llpc1_activity", "llpcL", "llpcR"),
    ):
        pooled_deltas.extend(
            (
                abs(
                    float(readout[runtime_left])
                    - float(pooled[trace_name]["raw_app_L"][sample_index])
                ),
                abs(
                    float(readout[runtime_right])
                    - float(pooled[trace_name]["raw_app_R"][sample_index])
                ),
            )
        )
    for trace_name, runtime_left, runtime_right in (
        ("nod1_voltage_v", "nod1L", "nod1R"),
        ("vch_voltage_v", "vchL", "vchR"),
        ("dch_voltage_v", "dchL", "dchR"),
    ):
        pooled_deltas.extend(
            (
                abs(
                    float(readout["mv"][runtime_left]) / 1000.0
                    - float(pooled[trace_name]["raw_app_L"][sample_index])
                ),
                abs(
                    float(readout["mv"][runtime_right]) / 1000.0
                    - float(pooled[trace_name]["raw_app_R"][sample_index])
                ),
            )
        )
    return {
        "time": time_delta,
        "nod1": nod1_delta,
        "voltage": voltage_delta,
        "activity": activity_delta,
        "retinal": retinal_delta,
        "pooled": max(pooled_deltas, default=_COMPARISON_SHAPE_MISMATCH),
    }


def evaluate_fly_fgs_incremental_runtime_parity(
    case: BenchmarkCase,
) -> EvaluatorResult:
    """Re-execute the SHA-locked engine and compare every registered sample."""

    if shutil.which("node") is None:
        return EvaluatorResult.blocked(
            "the canonical fly-FGS incremental runtime requires the pinned "
            "Node.js sidecar in the worker image"
        )
    from .fly_fgs import (
        FLY_FGS_SOURCE_MANIFEST_SHA256,
        load_registered_fly_fgs_fixture,
    )
    from .fly_fgs_runtime import NodeFlyFGSCircuitRuntime

    fixture_path = _verify_fixture(case)
    fixture = load_registered_fly_fgs_fixture(capture_path=fixture_path)
    maxima = {
        "time": 0.0,
        "nod1": 0.0,
        "voltage": 0.0,
        "activity": 0.0,
        "retinal": 0.0,
        "pooled": 0.0,
    }
    with NodeFlyFGSCircuitRuntime(request_timeout_s=180.0) as runtime:
        samples = [
            runtime.initialize(
                include_retinal_input=True,
                include_full_cell_state=True,
            )
        ]
        for sample_index in range(1, 100):
            samples.append(
                runtime.advance(
                    _registered_fly_fgs_runtime_control(fixture, sample_index),
                    include_retinal_input=True,
                    include_full_cell_state=True,
                )
            )
        finalization = runtime.finalize()
        receipt = runtime.ready_receipt
    for sample_index, sample in enumerate(samples):
        deltas = _fly_fgs_runtime_sample_deltas(sample, fixture, sample_index)
        for name, value in deltas.items():
            maxima[name] = max(maxima[name], value)
    source_receipt_match = float(
        receipt["source_manifest_sha256"] == FLY_FGS_SOURCE_MANIFEST_SHA256
        and receipt["snapshot_id"] == fixture.snapshot_id
        and receipt["dt_s"] == fixture.dt_s
        and receipt["sample_count"] == 100
        and receipt["pre_roll_steps"] == 48
        and receipt["executable_asset_ids"]
        == ["circuit_engine", "circuit_bundle"]
        and receipt["full_cell_state_eligible_motor_input"] is False
        and finalization["final_sample_index"] == 99
        and finalization["final_measurement_time_s"] == 0.495
    )
    return EvaluatorResult(
        values=(
            MetricValue("runtime_source_receipt_match", source_receipt_match),
            MetricValue("runtime_timebase_max_abs_delta_s", maxima["time"]),
            MetricValue("runtime_nod1_max_abs_delta_v", maxima["nod1"]),
            MetricValue(
                "runtime_full_voltage_max_abs_delta_v", maxima["voltage"]
            ),
            MetricValue(
                "runtime_full_activity_max_abs_delta", maxima["activity"]
            ),
            MetricValue(
                "runtime_t4a_luminance_max_abs_delta", maxima["retinal"]
            ),
            MetricValue("runtime_pooled_max_abs_delta", maxima["pooled"]),
        )
    )


def evaluate_fly_fgs_checkpoint_reentry(case: BenchmarkCase) -> EvaluatorResult:
    """Prove fresh-process restoration and corruption rejection at 250 ms."""

    if shutil.which("node") is None:
        return EvaluatorResult.blocked(
            "fly-FGS checkpoint re-entry requires the pinned Node.js sidecar "
            "in the worker image"
        )
    from .fly_fgs import load_registered_fly_fgs_fixture
    from .fly_fgs_runtime import (
        FlyFGSCircuitCheckpoint,
        FlyFGSRuntimeError,
        NodeFlyFGSCircuitRuntime,
    )

    fixture_path = _verify_fixture(case)
    fixture = load_registered_fly_fgs_fixture(capture_path=fixture_path)
    with NodeFlyFGSCircuitRuntime(request_timeout_s=180.0) as reference:
        reference.initialize()
        reference_checkpoint_sample = None
        checkpoint = None
        for sample_index in range(1, 100):
            reference_checkpoint_sample = reference.advance(
                _registered_fly_fgs_runtime_control(fixture, sample_index),
                include_retinal_input=sample_index == 50 or sample_index == 99,
                include_full_cell_state=sample_index == 50 or sample_index == 99,
            )
            if sample_index == 50:
                checkpoint = reference.checkpoint()
                checkpoint_sample = reference_checkpoint_sample
            if sample_index == 99:
                reference_final_sample = reference_checkpoint_sample
        reference_finalization = reference.finalize()
    assert checkpoint is not None

    corrupted_rejection = 0.0
    corrupted_payload = json.loads(
        json.dumps(checkpoint.to_dict(), allow_nan=False)
    )
    corrupted_payload["dynamic_state"]["sample_index"] = 49
    with NodeFlyFGSCircuitRuntime(request_timeout_s=180.0) as resumed:
        try:
            resumed.restore(FlyFGSCircuitCheckpoint(corrupted_payload))
        except FlyFGSRuntimeError:
            corrupted_rejection = 1.0
        restored = resumed.restore(
            checkpoint,
            include_retinal_input=True,
            include_full_cell_state=True,
        )
        for sample_index in range(51, 100):
            resumed_final_sample = resumed.advance(
                _registered_fly_fgs_runtime_control(fixture, sample_index),
                include_retinal_input=sample_index == 99,
                include_full_cell_state=sample_index == 99,
            )
        resumed_finalization = resumed.finalize()

    restored_deltas = _fly_fgs_runtime_sample_deltas(restored, fixture, 50)
    final_deltas = _fly_fgs_runtime_sample_deltas(
        resumed_final_sample, fixture, 99
    )
    checkpoint_sample_delta = max(
        _scaled_array_max_abs_delta(
            restored.full_cell_voltage_v,
            checkpoint_sample.full_cell_voltage_v,
        ),
        _scaled_array_max_abs_delta(
            restored.full_cell_activity,
            checkpoint_sample.full_cell_activity,
        ),
        _scaled_array_max_abs_delta(
            restored.retinal_input_luminance,
            checkpoint_sample.retinal_input_luminance,
        ),
    )
    resumed_final_delta = max(
        _scaled_array_max_abs_delta(
            resumed_final_sample.full_cell_voltage_v,
            reference_final_sample.full_cell_voltage_v,
        ),
        _scaled_array_max_abs_delta(
            resumed_final_sample.full_cell_activity,
            reference_final_sample.full_cell_activity,
        ),
        _scaled_array_max_abs_delta(
            resumed_final_sample.retinal_input_luminance,
            reference_final_sample.retinal_input_luminance,
        ),
    )
    digest_match = float(
        resumed_finalization["state_sha256"]
        == reference_finalization["state_sha256"]
    )
    return EvaluatorResult(
        values=(
            MetricValue("checkpoint_sample_index", 50.0),
            MetricValue(
                "corrupted_checkpoint_rejection_count", corrupted_rejection
            ),
            MetricValue(
                "restored_checkpoint_max_abs_delta", checkpoint_sample_delta
            ),
            MetricValue("resumed_tail_max_abs_delta", resumed_final_delta),
            MetricValue("resumed_final_state_digest_match", digest_match),
            MetricValue(
                "restored_registered_state_max_abs_delta",
                max(restored_deltas.values()),
            ),
            MetricValue(
                "resumed_registered_state_max_abs_delta",
                max(final_deltas.values()),
            ),
        )
    )


def evaluate_contract_roundtrip(case: BenchmarkCase) -> EvaluatorResult:
    """Round-trip the registered contract through canonical JSON."""

    canonical = case.to_json()
    rebuilt = BenchmarkCase.from_dict(json.loads(canonical))
    equal = (
        rebuilt == case
        and rebuilt.content_sha256 == case.content_sha256
        and rebuilt.to_json() == canonical
    )
    return EvaluatorResult(
        values=(MetricValue("canonical_digest_equal", float(equal)),)
    )


def evaluate_no_future_reads(_case: BenchmarkCase) -> EvaluatorResult:
    """Exercise queue ordering and the public circuit-to-bridge boundary."""

    queue = AvailabilityQueue[str]()
    samples = (
        AvailableValue(measurement_time_s=0.000, availability_time_s=0.003, value="a"),
        AvailableValue(measurement_time_s=0.002, availability_time_s=0.007, value="b"),
        AvailableValue(measurement_time_s=0.004, availability_time_s=0.005, value="c"),
    )
    # Deliberately insert out of availability order.
    for index in (1, 0, 2):
        queue.push(samples[index])

    future_read_count = 0
    for clock_s in (0.000, 0.002999, 0.003, 0.0049, 0.005, 0.006, 0.007):
        released = queue.advance(clock_s)
        if (
            released is not None
            and released.availability_time_s > clock_s + 1.0e-12
        ):
            future_read_count += 1

    # The queue invariant alone cannot prove that a schema adapter supplied the
    # declared source availability rather than its earlier measurement time.
    # Probe both sampled and event signals through the complete encoder.  The
    # bridge-local transport delay must compose with public schema availability.
    from .flight.bridge import CircuitToDNEncoder, CircuitToDNEncoderConfig
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
        Provenance,
    )

    dataset = DatasetRef(
        namespace=DatasetNamespace.SIMULATION,
        release="manufactured-bridge-availability-v1",
        source_uri="urn:fly-s2b:fixture:bridge-availability-v1",
    )
    provenance = Provenance(
        source_uri="urn:fly-s2b:fixture:bridge-availability-v1",
        method="manufactured causal timing probe",
        dataset_identity=dataset.identity_space,
    )
    confidence = Confidence(
        tier=EvidenceTier.MODEL_INFERENCE,
        level=ConfidenceLevel.LOW,
        score=0.0,
        basis="manufactured software invariant; no biological claim",
    )
    input_delay_s = 0.002

    def bridge_probe(signal_kind: CircuitSignalKind):
        sample_times_s = (0.0, 0.005, 0.010)
        signals = []
        if signal_kind is CircuitSignalKind.SPIKE_EVENTS:
            measurements = (0.001,)
            declared_availability = (0.006,)
        else:
            measurements = sample_times_s
            declared_availability = (0.004, 0.009, 0.014)
        for side in (AnatomicalSide.LEFT, AnatomicalSide.RIGHT):
            signals.append(
                CircuitSignal(
                    neuron=EntityRef(
                        dataset=dataset,
                        entity_id="nod1-%s-%s" % (signal_kind.value, side.value),
                        kind=EntityKind.NEURON,
                        cell_type="NOD1",
                        anatomical_side=side,
                    ),
                    signal_kind=signal_kind,
                    unit="1" if signal_kind is CircuitSignalKind.SPIKE_EVENTS else "V",
                    values=()
                    if signal_kind is CircuitSignalKind.SPIKE_EVENTS
                    else (-0.055, -0.060, -0.060),
                    spike_times_s=measurements
                    if signal_kind is CircuitSignalKind.SPIKE_EVENTS
                    else (),
                    availability_times_s=declared_availability,
                    provenance=provenance,
                    confidence=confidence,
                )
            )
        trace = CircuitOutputTrace(
            dataset=dataset,
            sample_times_s=sample_times_s,
            signals=tuple(signals),
            provenance=provenance,
            confidence=confidence,
            exact_timebase=True,
            sample_interval_end_s=0.015,
        )
        encoded = CircuitToDNEncoder(
            CircuitToDNEncoderConfig(
                update_dt_s=0.001,
                input_availability_delay_s=input_delay_s,
                encoder_delay_s=0.0,
            )
        ).encode(trace)
        availability_by_measurement = dict(zip(measurements, declared_availability))
        violations = 0
        for channel in encoded.channels:
            for sample in channel.samples:
                measured = sample.source_measurement_time_s
                if measured is None:
                    continue
                earliest_use_s = availability_by_measurement[measured] + input_delay_s
                if sample.measurement_time_s + 1.0e-12 < earliest_use_s:
                    violations += 1
        return violations

    bridge_future_read_count = sum(
        bridge_probe(kind)
        for kind in (CircuitSignalKind.VOLTAGE, CircuitSignalKind.SPIKE_EVENTS)
    )

    return EvaluatorResult(
        values=(
            MetricValue("future_read_count", float(future_read_count)),
            MetricValue(
                "bridge_future_read_count", float(bridge_future_read_count)
            ),
        )
    )


def evaluate_visual_causal_golden_suite(_case: BenchmarkCase) -> EvaluatorResult:
    """Run image-only visual invariants without semantic velocity access."""

    from .vision import (
        AnalyticGratingScene,
        FigureGroundScene,
        PanoramicRetina,
        ReichardtMotionDetector,
        UniformScene,
    )

    retina = PanoramicRetina(
        receptor_count=64,
        sensor_latency_s=0.0,
        integration_subsamples=3,
    )

    def response(scene):
        detector = ReichardtMotionDetector(output_latency_s=0.0)
        values = []
        for index in range(30):
            frame = retina.sample(
                scene,
                exposure_start_s=index * 0.002,
                exposure_end_s=(index + 1) * 0.002,
            )
            values.append(
                detector.update(
                    frame, current_time_s=frame.availability_time_s
                ).population_response
            )
        return np.asarray(values, dtype=float)

    uniform_max = float(np.max(np.abs(response(UniformScene(0.5)))))

    flicker_detector = ReichardtMotionDetector(output_latency_s=0.0)
    flicker = []
    for index in range(20):
        frame = retina.sample(
            UniformScene(0.2 if index % 2 else 0.8),
            exposure_start_s=index * 0.002,
            exposure_end_s=(index + 1) * 0.002,
        )
        flicker.append(flicker_detector.update(frame).population_response)
    flicker_max = float(np.max(np.abs(flicker)))

    positive = float(
        np.mean(response(AnalyticGratingScene(angular_velocity_rad_s=1.0))[10:])
    )
    negative = float(
        np.mean(response(AnalyticGratingScene(angular_velocity_rad_s=-1.0))[10:])
    )
    direction_error = abs(positive + negative)

    phase_zero = retina.sample(
        AnalyticGratingScene(phase_rad=0.0, angular_velocity_rad_s=0.7),
        exposure_start_s=0.010,
        exposure_end_s=0.012,
    )
    phase_wrap = retina.sample(
        AnalyticGratingScene(phase_rad=2.0 * math.pi, angular_velocity_rad_s=0.7),
        exposure_start_s=0.010,
        exposure_end_s=0.012,
    )
    phase_error = float(
        np.max(
            np.abs(
                np.asarray(phase_zero.samples) - np.asarray(phase_wrap.samples)
            )
        )
    )

    onset_scene = FigureGroundScene(
        background_velocity_rad_s=0.3,
        figure_velocity_rad_s=-0.8,
        figure_onset_s=0.25,
    )
    epsilon_s = 1.0e-9
    azimuth = np.linspace(-math.pi, math.pi, 257, endpoint=False)
    onset_jump = float(
        np.max(
            np.abs(
                onset_scene.luminance(azimuth, onset_scene.figure_onset_s - epsilon_s)
                - onset_scene.luminance(azimuth, onset_scene.figure_onset_s)
            )
        )
    )

    delayed_retina = PanoramicRetina(
        receptor_count=16,
        sensor_latency_s=0.005,
        integration_subsamples=1,
    )
    future_frame = delayed_retina.sample(
        UniformScene(), exposure_start_s=0.0, exposure_end_s=0.001
    )
    detector = ReichardtMotionDetector()
    future_read_count = 0
    try:
        detector.update(future_frame, current_time_s=0.001)
    except ValueError:
        pass
    else:
        future_read_count += 1

    return EvaluatorResult(
        values=(
            MetricValue("uniform_motion_abs_max", uniform_max),
            MetricValue("flicker_motion_abs_max", flicker_max),
            MetricValue("direction_mirror_abs_error", direction_error),
            MetricValue("phase_periodicity_abs_error", phase_error),
            MetricValue("delayed_onset_luminance_jump", onset_jump),
            MetricValue("future_frame_read_count", float(future_read_count)),
        )
    )


def evaluate_retinal_to_body_vertical_slice(_case: BenchmarkCase) -> EvaluatorResult:
    """Execute every available causal software stage from pixels to body state."""

    from .pipeline import NOD1FlightPipelineConfig, run_retinal_flight_pipeline
    from .vision import AnalyticGratingScene, PanoramicRetina

    retina = PanoramicRetina(
        receptor_count=16,
        sensor_latency_s=0.0005,
        integration_subsamples=3,
    )
    scene = AnalyticGratingScene(angular_velocity_rad_s=1.0)
    frames = tuple(
        retina.sample(
            scene,
            exposure_start_s=index * 0.001,
            exposure_end_s=(index + 1) * 0.001,
        )
        for index in range(21)
    )
    run = run_retinal_flight_pipeline(
        frames,
        config=NOD1FlightPipelineConfig(seed=20260717, neural_dt_s=0.001),
    )
    body_arrays = (
        run.flight.position_world_m,
        run.flight.velocity_world_m_s,
        run.flight.quaternion_body_to_world,
        run.flight.angular_velocity_body_rad_s,
    )
    nonfinite = sum(
        int(np.size(array) - np.count_nonzero(np.isfinite(array)))
        for array in body_arrays
    )
    preserved = float(run.retinal_frames == frames)
    return EvaluatorResult(
        values=(
            MetricValue("nonfinite_body_value_count", float(nonfinite)),
            MetricValue("retinal_frame_preserved", preserved),
            MetricValue(
                "individual_muscle_count", float(len(run.flight.muscle_force_n))
            ),
            MetricValue(
                "physics_step_count", float(run.flight.diagnostics.physics_steps)
            ),
        )
    )


def evaluate_retinal_to_flybody_vertical_slice(_case: BenchmarkCase) -> EvaluatorResult:
    """Execute the visual-to-muscle path against the pinned FlyBody worker.

    This is an end-to-end software integration claim.  The retinal/NOD1 and
    neuromuscular parameters remain exploratory and are not promoted by a
    successful worker execution.
    """

    if not _flybody_worker_available():
        return EvaluatorResult.blocked(
            "the retinal-to-FlyBody vertical slice requires flygym==2.1.0 and "
            "MuJoCo 3.9.x in the pinned dedicated worker"
        )

    from .flybody_adapter import (
        FlyBodyPhysicsAdapter,
        FlyBodyWorkerConfig,
        WingAxisTorqueMap,
    )

    return _evaluate_retinal_to_flybody_vertical_slice_with_adapter(
        FlyBodyPhysicsAdapter(
            WingAxisTorqueMap(),
            FlyBodyWorkerConfig(timestep_s=1.0e-4, spawn_height_m=0.100),
        )
    )


def _evaluate_retinal_to_flybody_vertical_slice_with_adapter(
    adapter: Any,
) -> EvaluatorResult:
    """Run the worker gate against an injected resettable physics adapter.

    Production calls inject the pinned FlyBody adapter above.  Keeping the
    causal comparison independent of worker construction lets dependency-light
    tests exercise every invariant with a deterministic manufactured adapter;
    such tests do not satisfy the registered worker case on their own.
    """

    from .flight import (
        BridgeIntervention,
        BridgeInterventionMode,
        BridgeTargetType,
        FlightEpisodeRunner,
    )
    from .pipeline import NOD1FlightPipelineConfig, run_retinal_flight_pipeline
    from .schema import AnatomicalSide
    from .vision import AnalyticGratingScene, PanoramicRetina, UniformScene

    retina = PanoramicRetina(
        receptor_count=16,
        sensor_latency_s=0.0005,
        integration_subsamples=3,
    )

    def retinal_frames(scene):
        return tuple(
            retina.sample(
                scene,
                exposure_start_s=index * 0.001,
                exposure_end_s=(index + 1) * 0.001,
            )
            for index in range(21)
        )

    # The reduced surrogate contains only its declared preferred-direction
    # compatibility path. Negative image velocity is a deterministic
    # manufactured excitation, not a physiological sign claim.
    moving_frames = retinal_frames(
        AnalyticGratingScene(angular_velocity_rad_s=-5.0)
    )
    neutral_frames = retinal_frames(UniformScene(luminance_level=0.5))
    expected_initial_root_state = adapter.default_initial_state()
    expected_initial_root_position_m = (
        expected_initial_root_state.position_world_m.copy()
    )
    # Establish the comparison reference at the same explicit airborne reset
    # boundary used by every episode.  Constructor-time MuJoCo qpos/COM is not
    # a valid proxy for the configured spawned state.
    adapter.reset(expected_initial_root_state)
    (
        expected_initial_wing_angle_rad,
        expected_initial_wing_velocity_rad_s,
    ) = adapter.wing_joint_state()
    expected_initial_wing_order = adapter.wing_joint_order
    expected_initial_com_position_world_m = adapter.whole_fly_com_position_m()
    runner = FlightEpisodeRunner(physics_adapter=adapter)
    common_config = NOD1FlightPipelineConfig(
        seed=20260718,
        physics_dt_s=1.0e-4,
        neural_dt_s=0.001,
        logging_dt_s=0.001,
    )
    neutral_run = run_retinal_flight_pipeline(
        neutral_frames,
        config=common_config,
        runner=runner,
    )
    run = run_retinal_flight_pipeline(
        moving_frames,
        config=common_config,
        runner=runner,
    )
    repeated_run = run_retinal_flight_pipeline(
        moving_frames,
        config=common_config,
        runner=runner,
    )
    target_motor_neuron = "MN-b3"
    target_muscle = "b3"
    target_side = AnatomicalSide.LEFT
    target_muscle_key = "%s:%s" % (target_side.value, target_muscle)
    target_runtime_key = "%s-%s" % (target_motor_neuron, target_side.value)
    mn_silence = BridgeIntervention(
        target_type=BridgeTargetType.MN,
        target_name=target_motor_neuron,
        side=target_side,
        mode=BridgeInterventionMode.SILENCE,
        start_s=0.0,
        end_s=0.020,
    )
    silenced_run = run_retinal_flight_pipeline(
        moving_frames,
        config=common_config,
        interventions=(mn_silence,),
        runner=runner,
    )
    repeated_silenced_run = run_retinal_flight_pipeline(
        moving_frames,
        config=common_config,
        interventions=(mn_silence,),
        runner=runner,
    )
    flight = run.flight
    repeated_flight = repeated_run.flight
    neutral_flight = neutral_run.flight
    silenced_flight = silenced_run.flight
    repeated_silenced_flight = repeated_silenced_run.flight
    steering_keys = {
        "%s:%s" % (channel.side.value, channel.muscle)
        for channel in run.bridge.wing_motor.channels
    }
    circuit_driven_motor_event_count = sum(
        len(channel.events) for channel in run.bridge.wing_motor.channels
    )
    neutral_motor_event_count = sum(
        len(channel.events) for channel in neutral_run.bridge.wing_motor.channels
    )
    circuit_driven_active_muscle_count = sum(
        float(np.max(flight.muscle_activation[key])) > 0.0
        for key in steering_keys
    )
    maximum_steering_activation_delta = max(
        float(
            np.max(
                np.abs(
                    flight.muscle_activation[key]
                    - neutral_flight.muscle_activation[key]
                )
            )
        )
        for key in steering_keys
    )
    nonsteering_activation_delta = _nonsteering_activation_max_abs_delta(
        flight, neutral_flight, tuple(steering_keys)
    )
    initial_state_delta = _paired_initial_state_standardized_delta(
        (
            neutral_flight,
            flight,
            repeated_flight,
            silenced_flight,
            repeated_silenced_flight,
        ),
        expected_initial_root_state,
        expected_initial_wing_angle_rad,
        expected_initial_wing_velocity_rad_s,
        expected_initial_wing_order,
        expected_initial_com_position_world_m,
    )
    timebase_delta_s = _paired_timebase_max_abs_delta_s(
        neutral_run,
        (run, repeated_run, silenced_run, repeated_silenced_run),
    )
    repeat_delta = _visual_pipeline_repeat_max_standardized_delta(
        run, repeated_run
    )
    silenced_repeat_delta = _visual_pipeline_repeat_max_standardized_delta(
        silenced_run, repeated_silenced_run
    )
    repeat_delta = max(repeat_delta, silenced_repeat_delta)
    intervention_upstream_mismatch_count = max(
        _intervention_upstream_mismatch_count(run, silenced_run),
        _intervention_upstream_mismatch_count(
            silenced_run, repeated_silenced_run
        ),
    )
    nontarget_event_mismatch_count = (
        _intervention_nontarget_event_mismatch_count(
            run,
            silenced_run,
            target_motor_neuron=target_motor_neuron,
            target_side=target_side.value,
        )
    )
    nontarget_muscle_state_mismatch_count = (
        _intervention_nontarget_muscle_state_mismatch_count(
            flight,
            silenced_flight,
            target_muscle_key=target_muscle_key,
        )
    )
    target_channel = run.bridge.wing_motor.channel(target_muscle, target_side)
    silenced_target_channel = silenced_run.bridge.wing_motor.channel(
        target_muscle, target_side
    )
    baseline_target_event_count = sum(
        event.availability_time_s < run.bridge.wing_motor.duration_s
        for event in target_channel.events
    )
    silenced_target_event_count = len(silenced_target_channel.events) + len(
        silenced_flight.motor_event_times_s[target_runtime_key]
    )
    silenced_target_max_abs_rate_hz = _motor_channel_max_abs_rate_hz(
        silenced_target_channel
    )
    target_event_availability_violations = (
        _target_event_availability_violation_count(
            run,
            target_muscle=target_muscle,
            target_side=target_side,
        )
    )
    target_activation_delta = _scaled_array_max_abs_delta(
        flight.muscle_activation[target_muscle_key],
        silenced_flight.muscle_activation[target_muscle_key],
    )
    desired_wing_delta_rad = float(
        np.max(np.abs(flight.wing_stroke_rad - neutral_flight.wing_stroke_rad))
    )
    intervention_desired_wing_delta_rad = max(
        _scaled_array_max_abs_delta(
            flight.wing_stroke_rad, silenced_flight.wing_stroke_rad
        ),
        _scaled_array_max_abs_delta(
            flight.wing_angle_of_attack_rad,
            silenced_flight.wing_angle_of_attack_rad,
        ),
    )
    if (
        flight.measured_wing_joint_angle_rad is None
        or neutral_flight.measured_wing_joint_angle_rad is None
    ):
        measured_wing_delta_rad = 0.0
    else:
        measured_wing_delta_rad = float(
            np.max(
                np.abs(
                    flight.measured_wing_joint_angle_rad
                    - neutral_flight.measured_wing_joint_angle_rad
                )
            )
        )
    if (
        flight.measured_wing_joint_angle_rad is None
        or silenced_flight.measured_wing_joint_angle_rad is None
    ):
        intervention_measured_wing_delta_rad = 0.0
    else:
        intervention_measured_wing_delta_rad = _scaled_array_max_abs_delta(
            flight.measured_wing_joint_angle_rad,
            silenced_flight.measured_wing_joint_angle_rad,
        )
    final_body_state_standardized_delta = _final_body_state_standardized_delta(
        flight, neutral_flight
    )
    intervention_final_body_state_standardized_delta = (
        _final_body_state_standardized_delta(flight, silenced_flight)
    )
    compared_flights = (
        neutral_flight,
        flight,
        repeated_flight,
        silenced_flight,
        repeated_silenced_flight,
    )
    body_arrays = tuple(
        array
        for compared in compared_flights
        for array in (
            compared.position_world_m,
            compared.velocity_world_m_s,
            compared.quaternion_body_to_world,
            compared.angular_velocity_body_rad_s,
            compared.aerodynamic_force_body_n,
            compared.aerodynamic_torque_body_n_m,
        )
    )
    optional_arrays = tuple(
        array
        for compared in compared_flights
        for array in (
            compared.measured_wing_joint_angle_rad,
            compared.measured_wing_joint_velocity_rad_s,
            compared.whole_fly_com_position_world_m,
        )
        if array is not None
    )
    nonfinite = sum(
        int(np.size(array) - np.count_nonzero(np.isfinite(array)))
        for array in body_arrays + optional_arrays
    )
    maximum_fluid_force_n = float(
        np.max(np.linalg.norm(flight.aerodynamic_force_body_n, axis=1))
    )
    return EvaluatorResult(
        values=(
            MetricValue(
                "flybody_backend_selected",
                float(flight.diagnostics.physics_backend == "flybody"),
            ),
            MetricValue("nonfinite_body_value_count", float(nonfinite)),
            MetricValue(
                "initial_root_position_error_m",
                float(
                    np.linalg.norm(
                        flight.position_world_m[0]
                        - expected_initial_root_position_m
                    )
                ),
            ),
            MetricValue(
                "paired_initial_state_standardized_delta",
                initial_state_delta,
            ),
            MetricValue(
                "paired_timebase_max_abs_delta_s",
                timebase_delta_s,
            ),
            MetricValue(
                "visual_repeat_max_standardized_delta",
                repeat_delta,
            ),
            MetricValue(
                "mn_silence_repeat_max_standardized_delta",
                silenced_repeat_delta,
            ),
            MetricValue(
                "mn_silence_upstream_mismatch_count",
                intervention_upstream_mismatch_count,
            ),
            MetricValue(
                "mn_silence_nontarget_event_mismatch_count",
                nontarget_event_mismatch_count,
            ),
            MetricValue(
                "mn_silence_nontarget_muscle_state_mismatch_count",
                nontarget_muscle_state_mismatch_count,
            ),
            MetricValue(
                "mn_silence_baseline_target_event_count",
                float(baseline_target_event_count),
            ),
            MetricValue(
                "mn_silence_target_event_count",
                float(silenced_target_event_count),
            ),
            MetricValue(
                "mn_silence_target_rate_max_abs_hz",
                float(silenced_target_max_abs_rate_hz),
            ),
            MetricValue(
                "mn_silence_target_event_availability_violation_count",
                target_event_availability_violations,
            ),
            MetricValue(
                "mn_silence_target_activation_max_abs_delta",
                target_activation_delta,
            ),
            MetricValue(
                "mn_silence_max_desired_wing_delta_rad",
                intervention_desired_wing_delta_rad,
            ),
            MetricValue(
                "mn_silence_max_measured_wing_delta_rad",
                intervention_measured_wing_delta_rad,
            ),
            MetricValue(
                "mn_silence_final_body_state_standardized_delta",
                intervention_final_body_state_standardized_delta,
            ),
            MetricValue(
                "individual_muscle_count", float(len(flight.muscle_force_n))
            ),
            MetricValue(
                "measured_wing_state_available",
                float(
                    flight.measured_wing_joint_angle_rad is not None
                    and flight.measured_wing_joint_velocity_rad_s is not None
                    and len(flight.measured_wing_joint_order) == 6
                ),
            ),
            MetricValue(
                "whole_fly_com_available",
                float(flight.whole_fly_com_position_world_m is not None),
            ),
            MetricValue(
                "maximum_external_actuator_torque_n_m",
                float(
                    flight.diagnostics.metrics[
                        "maximum_external_actuator_torque_n_m"
                    ]
                ),
            ),
            MetricValue("maximum_root_fluid_force_n", maximum_fluid_force_n),
            MetricValue(
                "maximum_measured_wing_excursion_rad",
                float(
                    flight.diagnostics.metrics.get(
                        "maximum_measured_wing_excursion_rad", 0.0
                    )
                ),
            ),
            MetricValue(
                "circuit_driven_motor_event_count",
                float(circuit_driven_motor_event_count),
            ),
            MetricValue(
                "neutral_circuit_driven_motor_event_count",
                float(neutral_motor_event_count),
            ),
            MetricValue(
                "circuit_driven_active_muscle_count",
                float(circuit_driven_active_muscle_count),
            ),
            MetricValue(
                "visual_ablation_max_steering_activation_delta",
                maximum_steering_activation_delta,
            ),
            MetricValue(
                "visual_ablation_nonsteering_activation_max_abs_delta",
                nonsteering_activation_delta,
            ),
            MetricValue(
                "visual_ablation_max_desired_wing_delta_rad",
                desired_wing_delta_rad,
            ),
            MetricValue(
                "visual_ablation_max_measured_wing_delta_rad",
                measured_wing_delta_rad,
            ),
            MetricValue(
                "visual_ablation_final_body_state_standardized_delta",
                final_body_state_standardized_delta,
            ),
        )
    )


def evaluate_registered_nod1_to_flybody_vertical_slice(
    case: BenchmarkCase,
) -> EvaluatorResult:
    """Run the registered 1,208-cell Chromium fixture through real FlyBody."""

    fixture_path = _verify_fixture(case)
    if not _flybody_worker_available():
        return EvaluatorResult.blocked(
            "the registered Chromium NOD1-to-FlyBody vertical slice requires "
            "flygym==2.1.0 and MuJoCo 3.9.x in the pinned dedicated worker"
        )

    from .flybody_adapter import (
        FlyBodyPhysicsAdapter,
        FlyBodyWorkerConfig,
        WingAxisTorqueMap,
    )

    return _evaluate_registered_nod1_to_flybody_with_adapter(
        FlyBodyPhysicsAdapter(
            WingAxisTorqueMap(),
            FlyBodyWorkerConfig(timestep_s=1.0e-4, spawn_height_m=0.100),
        ),
        case=case,
        fixture_path=fixture_path,
    )


def evaluate_registered_fly_fgs_to_flybody_vertical_slice(
    case: BenchmarkCase,
) -> EvaluatorResult:
    """Run the registered fly-FGS circuit output through native FlyBody.

    The fixture digest is always checked on the dependency-light host.  Native
    MuJoCo execution remains an explicit scheduled-worker capability and never
    falls back to the reduced rigid body under this case ID.
    """

    fixture_path = _verify_fixture(case)
    if not _flybody_worker_available():
        return EvaluatorResult.blocked(
            "the registered fly-FGS-to-FlyBody vertical slice requires "
            "flygym==2.1.0 and MuJoCo 3.9.x in the pinned dedicated worker"
        )

    from .flybody_adapter import (
        FlyBodyPhysicsAdapter,
        FlyBodyWorkerConfig,
        WingAxisTorqueMap,
    )

    return _evaluate_registered_fly_fgs_to_flybody_with_adapter(
        FlyBodyPhysicsAdapter(
            WingAxisTorqueMap(),
            FlyBodyWorkerConfig(timestep_s=1.0e-4, spawn_height_m=0.100),
        ),
        case=case,
        fixture_path=fixture_path,
    )


def _evaluate_registered_fly_fgs_to_flybody_with_adapter(
    adapter: Any,
    *,
    case: BenchmarkCase,
    fixture_path: Path,
) -> EvaluatorResult:
    """Exercise the exact fly-FGS circuit boundary and full physics adapter.

    The public evaluator supplies native FlyBody.  Tests may inject a
    deterministic protocol-compatible adapter to exercise these invariants,
    but such an injection does not satisfy the registered scheduled case.
    """

    from .flight import FlightEpisodeRunner
    from .pipeline import (
        NOD1FlightPipelineConfig,
        run_registered_fly_fgs_flight_pipeline,
    )

    expected_initial_state = adapter.default_initial_state()
    adapter.reset(expected_initial_state)
    expected_wing_angle_rad, expected_wing_velocity_rad_s = (
        adapter.wing_joint_state()
    )
    expected_wing_order = adapter.wing_joint_order
    expected_com_position_world_m = adapter.whole_fly_com_position_m()
    runner = FlightEpisodeRunner(physics_adapter=adapter)
    config = NOD1FlightPipelineConfig(
        seed=20260718,
        physics_dt_s=1.0e-4,
        neural_dt_s=0.005,
        logging_dt_s=0.001,
    )
    baseline = run_registered_fly_fgs_flight_pipeline(
        capture_path=fixture_path,
        config=config,
        runner=runner,
    )
    repeated = run_registered_fly_fgs_flight_pipeline(
        capture_path=fixture_path,
        config=config,
        runner=runner,
    )
    flight = baseline.flight
    repeated_flight = repeated.flight

    source_metadata = dict(baseline.source_metadata)
    source_scope_match = float(
        source_metadata.get("input_mode")
        == "registered_fly_fgs_fixed_step_circuit"
        and source_metadata.get("capture_sha256") == case.input_sha256
        and source_metadata.get("circuit_cell_count") == 1684
        and source_metadata.get("source_sample_count") == 100
        and source_metadata.get("source_duration_s") == 0.5
        and source_metadata.get("retinal_frames_present") is True
        and source_metadata.get("retinal_frame_count") == 99
        and source_metadata.get("circuit_replay_present") is True
        and source_metadata.get("full_cell_state_eligible_motor_input") is False
        and source_metadata.get("fly_fgs_downstream_mechanics_imported") is False
        and len(baseline.retinal_frames) == 99
        and len(baseline.circuit.signals) == 4
        and baseline.circuit_trace_sha256 == _FLY_FGS_REGISTERED_TRACE_SHA256
    )
    initial_state_delta = _paired_initial_state_standardized_delta(
        (flight, repeated_flight),
        expected_initial_state,
        expected_wing_angle_rad,
        expected_wing_velocity_rad_s,
        expected_wing_order,
        expected_com_position_world_m,
    )
    timebase_delta_s = _paired_timebase_max_abs_delta_s(
        baseline,
        (repeated,),
    )
    repeat_delta = _visual_pipeline_repeat_max_standardized_delta(
        baseline,
        repeated,
    )
    if baseline.circuit_replay != repeated.circuit_replay:
        repeat_delta = _COMPARISON_SHAPE_MISMATCH

    steering_keys = {
        "%s:%s" % (channel.side.value, channel.muscle)
        for channel in baseline.bridge.wing_motor.channels
    }
    circuit_event_count = sum(
        len(channel.events) for channel in baseline.bridge.wing_motor.channels
    )
    active_steering_count = sum(
        float(np.max(flight.muscle_activation[key])) > 0.0
        for key in steering_keys
    )
    arrays = []
    for compared in (flight, repeated_flight):
        arrays.extend(
            (
                compared.position_world_m,
                compared.velocity_world_m_s,
                compared.quaternion_body_to_world,
                compared.angular_velocity_body_rad_s,
                compared.wing_stroke_rad,
                compared.wing_angle_of_attack_rad,
                compared.aerodynamic_force_body_n,
                compared.aerodynamic_torque_body_n_m,
            )
        )
        arrays.extend(compared.muscle_activation.values())
        arrays.extend(compared.muscle_force_n.values())
        arrays.extend(compared.muscle_phase_effect.values())
        arrays.extend(compared.muscle_work_j.values())
        arrays.extend(
            value
            for value in (
                compared.measured_wing_joint_angle_rad,
                compared.measured_wing_joint_velocity_rad_s,
                compared.whole_fly_com_position_world_m,
                compared.external_actuator_torque_n_m,
                compared.physics_time_s,
                compared.external_actuator_torque_physics_n_m,
                compared.measured_wing_joint_angle_physics_rad,
                compared.measured_wing_joint_velocity_physics_rad_s,
                compared.aerodynamic_force_body_physics_n,
                compared.aerodynamic_torque_body_physics_n_m,
            )
            if value is not None
        )
    nonfinite_count = sum(
        int(np.size(array) - np.count_nonzero(np.isfinite(array)))
        for array in arrays
    )
    maximum_fluid_force_n = (
        0.0
        if flight.aerodynamic_force_body_physics_n is None
        else float(
            np.max(
                np.linalg.norm(
                    flight.aerodynamic_force_body_physics_n,
                    axis=1,
                )
            )
        )
    )
    return EvaluatorResult(
        values=(
            MetricValue(
                "registered_fly_fgs_source_scope_match", source_scope_match
            ),
            MetricValue(
                "flybody_backend_selected",
                float(
                    flight.diagnostics.physics_backend == "flybody"
                    and repeated_flight.diagnostics.physics_backend == "flybody"
                ),
            ),
            MetricValue("nonfinite_value_count", float(nonfinite_count)),
            MetricValue(
                "physics_step_count", float(flight.diagnostics.physics_steps)
            ),
            MetricValue(
                "paired_initial_state_standardized_delta", initial_state_delta
            ),
            MetricValue("paired_timebase_max_abs_delta_s", timebase_delta_s),
            MetricValue(
                "baseline_repeat_max_standardized_delta", repeat_delta
            ),
            MetricValue(
                "circuit_driven_motor_event_count", float(circuit_event_count)
            ),
            MetricValue(
                "circuit_driven_active_muscle_count",
                float(active_steering_count),
            ),
            MetricValue(
                "maximum_external_actuator_torque_n_m",
                float(
                    flight.diagnostics.metrics.get(
                        "maximum_external_actuator_torque_n_m", 0.0
                    )
                ),
            ),
            MetricValue("maximum_root_fluid_force_n", maximum_fluid_force_n),
            MetricValue(
                "maximum_measured_wing_excursion_rad",
                float(
                    flight.diagnostics.metrics.get(
                        "maximum_measured_wing_excursion_rad", 0.0
                    )
                ),
            ),
        )
    )


def _evaluate_registered_nod1_to_flybody_with_adapter(
    adapter: Any,
    *,
    case: BenchmarkCase,
    fixture_path: Path,
) -> EvaluatorResult:
    """Exercise the frozen browser output, a typed MN cut, and full physics.

    This gate establishes an executable, causal, non-dead software path.  It
    does not promote the frozen circuit, bridge gains, muscle model, or body
    trajectory as biologically calibrated.
    """

    from .flight import (
        BridgeIntervention,
        BridgeInterventionMode,
        BridgeTargetType,
        FlightEpisodeRunner,
    )
    from .pipeline import (
        NOD1FlightPipelineConfig,
        run_registered_nod1_browser_flight_pipeline,
    )
    from .schema import AnatomicalSide

    expected_initial_state = adapter.default_initial_state()
    adapter.reset(expected_initial_state)
    expected_wing_angle_rad, expected_wing_velocity_rad_s = (
        adapter.wing_joint_state()
    )
    expected_wing_order = adapter.wing_joint_order
    expected_com_position_world_m = adapter.whole_fly_com_position_m()
    runner = FlightEpisodeRunner(physics_adapter=adapter)
    config = NOD1FlightPipelineConfig(
        seed=20260718,
        physics_dt_s=1.0e-4,
        neural_dt_s=0.001,
        logging_dt_s=0.001,
    )
    baseline = run_registered_nod1_browser_flight_pipeline(
        fixture_path=fixture_path,
        config=config,
        runner=runner,
    )
    repeated_baseline = run_registered_nod1_browser_flight_pipeline(
        fixture_path=fixture_path,
        config=config,
        runner=runner,
    )
    target_motor_neuron = "MN-iv1"
    target_muscle = "iv1"
    target_side = AnatomicalSide.RIGHT
    target_muscle_key = "%s:%s" % (target_side.value, target_muscle)
    target_runtime_key = "%s-%s" % (target_motor_neuron, target_side.value)
    intervention = BridgeIntervention(
        target_type=BridgeTargetType.MN,
        target_name=target_motor_neuron,
        side=target_side,
        mode=BridgeInterventionMode.SILENCE,
        start_s=0.0,
        end_s=0.5,
    )
    silenced = run_registered_nod1_browser_flight_pipeline(
        fixture_path=fixture_path,
        config=config,
        interventions=(intervention,),
        runner=runner,
    )
    flight = baseline.flight
    repeated_flight = repeated_baseline.flight
    silenced_flight = silenced.flight

    source_metadata = dict(baseline.source_metadata)
    source_scope_match = float(
        source_metadata.get("input_mode")
        == "registered_frozen_browser_nod1_parity"
        and source_metadata.get("fixture_sha256") == case.input_sha256
        and source_metadata.get("circuit_cell_count") == 1208
        and source_metadata.get("source_sample_count") == 100
        and source_metadata.get("retinal_frames_present") is False
        and baseline.retinal_frames == ()
        and len(baseline.circuit.signals) == 4
    )
    duration_s = 0.5
    duration_error_s = max(
        abs(float(source_metadata.get("source_duration_s", math.inf)) - duration_s),
        abs(
            float(source_metadata.get("source_sample_interval_end_s", math.inf))
            - duration_s
        ),
        abs(baseline.bridge.descending.duration_s - duration_s),
        abs(baseline.bridge.wing_motor.duration_s - duration_s),
        abs(baseline.flight_config.duration_s - duration_s),
        abs(flight.diagnostics.final_time_s - duration_s),
        abs(float(flight.time_s[-1]) - duration_s),
    )
    latency_tail_complete = float(
        any(
            sample.source_measurement_time_s is not None
            and abs(sample.source_measurement_time_s - 0.495) <= 1.0e-12
            and abs(sample.availability_time_s - 0.498) <= 1.0e-12
            for channel in baseline.bridge.descending.channels
            for sample in channel.samples
        )
        and any(
            sample.source_measurement_time_s is not None
            and abs(sample.source_measurement_time_s - 0.495) <= 1.0e-12
            and abs(sample.availability_time_s - 0.500) <= 1.0e-12
            for channel in baseline.bridge.wing_motor.channels
            for sample in channel.rate_samples
        )
    )

    initial_state_delta = _paired_initial_state_standardized_delta(
        (flight, repeated_flight, silenced_flight),
        expected_initial_state,
        expected_wing_angle_rad,
        expected_wing_velocity_rad_s,
        expected_wing_order,
        expected_com_position_world_m,
    )
    timebase_delta_s = _paired_timebase_max_abs_delta_s(
        baseline, (repeated_baseline, silenced)
    )
    repeat_delta = _visual_pipeline_repeat_max_standardized_delta(
        baseline, repeated_baseline
    )
    upstream_mismatch_count = _intervention_upstream_mismatch_count(
        baseline, silenced
    )
    nontarget_event_mismatch_count = _intervention_nontarget_event_mismatch_count(
        baseline,
        silenced,
        target_motor_neuron=target_motor_neuron,
        target_side=target_side.value,
    )
    nontarget_muscle_mismatch_count = (
        _intervention_nontarget_muscle_state_mismatch_count(
            flight,
            silenced_flight,
            target_muscle_key=target_muscle_key,
        )
    )
    target_channel = baseline.bridge.wing_motor.channel(target_muscle, target_side)
    silenced_target_channel = silenced.bridge.wing_motor.channel(
        target_muscle, target_side
    )
    baseline_target_event_count = sum(
        event.availability_time_s < baseline.bridge.wing_motor.duration_s
        for event in target_channel.events
    )
    actionable_target_events = tuple(
        event
        for event in target_channel.events
        if event.availability_time_s < baseline.bridge.wing_motor.duration_s
    )
    first_target_event_time_s = min(
        (event.availability_time_s for event in actionable_target_events),
        default=math.inf,
    )
    silenced_target_event_count = len(silenced_target_channel.events) + len(
        silenced_flight.motor_event_times_s[target_runtime_key]
    )
    target_event_availability_violations = (
        _target_event_availability_violation_count(
            baseline,
            target_muscle=target_muscle,
            target_side=target_side,
        )
    )
    precontact_end_s = 0.050
    precontact_count = int(
        np.searchsorted(
            flight.time_s,
            precontact_end_s + 1.0e-12,
            side="right",
        )
    )
    if precontact_count < 2:
        raise RuntimeError("registered pre-contact window has too few samples")
    precontact = slice(0, precontact_count)
    target_activation_delta = _scaled_array_max_abs_delta(
        flight.muscle_activation[target_muscle_key][precontact],
        silenced_flight.muscle_activation[target_muscle_key][precontact],
    )
    desired_wing_delta_rad = max(
        _scaled_array_max_abs_delta(
            flight.wing_stroke_rad[precontact],
            silenced_flight.wing_stroke_rad[precontact],
        ),
        _scaled_array_max_abs_delta(
            flight.wing_angle_of_attack_rad[precontact],
            silenced_flight.wing_angle_of_attack_rad[precontact],
        ),
    )
    physics_timebases_match = (
        flight.physics_time_s is not None
        and silenced_flight.physics_time_s is not None
        and np.array_equal(flight.physics_time_s, silenced_flight.physics_time_s)
    )
    physics_precontact = (
        None
        if not physics_timebases_match
        else np.asarray(flight.physics_time_s)
        <= (precontact_end_s + 1.0e-12)
    )
    measured_wing_delta_rad = (
        0.0
        if physics_precontact is None
        or flight.measured_wing_joint_angle_physics_rad is None
        or silenced_flight.measured_wing_joint_angle_physics_rad is None
        else _scaled_array_max_abs_delta(
            flight.measured_wing_joint_angle_physics_rad[physics_precontact],
            silenced_flight.measured_wing_joint_angle_physics_rad[
                physics_precontact
            ],
        )
    )
    precontact_index = precontact_count - 1
    body_delta_values = np.concatenate(
        (
            (
                flight.position_world_m[precontact_index]
                - silenced_flight.position_world_m[precontact_index]
            )
            / 1.0e-3,
            (
                flight.velocity_world_m_s[precontact_index]
                - silenced_flight.velocity_world_m_s[precontact_index]
            )
            / 1.0,
            flight.quaternion_body_to_world[precontact_index]
            - silenced_flight.quaternion_body_to_world[precontact_index],
            (
                flight.angular_velocity_body_rad_s[precontact_index]
                - silenced_flight.angular_velocity_body_rad_s[precontact_index]
            )
            / 100.0,
        )
    )
    body_state_delta = (
        _COMPARISON_SHAPE_MISMATCH
        if not np.all(np.isfinite(body_delta_values))
        else float(np.linalg.norm(body_delta_values))
    )
    if (
        not physics_timebases_match
        or flight.external_actuator_torque_physics_n_m is None
        or silenced_flight.external_actuator_torque_physics_n_m is None
    ):
        actuator_torque_delta_n_m = 0.0
    else:
        assert physics_precontact is not None
        actuator_torque_delta_n_m = float(
            np.max(
                np.abs(
                    flight.external_actuator_torque_physics_n_m[physics_precontact]
                    - silenced_flight.external_actuator_torque_physics_n_m[
                        physics_precontact
                    ]
                )
            )
        )
    fluid_force_delta_n = (
        0.0
        if physics_precontact is None
        or flight.aerodynamic_force_body_physics_n is None
        or silenced_flight.aerodynamic_force_body_physics_n is None
        else float(
            np.max(
                np.linalg.norm(
                    flight.aerodynamic_force_body_physics_n[physics_precontact]
                    - silenced_flight.aerodynamic_force_body_physics_n[
                        physics_precontact
                    ],
                    axis=1,
                )
            )
        )
    )
    contact_telemetry_available = float(
        flight.ground_contact_count is not None
        and silenced_flight.ground_contact_count is not None
        and flight.ground_contact_transition_point_count is not None
        and silenced_flight.ground_contact_transition_point_count is not None
        and flight.physics_time_s is not None
        and silenced_flight.physics_time_s is not None
        and flight.diagnostics.metrics.get(
            "ground_contact_telemetry_available", 0.0
        )
        == 1.0
        and silenced_flight.diagnostics.metrics.get(
            "ground_contact_telemetry_available", 0.0
        )
        == 1.0
    )
    initial_ground_contact_count = max(
        float(flight.diagnostics.metrics.get("initial_ground_contact_count", math.inf)),
        float(
            silenced_flight.diagnostics.metrics.get(
                "initial_ground_contact_count", math.inf
            )
        ),
    )
    precontact_ground_contact_transition_count = max(
        float(
            flight.diagnostics.metrics.get(
                "ground_contact_transition_count_first_50ms", math.inf
            )
        ),
        float(
            silenced_flight.diagnostics.metrics.get(
                "ground_contact_transition_count_first_50ms", math.inf
            )
        ),
    )
    circuit_event_count = sum(
        len(channel.events) for channel in baseline.bridge.wing_motor.channels
    )
    steering_keys = {
        "%s:%s" % (channel.side.value, channel.muscle)
        for channel in baseline.bridge.wing_motor.channels
    }
    active_steering_count = sum(
        float(np.max(flight.muscle_activation[key])) > 0.0
        for key in steering_keys
    )
    compared_flights = (flight, repeated_flight, silenced_flight)
    arrays = []
    for compared in compared_flights:
        arrays.extend(
            (
                compared.position_world_m,
                compared.velocity_world_m_s,
                compared.quaternion_body_to_world,
                compared.angular_velocity_body_rad_s,
                compared.wing_stroke_rad,
                compared.wing_angle_of_attack_rad,
                compared.aerodynamic_force_body_n,
                compared.aerodynamic_torque_body_n_m,
            )
        )
        arrays.extend(compared.muscle_activation.values())
        arrays.extend(compared.muscle_force_n.values())
        arrays.extend(compared.muscle_phase_effect.values())
        arrays.extend(compared.muscle_work_j.values())
        arrays.extend(
            item
            for item in (
                compared.measured_wing_joint_angle_rad,
                compared.measured_wing_joint_velocity_rad_s,
                compared.whole_fly_com_position_world_m,
                compared.ground_contact_count,
                compared.external_actuator_torque_n_m,
                compared.physics_time_s,
                compared.ground_contact_transition_point_count,
                compared.external_actuator_torque_physics_n_m,
                compared.measured_wing_joint_angle_physics_rad,
                compared.measured_wing_joint_velocity_physics_rad_s,
                compared.aerodynamic_force_body_physics_n,
                compared.aerodynamic_torque_body_physics_n_m,
            )
            if item is not None
        )
    nonfinite_count = sum(
        int(np.size(array) - np.count_nonzero(np.isfinite(array)))
        for array in arrays
    )
    maximum_fluid_force_n = (
        0.0
        if flight.aerodynamic_force_body_physics_n is None
        else float(
            np.max(
                np.linalg.norm(
                    flight.aerodynamic_force_body_physics_n, axis=1
                )
            )
        )
    )

    return EvaluatorResult(
        values=(
            MetricValue("registered_source_scope_match", source_scope_match),
            MetricValue("fixture_runtime_duration_error_s", duration_error_s),
            MetricValue("latency_tail_complete", latency_tail_complete),
            MetricValue(
                "flybody_backend_selected",
                float(flight.diagnostics.physics_backend == "flybody"),
            ),
            MetricValue("nonfinite_value_count", float(nonfinite_count)),
            MetricValue(
                "physics_step_count", float(flight.diagnostics.physics_steps)
            ),
            MetricValue(
                "paired_initial_state_standardized_delta", initial_state_delta
            ),
            MetricValue("paired_timebase_max_abs_delta_s", timebase_delta_s),
            MetricValue(
                "baseline_repeat_max_standardized_delta", repeat_delta
            ),
            MetricValue(
                "ground_contact_telemetry_available",
                contact_telemetry_available,
            ),
            MetricValue(
                "initial_ground_contact_count", initial_ground_contact_count
            ),
            MetricValue(
                "ground_contact_transition_count_0_50ms",
                precontact_ground_contact_transition_count,
            ),
            MetricValue(
                "circuit_driven_motor_event_count", float(circuit_event_count)
            ),
            MetricValue(
                "circuit_driven_active_muscle_count", float(active_steering_count)
            ),
            MetricValue(
                "maximum_external_actuator_torque_n_m",
                float(
                    flight.diagnostics.metrics.get(
                        "maximum_external_actuator_torque_n_m", 0.0
                    )
                ),
            ),
            MetricValue("maximum_root_fluid_force_n", maximum_fluid_force_n),
            MetricValue(
                "maximum_measured_wing_excursion_rad",
                float(
                    flight.diagnostics.metrics.get(
                        "maximum_measured_wing_excursion_rad", 0.0
                    )
                ),
            ),
            MetricValue(
                "mn_silence_upstream_mismatch_count", upstream_mismatch_count
            ),
            MetricValue(
                "mn_silence_nontarget_event_mismatch_count",
                nontarget_event_mismatch_count,
            ),
            MetricValue(
                "mn_silence_nontarget_muscle_state_mismatch_count",
                nontarget_muscle_mismatch_count,
            ),
            MetricValue(
                "mn_silence_baseline_target_event_count",
                float(baseline_target_event_count),
            ),
            MetricValue(
                "mn_silence_first_target_event_time_s",
                float(first_target_event_time_s),
            ),
            MetricValue(
                "mn_silence_target_event_count",
                float(silenced_target_event_count),
            ),
            MetricValue(
                "mn_silence_target_rate_max_abs_hz",
                _motor_channel_max_abs_rate_hz(silenced_target_channel),
            ),
            MetricValue(
                "mn_silence_target_event_availability_violation_count",
                target_event_availability_violations,
            ),
            MetricValue(
                "mn_silence_target_activation_max_abs_delta",
                target_activation_delta,
            ),
            MetricValue(
                "mn_silence_max_desired_wing_delta_rad", desired_wing_delta_rad
            ),
            MetricValue(
                "mn_silence_max_measured_wing_delta_rad", measured_wing_delta_rad
            ),
            MetricValue(
                "mn_silence_max_external_actuator_torque_delta_n_m",
                actuator_torque_delta_n_m,
            ),
            MetricValue(
                "mn_silence_max_root_fluid_force_delta_n",
                fluid_force_delta_n,
            ),
            MetricValue(
                "mn_silence_precontact_body_state_standardized_delta",
                body_state_delta,
            ),
        )
    )


def evaluate_reduced_closed_loop_yaw(_case: BenchmarkCase) -> EvaluatorResult:
    """Exercise the causal visual/body loop using software-only metamorphic oracles.

    These metrics establish timing, sign symmetry, and deterministic negative
    feedback in the explicitly reduced yaw plant.  They are not evidence that
    the NOD1 pathway or a real fly has the same gains or recovery dynamics.
    """

    from .flight.closed_loop import ReducedClosedLoopYawSimulator, ReducedYawLoopConfig
    from .vision import UniformScene

    config = ReducedYawLoopConfig(physics_dt_s=2.0e-4)
    simulator = ReducedClosedLoopYawSimulator()
    closed = simulator.run(config)
    opened = simulator.run(
        ReducedYawLoopConfig(physics_dt_s=2.0e-4, feedback_enabled=False)
    )
    mirrored = simulator.run(
        ReducedYawLoopConfig(physics_dt_s=2.0e-4, initial_yaw_rate_rad_s=-3.0)
    )
    uniform = ReducedClosedLoopYawSimulator(UniformScene()).run(config)

    causal_violation_count = sum(
        len(result.causal_timing_violations())
        for result in (closed, opened, mirrored, uniform)
    )
    final_rate_ratio = abs(float(closed.yaw_rate_rad_s[-1])) / max(
        abs(float(opened.yaw_rate_rad_s[-1])), np.finfo(float).tiny
    )
    integrated_rate_ratio = closed.integrated_absolute_yaw_rate_rad() / max(
        opened.integrated_absolute_yaw_rate_rad(), np.finfo(float).tiny
    )
    mirror_error = float(
        np.max(np.abs(mirrored.yaw_rate_rad_s + closed.yaw_rate_rad_s))
    )
    uniform_torque = float(np.max(np.abs(uniform.control_yaw_torque_n_m)))
    return EvaluatorResult(
        values=(
            MetricValue("causal_timing_violation_count", float(causal_violation_count)),
            MetricValue("closed_loop_final_rate_ratio", final_rate_ratio),
            MetricValue("closed_loop_integrated_rate_ratio", integrated_rate_ratio),
            MetricValue("mirror_yaw_rate_max_abs_error", mirror_error),
            MetricValue("uniform_scene_max_abs_control_torque", uniform_torque),
        )
    )


def evaluate_reduced_timestep_convergence(_case: BenchmarkCase) -> EvaluatorResult:
    """Compare the deterministic test backend at 0.1 and 0.05 ms."""

    from .artifacts import run_scenario

    _, coarse = run_scenario(
        "baseline",
        duration_s=0.100,
        physics_dt_s=0.0001,
        logging_dt_s=0.001,
        seed=11,
    )
    _, fine = run_scenario(
        "baseline",
        duration_s=0.100,
        physics_dt_s=0.00005,
        logging_dt_s=0.001,
        seed=11,
    )
    coarse_impulse = np.asarray(
        coarse.diagnostics.integrated_aerodynamic_impulse_n_s, dtype=float
    )
    fine_impulse = np.asarray(
        fine.diagnostics.integrated_aerodynamic_impulse_n_s, dtype=float
    )
    impulse_change = float(
        np.linalg.norm(coarse_impulse - fine_impulse)
        / max(np.linalg.norm(fine_impulse), np.finfo(float).tiny)
    )
    coarse_state = np.concatenate(
        (
            coarse.position_world_m[-1],
            coarse.velocity_world_m_s[-1],
            coarse.angular_velocity_body_rad_s[-1],
        )
    )
    fine_state = np.concatenate(
        (
            fine.position_world_m[-1],
            fine.velocity_world_m_s[-1],
            fine.angular_velocity_body_rad_s[-1],
        )
    )
    # Declared nondimensionalization prevents nearly-zero coordinates from
    # dominating the convergence score: 1 mm, 1 m/s, and 10 rad/s scales.
    engineering_scale = np.asarray([1.0e-3] * 3 + [1.0] * 3 + [10.0] * 3)
    denominator = np.maximum(np.abs(fine_state), engineering_scale)
    body_change = float(np.max(np.abs(coarse_state - fine_state) / denominator))
    return EvaluatorResult(
        values=(
            MetricValue("reduced_impulse_relative_change", impulse_change),
            MetricValue("reduced_body_state_relative_change", body_change),
        )
    )


def evaluate_hines_dense_manufactured(_case: BenchmarkCase) -> EvaluatorResult:
    """Compare the arbitrary-order Hines solve with its dense matrix oracle."""

    parent = np.asarray([2, 4, -1, 2, 2, 4], dtype=int)
    coupling = np.asarray([0.8, 0.3, 0.0, 0.7, 0.5, 0.4], dtype=float)
    diagonal = np.asarray([1.5, 0.9, 3.0, 1.4, 2.0, 1.0], dtype=float)
    rhs = np.asarray([-0.05, 0.02, -0.12, 0.03, -0.08, 0.04], dtype=float)

    matrix = dense_tree_matrix(parent, diagonal, coupling)
    dense_solution = np.linalg.solve(matrix, rhs)
    hines_solution = hines_solve(parent, diagonal, coupling, rhs)
    max_error_v = float(np.max(np.abs(hines_solution - dense_solution)))

    # Convert each equation residual to an equivalent voltage by its absolute
    # row sum, then apply the preregistered 1e-9 V + 1e-8 relative scale.
    residual = matrix.dot(hines_solution) - rhs
    row_scale = np.sum(np.abs(matrix), axis=1)
    equivalent_residual_v = np.abs(residual) / row_scale
    acceptance_scale_v = 1.0e-9 + 1.0e-8 * np.abs(dense_solution)
    max_scaled_residual = float(
        np.max(equivalent_residual_v / acceptance_scale_v)
    )
    if not math.isfinite(max_scaled_residual):
        raise RuntimeError("manufactured Hines residual was non-finite")

    return EvaluatorResult(
        values=(
            MetricValue("max_abs_voltage_error_v", max_error_v),
            MetricValue("max_scaled_residual", max_scaled_residual),
        )
    )


def evaluate_cross_atlas_integrity(case: BenchmarkCase) -> EvaluatorResult:
    """Validate and count only explicit type-level cross-atlas transports."""

    # Resolve through the evidence package helper so source-tree and installed
    # layouts behave identically, while still enforcing the registry digest.
    fixture = _verify_fixture(case, default_seed_graph_path())
    graph = load_evidence_graph(fixture)
    direct_invalid = 0
    explicit_crosswalks = 0
    for edge in graph.edges:
        if not edge.is_cross_atlas:
            continue
        is_explicit_type_crosswalk = (
            edge.relation is EdgeRelation.TYPE_CROSSWALK
            and edge.source.kind is EntityKind.CELL_TYPE
            and edge.target.kind is EntityKind.CELL_TYPE
        )
        if is_explicit_type_crosswalk:
            explicit_crosswalks += 1
        else:
            direct_invalid += 1
    return EvaluatorResult(
        values=(
            MetricValue("direct_cross_atlas_join_count", float(direct_invalid)),
            MetricValue("explicit_type_crosswalk_count", float(explicit_crosswalks)),
        )
    )


def evaluate_banc_fanc_wing_pathway(case: BenchmarkCase) -> EvaluatorResult:
    """Recompute registered structural counts without inventing atlas identity.

    The fixture loader verifies every selected edge, identifier type, source
    receipt, summary total, and unresolved cross-atlas state.  Returned values
    are structural observations only; none are physiological weights or
    functional signs.
    """

    from .banc_fanc_evidence import (
        load_banc_fanc_evidence,
        validate_banc_fanc_evidence,
    )
    from .flight.bridge import default_steering_pathways

    fixture = _verify_fixture(case)
    observations = validate_banc_fanc_evidence(
        fixture, verify_registered_digest=True
    ).as_dict()
    evidence = load_banc_fanc_evidence(
        fixture, verify_registered_digest=True
    )
    steering_names = {
        pathway.muscle for pathway in default_steering_pathways()
    }
    evidence_counts: Dict[str, int] = {}
    for edge in evidence["banc_v888"]["dnp26_to_wing_motor_neuron_edges"]:
        muscle = str(edge["post_cell_type"])
        if muscle in steering_names:
            evidence_counts[muscle] = evidence_counts.get(muscle, 0) + int(
                edge["structural_synapse_count"]
            )
    runtime_counts = {
        pathway.muscle: pathway.structural_synapse_count
        for pathway in default_steering_pathways()
    }
    observations["runtime_structural_mask_mismatch_count"] = sum(
        evidence_counts.get(name) != runtime_counts.get(name)
        for name in evidence_counts.keys() | runtime_counts.keys()
    )
    return EvaluatorResult(
        values=tuple(
            MetricValue(metric.metric_id, float(observations[metric.metric_id]))
            for metric in case.metrics
        )
    )


def _logical_sha256_prefixed(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _parity_summary(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean_voltage_v": float(np.mean(values)),
        "minimum_voltage_v": float(np.min(values)),
        "maximum_voltage_v": float(np.max(values)),
        "final_voltage_v": float(values[-1]),
        "peak_to_peak_voltage_v": float(np.ptp(values)),
    }


def evaluate_browser_python_parity(case: BenchmarkCase) -> EvaluatorResult:
    """Recompute browser/Python differences from a strict captured fixture."""

    fixture_path = _verify_fixture(case)
    if fixture_path.suffix != ".gz":
        raise FixtureIntegrityError("browser parity fixture must be gzip-compressed JSON")
    with gzip.open(fixture_path, "rt", encoding="utf-8") as handle:
        fixture = json.load(handle)

    if fixture.get("fixture_kind") != "nod1_browser_python_numerical_parity":
        raise FixtureIntegrityError("browser parity fixture kind is invalid")
    source_receipt = fixture.get("source_receipt", {})
    if source_receipt.get("strict_match") is not True or source_receipt.get(
        "mismatches"
    ):
        raise FixtureIntegrityError("browser parity fixture did not use strict source hashes")
    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "manifests"
        / "legacy_nod1_v0.5.0.json"
    )
    if not manifest_path.is_file():
        manifest_path = (
            Path(sysconfig.get_path("data"))
            / "share"
            / "fly-sensor2behavior"
            / "manifests"
            / "legacy_nod1_v0.5.0.json"
        )
    if source_receipt.get("manifest_sha256") != "sha256:" + sha256_file(
        manifest_path
    ):
        raise FixtureIntegrityError("browser parity source manifest digest drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_deployment = manifest.get("deployed_capture", {})
    browser_runtime = fixture.get("browser_runtime", {})
    if browser_runtime.get("circuit_response_sha256") != expected_deployment.get(
        "circuit_response_sha256"
    ):
        raise FixtureIntegrityError("browser parity circuit response is not frozen")
    expected_worker_hashes = set(
        expected_deployment.get("worker_asset_sha256", ())
    )
    observed_worker_hashes = {
        item.get("sha256") for item in browser_runtime.get("worker_assets", ())
    }
    if not expected_worker_hashes or observed_worker_hashes != expected_worker_hashes:
        raise FixtureIntegrityError("browser parity Web Worker asset is not frozen")
    expected_index = expected_deployment.get("index_html", {})
    observed_index = browser_runtime.get("index_html", {})
    if (
        observed_index.get("sha256") != expected_index.get("sha256")
        or int(observed_index.get("bytes", -1))
        != int(expected_index.get("bytes", -2))
    ):
        raise FixtureIntegrityError("browser parity index HTML is not frozen")
    expected_main_assets = {
        (item.get("kind"), item.get("sha256"), int(item.get("bytes", -1)))
        for item in expected_deployment.get("main_assets", ())
    }
    observed_main_assets = {
        (item.get("kind"), item.get("sha256"), int(item.get("bytes", -2)))
        for item in browser_runtime.get("main_assets", ())
    }
    if not expected_main_assets or observed_main_assets != expected_main_assets:
        raise FixtureIntegrityError("browser parity main app assets are not frozen")

    dataset = fixture.get("dataset", {})
    expected_roots = (
        "720575940628438427",
        "720575940625528556",
        "720575940623997949",
        "720575940629456860",
    )
    if dataset.get("materialization") != 783 or tuple(
        dataset.get("root_ids", ())
    ) != expected_roots:
        raise FixtureIntegrityError("browser parity dataset identity is invalid")
    workers = browser_runtime.get("worker_assets", ())
    if not workers or any(
        not str(item.get("sha256", "")).startswith("sha256:")
        or int(item.get("bytes", 0)) <= 0
        for item in workers
    ):
        raise FixtureIntegrityError("browser parity fixture has no valid Web Worker receipt")

    replay = fixture.get("replay_input", {})
    saved_config = replay.get("saved_config")
    if not isinstance(saved_config, dict) or _logical_sha256_prefixed(
        saved_config
    ) != fixture.get("configuration", {}).get("saved_config_sha256"):
        raise FixtureIntegrityError("browser parity replay configuration digest drifted")
    if len(saved_config.get("runCells", ())) != 1208 or len(
        saved_config.get("runEdges", ())
    ) != 5188:
        raise FixtureIntegrityError("browser parity replay circuit inventory is invalid")
    prepared = replay.get("prepared_bundle", {})
    event_count = len(prepared.get("events", {}).get("pre", ()))
    mapping = prepared.get("meta", {}).get("synapseMapping", {})
    synthetic_count = int(mapping.get("syntheticFallbackSynapses", -1))
    if (
        event_count != 53715
        or int(mapping.get("events", -1)) != event_count
        or event_count - synthetic_count != 8920
    ):
        raise FixtureIntegrityError("browser parity prepared-event accounting is invalid")

    time_s = np.asarray(fixture.get("time_s", ()), dtype=float)
    if (
        time_s.ndim != 1
        or len(time_s) < 2
        or not np.all(np.isfinite(time_s))
        or not np.all(np.diff(time_s) > 0.0)
    ):
        raise FixtureIntegrityError("browser parity time base is invalid")
    readouts = fixture.get("readouts", {})
    max_readout_error_v = 0.0
    max_summary_relative_error = 0.0
    sample_count = 0
    denominator_floor_v = 5.0e-5
    for root_id in expected_roots:
        record = readouts.get(root_id, {})
        browser = np.asarray(record.get("browser_voltage_v", ()), dtype=float)
        python = np.asarray(record.get("python_voltage_v", ()), dtype=float)
        if (
            browser.shape != time_s.shape
            or python.shape != time_s.shape
            or not np.all(np.isfinite(browser))
            or not np.all(np.isfinite(python))
        ):
            raise FixtureIntegrityError("browser parity readout/time shapes differ")
        max_readout_error_v = max(
            max_readout_error_v, float(np.max(np.abs(browser - python)))
        )
        browser_summary = _parity_summary(browser)
        python_summary = _parity_summary(python)
        max_summary_relative_error = max(
            max_summary_relative_error,
            max(
                abs(browser_summary[name] - python_summary[name])
                / max(abs(python_summary[name]), denominator_floor_v)
                for name in browser_summary
            ),
        )
        sample_count += len(browser)

    stored_metrics = fixture.get("metrics", {})
    if (
        stored_metrics.get("passed") is not True
        or stored_metrics.get("numerical_thresholds_satisfied") is not True
        or int(stored_metrics.get("sample_count", -1)) != sample_count
        or not np.isclose(
            float(stored_metrics.get("max_readout_error_v", math.inf)),
            max_readout_error_v,
            rtol=0.0,
            atol=1.0e-15,
        )
        or not np.isclose(
            float(stored_metrics.get("max_summary_relative_error", math.inf)),
            max_summary_relative_error,
            rtol=0.0,
            atol=1.0e-15,
        )
    ):
        raise FixtureIntegrityError("browser parity stored metrics do not recompute")
    return EvaluatorResult(
        values=(
            MetricValue("max_readout_error_v", max_readout_error_v),
            MetricValue(
                "max_summary_relative_error", max_summary_relative_error
            ),
        )
    )


def _flybody_worker_available() -> bool:
    from .flybody_adapter import dependency_versions

    versions = dependency_versions()
    return versions.get("flygym") == "2.1.0" and bool(
        versions.get("mujoco") and str(versions["mujoco"]).startswith("3.9.")
    )


def evaluate_flybody_smoke(case: BenchmarkCase) -> EvaluatorResult:
    """Execute the pinned worker when available, otherwise report capability block."""

    fixture_path = _verify_fixture(case)
    if not _flybody_worker_available():
        return EvaluatorResult.blocked(
            "the checked-in FlyBody fixture digest is verified, but flygym==2.1.0 "
            "and MuJoCo 3.9.x are not installed on this host; run this case in the "
            "pinned dedicated worker image"
        )

    from .flybody_adapter import FlyBodyWorkerConfig, run_analytic_wingbeat_smoke

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    reference = fixture["summary"]
    comparison = fixture["comparison"]
    trace = run_analytic_wingbeat_smoke(
        duration_s=0.005,
        wingbeat_frequency_hz=218.0,
        config=FlyBodyWorkerConfig(timestep_s=5.0e-5),
    )
    timestep_s = 5.0e-5
    held_samples = int(round(trace.metadata["control_timestep_s"] / timestep_s))
    first_hold = trace.actuator_torque_n_m[1 : held_samples + 1]
    actual = {
        "actuation_hold": trace.metadata["actuation_hold"],
        "actuation_source": trace.metadata["actuation_source"],
        "air_unit_correction_applied": trace.metadata[
            "flygym_2_1_air_unit_correction_applied"
        ],
        "authority_notice": (
            "This compatibility smoke proves that the pinned FlyBody worker "
            "executes. The analytic wingbeat is upstream's fallback formula, "
            "not the released learned straight-flight/saccade policy and not a "
            "validated neuromuscular result."
        ),
        "body_state_reference": trace.metadata["body_state_reference"],
        "calibration": "none",
        "compiled_model_fingerprint": trace.metadata[
            "compiled_model_fingerprint"
        ],
        "control_phase_semantics": trace.metadata[
            "control_phase_semantics"
        ],
        "control_timestep_s": trace.metadata["control_timestep_s"],
        "dependency_record_sha256": trace.metadata[
            "dependency_record_sha256"
        ],
        "final_body_position_m": trace.body_position_m[-1].tolist(),
        "final_body_quaternion_wxyz": trace.body_quaternion_wxyz[-1].tolist(),
        "first_control_actuator_torque_n_m": first_hold[0].tolist(),
        "first_control_hold_max_delta_n_m": float(
            np.max(np.abs(first_hold - first_hold[0]))
        ),
        "first_control_hold_sample_count": held_samples,
        "fluid_geom_names": trace.metadata["fluid_geom_names"],
        "fluid_geoms_contact_disabled": trace.metadata[
            "fluid_geoms_contact_disabled"
        ],
        "ground_contact_topology": trace.metadata["ground_contact_topology"],
        "kind": "flybody_worker_compatibility_smoke",
        "maximum_actuator_torque_n_m": float(
            np.max(np.abs(trace.actuator_torque_n_m))
        ),
        "maximum_fluid_force_n": float(np.max(np.abs(trace.root_fluid_force_n))),
        "maximum_measured_wing_excursion_rad": float(
            np.max(np.ptp(trace.wing_angles_rad, axis=0))
        ),
        "mode": "analytic-wingbeat",
        "released_policy_topology_equivalent": trace.metadata[
            "released_policy_topology_equivalent"
        ],
        "root_fluid_force_sample_semantics": trace.metadata[
            "root_fluid_force_sample_semantics"
        ],
        "integrated_root_fluid_impulse_n_s": (
            timestep_s * np.sum(trace.root_fluid_force_n[1:], axis=0)
        ).tolist(),
        "steps": len(trace.time_s) - 1,
        "schema_version": "1.0.0",
        "status": "exploratory",
        "tendon_count": trace.metadata["tendon_count"],
        "timestep_s": timestep_s,
        "wing_dof_order": trace.metadata["wing_dof_order"],
        "waveform_semantics": trace.metadata["waveform_semantics"],
        "worker_versions": trace.metadata["worker_versions"],
    }
    normalized_error = 0.0
    for field in comparison["required_exact_summary_fields"]:
        if actual[field] != reference[field]:
            normalized_error = max(normalized_error, 2.0)
    for field, tolerance in comparison["numeric_summary_tolerances"].items():
        observed = np.asarray(actual[field], dtype=float)
        expected = np.asarray(reference[field], dtype=float)
        scale = float(tolerance["atol"]) + float(tolerance["rtol"]) * np.abs(
            expected
        )
        normalized_error = max(
            normalized_error,
            float(np.max(np.abs(observed - expected) / scale)),
        )
    return EvaluatorResult(
        values=(MetricValue("normalized_fixture_error", normalized_error),)
    )


def _open_loop_flybody_torque_program(
    duration_s: float,
    physics_timestep_s: float,
    *,
    control_timestep_s: float = 2.0e-4,
    wingbeat_frequency_hz: float = 218.0,
) -> np.ndarray:
    """Return a deterministic, beat-periodic six-axis numerical stress input.

    The same command is held over each 0.2 ms control interval at every physics
    resolution.  Its amplitudes are intentionally modest and are not presented
    as muscle forces or as a released FlyBody controller.
    """

    steps_float = duration_s / physics_timestep_s
    n_steps = int(round(steps_float))
    control_ratio = control_timestep_s / physics_timestep_s
    steps_per_control = int(round(control_ratio))
    if n_steps < 1 or not np.isclose(steps_float, n_steps, rtol=0.0, atol=1e-9):
        raise ValueError("duration must be an integer multiple of physics timestep")
    if steps_per_control < 1 or not np.isclose(
        control_ratio, steps_per_control, rtol=0.0, atol=1e-9
    ):
        raise ValueError("control timestep must be an integer multiple of physics timestep")

    control_index = np.arange(n_steps, dtype=np.int64) // steps_per_control
    phase = np.mod(
        2.0
        * np.pi
        * wingbeat_frequency_hz
        * control_index.astype(float)
        * control_timestep_s,
        2.0 * np.pi,
    )
    per_side = np.column_stack(
        (
            8.0e-8 * np.sin(phase),
            2.0e-8 * np.sin(1.5 * phase),
            5.0e-8 * np.sin(phase + np.pi / 2.0),
        )
    )
    return np.concatenate((per_side, per_side), axis=1)


def _relative_vector_change(coarse: np.ndarray, fine: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(coarse, dtype=float) - np.asarray(fine, dtype=float))
        / max(np.linalg.norm(np.asarray(fine, dtype=float)), np.finfo(float).tiny)
    )


def _integrated_interval_fluid_impulse(trace: object) -> np.ndarray:
    """Integrate MuJoCo interval forces using their right-edge storage index."""

    time_s = np.asarray(trace.time_s, dtype=float)
    force_n = np.asarray(trace.root_fluid_force_n, dtype=float)
    if time_s.ndim != 1 or force_n.shape != (len(time_s), 3) or len(time_s) < 2:
        raise ValueError("FlyBody trace has invalid force/time shapes")
    intervals = np.diff(time_s)
    if not np.allclose(intervals, intervals[0], rtol=0.0, atol=1.0e-15):
        raise ValueError("FlyBody convergence trace must use a uniform physics step")
    return float(intervals[0]) * np.sum(force_n[1:], axis=0)


def _quaternion_geodesic_rad(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def _standardized_terminal_flybody_change(coarse: object, fine: object) -> float:
    """Compare free-body endpoints using declared, non-vanishing SI scales."""

    return max(
        float(
            np.linalg.norm(coarse.body_position_m[-1] - fine.body_position_m[-1])
            / 2.5e-3
        ),
        float(
            np.linalg.norm(
                coarse.body_linear_velocity_m_s[-1]
                - fine.body_linear_velocity_m_s[-1]
            )
            / 1.0
        ),
        float(
            np.linalg.norm(
                coarse.body_angular_velocity_rad_s[-1]
                - fine.body_angular_velocity_rad_s[-1]
            )
            / 200.0
        ),
        _quaternion_geodesic_rad(
            coarse.body_quaternion_wxyz[-1], fine.body_quaternion_wxyz[-1]
        ),
    )


def evaluate_flybody_open_loop_convergence(_case: BenchmarkCase) -> EvaluatorResult:
    """Run a three-level FlyBody convergence stress test with fixed commands.

    This establishes numerical behavior of the reviewed adapter/model stack
    without conflating timestep refinement with a feedback controller.  It does
    not establish stable flight or biological accuracy.
    """

    if not _flybody_worker_available():
        return EvaluatorResult.blocked(
            "the three-level FlyBody numerical stress test requires "
            "flygym==2.1.0 and MuJoCo 3.9.x in the pinned dedicated worker"
        )

    from .flybody_adapter import FlyBodyWorkerConfig, run_torque_trace

    duration_s = 0.100
    timesteps_s = (1.0e-4, 5.0e-5, 2.5e-5)
    traces = []
    for timestep_s in timesteps_s:
        torque = _open_loop_flybody_torque_program(duration_s, timestep_s)
        traces.append(
            run_torque_trace(
                torque,
                config=FlyBodyWorkerConfig(
                    timestep_s=timestep_s,
                    spawn_height_m=0.100,
                ),
            )
        )

    impulses = [_integrated_interval_fluid_impulse(trace) for trace in traces]
    coarse_fine_impulse = _relative_vector_change(impulses[0], impulses[1])
    fine_ultrafine_impulse = _relative_vector_change(impulses[1], impulses[2])
    coarse_fine_body = _standardized_terminal_flybody_change(traces[0], traces[1])
    fine_ultrafine_body = _standardized_terminal_flybody_change(traces[1], traces[2])
    monotonicity_violation_count = float(
        int(fine_ultrafine_impulse > coarse_fine_impulse + 1.0e-12)
        + int(fine_ultrafine_body > coarse_fine_body + 1.0e-12)
    )
    minimum_height_m = float(
        min(np.min(trace.body_position_m[:, 2]) for trace in traces)
    )
    minimum_rms_fluid_force_n = float(
        min(
            np.sqrt(np.mean(np.sum(trace.root_fluid_force_n[1:] ** 2, axis=1)))
            for trace in traces
        )
    )
    minimum_wing_excursion_rad = float(
        min(
            np.max(np.ptp(trace.wing_angles_rad, axis=0))
            for trace in traces
        )
    )
    return EvaluatorResult(
        values=(
            MetricValue(
                "coarse_fine_impulse_relative_change", coarse_fine_impulse
            ),
            MetricValue(
                "fine_ultrafine_impulse_relative_change", fine_ultrafine_impulse
            ),
            MetricValue("coarse_fine_body_state_change", coarse_fine_body),
            MetricValue("fine_ultrafine_body_state_change", fine_ultrafine_body),
            MetricValue(
                "convergence_monotonicity_violation_count",
                monotonicity_violation_count,
            ),
            MetricValue("minimum_body_height_m", minimum_height_m),
            MetricValue("minimum_rms_fluid_force_n", minimum_rms_fluid_force_n),
            MetricValue("minimum_wing_excursion_rad", minimum_wing_excursion_rad),
        )
    )


def evaluate_flybody_timestep_convergence(_case: BenchmarkCase) -> EvaluatorResult:
    """Execute paired 100 ms *stable-flight* rollouts in the dedicated worker.

    The upstream analytic wing-pattern fallback is deliberately excluded: the
    FlyBody publication states that this open-loop pattern cannot sustain a
    stable hover and requires the released policy for stabilization.  Comparing
    its rapidly diverging free-body endpoints would conflate controller
    instability with integrator convergence.
    """

    if not _flybody_worker_available():
        return EvaluatorResult.blocked(
            "paired authoritative FlyBody rollouts require flygym==2.1.0 and "
            "MuJoCo 3.9.x in the pinned dedicated worker; reduced-order physics "
            "cannot satisfy this claim"
        )
    from .flybody_ordinary_flight_release import (
        OrdinaryFlightReleaseError,
        default_expected_manifest_path,
        load_expected_manifest,
        load_runtime_audit,
        validate_staged_release,
    )

    staged_root_text = os.environ.get("FLY_S2B_FLYBODY_RELEASE_ROOT")
    audit_path_text = os.environ.get("FLY_S2B_FLYBODY_RELEASE_AUDIT")
    # A fixed impossible path is used when no stage was declared so the report
    # remains deterministic and never scans an arbitrary current directory.
    staged_root = Path(
        staged_root_text
        if staged_root_text is not None
        else "/__fly_s2b_official_flybody_release_not_staged__"
    )
    try:
        manifest = load_expected_manifest(default_expected_manifest_path())
        audit = (
            load_runtime_audit(Path(audit_path_text))
            if audit_path_text is not None
            else None
        )
        intake = validate_staged_release(manifest, staged_root, audit=audit)
    except OrdinaryFlightReleaseError as exc:
        return EvaluatorResult.blocked(
            "official ordinary-flight intake contract is invalid or unavailable: %s"
            % exc
        )
    if not intake.ready:
        codes = ", ".join(sorted(set(intake.blocker_codes)))
        return EvaluatorResult.blocked(
            "official FlyBody ordinary-flight intake is not ready (%s). Required "
            "sources are release archives 51196859 and 44815195; controller-reuse "
            "archive 51196886 is excluded. No standalone normalization artifact "
            "is assumed, and the analytic compatibility fallback cannot satisfy "
            "the stable-policy convergence claim" % codes
        )
    return EvaluatorResult.blocked(
        "official FlyBody ordinary-flight intake passed, but the sealed paired "
        "steady-flight/saccade 0.1/0.05 ms rollout and immutable metric artifact "
        "have not yet been executed by this evaluator"
    )


def _flybody_checkpoint_command(step: int) -> Any:
    """Return a deterministic, bounded six-axis checkpoint-test command."""

    from .flight.types import WingKinematics

    torque = np.array(
        [
            1.0e-8 * math.sin(0.31 * step),
            -0.7e-8 * math.cos(0.23 * step),
            0.4e-8 * math.sin(0.17 * step),
            -0.9e-8 * math.sin(0.29 * step),
            0.6e-8 * math.cos(0.19 * step),
            -0.3e-8 * math.sin(0.13 * step),
        ],
        dtype=float,
    )
    return WingKinematics(
        phase_rad=0.1 * step,
        frequency_hz=200.0,
        stroke_rad=np.zeros(2, dtype=float),
        stroke_velocity_rad_s=np.zeros(2, dtype=float),
        stroke_acceleration_rad_s2=np.zeros(2, dtype=float),
        angle_of_attack_rad=np.zeros(2, dtype=float),
        deviation_rad=np.zeros(2, dtype=float),
        generalized_torque_n_m=np.zeros(2, dtype=float),
        wing_axis_torque_n_m=torque,
    )


def _flybody_checkpoint_observation(adapter: Any, state: Any) -> np.ndarray:
    """Flatten every state/telemetry value exposed at the adapter boundary."""

    wing_angles, wing_velocities = adapter.wing_joint_state()
    wrench = adapter.aerodynamic_wrench()
    return np.concatenate(
        (
            state.position_world_m,
            state.velocity_world_m_s,
            state.quaternion_body_to_world,
            state.angular_velocity_body_rad_s,
            wing_angles,
            wing_velocities,
            adapter.last_actuator_torque_n_m,
            wrench.force_body_n,
            wrench.torque_body_n_m,
            wrench.left_force_body_n,
            wrench.right_force_body_n,
            [wrench.mechanical_power_w],
        )
    )


def evaluate_flybody_checkpoint_reentry(_case: BenchmarkCase) -> EvaluatorResult:
    """Require exact fresh-adapter continuation of MuJoCo integration state."""

    if not _flybody_worker_available():
        return EvaluatorResult.blocked(
            "FlyBody checkpoint re-entry requires flygym==2.1.0 and MuJoCo "
            "3.9.x in the pinned dedicated worker"
        )
    from .flybody_adapter import (
        FlyBodyPhysicsAdapter,
        FlyBodyPhysicsCheckpoint,
        FlyBodyWorkerConfig,
        WingAxisTorqueMap,
    )

    config = FlyBodyWorkerConfig(timestep_s=1.0e-4)
    first = FlyBodyPhysicsAdapter(WingAxisTorqueMap(), config)
    first.reset(first.default_initial_state())
    state = first.default_initial_state()
    for step in range(5):
        state = first.step(
            _flybody_checkpoint_command(step),
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
            config.timestep_s,
        )
    checkpoint_observation = _flybody_checkpoint_observation(first, state)
    checkpoint = first.checkpoint()
    expected_tail = []
    for step in range(5, 11):
        state = first.step(
            _flybody_checkpoint_command(step),
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
            config.timestep_s,
        )
        expected_tail.append(_flybody_checkpoint_observation(first, state))

    corrupt_rejection_count = 0
    corrupt = checkpoint.to_dict()
    corrupt["simulation_state_b64"] = "AAAA"
    try:
        FlyBodyPhysicsCheckpoint(corrupt)
    except ValueError:
        corrupt_rejection_count = 1

    resumed = FlyBodyPhysicsAdapter(WingAxisTorqueMap(), config)
    state = resumed.restore_checkpoint(checkpoint)
    restored_observation = _flybody_checkpoint_observation(resumed, state)

    # A producer can recompute a valid envelope digest, so model identity is
    # checked independently by the adapter before any state mutation.
    incompatible_payload = checkpoint.to_dict()
    incompatible_payload["compiled_model_sha256"] = "sha256:" + "0" * 64
    incompatible_payload.pop("payload_sha256")
    incompatible_payload["payload_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(
            incompatible_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    before_rejection = _flybody_checkpoint_observation(resumed, state)
    cross_model_rejection_count = 0
    try:
        resumed.restore_checkpoint(FlyBodyPhysicsCheckpoint(incompatible_payload))
    except ValueError:
        cross_model_rejection_count = 1
    after_rejection = _flybody_checkpoint_observation(resumed, state)

    observed_tail = []
    for step in range(5, 11):
        state = resumed.step(
            _flybody_checkpoint_command(step),
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
            config.timestep_s,
        )
        observed_tail.append(_flybody_checkpoint_observation(resumed, state))

    return EvaluatorResult(
        values=(
            MetricValue(
                "restored_checkpoint_max_abs_delta",
                _scaled_array_max_abs_delta(
                    checkpoint_observation, restored_observation
                ),
            ),
            MetricValue(
                "restored_tail_max_abs_delta",
                _scaled_array_max_abs_delta(
                    np.asarray(expected_tail), np.asarray(observed_tail)
                ),
            ),
            MetricValue(
                "corrupted_checkpoint_rejection_count",
                float(corrupt_rejection_count),
            ),
            MetricValue(
                "cross_model_checkpoint_rejection_count",
                float(cross_model_rejection_count),
            ),
            MetricValue(
                "rejected_restore_state_max_abs_delta",
                _scaled_array_max_abs_delta(before_rejection, after_rejection),
            ),
        )
    )


def _numeric_tree_values(value: Any) -> list[float]:
    """Flatten public numeric state without pretending text labels are scalars."""

    if value is None or isinstance(value, (str, bytes)):
        return []
    if dataclasses.is_dataclass(value):
        output: list[float] = []
        for item in dataclasses.fields(value):
            output.extend(_numeric_tree_values(getattr(value, item.name)))
        return output
    if isinstance(value, Mapping):
        output = []
        for key in sorted(value, key=lambda item: str(item)):
            output.extend(_numeric_tree_values(value[key]))
        return output
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float).reshape(-1).tolist()
    if isinstance(value, (tuple, list)):
        output = []
        for child in value:
            output.extend(_numeric_tree_values(child))
        return output
    if isinstance(value, (bool, np.bool_)):
        return [float(value)]
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [float(value)]
    return []


def _streaming_gate_sample(sample_index: int) -> Any:
    from .fly_fgs import FLY_FGS_NOD1_ROOT_IDS
    from .fly_fgs_runtime import FlyFGSCircuitSample, FlyFGSSceneBodyInput

    time_s = sample_index * 0.005
    control = FlyFGSSceneBodyInput(
        heading_rad=0.0,
        heading_velocity_rad_s=0.0,
        figure_world_azimuth_rad=0.0,
        figure_velocity_rad_s=0.0,
        ground_velocity_rad_s=0.0,
    )
    return FlyFGSCircuitSample(
        sample_index=sample_index,
        measurement_time_s=time_s,
        availability_time_s=time_s,
        nod1_voltage_v={root_id: -0.055 for root_id in FLY_FGS_NOD1_ROOT_IDS},
        pooled_readout={},
        last_control=control,
    )


def _streaming_gate_config() -> Any:
    from .flight.streaming_bridge import (
        StreamingBridgeConfig,
        default_streaming_motor_pathways,
    )

    # This manufactured high-drive case guarantees several phase crossings.
    # Its gains are an engineering stimulus, never a calibration claim.
    pathways = tuple(
        dataclasses.replace(pathway, functional_rate_gain=1.0)
        for pathway in default_streaming_motor_pathways()
    )
    return StreamingBridgeConfig(
        encoder_delay_s=0.0,
        vnc_delay_s=0.0,
        nmj_delay_s=0.0005,
        dn_functional_gain_hz=200.0,
        maximum_dn_rate_hz=200.0,
        maximum_motor_rate_hz=200.0,
        pathways=pathways,
    )


def _advance_streaming_gate_interval(
    bridge: Any,
    mechanics: Any,
    tick_index: int,
) -> tuple[Any, tuple[Any, ...], Any, float]:
    sample = _streaming_gate_sample(tick_index // 10) if tick_index % 10 == 0 else None
    projected_phase = mechanics.project_phase_end_unwrapped_rad()
    interval = bridge.begin_interval(circuit_sample=sample)
    mechanics_frames = mechanics.advance_bridge_interval(interval)
    phase_path = (
        interval.wing_phase_start_unwrapped_rad,
        *(frame.phase_end_unwrapped_rad for frame in mechanics_frames),
    )
    bridge_frame = bridge.end_interval(phase_path)
    return interval, mechanics_frames, bridge_frame, projected_phase


def _piecewise_crossing_oracle(
    bridge_frame: Any,
    event: Any,
) -> tuple[float, float]:
    """Independently locate one preferred-phase crossing on the six-point path."""

    two_pi = 2.0 * math.pi
    path = bridge_frame.wing_phase_path_unwrapped_rad
    # The registered six-point path is five exact 0.1 ms mechanics intervals.
    segment_dt_s = 0.0001
    candidates: list[tuple[float, float]] = []
    for segment_index, (phase_start, phase_end) in enumerate(zip(path, path[1:])):
        if phase_end <= phase_start + 1.0e-15:
            continue
        first_cycle = math.ceil(
            (phase_start - event.wingbeat_phase_rad) / two_pi - 1.0e-12
        )
        for cycle in range(first_cycle, first_cycle + 3):
            target_phase = event.wingbeat_phase_rad + cycle * two_pi
            if target_phase < phase_start - 1.0e-12:
                continue
            if target_phase >= phase_end - 1.0e-12:
                break
            fraction = (target_phase - phase_start) / (phase_end - phase_start)
            piecewise_time = (
                bridge_frame.interval_start_s
                + segment_index * segment_dt_s
                + fraction * segment_dt_s
            )
            total_delta = path[-1] - path[0]
            linear_time = bridge_frame.interval_start_s
            if total_delta > 0.0:
                linear_time += (
                    (target_phase - path[0]) / total_delta
                    * (bridge_frame.interval_end_s - bridge_frame.interval_start_s)
                )
            candidates.append((piecewise_time, linear_time))
    if not candidates:
        return _COMPARISON_SHAPE_MISMATCH, _COMPARISON_SHAPE_MISMATCH
    return min(candidates, key=lambda item: abs(item[0] - event.event_time_s))


def evaluate_streaming_causal_neuromuscular_runtime(
    case: BenchmarkCase,
) -> EvaluatorResult:
    """Exercise the online bridge/mechanics causal and re-entry contracts."""

    from .flight.streaming_bridge import (
        StreamingBridgeCheckpoint,
        StreamingNOD1MotorBridge,
    )
    from .flight.streaming_mechanics import (
        StreamingMechanicsCheckpoint,
        StreamingMuscleWingStepper,
    )

    config = _streaming_gate_config()
    seed = 73
    bridge = StreamingNOD1MotorBridge(config, seed=seed)
    mechanics = StreamingMuscleWingStepper()
    records: list[tuple[Any, tuple[Any, ...], Any, float]] = []

    strict_receipt_rejection_count = 0
    canonical_sample_rejection_count = 0
    bridge_checkpoint = None
    mechanics_checkpoint = None
    checkpoint_tick = 7
    for tick_index in range(16):
        if tick_index == 10:
            before = bridge.checkpoint().to_dict()
            try:
                bridge.begin_interval(circuit_sample=_streaming_gate_sample(2))
            except ValueError:
                if bridge.checkpoint().to_dict() == before:
                    canonical_sample_rejection_count = 1
        interval, mechanics_frames, bridge_frame, projected = (
            _advance_streaming_gate_interval(bridge, mechanics, tick_index)
        )
        if tick_index == 0:
            receipt_tester = StreamingMuscleWingStepper()
            try:
                receipt_tester.advance_bridge_interval(
                    dataclasses.replace(interval, tick_index=1)
                )
            except ValueError:
                strict_receipt_rejection_count = 1
        records.append((interval, mechanics_frames, bridge_frame, projected))
        if tick_index + 1 == checkpoint_tick:
            bridge_checkpoint = bridge.checkpoint()
            mechanics_checkpoint = mechanics.checkpoint()

    assert bridge_checkpoint is not None and mechanics_checkpoint is not None
    expected_tail = records[checkpoint_tick:]
    expected_final_bridge = bridge.checkpoint().to_dict()
    expected_final_mechanics = mechanics.checkpoint().to_dict()

    resumed_bridge = StreamingNOD1MotorBridge(config, seed=seed)
    resumed_bridge.restore(bridge_checkpoint)
    resumed_mechanics = StreamingMuscleWingStepper.from_checkpoint(
        mechanics_checkpoint
    )
    observed_tail = [
        _advance_streaming_gate_interval(resumed_bridge, resumed_mechanics, tick)
        for tick in range(checkpoint_tick, 16)
    ]
    checkpoint_tail_delta = _scaled_array_max_abs_delta(
        _numeric_tree_values(expected_tail),
        _numeric_tree_values(observed_tail),
    )
    checkpoint_digest_match = float(
        resumed_bridge.checkpoint().to_dict() == expected_final_bridge
        and resumed_mechanics.checkpoint().to_dict() == expected_final_mechanics
    )

    corrupt_checkpoint_rejection_count = 0
    corrupted_bridge = json.loads(json.dumps(bridge_checkpoint.to_dict()))
    corrupted_bridge["state"]["tick_index"] += 1
    try:
        StreamingBridgeCheckpoint(corrupted_bridge)
    except ValueError:
        corrupt_checkpoint_rejection_count += 1
    corrupted_mechanics = json.loads(json.dumps(mechanics_checkpoint.to_dict()))
    corrupted_mechanics["state"]["tick_index"] += 1
    try:
        StreamingMechanicsCheckpoint(corrupted_mechanics)
    except ValueError:
        corrupt_checkpoint_rejection_count += 1

    clock_violations = 0
    same_interval_future_effect_count = 0
    receipt_availability_violations = 0
    piecewise_errors: list[float] = []
    nontrivial_piecewise_count = 0
    generated_tick: dict[str, int] = {}
    for tick_index, (interval, mechanics_frames, bridge_frame, projected) in enumerate(
        records
    ):
        expected_start = tick_index * 0.0005
        clock_violations += int(
            interval.tick_index != tick_index
            or bridge_frame.tick_index != tick_index
            or abs(interval.interval_start_s - expected_start) > 1.0e-12
            or abs(interval.interval_end_s - expected_start - 0.0005) > 1.0e-12
            or abs(bridge_frame.interval_start_s - expected_start) > 1.0e-12
            or len(mechanics_frames) != 5
            or len(bridge_frame.wing_phase_path_unwrapped_rad) != 6
            or abs(bridge_frame.wing_phase_end_unwrapped_rad - projected) > 1.0e-12
        )
        applied_ids = {
            event_id for frame in mechanics_frames for event_id in frame.applied_event_ids
        }
        current_generated_ids = {event.event_id for event in bridge_frame.generated_events}
        same_interval_future_effect_count += len(current_generated_ids & applied_ids)
        for frame_index, frame in enumerate(mechanics_frames):
            expected_physics_tick = tick_index * 5 + frame_index
            expected_physics_start = expected_physics_tick * 0.0001
            clock_violations += int(
                frame.tick_index != expected_physics_tick
                or abs(frame.interval_start_s - expected_physics_start) > 1.0e-12
                or abs(frame.interval_end_s - expected_physics_start - 0.0001)
                > 1.0e-12
            )
        for event in interval.delivered_events:
            containing_frames = [
                frame
                for frame in mechanics_frames
                if frame.interval_start_s - 1.0e-12
                <= event.availability_time_s
                < frame.interval_end_s - 1.0e-12
            ]
            receipt_availability_violations += int(
                event.availability_time_s < interval.interval_start_s - 1.0e-12
                or event.availability_time_s >= interval.interval_end_s - 1.0e-12
                or generated_tick.get(event.event_id, tick_index) >= tick_index
                or event.event_id not in applied_ids
                or len(containing_frames) != 1
                or event.event_id not in containing_frames[0].applied_event_ids
                or event.source_measurement_time_s is None
                or event.source_measurement_time_s > event.event_time_s + 1.0e-12
            )
        for event in bridge_frame.generated_events:
            generated_tick[event.event_id] = tick_index
            piecewise_time, linear_time = _piecewise_crossing_oracle(
                bridge_frame, event
            )
            piecewise_errors.append(abs(piecewise_time - event.event_time_s))
            nontrivial_piecewise_count += int(
                abs(piecewise_time - linear_time) > 1.0e-15
            )
            receipt_availability_violations += int(
                abs(event.availability_time_s - event.event_time_s - 0.0005)
                > 1.0e-12
            )

    numeric_values = np.asarray(_numeric_tree_values(records), dtype=float)
    nonfinite_count = int(
        numeric_values.size - np.count_nonzero(np.isfinite(numeric_values))
    )
    return EvaluatorResult(
        values=(
            MetricValue("streaming_clock_violation_count", float(clock_violations)),
            MetricValue(
                "same_interval_future_effect_count",
                float(same_interval_future_effect_count),
            ),
            MetricValue(
                "piecewise_phase_crossing_max_abs_error_s",
                max(piecewise_errors, default=_COMPARISON_SHAPE_MISMATCH),
            ),
            MetricValue(
                "piecewise_nontrivial_crossing_count",
                float(nontrivial_piecewise_count),
            ),
            MetricValue(
                "receipt_availability_violation_count",
                float(receipt_availability_violations),
            ),
            MetricValue(
                "strict_receipt_rejection_count",
                float(strict_receipt_rejection_count),
            ),
            MetricValue(
                "canonical_sample_rejection_count",
                float(canonical_sample_rejection_count),
            ),
            MetricValue(
                "streaming_checkpoint_tail_max_abs_delta",
                checkpoint_tail_delta,
            ),
            MetricValue(
                "streaming_checkpoint_digest_match",
                checkpoint_digest_match,
            ),
            MetricValue(
                "streaming_corrupt_checkpoint_rejection_count",
                float(corrupt_checkpoint_rejection_count),
            ),
            MetricValue(
                "streaming_nonfinite_value_count",
                float(nonfinite_count),
            ),
        )
    )


def _muscle_gate_event(
    event_id: str,
    availability_time_s: float,
    *,
    raw_app_side: Any,
    muscle: str,
) -> Any:
    from .flight.streaming_bridge import StreamingMotorEvent
    from .schema import AnatomicalSide

    motor_by_muscle = {
        "iv2": "MN-iv2",
        "i1": "MN-i1",
        "iv1": "MN-iv1",
        "b3": "MN-b3",
    }
    phase_by_muscle = {
        "iv2": 0.20 * 2.0 * math.pi,
        "i1": 0.35 * 2.0 * math.pi,
        "iv1": 0.55 * 2.0 * math.pi,
        "b3": 0.75 * 2.0 * math.pi,
    }
    return StreamingMotorEvent(
        event_id=event_id,
        motor_neuron=motor_by_muscle[muscle],
        muscle=muscle,
        raw_app_side=raw_app_side,
        anatomical_side=AnatomicalSide.UNKNOWN,
        event_time_s=max(0.0, availability_time_s - 0.0005),
        availability_time_s=availability_time_s,
        wingbeat_phase_rad=phase_by_muscle[muscle],
        rate_hz=200.0,
        emission_probability=1.0,
        source_measurement_time_s=0.0,
        generator_seed=29,
        phase_crossing_index=0,
    )


def _non_target_muscle_values(snapshot: Any, target_key: str) -> list[float]:
    output: list[float] = []
    for key in sorted(snapshot.individual):
        if key != target_key:
            output.extend(_numeric_tree_values(snapshot.individual[key]))
    return output


def evaluate_muscle_force_stage_intervention_contract(
    case: BenchmarkCase,
) -> EvaluatorResult:
    """Exercise true force-stage SILENCE/SCALE semantics and re-entry."""

    from .flight.streaming_bridge import RawAppSide
    from .flight.streaming_mechanics import (
        MechanicsInterventionMode,
        StreamingMechanicsCheckpoint,
        StreamingMechanicsConfig,
        StreamingMuscleIntervention,
        StreamingMuscleWingStepper,
    )

    def intervention(
        intervention_id: str,
        *,
        start_s: float,
        end_s: float,
        mode: Any = MechanicsInterventionMode.SILENCE,
        output_scale: float = 0.0,
    ) -> Any:
        return StreamingMuscleIntervention(
            intervention_id=intervention_id,
            muscle="iv2",
            start_s=start_s,
            end_s=end_s,
            mode=mode,
            raw_app_side=RawAppSide.L,
            output_scale=output_scale,
        )

    grid_rejection_count = 0
    for start_s, end_s in ((0.00015, 0.0003), (0.0001, 0.00025)):
        try:
            intervention("off-grid-%d" % grid_rejection_count, start_s=start_s, end_s=end_s)
        except ValueError:
            grid_rejection_count += 1

    one_tick_silence = intervention(
        "one-tick-force-cut", start_s=0.0001, end_s=0.0002
    )
    silence_config = StreamingMechanicsConfig(
        muscle_interventions=(one_tick_silence,)
    )
    silence = StreamingMuscleWingStepper(silence_config)
    control = StreamingMuscleWingStepper()
    silence_events = (
        _muscle_gate_event(
            "preload", 0.0, raw_app_side=RawAppSide.L, muscle="iv2"
        ),
        _muscle_gate_event(
            "at-start", 0.0001, raw_app_side=RawAppSide.L, muscle="iv2"
        ),
        _muscle_gate_event(
            "at-end", 0.0002, raw_app_side=RawAppSide.L, muscle="iv2"
        ),
        _muscle_gate_event(
            "nontarget", 0.0, raw_app_side=RawAppSide.L, muscle="i1"
        ),
    )
    silence.push_delivered_events(silence_events)
    control.push_delivered_events(silence_events)
    silence_frames = tuple(silence.step() for _ in range(3))
    control_frames = tuple(control.step() for _ in range(3))
    target_key = "left:iv2"
    target_before = silence_frames[0].natural_muscle_snapshot.individual[target_key]
    target_active_natural = silence_frames[1].natural_muscle_snapshot.individual[
        target_key
    ]
    target_active_effective = silence_frames[1].actuation_muscle_snapshot.individual[
        target_key
    ]
    natural_decay_violation_count = int(
        not (
            target_before.force_n > target_active_natural.force_n > 0.0
            and target_before.activation > target_active_natural.activation > 0.0
        )
    )
    suppression_mismatch_count = int(
        silence_frames[1].active_intervention_ids != ("one-tick-force-cut",)
        or silence_frames[1].applied_event_ids != ()
        or silence_frames[1].suppressed_event_ids != ("at-start",)
        or "at-start" in silence.applied_event_ids
        or silence.suppressed_event_ids != ("at-start",)
    )
    half_open_boundary_violation_count = int(
        not one_tick_silence.active(0.0001)
        or one_tick_silence.active(0.0002)
        or silence_frames[2].active_intervention_ids != ()
        or silence_frames[2].applied_event_ids != ("at-end",)
        or silence_frames[1].muscle_snapshot.individual[target_key].force_n <= 0.0
    )
    nontarget_mismatch_count = 0
    for observed, expected in zip(silence_frames, control_frames):
        for snapshot_name in (
            "actuation_muscle_snapshot",
            "natural_muscle_snapshot",
            "muscle_snapshot",
        ):
            nontarget_mismatch_count += int(
                _scaled_array_max_abs_delta(
                    _non_target_muscle_values(
                        getattr(observed, snapshot_name), target_key
                    ),
                    _non_target_muscle_values(
                        getattr(expected, snapshot_name), target_key
                    ),
                )
                != 0.0
            )

    scale_contract = intervention(
        "quarter-force",
        start_s=0.0,
        end_s=0.0003,
        mode=MechanicsInterventionMode.SCALE,
        output_scale=0.25,
    )
    scaled = StreamingMuscleWingStepper(
        StreamingMechanicsConfig(muscle_interventions=(scale_contract,))
    )
    scale_control = StreamingMuscleWingStepper()
    scale_events = (
        _muscle_gate_event(
            "scale-target", 0.0, raw_app_side=RawAppSide.L, muscle="iv2"
        ),
        _muscle_gate_event(
            "scale-same-lane-control",
            0.0,
            raw_app_side=RawAppSide.L,
            muscle="i1",
        ),
        _muscle_gate_event(
            "scale-other-lane-control",
            0.0,
            raw_app_side=RawAppSide.R,
            muscle="iv2",
        ),
    )
    scaled.push_delivered_events(scale_events)
    scale_control.push_delivered_events(scale_events)
    scaled_frame = scaled.step()
    scale_control_frame = scale_control.step()
    scale_natural = scaled_frame.natural_muscle_snapshot.individual[target_key]
    scale_effective = scaled_frame.muscle_snapshot.individual[target_key]
    scale_ratio_error = max(
        abs(scale_effective.force_n / scale_natural.force_n - 0.25),
        abs(scale_effective.activation / scale_natural.activation - 0.25),
    )
    scale_hidden_state_mismatch_count = int(
        scaled_frame.natural_muscle_snapshot
        != scale_control_frame.natural_muscle_snapshot
    )
    for snapshot_name in (
        "actuation_muscle_snapshot",
        "natural_muscle_snapshot",
        "muscle_snapshot",
    ):
        nontarget_mismatch_count += int(
            _scaled_array_max_abs_delta(
                _non_target_muscle_values(
                    getattr(scaled_frame, snapshot_name), target_key
                ),
                _non_target_muscle_values(
                    getattr(scale_control_frame, snapshot_name), target_key
                ),
            )
            != 0.0
        )

    checkpoint_silence = intervention(
        "checkpoint-force-cut", start_s=0.0001, end_s=0.0006
    )
    checkpoint_config = StreamingMechanicsConfig(
        muscle_interventions=(checkpoint_silence,)
    )
    source = StreamingMuscleWingStepper(checkpoint_config)
    source.push_delivered_events(
        (
            _muscle_gate_event(
                "checkpoint-suppressed",
                0.0001,
                raw_app_side=RawAppSide.L,
                muscle="iv2",
            ),
            _muscle_gate_event(
                "checkpoint-nontarget",
                0.0002,
                raw_app_side=RawAppSide.L,
                muscle="i1",
            ),
            _muscle_gate_event(
                "checkpoint-after-cut",
                0.0006,
                raw_app_side=RawAppSide.L,
                muscle="iv2",
            ),
        )
    )
    for _ in range(3):
        source.step()
    checkpoint = source.checkpoint()
    expected_tail = tuple(source.step() for _ in range(5))
    expected_final = source.checkpoint().to_dict()
    resumed = StreamingMuscleWingStepper.from_checkpoint(checkpoint)
    observed_tail = tuple(resumed.step() for _ in range(5))
    checkpoint_tail_mismatch_count = int(
        _scaled_array_max_abs_delta(
            _numeric_tree_values(expected_tail), _numeric_tree_values(observed_tail)
        )
        != 0.0
    )
    checkpoint_digest_match = float(resumed.checkpoint().to_dict() == expected_final)

    rejection_target = StreamingMuscleWingStepper(checkpoint_config)
    pristine_target = rejection_target.checkpoint().to_dict()

    def semantically_rejected(payload: Mapping[str, Any]) -> bool:
        candidate = json.loads(json.dumps(payload))
        unsigned = {
            key: value for key, value in candidate.items() if key != "payload_sha256"
        }
        candidate["payload_sha256"] = _canonical_gate_sha256(unsigned)
        try:
            rejection_target.restore(StreamingMechanicsCheckpoint(candidate))
        except ValueError:
            return rejection_target.checkpoint().to_dict() == pristine_target
        return False

    bad_active = checkpoint.to_dict()
    bad_active["state"]["active_intervention_ids"] = []
    bad_contract = checkpoint.to_dict()
    bad_contract["config"]["muscle_interventions"][0]["end_s"] = 0.0005
    schedule_corruption_rejection_count = sum(
        (semantically_rejected(bad_active), semantically_rejected(bad_contract))
    )
    bad_ledger = checkpoint.to_dict()
    bad_ledger["state"]["applied_event_ids"] = bad_ledger["state"].pop(
        "suppressed_event_ids"
    )
    bad_ledger["state"]["applied_events"] = bad_ledger["state"].pop(
        "suppressed_events"
    )
    bad_ledger["state"]["suppressed_event_ids"] = []
    bad_ledger["state"]["suppressed_events"] = []
    ledger_corruption_rejection_count = int(semantically_rejected(bad_ledger))

    contract_receipt_match = float(
        len(one_tick_silence.contract_sha256) == 64
        and one_tick_silence.to_dict()["start_s"] == 0.0001
        and one_tick_silence.to_dict()["end_s"] == 0.0002
        and "anatomical_side" not in one_tick_silence.to_dict()
        and len(silence.intervention_schedule_sha256) == 64
    )
    all_frames = (
        *silence_frames,
        *control_frames,
        scaled_frame,
        scale_control_frame,
        *expected_tail,
        *observed_tail,
    )
    numeric = np.asarray(_numeric_tree_values(all_frames), dtype=float)
    nonfinite_count = int(numeric.size - np.count_nonzero(np.isfinite(numeric)))
    return EvaluatorResult(
        values=(
            MetricValue(
                "muscle_intervention_grid_rejection_count",
                float(grid_rejection_count),
            ),
            MetricValue(
                "muscle_intervention_contract_receipt_match",
                contract_receipt_match,
            ),
            MetricValue(
                "silence_onset_effective_force_n",
                float(target_active_effective.force_n),
            ),
            MetricValue(
                "silence_natural_decay_violation_count",
                float(natural_decay_violation_count),
            ),
            MetricValue(
                "silence_nmj_suppression_mismatch_count",
                float(suppression_mismatch_count),
            ),
            MetricValue(
                "silence_half_open_boundary_violation_count",
                float(half_open_boundary_violation_count),
            ),
            MetricValue("scale_target_fraction_max_abs_error", scale_ratio_error),
            MetricValue(
                "scale_hidden_state_mismatch_count",
                float(scale_hidden_state_mismatch_count),
            ),
            MetricValue(
                "intervention_nontarget_mismatch_count",
                float(nontarget_mismatch_count),
            ),
            MetricValue(
                "intervention_checkpoint_tail_mismatch_count",
                float(checkpoint_tail_mismatch_count),
            ),
            MetricValue(
                "intervention_checkpoint_digest_match", checkpoint_digest_match
            ),
            MetricValue(
                "intervention_schedule_corruption_rejection_count",
                float(schedule_corruption_rejection_count),
            ),
            MetricValue(
                "intervention_ledger_corruption_rejection_count",
                float(ledger_corruption_rejection_count),
            ),
            MetricValue(
                "intervention_nonfinite_value_count", float(nonfinite_count)
            ),
        )
    )


def _canonical_gate_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_gate_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_gate_json(value).encode("utf-8")).hexdigest()


def _canonical_gate_tree(value: Any) -> Any:
    """Convert a public result tree to strict, canonical-JSON values."""

    if dataclasses.is_dataclass(value):
        return {
            field.name: _canonical_gate_tree(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_gate_tree(child)
            for key, child in value.items()
        }
    if isinstance(value, np.ndarray):
        return _canonical_gate_tree(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_canonical_gate_tree(child) for child in value]
    return value


def _canonical_gate_tree_sha256(value: Any) -> str:
    return hashlib.sha256(
        _canonical_gate_json(_canonical_gate_tree(value)).encode("utf-8")
    ).hexdigest()


def _invoke_streaming_fresh_process_probe(
    probe_path: Path,
    repo_root: Path,
    request: Mapping[str, Any],
) -> tuple[int, int, Mapping[str, Any], str]:
    """Invoke one probe in a new interpreter and return its process receipt."""

    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    process = subprocess.Popen(
        (sys.executable, str(probe_path)),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=repo_root,
        env=environment,
    )
    process_id = process.pid
    try:
        stdout, stderr = process.communicate(
            _canonical_gate_json(request), timeout=90
        )
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise RuntimeError("fresh-process streaming probe timed out") from None
    if not stdout.strip():
        raise RuntimeError(
            "fresh-process streaming probe returned empty stdout: " + stderr.strip()
        )

    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"non-finite JSON token returned by probe: {token}")

    response = json.loads(stdout, parse_constant=reject_nonfinite)
    if not isinstance(response, Mapping):
        raise RuntimeError("fresh-process streaming probe response is not an object")
    return process_id, process.returncode, response, stderr


def _expected_streaming_fresh_process_receipt(
    repo_root: Path,
    probe_path: Path,
) -> Mapping[str, Any]:
    """Compute the probe receipt independently from checked-in source bytes."""

    from .flight.streaming_bridge import (
        STREAMING_BRIDGE_RUNTIME_VERSION,
        STREAMING_BRIDGE_SCHEMA_VERSION,
    )
    from .flight.streaming_mechanics import (
        STREAMING_MECHANICS_RUNTIME_VERSION,
        STREAMING_MECHANICS_SCHEMA_VERSION,
    )

    source_root = repo_root / "src" / "fly_sensor2behavior"
    module_paths = {
        "fly_fgs_runtime.py": source_root / "fly_fgs_runtime.py",
        "flight/hinge.py": source_root / "flight" / "hinge.py",
        "flight/muscles.py": source_root / "flight" / "muscles.py",
        "flight/streaming_bridge.py": source_root
        / "flight"
        / "streaming_bridge.py",
        "flight/streaming_mechanics.py": source_root
        / "flight"
        / "streaming_mechanics.py",
        "flight/types.py": source_root / "flight" / "types.py",
        "scripts/probe_streaming_fresh_process.py": probe_path,
    }
    module_sha256 = {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in sorted(module_paths.items())
    }
    return {
        "probe_schema_version": "1.0.0",
        "probe_kind": "manufactured_streaming_software_reentry",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_hexversion": sys.hexversion,
        "numpy_version": np.__version__,
        "bridge_schema_version": STREAMING_BRIDGE_SCHEMA_VERSION,
        "bridge_runtime_version": STREAMING_BRIDGE_RUNTIME_VERSION,
        "mechanics_schema_version": STREAMING_MECHANICS_SCHEMA_VERSION,
        "mechanics_runtime_version": STREAMING_MECHANICS_RUNTIME_VERSION,
        "module_sha256": module_sha256,
        "module_set_sha256": hashlib.sha256(
            _canonical_gate_json(module_sha256).encode("ascii")
        ).hexdigest(),
    }


def evaluate_streaming_fresh_process_checkpoint_reentry(
    case: BenchmarkCase,
) -> EvaluatorResult:
    """Prove exact streaming continuation across independent OS processes."""

    repo_root = Path(__file__).resolve().parents[2]
    probe_path = repo_root / "scripts" / "probe_streaming_fresh_process.py"
    if not probe_path.is_file():
        return EvaluatorResult.blocked(
            "fresh-process streaming probe is absent from the source checkout"
        )

    seed = 73021
    split_intervals = 31
    tail_intervals = 44
    baseline_record = _invoke_streaming_fresh_process_probe(
        probe_path,
        repo_root,
        {
            "operation": "run",
            "seed": seed,
            "interval_count": split_intervals + tail_intervals,
        },
    )
    prefix_record = _invoke_streaming_fresh_process_probe(
        probe_path,
        repo_root,
        {
            "operation": "run",
            "seed": seed,
            "interval_count": split_intervals,
        },
    )
    baseline = baseline_record[2]
    prefix = prefix_record[2]
    resume_record = _invoke_streaming_fresh_process_probe(
        probe_path,
        repo_root,
        {
            "operation": "resume",
            "seed": seed,
            "interval_count": tail_intervals,
            "bridge_checkpoint": prefix.get("bridge_checkpoint"),
            "mechanics_checkpoint": prefix.get("mechanics_checkpoint"),
            "expected_runtime_receipt": prefix.get("runtime_receipt"),
        },
    )
    resumed = resume_record[2]

    corrupt_bridge = json.loads(
        _canonical_gate_json(prefix.get("bridge_checkpoint"))
    )
    if isinstance(corrupt_bridge, Mapping):
        corrupt_bridge["state"]["tick_index"] += 1
    corrupt_record = _invoke_streaming_fresh_process_probe(
        probe_path,
        repo_root,
        {
            "operation": "resume",
            "seed": seed,
            "interval_count": 2,
            "bridge_checkpoint": corrupt_bridge,
            "mechanics_checkpoint": prefix.get("mechanics_checkpoint"),
            "expected_runtime_receipt": prefix.get("runtime_receipt"),
        },
    )
    corrupt_response = corrupt_record[2]

    wrong_receipt = json.loads(_canonical_gate_json(prefix.get("runtime_receipt")))
    if isinstance(wrong_receipt, Mapping):
        wrong_receipt["module_sha256"]["flight/streaming_bridge.py"] = "0" * 64
    receipt_record = _invoke_streaming_fresh_process_probe(
        probe_path,
        repo_root,
        {
            "operation": "resume",
            "seed": seed,
            "interval_count": 1,
            "bridge_checkpoint": prefix.get("bridge_checkpoint"),
            "mechanics_checkpoint": prefix.get("mechanics_checkpoint"),
            "expected_runtime_receipt": wrong_receipt,
        },
    )
    receipt_response = receipt_record[2]

    records = (
        baseline_record,
        prefix_record,
        resume_record,
        corrupt_record,
        receipt_record,
    )
    success_records = (baseline_record, prefix_record, resume_record)
    success_responses = (baseline, prefix, resumed)
    unique_pid_count = len({record[0] for record in records})
    success_count = sum(
        record[1] == 0 and record[2].get("ok") is True
        for record in success_records
    )

    expected_receipt = _expected_streaming_fresh_process_receipt(
        repo_root, probe_path
    )
    runtime_source_receipt_match = float(
        all(
            response.get("runtime_receipt") == expected_receipt
            for response in success_responses
        )
    )

    baseline_tail = baseline.get("tail", [])
    prefix_tail = prefix.get("tail", [])
    resumed_tail = resumed.get("tail", [])
    tail_exact_match = float(
        prefix_tail == baseline_tail[:split_intervals]
        and resumed_tail == baseline_tail[split_intervals:]
        and all(
            response.get("tail_sha256")
            == hashlib.sha256(
                _canonical_gate_json(response.get("tail", [])).encode("ascii")
            ).hexdigest()
            for response in success_responses
        )
    )
    transition_digest_mismatch_count = 0
    for response in success_responses:
        for item in response.get("tail", []):
            transition_digest_mismatch_count += int(
                item.get("transition_sha256")
                != hashlib.sha256(
                    _canonical_gate_json(item.get("transition")).encode("ascii")
                ).hexdigest()
            )

    final_bridge_checkpoint_match = float(
        baseline.get("bridge_checkpoint") == resumed.get("bridge_checkpoint")
    )
    final_mechanics_checkpoint_match = float(
        baseline.get("mechanics_checkpoint")
        == resumed.get("mechanics_checkpoint")
    )
    final_checkpoint_digest_match = 1.0
    for response in success_responses:
        for checkpoint_key, response_digest_key in (
            ("bridge_checkpoint", "bridge_checkpoint_sha256"),
            ("mechanics_checkpoint", "mechanics_checkpoint_sha256"),
        ):
            checkpoint = response.get(checkpoint_key, {})
            unsigned = {
                key: value
                for key, value in checkpoint.items()
                if key != "payload_sha256"
            }
            final_checkpoint_digest_match *= float(
                response.get(response_digest_key)
                == checkpoint.get("payload_sha256")
                == _canonical_gate_sha256(unsigned)
            )

    clock_violation_count = 0
    expected_runs = (
        (baseline, "run", 0, split_intervals + tail_intervals),
        (prefix, "run", 0, split_intervals),
        (resumed, "resume", split_intervals, tail_intervals),
    )
    for response, operation, start_tick, interval_count in expected_runs:
        final_tick = start_tick + interval_count
        tail = response.get("tail", [])
        clock_violation_count += int(
            response.get("start_operation") != operation
            or response.get("intervals_advanced") != interval_count
            or response.get("final_bridge_tick") != final_tick
            or response.get("final_mechanics_tick") != 5 * final_tick
            or len(tail) != interval_count
        )
        for offset, item in enumerate(tail):
            tick = start_tick + offset
            transition = item.get("transition", {})
            mechanics = transition.get("mechanics", [])
            clock_violation_count += int(
                item.get("bridge_tick") != tick
                or transition.get("bridge_tick") != tick
                or abs(transition.get("start_s", math.inf) - tick * 0.0005)
                > 1.0e-15
                or abs(
                    transition.get("end_s", math.inf) - (tick + 1) * 0.0005
                )
                > 1.0e-15
                or len(transition.get("phase_path_unwrapped_rad", [])) != 6
                or len(mechanics) != 5
            )
            for mechanics_offset, frame in enumerate(mechanics):
                clock_violation_count += int(
                    frame.get("tick") != 5 * tick + mechanics_offset
                )

    corrupt_checkpoint_rejection_count = int(
        corrupt_record[1] != 0
        and corrupt_response.get("ok") is False
        and corrupt_response.get("probe_schema_version") == "1.0.0"
        and corrupt_response.get("error_type") == "ValueError"
        and "SHA-256 mismatch" in corrupt_response.get("error", "")
    )
    receipt_mismatch_rejection_count = int(
        receipt_record[1] != 0
        and receipt_response.get("ok") is False
        and receipt_response.get("probe_schema_version") == "1.0.0"
        and receipt_response.get("error_type") == "ValueError"
        and receipt_response.get("error") == "runtime receipt mismatch"
    )
    scientific_status_match = float(
        all(
            response.get("probe_schema_version") == "1.0.0"
            and response.get("probe_kind")
            == "manufactured_streaming_software_reentry"
            and response.get("scientific_status")
            == "software_only_manufactured_physics"
            for response in success_responses
        )
    )
    numeric_values = np.asarray(
        _numeric_tree_values(success_responses), dtype=float
    )
    nonfinite_count = int(
        numeric_values.size - np.count_nonzero(np.isfinite(numeric_values))
    )

    return EvaluatorResult(
        values=(
            MetricValue(
                "fresh_process_unique_pid_count", float(unique_pid_count)
            ),
            MetricValue("fresh_process_success_count", float(success_count)),
            MetricValue(
                "fresh_process_runtime_source_receipt_match",
                runtime_source_receipt_match,
            ),
            MetricValue("fresh_process_tail_exact_match", tail_exact_match),
            MetricValue(
                "fresh_process_transition_digest_mismatch_count",
                float(transition_digest_mismatch_count),
            ),
            MetricValue(
                "fresh_process_final_bridge_checkpoint_match",
                final_bridge_checkpoint_match,
            ),
            MetricValue(
                "fresh_process_final_mechanics_checkpoint_match",
                final_mechanics_checkpoint_match,
            ),
            MetricValue(
                "fresh_process_final_checkpoint_digest_match",
                final_checkpoint_digest_match,
            ),
            MetricValue(
                "fresh_process_clock_violation_count",
                float(clock_violation_count),
            ),
            MetricValue(
                "fresh_process_corrupt_checkpoint_rejection_count",
                float(corrupt_checkpoint_rejection_count),
            ),
            MetricValue(
                "fresh_process_receipt_mismatch_rejection_count",
                float(receipt_mismatch_rejection_count),
            ),
            MetricValue(
                "fresh_process_scientific_status_match", scientific_status_match
            ),
            MetricValue(
                "fresh_process_nonfinite_value_count", float(nonfinite_count)
            ),
        )
    )


def evaluate_effector_raw_lane_physical_hypotheses(
    case: BenchmarkCase,
) -> EvaluatorResult:
    """Exercise both explicit raw-lane assignments without resolving anatomy."""

    from .flight.canonical_closed_loop import (
        CANONICAL_CLOSED_LOOP_RUNTIME_VERSION,
        CANONICAL_CLOSED_LOOP_SCHEMA_VERSION,
        CANONICAL_SOURCE_LIMITATIONS,
        CanonicalClosedLoopCheckpoint,
        CanonicalClosedLoopConfig,
        CanonicalClosedLoopSimulator,
    )
    from .flight.effector_mapping import (
        EFFECTOR_MAPPING_PROVENANCE,
        RAW_APP_LATERALITY_STATUS,
        RAW_L_TO_PHYSICAL_LEFT,
        RAW_L_TO_PHYSICAL_RIGHT,
        SIGNED_BEHAVIOR_CLAIM_POLICY,
        effector_mapping_receipt,
        map_raw_app_wing_kinematics_to_physical,
    )
    from .flight.types import WingKinematics

    two_lane_fields = (
        "stroke_rad",
        "stroke_velocity_rad_s",
        "stroke_acceleration_rad_s2",
        "angle_of_attack_rad",
        "deviation_rad",
        "generalized_torque_n_m",
    )
    raw = WingKinematics(
        phase_rad=1.25,
        frequency_hz=218.0,
        stroke_rad=np.array((1.0, -2.0)),
        stroke_velocity_rad_s=np.array((3.0, -5.0)),
        stroke_acceleration_rad_s2=np.array((7.0, -11.0)),
        angle_of_attack_rad=np.array((13.0, -17.0)),
        deviation_rad=np.array((19.0, -23.0)),
        generalized_torque_n_m=np.array((29.0, -31.0)),
        wing_axis_torque_n_m=np.array(
            (37.0, 41.0, 43.0, -47.0, -53.0, -59.0)
        ),
    )
    raw_before = {
        name: np.asarray(getattr(raw, name), dtype=float).copy()
        for name in (*two_lane_fields, "wing_axis_torque_n_m")
    }
    left = map_raw_app_wing_kinematics_to_physical(
        raw, RAW_L_TO_PHYSICAL_LEFT
    )
    right = map_raw_app_wing_kinematics_to_physical(
        raw, RAW_L_TO_PHYSICAL_RIGHT
    )

    two_lane_mismatch_count = int(
        left.phase_rad != raw.phase_rad
        or right.phase_rad != raw.phase_rad
        or left.frequency_hz != raw.frequency_hz
        or right.frequency_hz != raw.frequency_hz
    )
    mirror_mismatch_count = 0
    source_mutation_count = 0
    output_alias_count = 0
    for name in two_lane_fields:
        source = raw_before[name]
        two_lane_mismatch_count += int(
            not np.array_equal(getattr(left, name), source)
            or not np.array_equal(getattr(right, name), source[::-1])
        )
        mirror_mismatch_count += int(
            not np.array_equal(getattr(right, name), getattr(left, name)[::-1])
        )
        source_mutation_count += int(
            not np.array_equal(np.asarray(getattr(raw, name)), source)
        )
        output_alias_count += int(
            np.shares_memory(getattr(left, name), getattr(raw, name))
        )
        output_alias_count += int(
            np.shares_memory(getattr(right, name), getattr(raw, name))
        )
    axis = raw_before["wing_axis_torque_n_m"]
    six_axis_mismatch_count = int(
        not np.array_equal(left.wing_axis_torque_n_m, axis)
        or not np.array_equal(
            right.wing_axis_torque_n_m,
            np.concatenate((axis[3:], axis[:3])),
        )
    )
    mirror_mismatch_count += int(
        not np.array_equal(
            right.wing_axis_torque_n_m[:3], left.wing_axis_torque_n_m[3:]
        )
        or not np.array_equal(
            right.wing_axis_torque_n_m[3:], left.wing_axis_torque_n_m[:3]
        )
    )
    source_mutation_count += int(
        not np.array_equal(raw.wing_axis_torque_n_m, axis)
    )
    output_alias_count += int(
        np.shares_memory(left.wing_axis_torque_n_m, raw.wing_axis_torque_n_m)
    )
    output_alias_count += int(
        np.shares_memory(right.wing_axis_torque_n_m, raw.wing_axis_torque_n_m)
    )
    twice_swapped = map_raw_app_wing_kinematics_to_physical(
        right, RAW_L_TO_PHYSICAL_RIGHT
    )
    involution_mismatch_count = int(
        raw.phase_rad != twice_swapped.phase_rad
        or raw.frequency_hz != twice_swapped.frequency_hz
        or any(
            not np.array_equal(getattr(raw, name), getattr(twice_swapped, name))
            for name in (*two_lane_fields, "wing_axis_torque_n_m")
        )
    )
    without_axis = dataclasses.replace(raw, wing_axis_torque_n_m=None)
    optional_axis_absence_match = float(
        map_raw_app_wing_kinematics_to_physical(
            without_axis, RAW_L_TO_PHYSICAL_LEFT
        ).wing_axis_torque_n_m
        is None
        and map_raw_app_wing_kinematics_to_physical(
            without_axis, RAW_L_TO_PHYSICAL_RIGHT
        ).wing_axis_torque_n_m
        is None
    )

    left_receipt = effector_mapping_receipt(RAW_L_TO_PHYSICAL_LEFT)
    right_receipt = effector_mapping_receipt(RAW_L_TO_PHYSICAL_RIGHT)
    receipt_mismatch_count = int(
        left_receipt.anatomical_status != RAW_APP_LATERALITY_STATUS
        or right_receipt.anatomical_status != RAW_APP_LATERALITY_STATUS
        or left_receipt.provenance != EFFECTOR_MAPPING_PROVENANCE
        or right_receipt.provenance != EFFECTOR_MAPPING_PROVENANCE
        or left_receipt.signed_behavior_claim_policy
        != SIGNED_BEHAVIOR_CLAIM_POLICY
        or right_receipt.signed_behavior_claim_policy
        != SIGNED_BEHAVIOR_CLAIM_POLICY
        or left_receipt.hypothesis is not RAW_L_TO_PHYSICAL_LEFT
        or right_receipt.hypothesis is not RAW_L_TO_PHYSICAL_RIGHT
    )
    signed_claim_prohibition_preserved = float(
        "prohibited" in left_receipt.signed_behavior_claim_policy
        and "yaw and roll" in left_receipt.signed_behavior_claim_policy
        and "hypothesis" in left_receipt.provenance[-1]
    )

    left_config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=RAW_L_TO_PHYSICAL_LEFT,
        duration_s=0.001,
    )
    right_config = dataclasses.replace(
        left_config,
        effector_laterality_hypothesis=RAW_L_TO_PHYSICAL_RIGHT,
    )
    unsigned_checkpoint: dict[str, Any] = {
        "schema_version": CANONICAL_CLOSED_LOOP_SCHEMA_VERSION,
        "runtime_version": CANONICAL_CLOSED_LOOP_RUNTIME_VERSION,
        "config": dict(left_config.to_dict()),
        "components": {
            "circuit": {},
            "bridge": {},
            "mechanics": {},
            "physics": {},
        },
        "state": {
            "bridge_tick_index": 1,
            "physics_tick_index": 5,
            "circuit_sample_count": 1,
            "last_circuit_sample_index": 0,
            "body_state": {
                "position_world_m": [0.0, 0.0, 0.01],
                "velocity_world_m_s": [0.0, 0.0, 0.0],
                "quaternion_body_to_world": [1.0, 0.0, 0.0, 0.0],
                "angular_velocity_body_rad_s": [0.0, 0.0, 0.0],
            },
            "yaw_wrapped_rad": 0.0,
            "yaw_unwrapped_rad": 0.0,
            "figure_world_azimuth_rad": 0.0,
            "figure_velocity_rad_s": left_config.figure_velocity_rad_s,
            "ground_velocity_rad_s": left_config.ground_velocity_rad_s,
        },
        "receipts": {
            "backend_name": "manufactured-effector-contract",
            "aerodynamic_owner": "manufactured-effector-contract",
            "articulated_physics_telemetry_available": False,
            "effector_mapping": dict(left_receipt.to_dict()),
            "source_limitations": list(CANONICAL_SOURCE_LIMITATIONS),
        },
    }
    checkpoint = CanonicalClosedLoopCheckpoint(
        {
            **unsigned_checkpoint,
            "payload_sha256": _canonical_gate_sha256(unsigned_checkpoint),
        }
    )
    factory_calls: list[str] = []
    cross_hypothesis_rejection_count = 0
    try:
        CanonicalClosedLoopSimulator.from_checkpoint(
            checkpoint,
            right_config,
            circuit_factory=lambda: factory_calls.append("circuit"),
            physics_factory=lambda: factory_calls.append("physics"),
        )
    except ValueError:
        cross_hypothesis_rejection_count = 1

    tampered_receipt = checkpoint.to_dict()
    tampered_receipt["receipts"]["effector_mapping"] = dict(
        right_receipt.to_dict()
    )
    tampered_unsigned = {
        key: value
        for key, value in tampered_receipt.items()
        if key != "payload_sha256"
    }
    tampered_receipt["payload_sha256"] = _canonical_gate_sha256(
        tampered_unsigned
    )
    receipt_tamper_rejection_count = 0
    try:
        CanonicalClosedLoopCheckpoint(tampered_receipt)
    except ValueError:
        receipt_tamper_rejection_count = 1

    numeric = np.asarray(
        _numeric_tree_values((raw, left, right, twice_swapped)), dtype=float
    )
    nonfinite_count = int(numeric.size - np.count_nonzero(np.isfinite(numeric)))
    return EvaluatorResult(
        values=(
            MetricValue(
                "effector_two_lane_mapping_mismatch_count",
                float(two_lane_mismatch_count),
            ),
            MetricValue(
                "effector_six_axis_mapping_mismatch_count",
                float(six_axis_mismatch_count),
            ),
            MetricValue(
                "effector_hypothesis_mirror_mismatch_count",
                float(mirror_mismatch_count),
            ),
            MetricValue(
                "effector_mapping_involution_mismatch_count",
                float(involution_mismatch_count),
            ),
            MetricValue(
                "effector_source_mutation_count", float(source_mutation_count)
            ),
            MetricValue("effector_output_alias_count", float(output_alias_count)),
            MetricValue(
                "effector_optional_axis_absence_match", optional_axis_absence_match
            ),
            MetricValue(
                "effector_receipt_mismatch_count", float(receipt_mismatch_count)
            ),
            MetricValue(
                "effector_signed_claim_prohibition_preserved",
                signed_claim_prohibition_preserved,
            ),
            MetricValue(
                "effector_cross_hypothesis_checkpoint_rejection_count",
                float(cross_hypothesis_rejection_count),
            ),
            MetricValue(
                "effector_cross_hypothesis_factory_call_count",
                float(len(factory_calls)),
            ),
            MetricValue(
                "effector_receipt_tamper_rejection_count",
                float(receipt_tamper_rejection_count),
            ),
            MetricValue(
                "effector_nonfinite_value_count", float(nonfinite_count)
            ),
        )
    )


class _CanonicalGatePhysics:
    """Checkpointable manufactured physics used only to test orchestration.

    It intentionally makes no FlyBody, aerodynamic, or biological-accuracy
    claim.  Its articulated state is deterministic and sufficiently complete
    to prove that the canonical runner transports measured telemetry rather
    than reconstructing it from desired wing commands.
    """

    backend_name = "manufactured-canonical-gate"
    aerodynamic_owner = "manufactured-canonical-gate"
    wing_joint_order = (
        "left_stroke",
        "left_deviation",
        "left_angle_of_attack",
        "right_stroke",
        "right_deviation",
        "right_angle_of_attack",
    )

    def __init__(self) -> None:
        self._state = self.default_initial_state()
        self._yaw_rad = 0.0
        self._step_index = 0
        self._wing_position_rad = np.zeros(6, dtype=float)
        self._wing_velocity_rad_s = np.zeros(6, dtype=float)
        self.last_actuator_torque_n_m = np.zeros(6, dtype=float)
        self._wrench = self._make_wrench(self.last_actuator_torque_n_m)

    @staticmethod
    def default_initial_state() -> Any:
        from .flight.types import RigidBodyState

        return RigidBodyState(
            position_world_m=np.array((0.0, 0.0, 0.01), dtype=float),
            velocity_world_m_s=np.array((0.02, -0.01, 0.0), dtype=float),
            quaternion_body_to_world=np.array((1.0, 0.0, 0.0, 0.0), dtype=float),
            angular_velocity_body_rad_s=np.array((0.0, 0.0, 1.5), dtype=float),
        )

    @staticmethod
    def _make_wrench(axis_torque_n_m: Any) -> Any:
        from .flight.types import AerodynamicWrench

        axis = np.asarray(axis_torque_n_m, dtype=float)
        asymmetry = float(np.sum(axis[:3]) - np.sum(axis[3:]))
        return AerodynamicWrench(
            force_body_n=np.array((0.0, asymmetry * 1.0e-2, 1.0e-6)),
            torque_body_n_m=np.array((0.0, 0.0, asymmetry)),
            left_force_body_n=np.array((0.0, 0.0, 0.5e-6)),
            right_force_body_n=np.array((0.0, 0.0, 0.5e-6)),
            mechanical_power_w=float(np.sum(np.abs(axis))),
        )

    def reset(self, initial_state: Any) -> None:
        from .flight.types import RigidBodyState

        if not isinstance(initial_state, RigidBodyState):
            raise TypeError("initial_state must be RigidBodyState")
        initial_state.validate()
        self._state = initial_state.copy()
        quaternion = self._state.quaternion_body_to_world
        self._yaw_rad = math.atan2(
            2.0 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
            1.0 - 2.0 * (quaternion[2] ** 2 + quaternion[3] ** 2),
        )
        self._step_index = 0
        self._wing_position_rad = np.zeros(6, dtype=float)
        self._wing_velocity_rad_s = np.zeros(6, dtype=float)
        self.last_actuator_torque_n_m = np.zeros(6, dtype=float)
        self._wrench = self._make_wrench(self.last_actuator_torque_n_m)

    def step(
        self,
        wings: Any,
        force_body_n: Any,
        torque_body_n_m: Any,
        dt_s: float,
    ) -> Any:
        from .flight.types import WingKinematics

        if not isinstance(wings, WingKinematics):
            raise TypeError("wings must be WingKinematics")
        if float(dt_s) != 0.0001:
            raise ValueError("manufactured canonical physics requires 0.1 ms")
        if not np.array_equal(np.asarray(force_body_n, dtype=float), np.zeros(3)):
            raise ValueError("canonical runner must not inject an extra body force")
        if not np.array_equal(np.asarray(torque_body_n_m, dtype=float), np.zeros(3)):
            raise ValueError("canonical runner must not inject an extra body torque")
        axis = (
            np.zeros(6, dtype=float)
            if wings.wing_axis_torque_n_m is None
            else np.asarray(wings.wing_axis_torque_n_m, dtype=float).copy()
        )
        self.last_actuator_torque_n_m = axis
        self._wing_position_rad = np.array(
            (
                wings.stroke_rad[0],
                wings.deviation_rad[0],
                wings.angle_of_attack_rad[0],
                wings.stroke_rad[1],
                wings.deviation_rad[1],
                wings.angle_of_attack_rad[1],
            ),
            dtype=float,
        )
        self._wing_velocity_rad_s = np.array(
            (
                wings.stroke_velocity_rad_s[0],
                0.0,
                0.0,
                wings.stroke_velocity_rad_s[1],
                0.0,
                0.0,
            ),
            dtype=float,
        )
        self._state.position_world_m += self._state.velocity_world_m_s * dt_s
        self._yaw_rad += self._state.angular_velocity_body_rad_s[2] * dt_s
        self._state.quaternion_body_to_world = np.array(
            (
                math.cos(0.5 * self._yaw_rad),
                0.0,
                0.0,
                math.sin(0.5 * self._yaw_rad),
            ),
            dtype=float,
        )
        self._step_index += 1
        self._wrench = self._make_wrench(axis)
        return self._state.copy()

    def aerodynamic_wrench(self) -> Any:
        return self._make_wrench(self.last_actuator_torque_n_m)

    def wing_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self._wing_position_rad.copy(), self._wing_velocity_rad_s.copy()

    def whole_fly_com_position_m(self) -> np.ndarray:
        return self._state.position_world_m + np.array((1.0e-4, 0.0, 2.0e-4))

    @staticmethod
    def ground_contact_count() -> int:
        return 0

    def checkpoint(self) -> Mapping[str, Any]:
        unsigned = {
            "schema_version": "1.0.0",
            "backend_name": self.backend_name,
            "state": {
                "step_index": self._step_index,
                "yaw_rad": self._yaw_rad,
                "position_world_m": self._state.position_world_m.tolist(),
                "velocity_world_m_s": self._state.velocity_world_m_s.tolist(),
                "quaternion_body_to_world": self._state.quaternion_body_to_world.tolist(),
                "angular_velocity_body_rad_s": self._state.angular_velocity_body_rad_s.tolist(),
                "wing_position_rad": self._wing_position_rad.tolist(),
                "wing_velocity_rad_s": self._wing_velocity_rad_s.tolist(),
                "last_actuator_torque_n_m": self.last_actuator_torque_n_m.tolist(),
            },
        }
        return {**unsigned, "payload_sha256": _canonical_gate_sha256(unsigned)}

    def restore_checkpoint(self, checkpoint: Any) -> Any:
        from .flight.types import RigidBodyState

        copied = json.loads(_canonical_gate_json(checkpoint))
        if set(copied) != {
            "schema_version",
            "backend_name",
            "state",
            "payload_sha256",
        }:
            raise ValueError("manufactured physics checkpoint fields mismatch")
        if copied["schema_version"] != "1.0.0" or copied["backend_name"] != self.backend_name:
            raise ValueError("manufactured physics checkpoint identity mismatch")
        unsigned = {key: value for key, value in copied.items() if key != "payload_sha256"}
        if copied["payload_sha256"] != _canonical_gate_sha256(unsigned):
            raise ValueError("manufactured physics checkpoint digest mismatch")
        state = copied["state"]
        if set(state) != {
            "step_index",
            "yaw_rad",
            "position_world_m",
            "velocity_world_m_s",
            "quaternion_body_to_world",
            "angular_velocity_body_rad_s",
            "wing_position_rad",
            "wing_velocity_rad_s",
            "last_actuator_torque_n_m",
        }:
            raise ValueError("manufactured physics state fields mismatch")
        if (
            isinstance(state["step_index"], bool)
            or not isinstance(state["step_index"], int)
            or state["step_index"] < 0
        ):
            raise ValueError("manufactured physics step index is invalid")
        yaw = float(state["yaw_rad"])
        body = RigidBodyState(
            position_world_m=np.asarray(state["position_world_m"], dtype=float),
            velocity_world_m_s=np.asarray(state["velocity_world_m_s"], dtype=float),
            quaternion_body_to_world=np.asarray(
                state["quaternion_body_to_world"], dtype=float
            ),
            angular_velocity_body_rad_s=np.asarray(
                state["angular_velocity_body_rad_s"], dtype=float
            ),
        )
        wing_position = np.asarray(state["wing_position_rad"], dtype=float)
        wing_velocity = np.asarray(state["wing_velocity_rad_s"], dtype=float)
        torque = np.asarray(state["last_actuator_torque_n_m"], dtype=float)
        if (
            not math.isfinite(yaw)
            or wing_position.shape != (6,)
            or wing_velocity.shape != (6,)
            or torque.shape != (6,)
            or not np.all(np.isfinite(wing_position))
            or not np.all(np.isfinite(wing_velocity))
            or not np.all(np.isfinite(torque))
        ):
            raise ValueError("manufactured physics checkpoint contains invalid values")
        self._step_index = state["step_index"]
        self._yaw_rad = yaw
        self._state = body
        self._wing_position_rad = wing_position.copy()
        self._wing_velocity_rad_s = wing_velocity.copy()
        self.last_actuator_torque_n_m = torque.copy()
        self._wrench = self._make_wrench(torque)
        return self._state.copy()

    def close(self) -> None:
        return None


def _body_state_vector(state: Any) -> np.ndarray:
    return np.concatenate(
        (
            state.position_world_m,
            state.velocity_world_m_s,
            state.quaternion_body_to_world,
            state.angular_velocity_body_rad_s,
        )
    )


def evaluate_canonical_fly_fgs_closed_loop(case: BenchmarkCase) -> EvaluatorResult:
    """Run and re-enter the canonical live visual-to-physics state machine."""

    if shutil.which("node") is None:
        return EvaluatorResult.blocked(
            "the canonical fly-FGS closed-loop gate requires Node.js for the "
            "content-addressed circuit sidecar"
        )

    from .fly_fgs import FLY_FGS_SOURCE_MANIFEST_SHA256, FLY_FGS_SNAPSHOT_ID
    from .fly_fgs_runtime import NodeFlyFGSCircuitRuntime
    from .flight.canonical_closed_loop import (
        CANONICAL_SOURCE_LIMITATIONS,
        CanonicalClosedLoopCheckpoint,
        CanonicalClosedLoopConfig,
        CanonicalClosedLoopSimulator,
    )
    from .flight.effector_mapping import (
        RAW_APP_LATERALITY_STATUS,
        RAW_L_TO_PHYSICAL_LEFT,
        SIGNED_BEHAVIOR_CLAIM_POLICY,
        map_raw_app_wing_kinematics_to_physical,
    )
    from .flight.streaming_bridge import StreamingNOD1MotorBridge
    from .flight.streaming_mechanics import StreamingMuscleWingStepper

    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=RAW_L_TO_PHYSICAL_LEFT,
        duration_s=0.012,
        figure_velocity_rad_s=0.2,
        ground_velocity_rad_s=0.1,
        include_retinal_input=True,
        include_full_cell_state=False,
    )
    seed = 73

    def circuit_factory() -> Any:
        return NodeFlyFGSCircuitRuntime(request_timeout_s=180.0)

    def bridge_factory() -> Any:
        return StreamingNOD1MotorBridge(seed=seed)

    def mechanics_factory() -> Any:
        return StreamingMuscleWingStepper()

    reference = CanonicalClosedLoopSimulator(
        config,
        circuit_runtime=circuit_factory(),
        bridge=bridge_factory(),
        mechanics=mechanics_factory(),
        physics_adapter=_CanonicalGatePhysics(),
    )
    resumed = None
    try:
        reference.advance_intervals(13)
        checkpoint = reference.checkpoint()
        expected_tail = reference.advance_intervals(11)
        result = reference.result()
        expected_final_checkpoint = reference.checkpoint().to_dict()
        source_receipt = reference.circuit_runtime.ready_receipt

        resumed = CanonicalClosedLoopSimulator.from_checkpoint(
            checkpoint,
            config,
            circuit_factory=circuit_factory,
            physics_factory=_CanonicalGatePhysics,
            bridge_factory=bridge_factory,
            mechanics_factory=mechanics_factory,
            physics_checkpoint_loader=lambda value: value,
        )
        observed_tail = resumed.advance_intervals(11)
        observed_final_checkpoint = resumed.checkpoint().to_dict()
    finally:
        reference.close()
        if resumed is not None:
            resumed.close()

    tail_delta = _scaled_array_max_abs_delta(
        _numeric_tree_values(expected_tail), _numeric_tree_values(observed_tail)
    )
    digest_match = float(expected_final_checkpoint == observed_final_checkpoint)

    corrupt_rejection_count = 0
    corrupted = json.loads(json.dumps(checkpoint.to_dict()))
    corrupted["state"]["bridge_tick_index"] += 1
    try:
        CanonicalClosedLoopCheckpoint(corrupted)
    except ValueError:
        corrupt_rejection_count = 1

    cross_config_rejection_count = 0
    incompatible_config = dataclasses.replace(config, duration_s=0.0115)
    try:
        CanonicalClosedLoopSimulator.from_checkpoint(
            checkpoint,
            incompatible_config,
            circuit_factory=circuit_factory,
            physics_factory=_CanonicalGatePhysics,
            bridge_factory=bridge_factory,
            mechanics_factory=mechanics_factory,
            physics_checkpoint_loader=lambda value: value,
        )
    except ValueError:
        cross_config_rejection_count = 1

    clock_violations = 0
    feedback_mismatches = 0
    circuit_observations = 0
    previous_body = result.initial_body_state
    for tick_index, interval in enumerate(result.intervals):
        expected_start = tick_index * 0.0005
        clock_violations += int(
            interval.bridge_tick_index != tick_index
            or abs(interval.interval_start_s - expected_start) > 1.0e-12
            or abs(interval.interval_end_s - expected_start - 0.0005) > 1.0e-12
            or interval.bridge_start.tick_index != tick_index
            or interval.bridge.tick_index != tick_index
            or len(interval.mechanics) != 5
            or len(interval.physics) != 5
            or len(interval.bridge.wing_phase_path_unwrapped_rad) != 6
        )
        generated_ids = {event.event_id for event in interval.bridge.generated_events}
        applied_ids = {
            event_id
            for frame in interval.mechanics
            for event_id in frame.applied_event_ids
        }
        clock_violations += len(generated_ids & applied_ids)
        for step_index, (mechanics_frame, transition) in enumerate(
            zip(interval.mechanics, interval.physics)
        ):
            physics_tick = tick_index * 5 + step_index
            physics_start = physics_tick * 0.0001
            clock_violations += int(
                mechanics_frame.tick_index != physics_tick
                or transition.physics_tick_index != physics_tick
                or abs(mechanics_frame.interval_start_s - physics_start) > 1.0e-12
                or abs(transition.interval_start_s - physics_start) > 1.0e-12
                or abs(transition.interval_end_s - physics_start - 0.0001)
                > 1.0e-12
            )
            expected_physical = map_raw_app_wing_kinematics_to_physical(
                mechanics_frame.actuation_wing_kinematics,
                config.effector_laterality_hypothesis,
            )
            feedback_mismatches += int(
                _scaled_array_max_abs_delta(
                    _numeric_tree_values(expected_physical),
                    _numeric_tree_values(transition.physical_actuation_wing_kinematics),
                )
                != 0.0
            )
        observation = interval.circuit_observation
        if observation is not None:
            circuit_observations += 1
            expected_sample = tick_index // 10
            control = observation.requested_control
            feedback_mismatches += int(
                observation.bridge_tick_index != tick_index
                or observation.sample.sample_index != expected_sample
                or abs(observation.observation_time_s - expected_start) > 1.0e-12
                or abs(observation.sample.measurement_time_s - expected_start) > 1.0e-12
                or abs(observation.sample.availability_time_s - expected_start) > 1.0e-12
                or _scaled_array_max_abs_delta(
                    _body_state_vector(observation.body_state),
                    _body_state_vector(previous_body),
                )
                != 0.0
                or abs(control.heading_rad - observation.unwrapped_world_yaw_rad)
                > 1.0e-12
                or abs(
                    control.heading_velocity_rad_s
                    - observation.world_z_yaw_rate_rad_s
                )
                > 1.0e-12
                or abs(
                    control.figure_world_azimuth_rad
                    - (
                        config.figure_initial_world_azimuth_rad
                        + config.figure_velocity_rad_s * expected_start
                    )
                )
                > 1.0e-12
                or (expected_sample > 0 and observation.sample.last_control != control)
            )
        previous_body = interval.physics[-1].body_state_after

    source_receipt_match = float(
        source_receipt.get("snapshot_id") == FLY_FGS_SNAPSHOT_ID
        and source_receipt.get("source_manifest_sha256")
        == FLY_FGS_SOURCE_MANIFEST_SHA256
        and source_receipt.get("dt_s") == 0.005
        and source_receipt.get("sample_count") == 100
        and source_receipt.get("pre_roll_steps") == 48
        and source_receipt.get("full_cell_state_eligible_motor_input") is False
        and source_receipt.get("executable_asset_ids")
        == ["circuit_engine", "circuit_bundle"]
    )
    telemetry_complete = float(
        result.articulated_physics_telemetry_available
        and result.initial_articulated_physics_telemetry is not None
        and all(
            transition.articulated_telemetry_after is not None
            for interval in result.intervals
            for transition in interval.physics
        )
    )
    limitations_preserved = float(
        result.validation_status == "exploratory"
        and tuple(result.source_limitations) == CANONICAL_SOURCE_LIMITATIONS
        and result.effector_mapping_receipt.anatomical_status
        == RAW_APP_LATERALITY_STATUS
        and result.effector_mapping_receipt.signed_behavior_claim_policy
        == SIGNED_BEHAVIOR_CLAIM_POLICY
    )
    numeric = np.asarray(_numeric_tree_values(result), dtype=float)
    nonfinite_count = int(numeric.size - np.count_nonzero(np.isfinite(numeric)))

    return EvaluatorResult(
        values=(
            MetricValue("canonical_clock_violation_count", float(clock_violations)),
            MetricValue(
                "canonical_circuit_observation_count", float(circuit_observations)
            ),
            MetricValue(
                "canonical_feedback_mismatch_count", float(feedback_mismatches)
            ),
            MetricValue("canonical_source_receipt_match", source_receipt_match),
            MetricValue(
                "canonical_component_checkpoint_tail_max_abs_delta", tail_delta
            ),
            MetricValue(
                "canonical_component_checkpoint_digest_match", digest_match
            ),
            MetricValue(
                "canonical_corrupt_checkpoint_rejection_count",
                float(corrupt_rejection_count),
            ),
            MetricValue(
                "canonical_cross_config_rejection_count",
                float(cross_config_rejection_count),
            ),
            MetricValue(
                "canonical_nonfinite_telemetry_count", float(nonfinite_count)
            ),
            MetricValue(
                "canonical_articulated_telemetry_complete", telemetry_complete
            ),
            MetricValue(
                "canonical_scientific_limitations_preserved", limitations_preserved
            ),
        )
    )


def _load_canonical_online_native_pair_contract(
    scenario_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Load the exact paired native scenario and its referenced intervention."""

    scenario = json.loads(Path(scenario_path).read_text(encoding="utf-8"))
    expected_acceptance = {
        "baseline_checkpoint_reentry": (
            "fresh circuit, bridge, mechanics, and compiled FlyBody objects "
            "restored at 50 ms must reproduce the complete 50-100 ms tail and "
            "final composite checkpoint exactly"
        ),
        "causal_intervention": (
            "baseline and SILENCE arms must be numerically identical before 60 ms; "
            "targeted raw-app-L iv2 events become suppressed only while the "
            "force-stage interval is active; non-target delivered events remain applied"
        ),
        "physical_response": (
            "the intervention must produce nonzero measured-wing and final native "
            "root-state divergence after onset"
        ),
        "shared_receipts": (
            "both arms must use identical source, worker, compiled-model, clock, "
            "scene, seed, and effector-hypothesis receipts"
        ),
        "scientific_boundary": (
            "software integration only; no stable-flight, calibrated-muscle, "
            "anatomical-laterality, or signed yaw/roll claim"
        ),
    }
    expected_common = {
        "bridge_dt_s": 0.0005,
        "checkpoint_time_s": 0.05,
        "circuit_dt_s": 0.005,
        "duration_s": 0.1,
        "effector_laterality_hypothesis": "raw_l_to_physical_left",
        "figure_initial_world_azimuth_rad": 0.0,
        "figure_velocity_rad_s": math.pi / 6.0,
        "flybody_spawn_height_m": 0.1,
        "ground_velocity_rad_s": 0.0,
        "include_full_cell_state": False,
        "include_retinal_input": True,
        "physics_dt_s": 0.0001,
        "seed": 73,
    }
    expected_arms = [
        {
            "interventions": [],
            "label": "canonical online native baseline",
            "scenario_id": "canonical-online-native-baseline",
        },
        {
            "intervention_fixture_sha256": (
                "c99a59f33c763f4dd24728a69c910450811260b02c5983ed68f886d2fc7597ac"
            ),
            "intervention_fixture_uri": (
                "data/benchmarks/scenarios/"
                "canonical-online-raw-l-iv2-silence.v1.json"
            ),
            "label": "raw-app-L iv2 force-stage silence",
            "scenario_id": "canonical-online-native-raw-l-iv2-silence",
        },
    ]
    if (
        not isinstance(scenario, Mapping)
        or set(scenario)
        != {
            "acceptance_contract",
            "arms",
            "common_configuration",
            "scenario_family",
            "schema_version",
            "status",
        }
        or scenario.get("schema_version") != "1.0.0"
        or scenario.get("status") != "exploratory"
        or scenario.get("scenario_family")
        != "canonical_online_fly_fgs_to_native_flybody_paired_intervention"
        or scenario.get("acceptance_contract") != expected_acceptance
        or scenario.get("common_configuration") != expected_common
        or scenario.get("arms") != expected_arms
    ):
        raise FixtureIntegrityError(
            "canonical online native paired scenario contract is not exact"
        )

    relative = Path(expected_arms[1]["intervention_fixture_uri"])
    source_tree = Path(__file__).resolve().parents[2] / relative
    installed_relative = Path(*relative.parts[1:])
    installed = (
        Path(sysconfig.get_path("data"))
        / "share"
        / "fly-sensor2behavior"
        / installed_relative
    )
    intervention_path = next(
        (path for path in (source_tree, installed) if path.is_file()), None
    )
    if intervention_path is None:
        raise FixtureIntegrityError(
            "paired native intervention fixture is unavailable"
        )
    expected_digest = expected_arms[1]["intervention_fixture_sha256"]
    if sha256_file(intervention_path) != expected_digest:
        raise FixtureIntegrityError(
            "paired native intervention fixture digest mismatch"
        )
    intervention_payload = json.loads(
        intervention_path.read_text(encoding="utf-8")
    )
    expected_intervention = [
        {
            "end_s": 0.1,
            "intervention_id": "raw-l-iv2-silence-60-100ms",
            "mode": "silence",
            "muscle": "iv2",
            "output_scale": 0.0,
            "raw_app_side": "L",
            "start_s": 0.06,
        }
    ]
    if intervention_payload != expected_intervention:
        raise FixtureIntegrityError(
            "paired native intervention contract is not exact"
        )
    return scenario, expected_intervention[0]


def evaluate_canonical_online_fly_fgs_to_flybody(
    case: BenchmarkCase,
) -> EvaluatorResult:
    """Run the live canonical visual circuit through native FlyBody/MuJoCo."""

    scenario_path = _verify_fixture(case)
    scenario_contract, intervention_contract = (
        _load_canonical_online_native_pair_contract(scenario_path)
    )
    if shutil.which("node") is None:
        return EvaluatorResult.blocked(
            "the online canonical fly-FGS-to-FlyBody gate requires Node.js for "
            "the content-addressed live circuit sidecar"
        )
    if not _flybody_worker_available():
        return EvaluatorResult.blocked(
            "the online canonical fly-FGS-to-FlyBody gate requires flygym==2.1.0 "
            "and MuJoCo 3.9.x in the pinned dedicated worker; no reduced or "
            "manufactured physics fallback is permitted"
        )

    from .flybody_adapter import (
        FlyBodyPhysicsAdapter,
        FlyBodyPhysicsCheckpoint,
        FlyBodyWorkerConfig,
        WingAxisTorqueMap,
    )

    common = scenario_contract["common_configuration"]

    def physics_factory() -> Any:
        return FlyBodyPhysicsAdapter(
            WingAxisTorqueMap(),
            FlyBodyWorkerConfig(
                timestep_s=common["physics_dt_s"],
                spawn_height_m=common["flybody_spawn_height_m"],
            ),
        )

    return _evaluate_canonical_online_fly_fgs_to_flybody_with_factory(
        case,
        physics_factory=physics_factory,
        physics_checkpoint_loader=FlyBodyPhysicsCheckpoint,
        scenario_contract=scenario_contract,
        intervention_contract=intervention_contract,
    )


def _new_canonical_online_circuit_runtime() -> Any:
    """Construct the scheduled sidecar with its exact Node.js release pin."""

    from .fly_fgs_runtime import NodeFlyFGSCircuitRuntime

    return NodeFlyFGSCircuitRuntime(
        request_timeout_s=180.0,
        expected_node_version=_CANONICAL_ONLINE_PINNED_NODE_VERSION,
    )


def _online_native_source_receipt_matches(receipt: Mapping[str, Any]) -> bool:
    from .fly_fgs import FLY_FGS_SOURCE_MANIFEST_SHA256, FLY_FGS_SNAPSHOT_ID
    from .fly_fgs_runtime import FLY_FGS_RUNTIME_PROTOCOL_VERSION

    return bool(
        receipt.get("event") == "ready"
        and receipt.get("protocol_version") == FLY_FGS_RUNTIME_PROTOCOL_VERSION
        and receipt.get("node_version")
        == _CANONICAL_ONLINE_PINNED_NODE_VERSION
        and receipt.get("snapshot_id") == FLY_FGS_SNAPSHOT_ID
        and receipt.get("source_manifest_sha256")
        == FLY_FGS_SOURCE_MANIFEST_SHA256
        and isinstance(receipt.get("circuit_engine_sha256"), str)
        and len(receipt.get("circuit_engine_sha256", "")) == 64
        and isinstance(receipt.get("circuit_bundle_sha256"), str)
        and len(receipt.get("circuit_bundle_sha256", "")) == 64
        and receipt.get("dt_s") == 0.005
        and receipt.get("sample_count") == 100
        and receipt.get("pre_roll_steps") == 48
        and receipt.get("motor_input_policy")
        == "four individual NOD1 voltage channels only"
        and receipt.get("full_cell_state_eligible_motor_input") is False
        and receipt.get("executable_asset_ids")
        == ["circuit_engine", "circuit_bundle"]
    )


def _is_sha256_oci_digest(value: Any) -> bool:
    """Return whether ``value`` is one final lowercase OCI SHA-256 digest."""

    return bool(
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _online_native_compiled_receipt_matches(
    receipt: Mapping[str, Any],
    composite_checkpoint: Mapping[str, Any],
) -> bool:
    """Validate worker and compiled-model identity without trusting its label."""

    from .flybody_adapter import (
        PINNED_FLYGYM_VERSION,
        PINNED_MUJOCO_SERIES,
        WORKER_DEPENDENCY_LOCK_NAME,
        declared_worker_image_digest,
        dependency_record_fingerprints,
        dependency_versions,
        worker_dependency_lock_sha256,
    )
    from .schema import REVIEWED_FLYBODY_WING_AXIS_ORDER

    try:
        versions = dependency_versions()
        records = dependency_record_fingerprints()
        declared_image_digest = declared_worker_image_digest()
        model = receipt["compiled_model_fingerprint"]
        model_digest = model["sha256"]
        physics_checkpoint = composite_checkpoint["components"]["physics"]
        checkpoint_contract = receipt["checkpoint_contract"]
        worker_config = receipt["worker_config"]
        source = receipt["source"]
        worker_image_digest = receipt.get("worker_image_digest")
        return bool(
            receipt.get("schema_version") == "1.0.0"
            and receipt.get("engine") == "FlyGym/FlyBody with native MuJoCo"
            and receipt.get("worker_versions") == versions
            and versions.get("flygym") == PINNED_FLYGYM_VERSION
            and isinstance(versions.get("mujoco"), str)
            and versions["mujoco"].startswith(PINNED_MUJOCO_SERIES + ".")
            and receipt.get("worker_dependency_lock")
            == {
                "name": WORKER_DEPENDENCY_LOCK_NAME,
                "sha256": worker_dependency_lock_sha256(),
            }
            and receipt.get("dependency_record_sha256") == records
            and _is_sha256_oci_digest(worker_image_digest)
            and worker_image_digest == declared_image_digest
            and receipt.get("worker_image_digest_status")
            == "declared_oci_digest"
            and isinstance(model_digest, str)
            and model_digest.startswith("sha256:")
            and len(model_digest) == 71
            and physics_checkpoint.get("compiled_model_sha256") == model_digest
            and physics_checkpoint.get("worker_versions") == versions
            and physics_checkpoint.get("physics_timestep_s") == 0.0001
            and physics_checkpoint.get("wing_dof_order")
            == list(REVIEWED_FLYBODY_WING_AXIS_ORDER)
            and receipt.get("wing_dof_order")
            == list(REVIEWED_FLYBODY_WING_AXIS_ORDER)
            and worker_config
            == {
                "timestep_s": 0.0001,
                "spawn_height_m": 0.1,
                "max_abs_wing_torque_n_m": 3.0e-6,
                "add_aerodynamic_geoms": True,
            }
            and receipt.get("fluid_geoms_contact_disabled") is True
            and receipt.get("ground_contact_topology") == "legs_only"
            and receipt.get("released_policy_topology_equivalent") is False
            and receipt.get("tendon_count") == 8
            and source.get("flygym_tag") == "v2.1.0"
            and source.get("flygym_commit")
            == "ca65a510c2afe6ac61c51df4f274c8d190c2f95f"
            and checkpoint_contract
            == {
                "schema_version": "1.0.0",
                "state_spec": "mjSTATE_INTEGRATION",
                "float_encoding": "float64_le_base64",
                "compatibility_bound_to_compiled_model": True,
                "fresh_adapter_exact_reentry_required": True,
            }
            and receipt.get("public_units") == "SI"
            and receipt.get("root_fluid_wrench_scope")
            == "root-total; no reviewed per-wing decomposition"
        )
    except Exception:
        # Receipt acquisition is itself part of the hard gate. Missing package
        # metadata, malformed nested fields, or unreadable lock bytes all fail
        # closed as a non-matching compiled worker.
        return False


def _online_native_pre_onset_transition_applied_mismatch_count(
    baseline_mechanics_frame: Any,
    intervention_mechanics_frame: Any,
    baseline_transition: Any,
    intervention_transition: Any,
    *,
    onset_s: float,
) -> int:
    """Compare only commands and physics that were applied before onset.

    A mechanics frame owns the half-open interval ``[start, end)``.  Its
    ``actuation_*`` fields are the left-boundary command applied over that
    interval, while ``muscle_snapshot`` and ``wing_kinematics`` are endpoint
    state used to construct the *next* interval's command.  At an on-grid
    intervention onset, the endpoint effective state can therefore differ at
    ``end == onset`` without any pre-onset physical transition differing.

    This comparison deliberately includes the left-boundary muscle state, the
    wing command derived from it, and the complete resulting physics
    transition.  It deliberately excludes endpoint effective mechanics state.
    A transition whose left boundary is exactly the onset is intervention
    eligible under the declared half-open ``[onset, end)`` contract and is not
    a pre-onset identity observation.
    """

    try:
        baseline_start_s = float(baseline_transition.interval_start_s)
        intervention_start_s = float(intervention_transition.interval_start_s)
    except (AttributeError, TypeError, ValueError):
        return 1
    tolerance_s = 1.0e-12
    baseline_is_pre_onset = baseline_start_s < onset_s - tolerance_s
    intervention_is_pre_onset = intervention_start_s < onset_s - tolerance_s
    if not baseline_is_pre_onset and not intervention_is_pre_onset:
        return 0
    if (
        baseline_is_pre_onset != intervention_is_pre_onset
        or abs(baseline_start_s - intervention_start_s) > tolerance_s
    ):
        return 1
    baseline_applied = (
        baseline_mechanics_frame.actuation_muscle_snapshot,
        baseline_mechanics_frame.actuation_wing_kinematics,
        baseline_transition,
    )
    intervention_applied = (
        intervention_mechanics_frame.actuation_muscle_snapshot,
        intervention_mechanics_frame.actuation_wing_kinematics,
        intervention_transition,
    )
    return int(
        _scaled_array_max_abs_delta(
            _numeric_tree_values(baseline_applied),
            _numeric_tree_values(intervention_applied),
        )
        != 0.0
    )


def _online_native_event_to_next_wing_command_violation_count(
    mechanics_factory: Callable[[], Any],
) -> int:
    """Run an explicit on-grid event/no-event hinge counterfactual.

    The event is preloaded but becomes available exactly at the 0.1 ms right
    edge of the first frame.  Half-open semantics require it to be applied in
    the second frame, leave that frame's already-selected command unchanged,
    alter endpoint steering-muscle telemetry, and alter the command selected
    at the third frame's left boundary.  The final assertion is essential: a
    mutant that updates muscle telemetry while making the hinge ignore
    event-derived steering must fail this gate.
    """

    from .flight.streaming_bridge import RawAppSide, StreamingMotorEvent
    from .schema import AnatomicalSide

    event_id = "canonical-gate-on-grid-iv2-counterfactual"
    event = StreamingMotorEvent(
        event_id=event_id,
        motor_neuron="MN-iv2",
        muscle="iv2",
        raw_app_side=RawAppSide.L,
        anatomical_side=AnatomicalSide.UNKNOWN,
        event_time_s=0.0,
        availability_time_s=0.0001,
        wingbeat_phase_rad=0.20 * 2.0 * math.pi,
        rate_hz=200.0,
        emission_probability=1.0,
        source_measurement_time_s=0.0,
        generator_seed=73,
        phase_crossing_index=0,
    )
    try:
        event_path = mechanics_factory()
        no_event_path = mechanics_factory()
        event_path.push_delivered_events((event,))
        event_first = event_path.step()
        control_first = no_event_path.step()
        event_application = event_path.step()
        control_application = no_event_path.step()
        event_next = event_path.step()
        control_next = no_event_path.step()

        event_endpoint = event_application.natural_muscle_snapshot.individual[
            "left:iv2"
        ]
        control_endpoint = control_application.natural_muscle_snapshot.individual[
            "left:iv2"
        ]
        event_next_muscle = event_next.actuation_muscle_snapshot.individual[
            "left:iv2"
        ]
        control_next_muscle = control_next.actuation_muscle_snapshot.individual[
            "left:iv2"
        ]
        event_next_command = np.asarray(
            event_next.actuation_wing_kinematics.wing_axis_torque_n_m,
            dtype=float,
        )
        control_next_command = np.asarray(
            control_next.actuation_wing_kinematics.wing_axis_torque_n_m,
            dtype=float,
        )
        violations = 0
        violations += int(
            event_first.applied_event_ids != ()
            or event_first.suppressed_event_ids != ()
            or event_first.pending_event_count != 1
        )
        violations += int(
            event_application.applied_event_ids != (event_id,)
            or event_application.suppressed_event_ids != ()
        )
        # Neither the interval before availability nor the interval in which
        # the event is integrated may retroactively change its selected
        # left-boundary command.
        violations += int(
            _canonical_gate_tree_sha256(
                event_first.actuation_wing_kinematics
            )
            != _canonical_gate_tree_sha256(
                control_first.actuation_wing_kinematics
            )
        )
        violations += int(
            _canonical_gate_tree_sha256(
                event_application.actuation_wing_kinematics
            )
            != _canonical_gate_tree_sha256(
                control_application.actuation_wing_kinematics
            )
        )
        # The event must first be visible in muscle state, then propagate into
        # the command selected for the next physics interval.
        violations += int(
            event_endpoint.activation <= control_endpoint.activation
            or event_endpoint.force_n <= control_endpoint.force_n
            or event_next_muscle.activation <= control_next_muscle.activation
            or event_next_muscle.force_n <= control_next_muscle.force_n
        )
        violations += int(
            event_next_command.shape != (6,)
            or control_next_command.shape != (6,)
            or not np.all(np.isfinite(event_next_command))
            or not np.all(np.isfinite(control_next_command))
            or np.array_equal(event_next_command, control_next_command)
        )
        return violations
    except Exception:
        # Malformed or disconnected mechanics are a hard causal-contract
        # violation, not a reason to silently omit this counterfactual.
        return 1


def _evaluate_canonical_online_fly_fgs_to_flybody_with_factory(
    case: BenchmarkCase,
    *,
    physics_factory: Callable[[], Any],
    physics_checkpoint_loader: Callable[[Mapping[str, Any]], Any],
    scenario_contract: Mapping[str, Any],
    intervention_contract: Mapping[str, Any],
) -> EvaluatorResult:
    """Execute and audit the scheduled online native integration boundary."""

    del case
    from .flight.canonical_closed_loop import (
        CANONICAL_SOURCE_LIMITATIONS,
        CanonicalClosedLoopConfig,
        CanonicalClosedLoopSimulator,
    )
    from .flight.effector_mapping import (
        RAW_APP_LATERALITY_STATUS,
        RAW_L_TO_PHYSICAL_LEFT,
        SIGNED_BEHAVIOR_CLAIM_POLICY,
        map_raw_app_wing_kinematics_to_physical,
    )
    from .flight.streaming_bridge import (
        RawAppSide,
        StreamingNOD1MotorBridge,
        StreamingSignalSemantics,
    )
    from .flight.streaming_mechanics import (
        MechanicsInterventionMode,
        StreamingMechanicsConfig,
        StreamingMuscleIntervention,
        StreamingMuscleWingStepper,
    )

    common = scenario_contract["common_configuration"]
    duration_s = common["duration_s"]
    split_intervals = round(
        common["checkpoint_time_s"] / common["bridge_dt_s"]
    )
    total_intervals = round(duration_s / common["bridge_dt_s"])
    seed = common["seed"]
    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=RAW_L_TO_PHYSICAL_LEFT,
        duration_s=duration_s,
        circuit_dt_s=common["circuit_dt_s"],
        bridge_dt_s=common["bridge_dt_s"],
        physics_dt_s=common["physics_dt_s"],
        figure_initial_world_azimuth_rad=common[
            "figure_initial_world_azimuth_rad"
        ],
        figure_velocity_rad_s=common["figure_velocity_rad_s"],
        ground_velocity_rad_s=common["ground_velocity_rad_s"],
        include_retinal_input=common["include_retinal_input"],
        include_full_cell_state=common["include_full_cell_state"],
    )

    def circuit_factory() -> Any:
        return _new_canonical_online_circuit_runtime()

    def bridge_factory() -> Any:
        return StreamingNOD1MotorBridge(seed=seed)

    def mechanics_factory() -> Any:
        return StreamingMuscleWingStepper()

    silence = StreamingMuscleIntervention(
        intervention_id=intervention_contract["intervention_id"],
        muscle=intervention_contract["muscle"],
        start_s=intervention_contract["start_s"],
        end_s=intervention_contract["end_s"],
        mode=MechanicsInterventionMode.SILENCE,
        raw_app_side=RawAppSide.L,
        output_scale=intervention_contract["output_scale"],
    )

    def intervention_mechanics_factory() -> Any:
        return StreamingMuscleWingStepper(
            StreamingMechanicsConfig(muscle_interventions=(silence,))
        )

    reference = CanonicalClosedLoopSimulator(
        config,
        circuit_runtime=circuit_factory(),
        bridge=bridge_factory(),
        mechanics=mechanics_factory(),
        physics_adapter=physics_factory(),
    )
    resumed = None
    intervention = None
    try:
        reference.advance_intervals(split_intervals)
        checkpoint = reference.checkpoint()
        checkpoint_dict = checkpoint.to_dict()
        expected_tail = reference.advance_intervals(
            total_intervals - split_intervals
        )
        result = reference.result()
        expected_final_checkpoint = reference.checkpoint().to_dict()
        source_receipt = reference.circuit_runtime.ready_receipt
        physics_receipt = dict(reference.physics_adapter.provenance_metadata())

        resumed = CanonicalClosedLoopSimulator.from_checkpoint(
            checkpoint,
            config,
            circuit_factory=circuit_factory,
            physics_factory=physics_factory,
            bridge_factory=bridge_factory,
            mechanics_factory=mechanics_factory,
            physics_checkpoint_loader=physics_checkpoint_loader,
        )
        resumed_source_receipt = resumed.circuit_runtime.ready_receipt
        resumed_physics_receipt = dict(
            resumed.physics_adapter.provenance_metadata()
        )
        observed_tail = resumed.advance_intervals(
            total_intervals - split_intervals
        )
        observed_final_checkpoint = resumed.checkpoint().to_dict()

        intervention = CanonicalClosedLoopSimulator(
            config,
            circuit_runtime=circuit_factory(),
            bridge=bridge_factory(),
            mechanics=intervention_mechanics_factory(),
            physics_adapter=physics_factory(),
        )
        intervention_result = intervention.run()
        intervention_final_checkpoint = intervention.checkpoint().to_dict()
        intervention_source_receipt = intervention.circuit_runtime.ready_receipt
        intervention_physics_receipt = dict(
            intervention.physics_adapter.provenance_metadata()
        )
    finally:
        reference.close()
        if resumed is not None:
            resumed.close()
        if intervention is not None:
            intervention.close()

    clock_violations = 0
    feedback_mismatches = 0
    circuit_sample_count = 0
    physics_transition_count = 0
    six_axis_command_violations = 0
    generated_by_id: Dict[str, tuple[int, Any]] = {}
    delivered_ids: set[str] = set()
    applied_ids: set[str] = set()
    causal_event_violations = 0
    natural_response_count = 0
    effective_response_count = 0
    max_dn_rate_hz = 0.0
    max_mn_rate_hz = 0.0
    max_command_torque_n_m = 0.0
    max_actuator_torque_n_m = 0.0
    command_actuator_max_abs_delta_n_m = 0.0
    max_root_fluid_force_n = 0.0
    max_wing_excursion_rad = 0.0
    ground_contact_count = 0
    initial_telemetry = result.initial_articulated_physics_telemetry
    telemetry_complete = bool(
        result.backend_name == "flybody"
        and result.aerodynamic_owner == "flybody"
        and result.articulated_physics_telemetry_available
        and initial_telemetry is not None
    )
    if initial_telemetry is not None:
        initial_wing_position = initial_telemetry.measured_wing_position_rad
        initial_com_position = initial_telemetry.whole_fly_com_position_world_m
        ground_contact_count += initial_telemetry.ground_contact_count
    else:
        initial_wing_position = np.zeros(6, dtype=float)
        initial_com_position = np.zeros(3, dtype=float)
    final_com_position = initial_com_position
    previous_body_state = result.initial_body_state

    for tick_index, interval in enumerate(result.intervals):
        expected_start_s = tick_index * 0.0005
        clock_violations += int(
            interval.bridge_tick_index != tick_index
            or abs(interval.interval_start_s - expected_start_s) > 1.0e-12
            or abs(interval.interval_end_s - expected_start_s - 0.0005)
            > 1.0e-12
            or interval.bridge_start.tick_index != tick_index
            or interval.bridge.tick_index != tick_index
            or len(interval.mechanics) != 5
            or len(interval.physics) != 5
            or len(interval.bridge.wing_phase_path_unwrapped_rad) != 6
        )
        observation = interval.circuit_observation
        if observation is not None:
            circuit_sample_count += 1
            expected_sample_index = tick_index // 10
            clock_violations += int(
                tick_index % 10 != 0
                or observation.sample.sample_index != expected_sample_index
                or abs(observation.observation_time_s - expected_start_s)
                > 1.0e-12
                or abs(observation.sample.measurement_time_s - expected_start_s)
                > 1.0e-12
                or abs(observation.sample.availability_time_s - expected_start_s)
                > 1.0e-12
                or abs(
                    observation.requested_control.figure_world_azimuth_rad
                    - config.figure_velocity_rad_s * expected_start_s
                )
                > 1.0e-12
                or observation.requested_control.figure_velocity_rad_s
                != config.figure_velocity_rad_s
                or (expected_sample_index > 0 and observation.sample.last_control
                    != observation.requested_control)
            )
            feedback_mismatches += int(
                _scaled_array_max_abs_delta(
                    _body_state_vector(observation.body_state),
                    _body_state_vector(previous_body_state),
                )
                != 0.0
                or abs(
                    observation.requested_control.heading_rad
                    - observation.unwrapped_world_yaw_rad
                )
                > 1.0e-12
                or abs(
                    observation.requested_control.heading_velocity_rad_s
                    - observation.world_z_yaw_rate_rad_s
                )
                > 1.0e-12
            )
        elif tick_index % 10 == 0:
            clock_violations += 1

        max_dn_rate_hz = max(
            max_dn_rate_hz,
            *(sample.rate_hz for sample in interval.bridge.generated_dn_rates),
        )
        max_mn_rate_hz = max(
            max_mn_rate_hz,
            *(sample.rate_hz for sample in interval.bridge.generated_motor_rates),
        )
        causal_event_violations += sum(
            sample.semantics is not StreamingSignalSemantics.INFERRED_RATE
            for sample in (
                *interval.bridge.generated_dn_rates,
                *interval.bridge.generated_motor_rates,
            )
        )
        for event in interval.bridge.generated_events:
            causal_event_violations += int(
                event.event_id in generated_by_id
                or event.semantics
                is not StreamingSignalSemantics.SEEDED_SYNTHETIC
            )
            generated_by_id[event.event_id] = (tick_index, event)

        interval_delivered = {
            event.event_id: event for event in interval.bridge_start.delivered_events
        }
        causal_event_violations += int(
            tuple(interval.bridge_start.delivered_events)
            != tuple(interval.bridge.delivered_events)
        )
        causal_event_violations += len(
            set(interval_delivered).intersection(delivered_ids)
        )
        delivered_ids.update(interval_delivered)
        generated_this_interval = {
            event.event_id for event in interval.bridge.generated_events
        }

        for step_index, (mechanics_frame, transition) in enumerate(
            zip(interval.mechanics, interval.physics)
        ):
            physics_tick = tick_index * 5 + step_index
            expected_physics_start_s = physics_tick * 0.0001
            physics_transition_count += 1
            clock_violations += int(
                mechanics_frame.tick_index != physics_tick
                or transition.physics_tick_index != physics_tick
                or abs(
                    mechanics_frame.interval_start_s
                    - expected_physics_start_s
                )
                > 1.0e-12
                or abs(
                    transition.interval_start_s - expected_physics_start_s
                )
                > 1.0e-12
                or abs(
                    transition.interval_end_s
                    - expected_physics_start_s
                    - 0.0001
                )
                > 1.0e-12
            )
            physical = transition.physical_actuation_wing_kinematics
            expected_physical = map_raw_app_wing_kinematics_to_physical(
                mechanics_frame.actuation_wing_kinematics,
                config.effector_laterality_hypothesis,
            )
            command = physical.wing_axis_torque_n_m
            six_axis_command_violations += int(
                command is None
                or np.asarray(command).shape != (6,)
                or not np.all(np.isfinite(command))
                or _canonical_gate_tree_sha256(physical)
                != _canonical_gate_tree_sha256(expected_physical)
            )
            if command is not None and np.asarray(command).shape == (6,):
                max_command_torque_n_m = max(
                    max_command_torque_n_m,
                    float(np.max(np.abs(np.asarray(command, dtype=float)))),
                )

            telemetry = transition.articulated_telemetry_after
            telemetry_complete = telemetry_complete and telemetry is not None
            if telemetry is not None:
                six_axis_command_violations += int(
                    len(telemetry.wing_joint_order) != 6
                    or telemetry.actuator_torque_n_m.shape != (6,)
                    or telemetry.measured_wing_position_rad.shape != (6,)
                    or telemetry.measured_wing_velocity_rad_s.shape != (6,)
                )
                max_actuator_torque_n_m = max(
                    max_actuator_torque_n_m,
                    float(np.max(np.abs(telemetry.actuator_torque_n_m))),
                )
                if command is not None and np.asarray(command).shape == (6,):
                    command_actuator_max_abs_delta_n_m = max(
                        command_actuator_max_abs_delta_n_m,
                        float(
                            np.max(
                                np.abs(
                                    np.asarray(command, dtype=float)
                                    - telemetry.actuator_torque_n_m
                                )
                            )
                        ),
                    )
                max_wing_excursion_rad = max(
                    max_wing_excursion_rad,
                    float(
                        np.max(
                            np.abs(
                                telemetry.measured_wing_position_rad
                                - initial_wing_position
                            )
                        )
                    ),
                )
                ground_contact_count += telemetry.ground_contact_count
                final_com_position = telemetry.whole_fly_com_position_world_m
            wrench = transition.aerodynamic_wrench_after
            max_root_fluid_force_n = max(
                max_root_fluid_force_n,
                float(np.linalg.norm(wrench.force_body_n)),
            )

            for event_id in mechanics_frame.applied_event_ids:
                causal_event_violations += int(
                    event_id in applied_ids
                    or event_id not in interval_delivered
                    or event_id not in generated_by_id
                    or event_id in generated_this_interval
                )
                applied_ids.add(event_id)
                generated = generated_by_id.get(event_id)
                if generated is None:
                    continue
                generated_tick, event = generated
                causal_event_violations += int(
                    generated_tick >= tick_index
                    or event.availability_time_s
                    < mechanics_frame.interval_start_s - 1.0e-12
                    or event.availability_time_s
                    >= mechanics_frame.interval_end_s - 1.0e-12
                    or event.source_measurement_time_s is None
                    or event.source_measurement_time_s
                    > event.event_time_s + 1.0e-12
                )
                virtual_side = (
                    "left" if event.raw_app_side is RawAppSide.L else "right"
                )
                muscle_key = f"{virtual_side}:{event.muscle}"
                before = mechanics_frame.actuation_muscle_snapshot.individual.get(
                    muscle_key
                )
                natural = mechanics_frame.natural_muscle_snapshot.individual.get(
                    muscle_key
                )
                effective = mechanics_frame.muscle_snapshot.individual.get(
                    muscle_key
                )
                natural_response_count += int(
                    before is not None
                    and natural is not None
                    and natural.activation > before.activation
                    and natural.force_n > before.force_n
                )
                effective_response_count += int(
                    before is not None
                    and effective is not None
                    and effective.activation > before.activation
                    and effective.force_n > before.force_n
                )
            causal_event_violations += len(
                set(mechanics_frame.suppressed_event_ids)
            )
        previous_body_state = interval.physics[-1].body_state_after

    causal_event_violations += len(delivered_ids - applied_ids)
    paired_clock_violations = 0
    pre_onset_numeric_mismatch_count = 0
    intervention_pre_onset_generated_ids: list[str] = []
    intervention_delivered_by_id: Dict[str, Any] = {}
    intervention_delivered_duplicate_count = 0
    intervention_applied_count_by_id: Dict[str, int] = {}
    intervention_suppressed_count_by_id: Dict[str, int] = {}
    max_target_actuation_force_n = 0.0
    post_onset_measured_wing_delta_rad = 0.0
    intervention_physics_transition_count = 0
    intervention_circuit_sample_count = 0
    intervention_initial_telemetry = (
        intervention_result.initial_articulated_physics_telemetry
    )
    telemetry_complete = bool(
        telemetry_complete
        and intervention_result.backend_name == "flybody"
        and intervention_result.aerodynamic_owner == "flybody"
        and intervention_result.articulated_physics_telemetry_available
        and intervention_initial_telemetry is not None
    )
    if intervention_initial_telemetry is not None:
        ground_contact_count += intervention_initial_telemetry.ground_contact_count
    previous_intervention_body_state = intervention_result.initial_body_state

    for tick_index, (baseline_interval, intervention_interval) in enumerate(
        zip(result.intervals, intervention_result.intervals)
    ):
        expected_start_s = tick_index * 0.0005
        paired_clock_violations += int(
            intervention_interval.bridge_tick_index != tick_index
            or abs(intervention_interval.interval_start_s - expected_start_s)
            > 1.0e-12
            or abs(
                intervention_interval.interval_end_s
                - expected_start_s
                - 0.0005
            )
            > 1.0e-12
            or len(intervention_interval.mechanics) != 5
            or len(intervention_interval.physics) != 5
        )
        intervention_observation = intervention_interval.circuit_observation
        if intervention_observation is not None:
            intervention_circuit_sample_count += 1
            expected_sample_index = tick_index // 10
            paired_clock_violations += int(
                tick_index % 10 != 0
                or intervention_observation.sample.sample_index
                != expected_sample_index
                or _scaled_array_max_abs_delta(
                    _body_state_vector(intervention_observation.body_state),
                    _body_state_vector(previous_intervention_body_state),
                )
                != 0.0
                or abs(
                    intervention_observation.requested_control.heading_rad
                    - intervention_observation.unwrapped_world_yaw_rad
                )
                > 1.0e-12
                or abs(
                    intervention_observation.requested_control.heading_velocity_rad_s
                    - intervention_observation.world_z_yaw_rate_rad_s
                )
                > 1.0e-12
            )
        elif tick_index % 10 == 0:
            paired_clock_violations += 1
        intervention_pre_onset_generated_ids.extend(
            event.event_id
            for event in intervention_interval.bridge.generated_events
            if event.event_time_s < silence.start_s - 1.0e-12
        )
        for event in intervention_interval.bridge_start.delivered_events:
            intervention_delivered_duplicate_count += int(
                event.event_id in intervention_delivered_by_id
            )
            intervention_delivered_by_id[event.event_id] = event

        for step_index, (
            baseline_mechanics_frame,
            intervention_mechanics_frame,
            baseline_transition,
            intervention_transition,
        ) in enumerate(
            zip(
                baseline_interval.mechanics,
                intervention_interval.mechanics,
                baseline_interval.physics,
                intervention_interval.physics,
            )
        ):
            expected_physics_tick = tick_index * 5 + step_index
            paired_clock_violations += int(
                intervention_transition.physics_tick_index
                != expected_physics_tick
                or abs(
                    intervention_transition.interval_start_s
                    - expected_physics_tick * 0.0001
                )
                > 1.0e-12
                or abs(
                    intervention_transition.interval_end_s
                    - (expected_physics_tick + 1) * 0.0001
                )
                > 1.0e-12
            )
            intervention_physics_transition_count += 1
            pre_onset_numeric_mismatch_count += (
                _online_native_pre_onset_transition_applied_mismatch_count(
                    baseline_mechanics_frame,
                    intervention_mechanics_frame,
                    baseline_transition,
                    intervention_transition,
                    onset_s=silence.start_s,
                )
            )
            telemetry = intervention_transition.articulated_telemetry_after
            baseline_telemetry = baseline_transition.articulated_telemetry_after
            telemetry_complete = bool(
                telemetry_complete
                and telemetry is not None
                and baseline_telemetry is not None
            )
            if telemetry is not None:
                ground_contact_count += telemetry.ground_contact_count
            if (
                intervention_transition.interval_start_s
                >= silence.start_s - 1.0e-12
                and telemetry is not None
                and baseline_telemetry is not None
            ):
                post_onset_measured_wing_delta_rad = max(
                    post_onset_measured_wing_delta_rad,
                    float(
                        np.max(
                            np.abs(
                                telemetry.measured_wing_position_rad
                                - baseline_telemetry.measured_wing_position_rad
                            )
                        )
                    ),
                )

        for mechanics_frame in intervention_interval.mechanics:
            for event_id in mechanics_frame.applied_event_ids:
                intervention_applied_count_by_id[event_id] = (
                    intervention_applied_count_by_id.get(event_id, 0) + 1
                )
            for event_id in mechanics_frame.suppressed_event_ids:
                intervention_suppressed_count_by_id[event_id] = (
                    intervention_suppressed_count_by_id.get(event_id, 0) + 1
                )
            if (
                silence.start_s - 1.0e-12
                <= mechanics_frame.interval_start_s
                < silence.end_s - 1.0e-12
            ):
                target = (
                    mechanics_frame.actuation_muscle_snapshot.individual.get(
                        "left:iv2"
                    )
                )
                if target is None:
                    paired_clock_violations += 1
                else:
                    max_target_actuation_force_n = max(
                        max_target_actuation_force_n, target.force_n
                    )
        previous_intervention_body_state = (
            intervention_interval.physics[-1].body_state_after
        )

    causal_event_violations += (
        _online_native_event_to_next_wing_command_violation_count(
            mechanics_factory
        )
    )

    paired_clock_violations += int(
        len(intervention_result.intervals) != total_intervals
        or intervention_circuit_sample_count != 20
        or intervention_physics_transition_count != total_intervals * 5
    )

    baseline_pre_onset_generated_ids = [
        event.event_id
        for interval in result.intervals
        for event in interval.bridge.generated_events
        if event.event_time_s < silence.start_s - 1.0e-12
    ]
    pre_onset_generated_event_mismatch_count = int(
        baseline_pre_onset_generated_ids
        != intervention_pre_onset_generated_ids
    )
    target_event_ids = {
        event_id
        for event_id, event in intervention_delivered_by_id.items()
        if event.raw_app_side is RawAppSide.L
        and event.muscle == "iv2"
        and silence.start_s - 1.0e-12
        <= event.availability_time_s
        < silence.end_s - 1.0e-12
    }
    event_locality_mismatch_count = intervention_delivered_duplicate_count
    for event_id in set(intervention_applied_count_by_id).union(
        intervention_suppressed_count_by_id
    ):
        if event_id not in intervention_delivered_by_id:
            event_locality_mismatch_count += (
                intervention_applied_count_by_id.get(event_id, 0)
                + intervention_suppressed_count_by_id.get(event_id, 0)
            )
    for event_id, event in intervention_delivered_by_id.items():
        applied_count = intervention_applied_count_by_id.get(event_id, 0)
        suppressed_count = intervention_suppressed_count_by_id.get(event_id, 0)
        is_target = (
            event.raw_app_side is RawAppSide.L
            and event.muscle == "iv2"
            and silence.start_s - 1.0e-12
            <= event.availability_time_s
            < silence.end_s - 1.0e-12
        )
        event_locality_mismatch_count += int(
            (is_target and (applied_count != 0 or suppressed_count != 1))
            or (
                not is_target
                and (applied_count != 1 or suppressed_count != 0)
            )
        )
    target_suppressed_count = sum(
        intervention_suppressed_count_by_id.get(event_id, 0) == 1
        for event_id in target_event_ids
    )
    intervention_final_root_position_delta_m = float(
        np.linalg.norm(
            intervention_result.final_body_state.position_world_m
            - result.final_body_state.position_world_m
        )
    )
    source_receipt_match = float(
        source_receipt == resumed_source_receipt
        == intervention_source_receipt
        and _online_native_source_receipt_matches(source_receipt)
    )
    compiled_receipt_match = float(
        physics_receipt == resumed_physics_receipt
        == intervention_physics_receipt
        and _online_native_compiled_receipt_matches(
            physics_receipt, checkpoint_dict
        )
        and _online_native_compiled_receipt_matches(
            resumed_physics_receipt, observed_final_checkpoint
        )
        and _online_native_compiled_receipt_matches(
            intervention_physics_receipt, intervention_final_checkpoint
        )
    )
    tail_digest_match = float(
        _canonical_gate_tree_sha256(expected_tail)
        == _canonical_gate_tree_sha256(observed_tail)
    )
    final_checkpoint_match = float(
        expected_final_checkpoint == observed_final_checkpoint
    )
    limitations_preserved = float(
        result.validation_status == "exploratory"
        and intervention_result.validation_status == "exploratory"
        and tuple(result.source_limitations) == CANONICAL_SOURCE_LIMITATIONS
        and tuple(intervention_result.source_limitations)
        == CANONICAL_SOURCE_LIMITATIONS
        and result.effector_mapping_receipt.anatomical_status
        == RAW_APP_LATERALITY_STATUS
        and result.effector_mapping_receipt.signed_behavior_claim_policy
        == SIGNED_BEHAVIOR_CLAIM_POLICY
        and intervention_result.effector_mapping_receipt
        == result.effector_mapping_receipt
        and any("stable flight" in item for item in result.source_limitations)
    )
    root_displacement_m = float(
        np.linalg.norm(
            result.final_body_state.position_world_m
            - result.initial_body_state.position_world_m
        )
    )
    com_displacement_m = float(
        np.linalg.norm(final_com_position - initial_com_position)
    )
    numeric = np.asarray(
        _numeric_tree_values(
            (result, intervention_result, expected_tail, observed_tail)
        ),
        dtype=float,
    )
    nonfinite_count = int(
        numeric.size - np.count_nonzero(np.isfinite(numeric))
    )

    return EvaluatorResult(
        values=(
            MetricValue(
                "online_native_clock_violation_count", float(clock_violations)
            ),
            MetricValue(
                "online_native_circuit_sample_count",
                float(circuit_sample_count),
            ),
            MetricValue(
                "online_native_bridge_interval_count",
                float(len(result.intervals)),
            ),
            MetricValue(
                "online_native_physics_transition_count",
                float(physics_transition_count),
            ),
            MetricValue(
                "online_native_feedback_mismatch_count",
                float(feedback_mismatches),
            ),
            MetricValue(
                "online_native_intervention_clock_violation_count",
                float(paired_clock_violations),
            ),
            MetricValue(
                "online_native_intervention_physics_transition_count",
                float(intervention_physics_transition_count),
            ),
            MetricValue(
                "online_native_source_runtime_receipt_match",
                source_receipt_match,
            ),
            MetricValue(
                "online_native_compiled_model_receipt_match",
                compiled_receipt_match,
            ),
            MetricValue(
                "online_native_max_generated_dn_rate_hz", max_dn_rate_hz
            ),
            MetricValue(
                "online_native_max_generated_mn_rate_hz", max_mn_rate_hz
            ),
            MetricValue(
                "online_native_generated_event_count",
                float(len(generated_by_id)),
            ),
            MetricValue(
                "online_native_applied_event_count", float(len(applied_ids))
            ),
            MetricValue(
                "online_native_event_causality_violation_count",
                float(causal_event_violations),
            ),
            MetricValue(
                "online_native_targeted_natural_muscle_response_count",
                float(natural_response_count),
            ),
            MetricValue(
                "online_native_targeted_effective_muscle_response_count",
                float(effective_response_count),
            ),
            MetricValue(
                "online_native_intervention_pre_onset_numeric_mismatch_count",
                float(pre_onset_numeric_mismatch_count),
            ),
            MetricValue(
                "online_native_intervention_pre_onset_generated_event_mismatch_count",
                float(pre_onset_generated_event_mismatch_count),
            ),
            MetricValue(
                "online_native_intervention_target_suppressed_count",
                float(target_suppressed_count),
            ),
            MetricValue(
                "online_native_intervention_event_locality_mismatch_count",
                float(event_locality_mismatch_count),
            ),
            MetricValue(
                "online_native_intervention_max_target_actuation_force_n",
                max_target_actuation_force_n,
            ),
            MetricValue(
                "online_native_intervention_post_onset_measured_wing_delta_rad",
                post_onset_measured_wing_delta_rad,
            ),
            MetricValue(
                "online_native_intervention_final_root_position_delta_m",
                intervention_final_root_position_delta_m,
            ),
            MetricValue(
                "online_native_six_axis_command_violation_count",
                float(six_axis_command_violations),
            ),
            MetricValue(
                "online_native_max_physical_command_torque_n_m",
                max_command_torque_n_m,
            ),
            MetricValue(
                "online_native_max_actuator_torque_n_m",
                max_actuator_torque_n_m,
            ),
            MetricValue(
                "online_native_command_actuator_max_abs_delta_n_m",
                command_actuator_max_abs_delta_n_m,
            ),
            MetricValue(
                "online_native_max_root_fluid_force_n",
                max_root_fluid_force_n,
            ),
            MetricValue(
                "online_native_max_measured_wing_excursion_rad",
                max_wing_excursion_rad,
            ),
            MetricValue(
                "online_native_root_position_displacement_m",
                root_displacement_m,
            ),
            MetricValue(
                "online_native_whole_fly_com_displacement_m",
                com_displacement_m,
            ),
            MetricValue(
                "online_native_ground_contact_count",
                float(ground_contact_count),
            ),
            MetricValue(
                "online_native_articulated_telemetry_complete",
                float(telemetry_complete),
            ),
            MetricValue(
                "online_native_composite_checkpoint_tail_digest_match",
                tail_digest_match,
            ),
            MetricValue(
                "online_native_composite_checkpoint_final_match",
                final_checkpoint_match,
            ),
            MetricValue(
                "online_native_nonfinite_telemetry_count",
                float(nonfinite_count),
            ),
            MetricValue(
                "online_native_scientific_limitations_preserved",
                limitations_preserved,
            ),
        )
    )


_EMPIRICAL_BLOCKERS = {
    "hinge.heldout_wing_prediction": (
        "the CC0 Melis HDF5 has been located but is not staged with an exact SHA-256; "
        "the v2 completed-wingbeat availability audit, permanent acquisition-date "
        "split, reconciled matched causal CNN, preprocessing receipts, and sealed "
        "held-out execution are unavailable, while the published within-movie "
        "first-30/future-window score is ineligible because it leaks"
    ),
    "power_muscle.heldout_calcium_flight_state": (
        "public asynchronous-motor firing and splay recordings have been located, "
        "but complete per-animal calcium/ROI trials and their processing provenance "
        "are unavailable and reported calcium cohort sizes conflict; no immutable "
        "animal-held-out split, artifact receipt, or comprehensive evaluator exists"
    ),
    "dng02.heldout_wingbeat_amplitude": (
        "the expected public DNg02 artifact identities and leakage-safe v3 permanent "
        "driver-line split are frozen without outcome access, but promotion remains "
        "blocked pending a reviewed activation-onset mapping receipt, exact artifact "
        "verification and calibration/validation release by a trusted splitter, "
        "frozen finite predictions for both protocols, and one-shot sealed evaluation; "
        "targeted-pair count remains a driver-line proxy, and the upstream hinge and "
        "power-muscle gates remain independently blocked"
    ),
}


def _blocked_empirical(case: BenchmarkCase) -> EvaluatorResult:
    return EvaluatorResult.blocked(_EMPIRICAL_BLOCKERS[case.case_id])


def default_evaluators(
    registry: Optional[BenchmarkRegistry] = None,
) -> Dict[str, BenchmarkEvaluator]:
    """Return built-ins applicable to ``registry`` without importing workers."""

    implementations: Dict[str, Callable[[BenchmarkCase], EvaluatorResult]] = {
        "contracts.json_roundtrip": evaluate_contract_roundtrip,
        "timing.no_future_reads": evaluate_no_future_reads,
        "vision.causal_golden_suite": evaluate_visual_causal_golden_suite,
        "pipeline.retinal_to_body_vertical_slice": evaluate_retinal_to_body_vertical_slice,
        "pipeline.retinal_to_flybody_vertical_slice": evaluate_retinal_to_flybody_vertical_slice,
        "fly_fgs.fixed_step_integrity": evaluate_fly_fgs_fixed_step_integrity,
        "fly_fgs.incremental_runtime_parity": (
            evaluate_fly_fgs_incremental_runtime_parity
        ),
        "fly_fgs.checkpoint_reentry": evaluate_fly_fgs_checkpoint_reentry,
        "streaming.causal_neuromuscular_runtime": (
            evaluate_streaming_causal_neuromuscular_runtime
        ),
        "streaming.fresh_process_checkpoint_reentry": (
            evaluate_streaming_fresh_process_checkpoint_reentry
        ),
        "muscle.force_stage_intervention_contract": (
            evaluate_muscle_force_stage_intervention_contract
        ),
        "effector.raw_lane_physical_hypotheses": (
            evaluate_effector_raw_lane_physical_hypotheses
        ),
        "feedback.canonical_fly_fgs_closed_loop": (
            evaluate_canonical_fly_fgs_closed_loop
        ),
        "pipeline.canonical_online_fly_fgs_to_flybody": (
            evaluate_canonical_online_fly_fgs_to_flybody
        ),
        "pipeline.registered_fly_fgs_to_flybody_vertical_slice": (
            evaluate_registered_fly_fgs_to_flybody_vertical_slice
        ),
        "pipeline.registered_nod1_to_flybody_vertical_slice": (
            evaluate_registered_nod1_to_flybody_vertical_slice
        ),
        "feedback.reduced_closed_loop_yaw": evaluate_reduced_closed_loop_yaw,
        "nod1.hines_dense_manufactured": evaluate_hines_dense_manufactured,
        "physics.reduced_timestep_convergence": evaluate_reduced_timestep_convergence,
        "nod1.browser_python_parity": evaluate_browser_python_parity,
        "flybody.adapter_analytic_smoke": evaluate_flybody_smoke,
        "physics.flybody_open_loop_convergence": evaluate_flybody_open_loop_convergence,
        "flybody.checkpoint_reentry": evaluate_flybody_checkpoint_reentry,
        "physics.timestep_convergence": evaluate_flybody_timestep_convergence,
        "evidence.cross_atlas_integrity": evaluate_cross_atlas_integrity,
        "evidence.banc_fanc_wing_pathway": evaluate_banc_fanc_wing_pathway,
        "hinge.heldout_wing_prediction": _blocked_empirical,
        "power_muscle.heldout_calcium_flight_state": _blocked_empirical,
        "dng02.heldout_wingbeat_amplitude": _blocked_empirical,
    }
    if registry is None:
        return dict(implementations)
    known = {case.case_id for case in registry.cases}
    return {
        case_id: evaluator
        for case_id, evaluator in implementations.items()
        if case_id in known
    }


__all__ = [
    "FixtureIntegrityError",
    "default_evaluators",
    "evaluate_contract_roundtrip",
    "evaluate_banc_fanc_wing_pathway",
    "evaluate_cross_atlas_integrity",
    "evaluate_canonical_fly_fgs_closed_loop",
    "evaluate_canonical_online_fly_fgs_to_flybody",
    "evaluate_effector_raw_lane_physical_hypotheses",
    "evaluate_fly_fgs_fixed_step_integrity",
    "evaluate_fly_fgs_incremental_runtime_parity",
    "evaluate_fly_fgs_checkpoint_reentry",
    "evaluate_flybody_smoke",
    "evaluate_flybody_open_loop_convergence",
    "evaluate_flybody_checkpoint_reentry",
    "evaluate_flybody_timestep_convergence",
    "evaluate_hines_dense_manufactured",
    "evaluate_muscle_force_stage_intervention_contract",
    "evaluate_no_future_reads",
    "evaluate_browser_python_parity",
    "evaluate_reduced_timestep_convergence",
    "evaluate_reduced_closed_loop_yaw",
    "evaluate_retinal_to_body_vertical_slice",
    "evaluate_retinal_to_flybody_vertical_slice",
    "evaluate_registered_fly_fgs_to_flybody_vertical_slice",
    "evaluate_registered_nod1_to_flybody_vertical_slice",
    "evaluate_streaming_causal_neuromuscular_runtime",
    "evaluate_streaming_fresh_process_checkpoint_reentry",
    "evaluate_visual_causal_golden_suite",
    "sha256_file",
]
