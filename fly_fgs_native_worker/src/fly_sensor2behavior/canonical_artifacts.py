"""Immutable artifacts and browser projections for the canonical online stack.

This module is deliberately separate from the older open-loop episode writer.
The canonical runtime has three authoritative clocks, discrete NMJ events, a
model-owned phase trace, and an unresolved raw-app-lane-to-anatomy boundary.
Flattening it into the legacy logging-clock contract would erase those facts.

The scientific artifact therefore retains transition-rate arrays and causal
event tables.  The browser projection is a derived view only.  In particular,
it never interpolates event timing and never reconstructs wing phase from a
nominal constant frequency.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifacts import (
    WEB_REPLAY_SCHEMA_VERSION,
    _json_dump,
    _quaternion_to_euler,
    _rolling_wing_envelope,
    _write_chunked_array,
    sha256_file,
)
from .flight.canonical_closed_loop import (
    CANONICAL_CLOSED_LOOP_RUNTIME_VERSION,
    CANONICAL_CLOSED_LOOP_SCHEMA_VERSION,
    CanonicalClosedLoopCheckpoint,
    CanonicalClosedLoopResult,
    CanonicalPhysicsTransition,
    _body_from_dict,
)
from .fly_fgs import (
    FLY_FGS_NOD1_RAW_APP_SIDES,
    FLY_FGS_NOD1_ROOT_IDS,
    FLY_FGS_SNAPSHOT_ID,
    FLY_FGS_SOURCE_MANIFEST_SHA256,
    load_registered_fly_fgs_fixture,
)
from .flight.rigid_body import quaternion_to_matrix
from .flight.streaming_bridge import (
    RawAppSide,
    StreamingBridgeStage,
    StreamingMotorEvent,
    StreamingRateSample,
)
from .flight.types import AerodynamicWrench, MuscleSnapshot, RigidBodyState, WingKinematics


CANONICAL_ARTIFACT_SCHEMA_VERSION = "1.1.0"
CANONICAL_WEB_TRACE_SCHEMA_VERSION = "1.0.0"
ONLINE_CIRCUIT_REPLAY_SCHEMA_VERSION = "1.0.0"
CANONICAL_WEB_SOURCE_KIND = "canonical_online_fly_fgs_streaming_flybody"
CANONICAL_ARTIFACT_SOURCE_KIND = "canonical_online_fly_fgs_streaming_flight"
CANONICAL_WEB_PROJECTION_CANONICALIZATION = (
    "RFC8259 JSON; UTF-8; sorted keys; separators comma/colon; NaN/Infinity rejected; "
    "source_artifact_manifest_sha256 and source_artifact_schema_version excluded"
)
_WEB_PROJECTION_EXCLUDED_FIELDS = (
    "source_artifact_manifest_sha256",
    "source_artifact_schema_version",
)

_TIME_TOLERANCE_S = 1.0e-12
_RAW_LANE_ORDER = (RawAppSide.L, RawAppSide.R)
_MOTOR_ORDER = ("MN-iv2", "MN-i1", "MN-iv1", "MN-b3")
_MUSCLE_ORDER = ("iv2", "i1", "iv1", "b3")
_DN_RATE_AXIS = tuple((lane, "DNp26") for lane in _RAW_LANE_ORDER)
_MN_RATE_AXIS = tuple(
    (lane, motor) for lane in _RAW_LANE_ORDER for motor in _MOTOR_ORDER
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _strict_json_copy(value: Any, label: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must contain only finite JSON values" % label) from exc


def _json_content_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _javascript_number_text(value: float) -> str:
    """Serialize one finite binary64 like ``JSON.stringify``.

    The fly-FGS sidecar signs a recursively key-sorted JavaScript JSON object.
    Python and JavaScript differ for ``-0``, integer-valued floats, and the
    fixed/exponential notation thresholds.  Python's ``repr`` supplies the
    shortest round-tripping digit sequence; this function applies ECMAScript's
    JSON number layout to that sequence.
    """

    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JavaScript checkpoint canonicalization requires finite numbers")
    if number == 0.0:
        return "0"
    negative = number < 0.0
    raw = repr(abs(number)).lower()
    if "e" in raw:
        mantissa, exponent_text = raw.split("e", 1)
        exponent = int(exponent_text)
    else:
        mantissa = raw
        exponent = 0
    if "." in mantissa:
        integer_part, fractional_part = mantissa.split(".", 1)
    else:
        integer_part, fractional_part = mantissa, ""
    digits = integer_part + fractional_part
    decimal_position = len(integer_part) + exponent
    leading_zero_count = len(digits) - len(digits.lstrip("0"))
    digits = digits.lstrip("0")
    decimal_position -= leading_zero_count
    if not digits:
        return "0"
    digits = digits.rstrip("0") or "0"
    digit_count = len(digits)
    if decimal_position <= 0 and decimal_position > -6:
        rendered = "0." + ("0" * (-decimal_position)) + digits
    elif 0 < decimal_position <= 21:
        if decimal_position >= digit_count:
            rendered = digits + ("0" * (decimal_position - digit_count))
        else:
            rendered = (
                digits[:decimal_position]
                + "."
                + digits[decimal_position:]
            )
    else:
        rendered = digits[0]
        if digit_count > 1:
            rendered += "." + digits[1:]
        scientific_exponent = decimal_position - 1
        rendered += "e%s%d" % (
            "+" if scientific_exponent >= 0 else "",
            scientific_exponent,
        )
    return ("-" if negative else "") + rendered


def _javascript_sorted_json_text(value: Any) -> str:
    """Mirror the sidecar's ``JSON.stringify(sortedValue(value))`` contract."""

    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return _javascript_number_text(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, list):
        return "[" + ",".join(_javascript_sorted_json_text(item) for item in value) + "]"
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JavaScript checkpoint object keys must be strings")
        # ``sortedValue`` inserts keys in lexical order, but ECMAScript object
        # enumeration always emits canonical array-index properties first in
        # numeric order. The cable-state object has keys such as "2" and
        # "108", so ordinary Python lexical sorting would sign different
        # bytes despite identical state.
        array_index_keys = []
        ordinary_keys = []
        for key in value:
            is_canonical_index = (
                key == "0"
                or (
                    key.isdigit()
                    and not key.startswith("0")
                    and int(key) <= 4294967294
                )
            )
            if is_canonical_index:
                array_index_keys.append(key)
            else:
                ordinary_keys.append(key)
        ordered_keys = sorted(array_index_keys, key=int) + sorted(ordinary_keys)
        return "{" + ",".join(
            "%s:%s"
            % (
                json.dumps(key, ensure_ascii=False, allow_nan=False),
                _javascript_sorted_json_text(value[key]),
            )
            for key in ordered_keys
        ) + "}"
    raise ValueError(
        "JavaScript checkpoint canonicalization does not support %s"
        % type(value).__name__
    )


_FLY_FGS_CIRCUIT_CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "snapshot_id",
        "source_manifest_sha256",
        "circuit_engine_sha256",
        "circuit_bundle_sha256",
        "dt_s",
        "float_encoding",
        "dynamic_state",
        "payload_sha256",
    }
)


def _checkpoint_component_payload_sha256(
    name: str, child: Mapping[str, Any]
) -> str:
    unsigned = {key: value for key, value in child.items() if key != "payload_sha256"}
    if name == "circuit" and frozenset(child) == _FLY_FGS_CIRCUIT_CHECKPOINT_FIELDS:
        contract = _registered_fly_fgs_display_contract()
        receipts = contract["asset_receipts"]
        if (
            child["schema_version"] != "1.0.0"
            or child["protocol_version"] != "1.0.0"
            or child["snapshot_id"] != contract["snapshot_id"]
            or child["source_manifest_sha256"]
            != contract["source_manifest_sha256"]
            or child["circuit_engine_sha256"]
            != receipts["circuit_engine"]["sha256"]
            or child["circuit_bundle_sha256"]
            != receipts["circuit_bundle"]["sha256"]
            or child["dt_s"] != 0.005
            or child["float_encoding"] != "float64_le_base64"
            or not isinstance(child["dynamic_state"], Mapping)
        ):
            raise ValueError("fly-FGS circuit checkpoint source contract mismatch")
        encoded = _javascript_sorted_json_text(unsigned).encode("utf-8")
    else:
        encoded = json.dumps(
            unsigned,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1)
def _registered_fly_fgs_display_contract_json() -> str:
    """Return the verified bundle axis/topology as immutable cached JSON.

    Loading the registered fixture validates the source manifest, every asset
    digest, the exact 1,684-cell bundle order, and the 1,441-cell retinotopic
    T4a sub-axis.  Cache serialized JSON rather than a mutable mapping so no
    caller can mutate the process-global source contract.
    """

    fixture = load_registered_fly_fgs_fixture()
    topology = _strict_json_copy(
        dict(fixture.circuit_topology), "registered fly-FGS topology"
    )
    cell_axis = topology["cell_axis"]
    retinal_axis = topology["retinotopic_t4a"]
    cell_ids = [str(cell["id"]) for cell in cell_axis]
    if len(cell_ids) != 1684 or len(set(cell_ids)) != len(cell_ids):
        raise ValueError("registered fly-FGS cell axis is not the exact 1,684-cell inventory")
    if (
        retinal_axis["count"] != 1441
        or len(retinal_axis["cell_ids"]) != 1441
        or len(set(retinal_axis["cell_ids"])) != 1441
    ):
        raise ValueError("registered fly-FGS retinal axis is not the exact 1,441-cell inventory")
    nod1_cell_indices = {}
    for root_id in FLY_FGS_NOD1_ROOT_IDS:
        cell_id = "r%s" % root_id
        try:
            nod1_cell_indices[root_id] = cell_ids.index(cell_id)
        except ValueError as exc:
            raise ValueError(
                "registered fly-FGS NOD1 root %s is absent from the cell axis"
                % root_id
            ) from exc
    asset_receipts = {
        receipt.asset_id: dict(receipt.to_dict())
        for receipt in fixture.asset_receipts
    }
    asset_receipts["source_manifest"] = {
        "asset_id": "source_manifest",
        "relative_path": fixture.source_manifest_uri,
        "sha256": fixture.source_manifest_sha256,
        "bytes": fixture.source_manifest_bytes,
        "media_type": "application/json",
        "role": "strict_source_contract",
        "eligible_circuit_input": False,
    }
    contract = {
        "snapshot_id": fixture.snapshot_id,
        "source_manifest_sha256": fixture.source_manifest_sha256,
        "source_manifest_uri": fixture.source_manifest_uri,
        "asset_receipts": asset_receipts,
        "dataset": _strict_json_copy(
            dict(fixture.dataset_metadata), "registered fly-FGS dataset metadata"
        ),
        "cell_inventory": _strict_json_copy(
            dict(fixture.circuit_inventory), "registered fly-FGS cell inventory"
        ),
        "circuit_topology": topology,
        "cell_ids": cell_ids,
        "cell_axis_sha256": _json_content_sha256(cell_axis),
        "cell_ids_sha256": _json_content_sha256(cell_ids),
        "retinal_axis_sha256": _json_content_sha256(retinal_axis),
        "nod1_cell_indices": nod1_cell_indices,
        "rejected_downstream_fields": list(fixture.rejected_downstream_fields),
        "claim_scope": _strict_json_copy(
            dict(fixture.claim_scope), "registered fly-FGS claim scope"
        ),
    }
    if (
        contract["snapshot_id"] != FLY_FGS_SNAPSHOT_ID
        or contract["source_manifest_sha256"]
        != FLY_FGS_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("registered fly-FGS display contract identity drifted")
    return json.dumps(
        contract,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _registered_fly_fgs_display_contract() -> Mapping[str, Any]:
    return json.loads(_registered_fly_fgs_display_contract_json())


def _display_array_axis_metadata(
    observations: Sequence[Any],
) -> Mapping[str, Mapping[str, Any]]:
    """Name every display-state axis and bind canonical widths to the bundle.

    Small fake/adaptor runtimes used by unit tests remain publishable, but are
    labelled as unregistered index axes.  A runtime that emits the canonical
    widths is required to agree with the registered cell order at all four
    NOD1 positions; those arrays then receive the exact source IDs and axis
    digests rather than a shape-only interpretation.
    """

    first = observations[0].sample
    retinal_width = (
        None
        if first.retinal_input_luminance is None
        else len(first.retinal_input_luminance)
    )
    full_width = (
        None if first.full_cell_voltage_v is None else len(first.full_cell_voltage_v)
    )
    needs_registered_contract = retinal_width == 1441 or full_width == 1684
    contract = (
        _registered_fly_fgs_display_contract()
        if needs_registered_contract
        else None
    )
    metadata: Dict[str, Mapping[str, Any]] = {}
    if retinal_width is not None:
        if retinal_width == 1441:
            assert contract is not None
            retinal_axis = contract["circuit_topology"]["retinotopic_t4a"]
            metadata["retinal_input_luminance"] = {
                "axis_name": "registered_retinotopic_t4a_cell",
                "axis_labels": list(retinal_axis["cell_ids"]),
                "parent_cell_indices": list(retinal_axis["cell_indices"]),
                "azimuth_deg": list(retinal_axis["azimuth_deg"]),
                "elevation_deg": list(retinal_axis["elevation_deg"]),
                "axis_registration": "registered_fly_fgs_circuit_bundle_order",
                "axis_sha256": contract["retinal_axis_sha256"],
                "source_manifest_sha256": contract["source_manifest_sha256"],
                "sample_axis": "circuit_measurement_time_s",
                "eligible_motor_input": False,
            }
        else:
            metadata["retinal_input_luminance"] = {
                "axis_name": "runtime_retinal_index",
                "axis_labels": [
                    "unregistered_retinal_index:%d" % index
                    for index in range(retinal_width)
                ],
                "axis_registration": "unregistered_runtime_axis",
                "sample_axis": "circuit_measurement_time_s",
                "eligible_motor_input": False,
            }
    if full_width is not None:
        if full_width == 1684:
            assert contract is not None
            if retinal_width is not None and retinal_width != 1441:
                raise ValueError(
                    "registered full-cell state cannot be paired with an unregistered retinal axis"
                )
            nod1_indices = contract["nod1_cell_indices"]
            for observation in observations:
                sample = observation.sample
                assert sample.full_cell_voltage_v is not None
                for root_id in FLY_FGS_NOD1_ROOT_IDS:
                    if (
                        sample.full_cell_voltage_v[nod1_indices[root_id]]
                        != sample.nod1_voltage_v[root_id]
                    ):
                        raise ValueError(
                            "full-cell state does not preserve the registered NOD1 cell axis"
                        )
            full_metadata = {
                "axis_name": "registered_fly_fgs_circuit_cell",
                "axis_labels": list(contract["cell_ids"]),
                "axis_registration": "registered_fly_fgs_circuit_bundle_order",
                "axis_sha256": contract["cell_axis_sha256"],
                "axis_labels_sha256": contract["cell_ids_sha256"],
                "source_manifest_sha256": contract["source_manifest_sha256"],
                "sample_axis": "circuit_measurement_time_s",
                "eligible_motor_input": False,
            }
        else:
            full_metadata = {
                "axis_name": "runtime_circuit_cell_index",
                "axis_labels": [
                    "unregistered_circuit_cell_index:%d" % index
                    for index in range(full_width)
                ],
                "axis_registration": "unregistered_runtime_axis",
                "sample_axis": "circuit_measurement_time_s",
                "eligible_motor_input": False,
            }
        metadata["full_circuit_voltage_v"] = dict(full_metadata)
        metadata["full_circuit_activity"] = dict(full_metadata)
    return metadata


def _flatten_transitions(
    result: CanonicalClosedLoopResult,
) -> Tuple[CanonicalPhysicsTransition, ...]:
    if result.initial_bridge_tick_index != 0 or not result.completed:
        raise ValueError("canonical publication requires a complete result from reset")
    if len(result.intervals) != result.config.bridge_interval_count:
        raise ValueError("canonical interval inventory is incomplete")
    telemetry_present: List[bool] = [
        result.initial_articulated_physics_telemetry is not None
    ]
    previous_phase: Optional[float] = None
    for interval_index, interval in enumerate(result.intervals):
        expected_start = interval_index * result.config.bridge_dt_s
        expected_end = expected_start + result.config.bridge_dt_s
        if (
            interval.bridge_tick_index != interval_index
            or abs(interval.interval_start_s - expected_start) > _TIME_TOLERANCE_S
            or abs(interval.interval_end_s - expected_end) > _TIME_TOLERANCE_S
        ):
            raise ValueError("canonical bridge interval clock is inconsistent")
        bridge_start = interval.bridge_start
        bridge = interval.bridge
        if (
            bridge_start.tick_index != interval_index
            or bridge.tick_index != interval_index
            or abs(bridge_start.interval_start_s - expected_start)
            > _TIME_TOLERANCE_S
            or abs(bridge_start.interval_end_s - expected_end)
            > _TIME_TOLERANCE_S
            or abs(bridge.interval_start_s - expected_start) > _TIME_TOLERANCE_S
            or abs(bridge.interval_end_s - expected_end) > _TIME_TOLERANCE_S
        ):
            raise ValueError("canonical bridge component clock is inconsistent")
        for name in (
            "source_hold",
            "generated_dn_rates",
            "held_dn_rates",
            "generated_motor_rates",
            "held_motor_rates",
            "delivered_events",
            "active_intervention_ids",
        ):
            if tuple(getattr(bridge_start, name)) != tuple(getattr(bridge, name)):
                raise ValueError("bridge start/end %s receipts disagree" % name)
        if len(interval.mechanics) != 5 or len(interval.physics) != 5:
            raise ValueError("each canonical bridge interval must contain five transitions")
        phase_path = (
            bridge_start.wing_phase_start_unwrapped_rad,
            *(frame.phase_end_unwrapped_rad for frame in interval.mechanics),
        )
        if not np.array_equal(
            np.asarray(phase_path, dtype=float),
            np.asarray(bridge.wing_phase_path_unwrapped_rad, dtype=float),
        ):
            raise ValueError("bridge phase path does not match mechanics endpoints")
        if (
            abs(interval.projected_model_phase_end_unwrapped_rad - phase_path[-1])
            > _TIME_TOLERANCE_S
            or abs(bridge.wing_phase_end_unwrapped_rad - phase_path[-1])
            > _TIME_TOLERANCE_S
        ):
            raise ValueError("canonical projected phase endpoint is inconsistent")
        if previous_phase is not None and abs(phase_path[0] - previous_phase) > _TIME_TOLERANCE_S:
            raise ValueError("canonical wing phase is discontinuous across bridge intervals")
        previous_phase = phase_path[-1]
        for local_index, (mechanics, transition) in enumerate(
            zip(interval.mechanics, interval.physics)
        ):
            expected_physics_tick = interval_index * 5 + local_index
            expected_physics_start = expected_physics_tick * result.config.physics_dt_s
            if transition.mechanics is not mechanics:
                raise ValueError("physics transition does not retain its mechanics receipt")
            if (
                mechanics.tick_index != expected_physics_tick
                or transition.physics_tick_index != expected_physics_tick
                or abs(mechanics.interval_start_s - expected_physics_start)
                > _TIME_TOLERANCE_S
                or abs(transition.interval_start_s - expected_physics_start)
                > _TIME_TOLERANCE_S
                or abs(mechanics.interval_end_s - transition.interval_end_s)
                > _TIME_TOLERANCE_S
            ):
                raise ValueError("canonical mechanics/physics clocks are inconsistent")
            telemetry_present.append(transition.articulated_telemetry_after is not None)
    if any(telemetry_present) != all(telemetry_present):
        raise ValueError("articulated physics telemetry cannot be partially populated")
    if result.articulated_physics_telemetry_available != all(telemetry_present):
        raise ValueError("articulated physics telemetry capability receipt is inconsistent")

    transitions = tuple(
        transition
        for interval in result.intervals
        for transition in interval.physics
    )
    expected_count = result.config.bridge_interval_count * 5
    if len(transitions) != expected_count:
        raise ValueError("each canonical bridge interval must contain five transitions")
    if not transitions:
        raise ValueError("canonical artifacts require at least one completed transition")
    expected_start_index = result.initial_bridge_tick_index * 5
    for local_index, transition in enumerate(transitions):
        expected_index = expected_start_index + local_index
        if transition.physics_tick_index != expected_index:
            raise ValueError("canonical transition indices are not contiguous")
        expected_start = expected_index * result.config.physics_dt_s
        if (
            abs(transition.interval_start_s - expected_start) > _TIME_TOLERANCE_S
            or abs(
                transition.interval_end_s
                - transition.interval_start_s
                - result.config.physics_dt_s
            )
            > _TIME_TOLERANCE_S
        ):
            raise ValueError("canonical transition clock is inconsistent")
        if local_index == 0:
            if not _body_equal(transition.body_state_before, result.initial_body_state):
                raise ValueError("first transition does not begin at result initial state")
        elif not _body_equal(
            transition.body_state_before, transitions[local_index - 1].body_state_after
        ):
            raise ValueError("canonical body-state transitions are discontinuous")
    if not _body_equal(transitions[-1].body_state_after, result.final_body_state):
        raise ValueError("last transition does not end at result final state")
    return transitions


def _body_equal(first: RigidBodyState, second: RigidBodyState) -> bool:
    return all(
        np.array_equal(getattr(first, name), getattr(second, name))
        for name in (
            "position_world_m",
            "velocity_world_m_s",
            "quaternion_body_to_world",
            "angular_velocity_body_rad_s",
        )
    )


def _body_arrays(
    result: CanonicalClosedLoopResult,
    transitions: Sequence[CanonicalPhysicsTransition],
) -> Mapping[str, np.ndarray]:
    states = (result.initial_body_state,) + tuple(
        transition.body_state_after for transition in transitions
    )
    return {
        "body_position_world_m": np.asarray(
            [state.position_world_m for state in states], dtype=float
        ),
        "body_velocity_world_m_s": np.asarray(
            [state.velocity_world_m_s for state in states], dtype=float
        ),
        "body_quaternion_body_to_world": np.asarray(
            [state.quaternion_body_to_world for state in states], dtype=float
        ),
        "body_angular_velocity_body_rad_s": np.asarray(
            [state.angular_velocity_body_rad_s for state in states], dtype=float
        ),
    }


def _wrench_row(wrench: AerodynamicWrench) -> Tuple[np.ndarray, np.ndarray, float]:
    return (
        np.asarray(wrench.force_body_n, dtype=float),
        np.asarray(wrench.torque_body_n_m, dtype=float),
        float(wrench.mechanical_power_w),
    )


def _wing_fields(prefix: str, wings: Sequence[WingKinematics]) -> Mapping[str, np.ndarray]:
    return {
        "%s_phase_rad" % prefix: np.asarray(
            [wing.phase_rad for wing in wings], dtype=float
        ),
        "%s_frequency_hz" % prefix: np.asarray(
            [wing.frequency_hz for wing in wings], dtype=float
        ),
        "%s_stroke_rad" % prefix: np.asarray(
            [wing.stroke_rad for wing in wings], dtype=float
        ),
        "%s_stroke_velocity_rad_s" % prefix: np.asarray(
            [wing.stroke_velocity_rad_s for wing in wings], dtype=float
        ),
        "%s_stroke_acceleration_rad_s2" % prefix: np.asarray(
            [wing.stroke_acceleration_rad_s2 for wing in wings], dtype=float
        ),
        "%s_angle_of_attack_rad" % prefix: np.asarray(
            [wing.angle_of_attack_rad for wing in wings], dtype=float
        ),
        "%s_deviation_rad" % prefix: np.asarray(
            [wing.deviation_rad for wing in wings], dtype=float
        ),
        "%s_generalized_torque_n_m" % prefix: np.asarray(
            [wing.generalized_torque_n_m for wing in wings], dtype=float
        ),
        "%s_wing_axis_torque_n_m" % prefix: np.asarray(
            [wing.wing_axis_torque_n_m for wing in wings], dtype=float
        ),
    }


def _raw_key_to_lane_id(key: str) -> str:
    side, muscle = key.split(":", 1)
    lane = "raw_app_L" if side == "left" else "raw_app_R"
    return "%s:%s" % (lane, muscle)


def _muscle_axis(snapshot: MuscleSnapshot) -> Tuple[str, ...]:
    keys = tuple(snapshot.individual)
    expected = tuple(
        "%s:%s" % (side, muscle)
        for side in ("left", "right")
        for muscle in ("DLM", "DVM", "iv2", "i1", "iv1", "b3", "tp1")
    )
    if set(keys) != set(expected):
        raise ValueError("streaming muscle inventory does not match the reviewed axis")
    return expected


def _snapshot_matrix(
    snapshots: Sequence[MuscleSnapshot],
    axis: Sequence[str],
    field: str,
) -> np.ndarray:
    return np.asarray(
        [
            [float(getattr(snapshot.individual[key], field)) for key in axis]
            for snapshot in snapshots
        ],
        dtype=float,
    )


def _circuit_observations(result: CanonicalClosedLoopResult) -> Tuple[Any, ...]:
    observations_list: List[Any] = []
    for interval in result.intervals:
        expected = interval.bridge_tick_index % 10 == 0
        observation = interval.circuit_observation
        if (observation is not None) != expected:
            raise ValueError("circuit observation presence does not match the 5 ms clock")
        if observation is not None:
            observations_list.append(observation)
    observations = tuple(observations_list)
    if (
        len(observations) != result.circuit_sample_count
        or len(observations) != result.config.circuit_sample_count
    ):
        raise ValueError("result circuit sample count does not match the complete clock")
    if not observations:
        raise ValueError("canonical result contains no circuit observations")
    pooled_keys = tuple(observations[0].sample.pooled_readout)
    retinal_width: Optional[int] = None
    full_width: Optional[int] = None
    for expected_index, current in enumerate(observations):
        sample = current.sample
        expected_tick = expected_index * 10
        expected_time = expected_index * result.config.circuit_dt_s
        if (
            current.bridge_tick_index != expected_tick
            or abs(current.observation_time_s - expected_time) > _TIME_TOLERANCE_S
            or sample.sample_index != expected_index
            or abs(sample.measurement_time_s - expected_time) > _TIME_TOLERANCE_S
            or abs(sample.availability_time_s - expected_time) > _TIME_TOLERANCE_S
        ):
            raise ValueError("circuit samples do not follow the exact 5 ms reset clock")
        interval = result.intervals[expected_tick]
        if not _body_equal(current.body_state, interval.physics[0].body_state_before):
            raise ValueError("circuit observation body state does not match its boundary")
        expected_mode = (
            "captured_static_preroll_gauge_matched"
            if expected_index == 0
            else "body_scene_control_applied"
        )
        if current.initialization_mode != expected_mode:
            raise ValueError("circuit initialization/reset mode receipt is inconsistent")
        if expected_index > 0 and sample.last_control != current.requested_control:
            raise ValueError("circuit sample does not retain its requested body/scene control")
        if tuple(sample.nod1_voltage_v) != FLY_FGS_NOD1_ROOT_IDS:
            raise ValueError("circuit NOD1 root inventory/order is inconsistent")
        if tuple(sample.pooled_readout) != pooled_keys:
            raise ValueError("pooled circuit readout inventory changes across samples")
        retinal = sample.retinal_input_luminance
        if (retinal is not None) != result.config.include_retinal_input:
            raise ValueError("retinal trace presence does not match the run configuration")
        if retinal is not None:
            if retinal_width is None:
                retinal_width = len(retinal)
            if not retinal or len(retinal) != retinal_width:
                raise ValueError("retinal trace inventory changes across circuit samples")
        voltage = sample.full_cell_voltage_v
        activity = sample.full_cell_activity
        if (voltage is None) != (activity is None):
            raise ValueError("full-cell voltage/activity traces must be paired")
        if (voltage is not None) != result.config.include_full_cell_state:
            raise ValueError("full-cell trace presence does not match the run configuration")
        if voltage is not None and activity is not None:
            if full_width is None:
                full_width = len(voltage)
            if (
                not voltage
                or len(voltage) != full_width
                or len(activity) != full_width
            ):
                raise ValueError("full-cell trace inventory changes across circuit samples")
        if sample.full_cell_state_eligible_motor_input is not False:
            raise ValueError("full circuit state cannot be motor eligible")
    return observations


def _event_dict(event: StreamingMotorEvent, disposition: str) -> Mapping[str, Any]:
    delivered = disposition in ("applied", "suppressed")
    return {
        "generated_by_bridge": True,
        "event_id": event.event_id,
        "motor_neuron": event.motor_neuron,
        "muscle": event.muscle,
        "raw_app_side": event.raw_app_side.value,
        "anatomical_side": event.anatomical_side.value,
        "event_time_s": event.event_time_s,
        "availability_time_s": event.availability_time_s,
        "wingbeat_phase_rad": event.wingbeat_phase_rad,
        "rate_hz": event.rate_hz,
        "emission_probability": event.emission_probability,
        "source_measurement_time_s": event.source_measurement_time_s,
        "generator_seed": event.generator_seed,
        "phase_crossing_index": event.phase_crossing_index,
        "semantics": event.semantics.value,
        "delivered_to_mechanics": delivered,
        "nmj_delivery_status": (
            "delivered_to_mechanics" if delivered else "pending_at_episode_end"
        ),
        "mechanics_disposition": disposition,
        "applied_to_muscle_state": disposition == "applied",
        "suppressed_by_mechanics_intervention": disposition == "suppressed",
    }


def _event_checkpoint_dict(event: StreamingMotorEvent) -> Mapping[str, Any]:
    record = dict(_event_dict(event, "pending_at_episode_end"))
    for key in (
        "generated_by_bridge",
        "delivered_to_mechanics",
        "nmj_delivery_status",
        "mechanics_disposition",
        "applied_to_muscle_state",
        "suppressed_by_mechanics_intervention",
    ):
        record.pop(key)
    return record


def _rate_key(sample: StreamingRateSample) -> Tuple[RawAppSide, str]:
    return sample.raw_app_side, sample.target_name


def _rate_map(
    samples: Sequence[StreamingRateSample],
    axis: Sequence[Tuple[RawAppSide, str]],
    stage: StreamingBridgeStage,
    *,
    complete: bool,
    label: str,
) -> Mapping[Tuple[RawAppSide, str], StreamingRateSample]:
    result: Dict[Tuple[RawAppSide, str], StreamingRateSample] = {}
    expected = set(axis)
    for sample in samples:
        if sample.stage is not stage or _rate_key(sample) not in expected:
            raise ValueError("%s contains an unexpected rate channel" % label)
        if _rate_key(sample) in result:
            raise ValueError("%s contains a duplicate rate channel" % label)
        result[_rate_key(sample)] = sample
    if complete and set(result) != expected:
        raise ValueError("%s rate inventory is incomplete" % label)
    return result


def _rate_record(
    sample: StreamingRateSample,
    *,
    record_kind: str,
    bridge_tick_index: int,
) -> Mapping[str, Any]:
    return {
        "record_kind": record_kind,
        "bridge_tick_index": bridge_tick_index,
        "stage": sample.stage.value,
        "target_name": sample.target_name,
        "raw_app_side": sample.raw_app_side.value,
        "anatomical_side": sample.anatomical_side.value,
        "measurement_time_s": sample.measurement_time_s,
        "availability_time_s": sample.availability_time_s,
        "rate_hz": sample.rate_hz,
        "source_measurement_time_s": sample.source_measurement_time_s,
        "semantics": sample.semantics.value,
    }


def _rate_array_name(prefix: str, stage_name: str, field: str) -> str:
    return "%s_%s_%s" % (
        prefix,
        stage_name,
        field if field == "rate_hz" else "rate_" + field,
    )


@dataclass(frozen=True)
class CanonicalArrayBundle:
    arrays: Mapping[str, Tuple[np.ndarray, str, str]]
    axis_metadata: Mapping[str, Mapping[str, Any]]
    event_records: Tuple[Mapping[str, Any], ...]
    rate_records: Tuple[Mapping[str, Any], ...]
    circuit_records: Tuple[Mapping[str, Any], ...]
    intervention_records: Tuple[Mapping[str, Any], ...]


def canonical_array_bundle(result: CanonicalClosedLoopResult) -> CanonicalArrayBundle:
    """Validate and flatten a complete canonical episode without writing."""

    if not isinstance(result, CanonicalClosedLoopResult):
        raise TypeError("result must be CanonicalClosedLoopResult")
    if result.initial_bridge_tick_index != 0 or not result.completed:
        raise ValueError(
            "canonical publication requires a complete result beginning at bridge tick 0"
        )
    if result.effector_mapping_receipt != result.config.effector_mapping_receipt:
        raise ValueError("result effector mapping receipt does not match its configuration")
    if not result.backend_name or not result.aerodynamic_owner:
        raise ValueError("result must retain non-empty physics authority receipts")

    transitions = _flatten_transitions(result)
    observations = _circuit_observations(result)
    start_physics_tick = result.initial_bridge_tick_index * 5
    # Publish the clock from integer tick identity, not from repeated floating
    # additions stored on component frames. The latter is validated against
    # this grid above but can differ by a last-bit rounding path.
    physics_time = (
        np.arange(
            start_physics_tick,
            start_physics_tick + len(transitions) + 1,
            dtype=float,
        )
        * result.config.physics_dt_s
    )
    arrays: Dict[str, Tuple[np.ndarray, str, str]] = {
        "physics_time_s": (
            physics_time,
            "s",
            "authoritative canonical physics boundary clock",
        )
    }
    for name, values in _body_arrays(result, transitions).items():
        unit = {
            "body_position_world_m": "m",
            "body_velocity_world_m_s": "m s^-1",
            "body_quaternion_body_to_world": "1",
            "body_angular_velocity_body_rad_s": "rad s^-1",
        }[name]
        arrays[name] = (values, unit, "authoritative external physics state")

    wrenches = (result.initial_aerodynamic_wrench,) + tuple(
        transition.aerodynamic_wrench_after for transition in transitions
    )
    arrays["aerodynamic_force_body_n"] = (
        np.asarray([_wrench_row(item)[0] for item in wrenches], dtype=float),
        "N",
        "external physics root-total aerodynamic wrench",
    )
    arrays["aerodynamic_torque_body_n_m"] = (
        np.asarray([_wrench_row(item)[1] for item in wrenches], dtype=float),
        "N m",
        "external physics root-total aerodynamic wrench",
    )
    arrays["aerodynamic_mechanical_power_w"] = (
        np.asarray([_wrench_row(item)[2] for item in wrenches], dtype=float),
        "W",
        "external physics reported mechanical power",
    )

    phase = np.asarray(
        [transitions[0].mechanics.phase_start_unwrapped_rad]
        + [item.mechanics.phase_end_unwrapped_rad for item in transitions],
        dtype=float,
    )
    if np.any(np.diff(phase) < -_TIME_TOLERANCE_S):
        raise ValueError("authoritative unwrapped wing phase decreases")
    arrays["wing_phase_unwrapped_rad"] = (
        phase,
        "rad",
        "model-owned thorax oscillator state; uncalibrated and not measured",
    )
    arrays["wing_frequency_hz"] = (
        np.asarray(
            [transitions[0].mechanics.kinematic_frequency_hz]
            + [item.mechanics.kinematic_frequency_hz for item in transitions],
            dtype=float,
        ),
        "Hz",
        "model-owned thorax oscillator state; uncalibrated and not measured",
    )

    raw_commands = tuple(
        transition.mechanics.actuation_wing_kinematics
        for transition in transitions
    )
    physical_commands = tuple(
        transition.physical_actuation_wing_kinematics
        for transition in transitions
    )
    units_by_suffix = {
        "phase_rad": "rad",
        "frequency_hz": "Hz",
        "stroke_rad": "rad",
        "stroke_velocity_rad_s": "rad s^-1",
        "stroke_acceleration_rad_s2": "rad s^-2",
        "angle_of_attack_rad": "rad",
        "deviation_rad": "rad",
        "generalized_torque_n_m": "N m",
        "wing_axis_torque_n_m": "N m",
    }
    for prefix, values, provenance in (
        (
            "raw_app_actuation",
            raw_commands,
            "left-boundary virtual-hinge command in unresolved raw app lane order",
        ),
        (
            "physical_actuation",
            physical_commands,
            "left-boundary command after the explicitly selected physical-wing hypothesis",
        ),
    ):
        for name, array in _wing_fields(prefix, values).items():
            suffix = name[len(prefix) + 1 :]
            arrays[name] = (array, units_by_suffix[suffix], provenance)

    initial_effective = transitions[0].mechanics.actuation_muscle_snapshot
    # The frame exposes the natural state at the *right* endpoint, but does
    # not expose a distinct natural left-boundary snapshot. Keep that trace on
    # the transition-end clock instead of copying a future value into t=0.
    natural_snapshots = tuple(
        transition.mechanics.natural_muscle_snapshot for transition in transitions
    )
    effective_snapshots = (initial_effective,) + tuple(
        transition.mechanics.muscle_snapshot for transition in transitions
    )
    muscle_axis = _muscle_axis(transitions[0].mechanics.natural_muscle_snapshot)
    for status, snapshots in (
        ("natural", natural_snapshots),
        ("effective", effective_snapshots),
    ):
        for field, unit in (
            ("activation", "1"),
            ("force_n", "N"),
            ("phase_effect", "1"),
            ("resonance_scale", "1"),
        ):
            arrays["muscle_%s_%s" % (status, field)] = (
                _snapshot_matrix(snapshots, muscle_axis, field),
                unit,
                "%s streaming muscle state in unresolved raw app lanes" % status,
            )
    arrays["muscle_natural_endpoint_time_s"] = (
        physics_time[1:],
        "s",
        "right-endpoint clock for natural muscle state; no t=0 state was invented",
    )

    circuit_time = np.asarray(
        [item.sample.measurement_time_s for item in observations], dtype=float
    )
    circuit_availability = np.asarray(
        [item.sample.availability_time_s for item in observations], dtype=float
    )
    nod_axis = tuple(observations[0].sample.nod1_voltage_v)
    arrays["circuit_measurement_time_s"] = (
        circuit_time,
        "s",
        "canonical fly-FGS measurement clock",
    )
    arrays["circuit_availability_time_s"] = (
        circuit_availability,
        "s",
        "canonical fly-FGS runtime availability clock",
    )
    arrays["nod1_voltage_v"] = (
        np.asarray(
            [
                [item.sample.nod1_voltage_v[root_id] for root_id in nod_axis]
                for item in observations
            ],
            dtype=float,
        ),
        "V",
        "exact four-channel motor-eligible canonical fly-FGS output",
    )
    retinal = [item.sample.retinal_input_luminance for item in observations]
    if all(item is not None for item in retinal):
        arrays["retinal_input_luminance"] = (
            np.asarray(retinal, dtype=float),
            "1",
            "canonical fly-FGS analytic T4a input; not calibrated irradiance",
        )
    full_voltage = [item.sample.full_cell_voltage_v for item in observations]
    full_activity = [item.sample.full_cell_activity for item in observations]
    if all(item is not None for item in full_voltage):
        arrays["full_circuit_voltage_v"] = (
            np.asarray(full_voltage, dtype=float),
            "V",
            "full canonical fly-FGS display state; ineligible as motor input",
        )
    if all(item is not None for item in full_activity):
        arrays["full_circuit_activity"] = (
            np.asarray(full_activity, dtype=float),
            "1",
            "full canonical fly-FGS display state; ineligible as motor input",
        )

    intervals = tuple(result.intervals)
    for interval in intervals:
        source_observation = observations[interval.bridge_tick_index // 10].sample
        holds = interval.bridge.source_hold
        if tuple(hold.root_id for hold in holds) != FLY_FGS_NOD1_ROOT_IDS:
            raise ValueError("bridge source hold does not retain all four NOD1 cells")
        for hold in holds:
            if (
                hold.raw_app_side.value != FLY_FGS_NOD1_RAW_APP_SIDES[hold.root_id]
                or hold.sample_index != source_observation.sample_index
                or abs(hold.measurement_time_s - source_observation.measurement_time_s)
                > _TIME_TOLERANCE_S
                or abs(hold.availability_time_s - source_observation.availability_time_s)
                > _TIME_TOLERANCE_S
                or hold.voltage_v != source_observation.nod1_voltage_v[hold.root_id]
            ):
                raise ValueError("bridge NOD1 hold is not the latest exact circuit sample")
    bridge_time = np.asarray(
        [item.interval_start_s for item in intervals], dtype=float
    )
    arrays["bridge_interval_start_s"] = (
        bridge_time,
        "s",
        "authoritative causal 0.5 ms bridge clock",
    )
    arrays["bridge_phase_path_unwrapped_rad"] = (
        np.asarray(
            [item.bridge.wing_phase_path_unwrapped_rad for item in intervals],
            dtype=float,
        ),
        "rad",
        "six-point exact mechanics phase path used for event crossings",
    )
    rate_records_list: List[Mapping[str, Any]] = []
    rate_arrays: Dict[str, List[List[float]]] = {}
    rate_presence: Dict[str, List[List[bool]]] = {}
    generated_history: Dict[
        Tuple[str, RawAppSide, str], List[StreamingRateSample]
    ] = {}
    rate_delay_by_stage: Dict[str, float] = {}
    for stage_name, stage, axis in (
        ("dn", StreamingBridgeStage.DN, _DN_RATE_AXIS),
        ("mn", StreamingBridgeStage.MN, _MN_RATE_AXIS),
    ):
        for prefix in ("generated", "held"):
            for field in ("rate_hz", "measurement_time_s", "availability_time_s"):
                rate_arrays[_rate_array_name(prefix, stage_name, field)] = []
            rate_presence["held_%s_rate_present" % stage_name] = []

    for interval in intervals:
        start_s = interval.interval_start_s
        interval_held_dn: Mapping[
            Tuple[RawAppSide, str], StreamingRateSample
        ] = {}
        for stage_name, stage, axis, generated_values, held_values in (
            (
                "dn",
                StreamingBridgeStage.DN,
                _DN_RATE_AXIS,
                interval.bridge.generated_dn_rates,
                interval.bridge.held_dn_rates,
            ),
            (
                "mn",
                StreamingBridgeStage.MN,
                _MN_RATE_AXIS,
                interval.bridge.generated_motor_rates,
                interval.bridge.held_motor_rates,
            ),
        ):
            generated = _rate_map(
                generated_values,
                axis,
                stage,
                complete=True,
                label="generated %s" % stage_name,
            )
            held = _rate_map(
                held_values,
                axis,
                stage,
                complete=False,
                label="held %s" % stage_name,
            )
            for key in axis:
                sample = generated[key]
                if (
                    abs(sample.measurement_time_s - start_s) > _TIME_TOLERANCE_S
                    or sample.availability_time_s + _TIME_TOLERANCE_S < start_s
                ):
                    raise ValueError("generated %s rate clock is inconsistent" % stage_name)
                delay = sample.availability_time_s - sample.measurement_time_s
                if stage_name not in rate_delay_by_stage:
                    rate_delay_by_stage[stage_name] = delay
                elif abs(delay - rate_delay_by_stage[stage_name]) > _TIME_TOLERANCE_S:
                    raise ValueError("generated %s rate delay changes during the run" % stage_name)
                generated_history.setdefault((stage_name, *key), []).append(sample)
                rate_records_list.append(
                    _rate_record(
                        sample,
                        record_kind="generated",
                        bridge_tick_index=interval.bridge_tick_index,
                    )
                )
            expected_held: Dict[
                Tuple[RawAppSide, str], StreamingRateSample
            ] = {}
            for key in axis:
                causal = [
                    sample
                    for sample in generated_history[(stage_name, *key)]
                    if sample.availability_time_s <= start_s + _TIME_TOLERANCE_S
                ]
                if causal:
                    expected_held[key] = causal[-1]
            if held != expected_held:
                raise ValueError("held %s rates are not the latest causal samples" % stage_name)
            if stage_name == "dn":
                source_measurement = observations[
                    interval.bridge_tick_index // 10
                ].sample.measurement_time_s
                if any(
                    sample.source_measurement_time_s != source_measurement
                    for sample in generated.values()
                ):
                    raise ValueError("DN rates do not retain the active NOD1 sample clock")
                interval_held_dn = held
            else:
                for lane, motor in axis:
                    held_dn = interval_held_dn.get((lane, "DNp26"))
                    expected_source = (
                        None
                        if held_dn is None
                        else held_dn.source_measurement_time_s
                    )
                    if generated[(lane, motor)].source_measurement_time_s != expected_source:
                        raise ValueError("MN rate source clock does not match its causal DN hold")
            for sample in held_values:
                rate_records_list.append(
                    _rate_record(
                        sample,
                        record_kind="held_at_interval_start",
                        bridge_tick_index=interval.bridge_tick_index,
                    )
                )
            for prefix, mapping in (("generated", generated), ("held", held)):
                present = [key in mapping for key in axis]
                for field in ("rate_hz", "measurement_time_s", "availability_time_s"):
                    rate_arrays[_rate_array_name(prefix, stage_name, field)].append(
                        [float(getattr(mapping[key], field)) if key in mapping else 0.0 for key in axis]
                    )
                if prefix == "held":
                    rate_presence["held_%s_rate_present" % stage_name].append(present)

    rate_records = tuple(rate_records_list)
    for name, rows in rate_arrays.items():
        unit = "Hz" if name.endswith("rate_hz") else "s"
        state = "generated" if name.startswith("generated") else "causally held"
        arrays[name] = (
            np.asarray(rows, dtype=float),
            unit,
            "%s inferred rate samples on the authoritative bridge clock" % state,
        )
    for name, rows in rate_presence.items():
        arrays[name] = (
            np.asarray(rows, dtype=bool),
            "1",
            "presence mask; held-rate numeric placeholders are zero where false",
        )

    generated_events: Dict[str, StreamingMotorEvent] = {}
    delivered_ids: set[str] = set()
    applied_ids: set[str] = set()
    suppressed_ids: set[str] = set()
    intervention_records_list: List[Mapping[str, Any]] = []
    mechanics_schedule_digest: Optional[str] = None
    for interval in intervals:
        held_motor_for_events = _rate_map(
            interval.bridge.held_motor_rates,
            _MN_RATE_AXIS,
            StreamingBridgeStage.MN,
            complete=False,
            label="held MN event source",
        )
        for event in interval.bridge.generated_events:
            if event.event_id in generated_events:
                raise ValueError("generated motor event ID was duplicated")
            if (
                event.event_time_s < interval.interval_start_s - _TIME_TOLERANCE_S
                or event.event_time_s >= interval.interval_end_s - _TIME_TOLERANCE_S
                or event.availability_time_s
                < interval.interval_end_s - _TIME_TOLERANCE_S
            ):
                raise ValueError("generated motor event clock is inconsistent")
            rate_source = held_motor_for_events.get(
                (event.raw_app_side, event.motor_neuron)
            )
            if (
                rate_source is None
                or event.muscle != event.motor_neuron.removeprefix("MN-")
                or event.rate_hz != rate_source.rate_hz
                or event.source_measurement_time_s
                != rate_source.source_measurement_time_s
            ):
                raise ValueError("generated motor event does not retain its held MN source")
            generated_events[event.event_id] = event
        interval_delivered = tuple(
            event.event_id for event in interval.bridge.delivered_events
        )
        if (
            len(set(interval_delivered)) != len(interval_delivered)
            or delivered_ids.intersection(interval_delivered)
        ):
            raise ValueError("motor event was delivered more than once")
        for event in interval.bridge.delivered_events:
            if (
                event.availability_time_s
                < interval.interval_start_s - _TIME_TOLERANCE_S
                or event.availability_time_s
                >= interval.interval_end_s - _TIME_TOLERANCE_S
            ):
                raise ValueError("delivered motor event lies outside its causal interval")
        delivered_ids.update(interval_delivered)
        interval_terminal_ids: List[str] = []
        for mechanics in interval.mechanics:
            frame_applied = tuple(mechanics.applied_event_ids)
            frame_suppressed = tuple(mechanics.suppressed_event_ids)
            if set(frame_applied).intersection(frame_suppressed):
                raise ValueError("mechanics event cannot be applied and suppressed")
            for event_id, disposition in (
                *((event_id, "applied") for event_id in frame_applied),
                *((event_id, "suppressed") for event_id in frame_suppressed),
            ):
                if event_id in applied_ids or event_id in suppressed_ids:
                    raise ValueError("mechanics event disposition was recorded more than once")
                event = generated_events.get(event_id)
                if event is None or event_id not in interval_delivered:
                    raise ValueError("mechanics disposition lacks a delivered event receipt")
                if (
                    event.availability_time_s
                    < mechanics.interval_start_s - _TIME_TOLERANCE_S
                    or event.availability_time_s
                    >= mechanics.interval_end_s - _TIME_TOLERANCE_S
                ):
                    raise ValueError("mechanics event disposition is on the wrong physics tick")
                (applied_ids if disposition == "applied" else suppressed_ids).add(
                    event_id
                )
                interval_terminal_ids.append(event_id)
            if mechanics_schedule_digest is None:
                mechanics_schedule_digest = mechanics.intervention_schedule_sha256
            elif mechanics.intervention_schedule_sha256 != mechanics_schedule_digest:
                raise ValueError("mechanics intervention schedule digest changes during a run")
            intervention_records_list.append(
                {
                    "stage": "mechanics_force",
                    "tick_index": mechanics.tick_index,
                    "interval_start_s": mechanics.interval_start_s,
                    "interval_end_s": mechanics.interval_end_s,
                    "active_intervention_ids": list(
                        mechanics.active_intervention_ids
                    ),
                    "schedule_sha256": mechanics.intervention_schedule_sha256,
                }
            )
        if set(interval_terminal_ids) != set(interval_delivered) or len(
            interval_terminal_ids
        ) != len(interval_delivered):
            raise ValueError("delivered events lack an exact mechanics disposition")
        intervention_records_list.append(
            {
                "stage": "bridge_command_generation",
                "tick_index": interval.bridge_tick_index,
                "interval_start_s": interval.interval_start_s,
                "interval_end_s": interval.interval_end_s,
                "active_intervention_ids": list(
                    interval.bridge.active_intervention_ids
                ),
                "schedule_sha256": None,
            }
        )
    unknown_delivery = delivered_ids.difference(generated_events)
    if unknown_delivery:
        raise ValueError("delivered motor event was absent from generated-event ledger")
    if applied_ids.intersection(suppressed_ids) or applied_ids.union(
        suppressed_ids
    ) != delivered_ids:
        raise ValueError("mechanics event disposition ledger is incomplete")
    event_records = tuple(
        _event_dict(
            event,
            "applied"
            if event_id in applied_ids
            else "suppressed"
            if event_id in suppressed_ids
            else "pending_at_episode_end",
        )
        for event_id, event in sorted(
            generated_events.items(), key=lambda item: (item[1].event_time_s, item[0])
        )
    )
    intervention_records = tuple(intervention_records_list)
    if event_records:
        arrays["motor_event_numeric"] = (
            np.asarray(
                [
                    [
                        record["event_time_s"],
                        record["availability_time_s"],
                        record["wingbeat_phase_rad"],
                        record["rate_hz"],
                        record["emission_probability"],
                        0.0
                        if record["source_measurement_time_s"] is None
                        else record["source_measurement_time_s"],
                        float(record["phase_crossing_index"]),
                        1.0 if record["delivered_to_mechanics"] else 0.0,
                        1.0 if record["applied_to_muscle_state"] else 0.0,
                        1.0
                        if record["suppressed_by_mechanics_intervention"]
                        else 0.0,
                    ]
                    for record in event_records
                ],
                dtype=float,
            ),
            "mixed; see column_units",
            "seeded-synthetic phase-gated event ledger with exact causal times",
        )

    circuit_records = tuple(
        {
            "sample_index": observation.sample.sample_index,
            "measurement_time_s": observation.sample.measurement_time_s,
            "availability_time_s": observation.sample.availability_time_s,
            "initialization_mode": observation.initialization_mode,
            "requested_control": dict(observation.requested_control.to_dict()),
            "wrapped_world_yaw_rad": observation.wrapped_world_yaw_rad,
            "unwrapped_world_yaw_rad": observation.unwrapped_world_yaw_rad,
            "world_z_yaw_rate_rad_s": observation.world_z_yaw_rate_rad_s,
            "nod1_voltage_v": dict(observation.sample.nod1_voltage_v),
            "pooled_readout": _strict_json_copy(
                dict(observation.sample.pooled_readout), "pooled readout"
            ),
            "retinal_input_included": observation.sample.retinal_input_luminance
            is not None,
            "full_cell_state_included": observation.sample.full_cell_voltage_v
            is not None,
            "full_cell_state_eligible_motor_input": False,
        }
        for observation in observations
    )

    axis_metadata: Dict[str, Mapping[str, Any]] = {
        "body_position_world_m": {"axis_labels": ["world_x", "world_y", "world_z"]},
        "body_velocity_world_m_s": {"axis_labels": ["world_x", "world_y", "world_z"]},
        "body_angular_velocity_body_rad_s": {
            "axis_labels": ["body_x", "body_y", "body_z"]
        },
        "aerodynamic_force_body_n": {"axis_labels": ["body_x", "body_y", "body_z"]},
        "aerodynamic_torque_body_n_m": {
            "axis_labels": ["body_x", "body_y", "body_z"]
        },
        "nod1_voltage_v": {
            "axis_name": "motor_eligible_nod1_root",
            "axis_labels": list(nod_axis),
            "root_ids": list(nod_axis),
            "side_semantics": "raw application L/R only; anatomy unknown",
            "sample_axis": "circuit_measurement_time_s",
            "eligible_motor_input": True,
        },
        "generated_dn_rate_hz": {
            "axis_labels": [
                "raw_app_%s:%s" % (lane.value, target)
                for lane, target in _DN_RATE_AXIS
            ]
        },
        "generated_mn_rate_hz": {
            "axis_labels": [
                "%s:%s" % ("raw_app_" + lane.value, motor)
                for lane in _RAW_LANE_ORDER
                for motor in _MOTOR_ORDER
            ]
        },
        "motor_event_numeric": {
            "column_labels": [
                "event_time_s",
                "availability_time_s",
                "wingbeat_phase_rad",
                "rate_hz",
                "emission_probability",
                "source_measurement_time_s_or_zero",
                "phase_crossing_index",
                "delivered_to_mechanics",
                "applied_to_muscle_state",
                "suppressed_by_mechanics_intervention",
            ],
            "column_units": [
                "s",
                "s",
                "rad",
                "Hz",
                "1",
                "s",
                "1",
                "1",
                "1",
                "1",
            ],
            "source_measurement_time_presence": [
                record["source_measurement_time_s"] is not None
                for record in event_records
            ],
        },
    }
    axis_metadata.update(_display_array_axis_metadata(observations))
    for prefix in ("generated", "held"):
        for suffix in (
            "rate_hz",
            "rate_measurement_time_s",
            "rate_availability_time_s",
        ):
            axis_metadata["%s_dn_%s" % (prefix, suffix)] = {
                "axis_labels": [
                    "raw_app_%s:%s" % (lane.value, target)
                    for lane, target in _DN_RATE_AXIS
                ]
            }
            axis_metadata["%s_mn_%s" % (prefix, suffix)] = {
                "axis_labels": [
                    "raw_app_%s:%s" % (lane.value, motor)
                    for lane, motor in _MN_RATE_AXIS
                ]
            }
    axis_metadata["held_dn_rate_present"] = axis_metadata["held_dn_rate_hz"]
    axis_metadata["held_mn_rate_present"] = axis_metadata["held_mn_rate_hz"]
    for name in tuple(arrays):
        if name.startswith("raw_app_actuation_") or name.startswith(
            "physical_actuation_"
        ):
            width = arrays[name][0].shape[-1]
            if width == 2:
                axis_metadata[name] = {
                    "axis_labels": (
                        ["raw_app_L", "raw_app_R"]
                        if name.startswith("raw_app_")
                        else ["physical_left", "physical_right"]
                    ),
                    "sample_semantics": "left-boundary zero-order hold over the transition",
                }
            elif width == 6:
                axis_metadata[name] = {
                    "axis_labels": (
                        [
                            "raw_app_L_x",
                            "raw_app_L_y",
                            "raw_app_L_z",
                            "raw_app_R_x",
                            "raw_app_R_y",
                            "raw_app_R_z",
                        ]
                        if name.startswith("raw_app_")
                        else [
                            "physical_left_x",
                            "physical_left_y",
                            "physical_left_z",
                            "physical_right_x",
                            "physical_right_y",
                            "physical_right_z",
                        ]
                    ),
                    "sample_semantics": "left-boundary zero-order hold over the transition",
                }
    for name in tuple(arrays):
        if name.startswith("muscle_") and name != "muscle_natural_endpoint_time_s":
            axis_metadata[name] = {
                "axis_labels": [_raw_key_to_lane_id(key) for key in muscle_axis],
                "anatomical_side": "unknown",
            }

    for name, (values, _unit, _provenance) in arrays.items():
        array = np.asarray(values)
        if array.dtype.hasobject or not np.all(np.isfinite(array)):
            raise ValueError("canonical array %s is not finite numeric data" % name)
        metadata = axis_metadata.get(name)
        if metadata is not None and "axis_labels" in metadata:
            labels = metadata["axis_labels"]
            if (
                not isinstance(labels, list)
                or array.ndim == 0
                or len(labels) != array.shape[-1]
                or len(set(labels)) != len(labels)
            ):
                raise ValueError(
                    "canonical array %s axis identity metadata is inconsistent" % name
                )

    return CanonicalArrayBundle(
        arrays=arrays,
        axis_metadata=axis_metadata,
        event_records=event_records,
        rate_records=rate_records,
        circuit_records=circuit_records,
        intervention_records=intervention_records,
    )


def _canonical_bundle_sha256(bundle: CanonicalArrayBundle) -> str:
    """Hash every authoritative value and its interpretation metadata."""

    digest = hashlib.sha256()
    for name in sorted(bundle.arrays):
        values, unit, provenance = bundle.arrays[name]
        array = np.asarray(values)
        if array.dtype.hasobject or not np.all(np.isfinite(array)):
            raise ValueError("canonical content hash requires finite numeric arrays")
        canonical_dtype = array.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
        header = {
            "name": name,
            "dtype": canonical.dtype.str,
            "shape": list(canonical.shape),
            "unit": unit,
            "provenance": provenance,
            "axis_metadata": bundle.axis_metadata.get(name),
        }
        digest.update(
            json.dumps(
                header,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(canonical.tobytes(order="C"))
        digest.update(b"\0")
    tables = {
        "event_records": bundle.event_records,
        "rate_records": bundle.rate_records,
        "circuit_records": bundle.circuit_records,
        "intervention_records": bundle.intervention_records,
    }
    digest.update(
        json.dumps(
            tables,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _canonical_result_content_sha256(
    result: CanonicalClosedLoopResult,
    bundle: Optional[CanonicalArrayBundle] = None,
) -> str:
    bundle_digest = _canonical_bundle_sha256(
        canonical_array_bundle(result) if bundle is None else bundle
    )
    receipt = {
        "bundle_sha256": bundle_digest,
        "validation_status": result.validation_status,
        "source_limitations": list(result.source_limitations),
        "backend_name": result.backend_name,
        "aerodynamic_owner": result.aerodynamic_owner,
        "effector_mapping": dict(result.effector_mapping_receipt.to_dict()),
    }
    return hashlib.sha256(
        json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _checkpoint_receipt(
    checkpoint: Optional[CanonicalClosedLoopCheckpoint],
    result: CanonicalClosedLoopResult,
) -> Optional[Mapping[str, Any]]:
    if checkpoint is None:
        return None
    if not isinstance(checkpoint, CanonicalClosedLoopCheckpoint):
        raise TypeError("checkpoint must be CanonicalClosedLoopCheckpoint")
    payload = checkpoint.to_dict()
    state = payload["state"]
    if (
        payload["config"] != _strict_json_copy(
            dict(result.config.to_dict()), "canonical result config"
        )
        or state["bridge_tick_index"] != result.final_bridge_tick_index
        or state["physics_tick_index"] != result.final_bridge_tick_index * 5
        or state["circuit_sample_count"] != result.circuit_sample_count
        or state["last_circuit_sample_index"] != result.circuit_sample_count - 1
    ):
        raise ValueError("checkpoint clocks/configuration do not match the result endpoint")
    if not _body_equal(_body_from_dict(state["body_state"]), result.final_body_state):
        raise ValueError("checkpoint body state does not match the result endpoint")
    receipts = payload["receipts"]
    if (
        receipts["backend_name"] != result.backend_name
        or receipts["aerodynamic_owner"] != result.aerodynamic_owner
        or receipts["articulated_physics_telemetry_available"]
        is not result.articulated_physics_telemetry_available
        or receipts["effector_mapping"]
        != _strict_json_copy(
            dict(result.effector_mapping_receipt.to_dict()),
            "result effector mapping",
        )
        or tuple(receipts["source_limitations"]) != tuple(result.source_limitations)
    ):
        raise ValueError("checkpoint runtime receipts do not match the result")
    component_digests: Dict[str, str] = {}
    for name, child in payload["components"].items():
        if not isinstance(child, Mapping):
            raise ValueError("checkpoint component payload must be a mapping")
        supplied = child.get("payload_sha256")
        if not isinstance(supplied, str):
            raise ValueError("checkpoint component digest is invalid")
        if name == "physics" and supplied.startswith("sha256:"):
            normalized_supplied = supplied.removeprefix("sha256:")
        else:
            normalized_supplied = supplied
        if (
            len(normalized_supplied) != 64
            or any(
                character not in "0123456789abcdef"
                for character in normalized_supplied
            )
            or (name != "physics" and normalized_supplied != supplied)
        ):
            raise ValueError("checkpoint component digest is invalid")
        computed = _checkpoint_component_payload_sha256(name, child)
        if computed != normalized_supplied:
            raise ValueError("checkpoint component payload SHA-256 mismatch: %s" % name)
        # Component implementations retain their native spelling in the full
        # checkpoint.  The receipt uses one uniform bare-hex representation;
        # FlyBody's native ``sha256:`` prefix is removed only after its payload
        # has been independently recomputed and verified above.
        component_digests[name] = normalized_supplied

    bridge_contracts = payload["components"]["bridge"].get("interventions", [])
    mechanics_contracts = payload["components"]["mechanics"]["config"].get(
        "muscle_interventions", []
    )
    for label, contracts in (
        ("bridge", bridge_contracts),
        ("mechanics", mechanics_contracts),
    ):
        ids = (
            []
            if not isinstance(contracts, list)
            else [
                contract.get("intervention_id")
                if isinstance(contract, Mapping)
                else None
                for contract in contracts
            ]
        )
        if (
            not isinstance(contracts, list)
            or any(not isinstance(item, str) or not item for item in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("checkpoint %s intervention inventory is invalid" % label)

    def expected_active_ids(contracts: Sequence[Mapping[str, Any]], time_s: float) -> Tuple[str, ...]:
        ids: List[str] = []
        for contract in contracts:
            intervention_id = contract.get("intervention_id")
            start = contract.get("start_s")
            end = contract.get("end_s")
            if (
                not isinstance(intervention_id, str)
                or not intervention_id
                or isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not math.isfinite(float(start))
                or not math.isfinite(float(end))
                or float(start) < 0.0
                or float(end) <= float(start)
            ):
                raise ValueError("checkpoint intervention contract is invalid")
            if float(start) <= time_s < float(end):
                ids.append(intervention_id)
        if len(set(ids)) != len(ids):
            raise ValueError("checkpoint intervention IDs are duplicated")
        return tuple(ids)

    for contract in mechanics_contracts:
        contract_digest = contract.get("contract_sha256")
        unsigned_contract = {
            key: value for key, value in contract.items() if key != "contract_sha256"
        }
        if contract_digest != hashlib.sha256(
            json.dumps(
                unsigned_contract,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest():
            raise ValueError("checkpoint mechanics intervention contract digest mismatch")
    expected_mechanics_schedule_digest = hashlib.sha256(
        json.dumps(
            mechanics_contracts,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    for interval in result.intervals:
        if tuple(interval.bridge.active_intervention_ids) != expected_active_ids(
            bridge_contracts, interval.interval_start_s
        ):
            raise ValueError("bridge intervention activity does not match checkpoint contracts")
        for mechanics in interval.mechanics:
            if (
                mechanics.intervention_schedule_sha256
                != expected_mechanics_schedule_digest
                or tuple(mechanics.active_intervention_ids)
                != expected_active_ids(mechanics_contracts, mechanics.interval_start_s)
            ):
                raise ValueError(
                    "mechanics intervention activity does not match checkpoint contracts"
                )

    delivered_ids = {
        event.event_id
        for interval in result.intervals
        for event in interval.bridge.delivered_events
    }
    applied_ids = {
        event_id
        for interval in result.intervals
        for mechanics in interval.mechanics
        for event_id in mechanics.applied_event_ids
    }
    suppressed_ids = {
        event_id
        for interval in result.intervals
        for mechanics in interval.mechanics
        for event_id in mechanics.suppressed_event_ids
    }
    generated_ids = {
        event.event_id
        for interval in result.intervals
        for event in interval.bridge.generated_events
    }
    generated_event_payloads = {
        event.event_id: _event_checkpoint_dict(event)
        for interval in result.intervals
        for event in interval.bridge.generated_events
    }
    bridge_state = payload["components"]["bridge"]["state"]
    mechanics_state = payload["components"]["mechanics"]["state"]
    endpoint_time_s = result.final_bridge_tick_index * result.config.bridge_dt_s
    expected_phase_history = [
        list(interval.bridge.wing_phase_path_unwrapped_rad)
        for interval in result.intervals
    ]
    final_observation = _circuit_observations(result)[-1].sample
    checkpoint_delivered_ids = {
        event["event_id"] for event in bridge_state["delivered_events"]
    }
    checkpoint_pending_ids = {
        item["value"]["event_id"] for item in bridge_state["event_pending"]
    }
    checkpoint_delivered_payloads = {
        event["event_id"]: event for event in bridge_state["delivered_events"]
    }
    checkpoint_pending_payloads = {
        item["value"]["event_id"]: item["value"]
        for item in bridge_state["event_pending"]
    }
    checkpoint_applied_payloads = {
        event["event_id"]: event for event in mechanics_state["applied_events"]
    }
    checkpoint_suppressed_payloads = {
        event["event_id"]: event for event in mechanics_state["suppressed_events"]
    }
    if (
        bridge_state["tick_index"] != result.final_bridge_tick_index
        or mechanics_state["tick_index"] != result.final_bridge_tick_index * 5
        or bridge_state["wing_phase_path_history"] != expected_phase_history
        or abs(
            float(bridge_state["wing_phase_unwrapped_rad"])
            - result.intervals[-1].bridge.wing_phase_end_unwrapped_rad
        )
        > _TIME_TOLERANCE_S
        or abs(
            float(mechanics_state["phase_unwrapped_rad"])
            - result.intervals[-1].mechanics[-1].phase_end_unwrapped_rad
        )
        > _TIME_TOLERANCE_S
        or bridge_state["last_pushed"]
        != {
            "sample_index": final_observation.sample_index,
            "measurement_time_s": final_observation.measurement_time_s,
            "availability_time_s": final_observation.availability_time_s,
        }
        or len(checkpoint_delivered_payloads)
        != len(bridge_state["delivered_events"])
        or len(checkpoint_pending_payloads) != len(bridge_state["event_pending"])
        or len(checkpoint_applied_payloads) != len(mechanics_state["applied_events"])
        or len(checkpoint_suppressed_payloads)
        != len(mechanics_state["suppressed_events"])
        or checkpoint_delivered_ids != delivered_ids
        or checkpoint_pending_ids != generated_ids.difference(delivered_ids)
        or checkpoint_delivered_payloads
        != {
            event_id: generated_event_payloads[event_id]
            for event_id in delivered_ids
        }
        or checkpoint_pending_payloads
        != {
            event_id: generated_event_payloads[event_id]
            for event_id in generated_ids.difference(delivered_ids)
        }
        or set(mechanics_state["applied_event_ids"]) != applied_ids
        or set(mechanics_state["suppressed_event_ids"]) != suppressed_ids
        or checkpoint_applied_payloads
        != {
            event_id: generated_event_payloads[event_id]
            for event_id in applied_ids
        }
        or checkpoint_suppressed_payloads
        != {
            event_id: generated_event_payloads[event_id]
            for event_id in suppressed_ids
        }
        or mechanics_state["pending_events"]
        or tuple(bridge_state["active_intervention_ids"])
        != expected_active_ids(bridge_contracts, endpoint_time_s)
        or tuple(mechanics_state["active_intervention_ids"])
        != expected_active_ids(mechanics_contracts, endpoint_time_s)
        or mechanics_state["intervention_schedule_sha256"]
        != expected_mechanics_schedule_digest
    ):
        raise ValueError("checkpoint event/component state does not match the result")
    return {
        "schema_version": payload["schema_version"],
        "runtime_version": payload["runtime_version"],
        "payload_sha256": payload["payload_sha256"],
        "bridge_tick_index": payload["state"]["bridge_tick_index"],
        "physics_tick_index": payload["state"]["physics_tick_index"],
        "component_payload_sha256": component_digests,
        "intervention_contracts": {
            "bridge_command_generation": bridge_contracts,
            "mechanics_force": mechanics_contracts,
            "mechanics_schedule_sha256": mechanics_state[
                "intervention_schedule_sha256"
            ],
        },
    }


def canonical_run_id(
    result: CanonicalClosedLoopResult,
    *,
    scenario_id: str = "canonical_closed_loop",
    checkpoint: Optional[CanonicalClosedLoopCheckpoint] = None,
) -> str:
    """Return the deterministic identity used by the immutable writer."""

    if not isinstance(result, CanonicalClosedLoopResult):
        raise TypeError("result must be CanonicalClosedLoopResult")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be a non-empty string")
    checkpoint_receipt = _checkpoint_receipt(checkpoint, result)
    scientific_content_sha256 = _canonical_result_content_sha256(result)
    identity = {
        "scenario_id": scenario_id,
        "config": dict(result.config.to_dict()),
        "backend_name": result.backend_name,
        "aerodynamic_owner": result.aerodynamic_owner,
        "effector_mapping": dict(result.effector_mapping_receipt.to_dict()),
        "initial_bridge_tick_index": result.initial_bridge_tick_index,
        "final_bridge_tick_index": result.final_bridge_tick_index,
        "scientific_content_sha256": scientific_content_sha256,
        "checkpoint_payload_sha256": (
            None if checkpoint_receipt is None else checkpoint_receipt["payload_sha256"]
        ),
    }
    encoded = json.dumps(
        identity,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "%s-%s" % (scenario_id, hashlib.sha256(encoded).hexdigest()[:20])


def canonical_web_projection_sha256(replay: Mapping[str, Any]) -> str:
    """Hash a replay while excluding its circular artifact-reference fields."""

    if not isinstance(replay, Mapping):
        raise TypeError("replay must be a mapping")
    projected = {
        key: value
        for key, value in replay.items()
        if key not in _WEB_PROJECTION_EXCLUDED_FIELDS
    }
    encoded = json.dumps(
        projected,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_canonical_closed_loop_artifact(
    output_dir: Path,
    result: CanonicalClosedLoopResult,
    *,
    checkpoint: Optional[CanonicalClosedLoopCheckpoint] = None,
    scenario_id: str = "canonical_closed_loop",
    created_at_utc: Optional[str] = None,
    chunk_samples: int = 256,
    runtime_receipts: Optional[Mapping[str, Any]] = None,
    web_replay: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Atomically write an immutable multi-clock canonical scientific run."""

    if not isinstance(result, CanonicalClosedLoopResult):
        raise TypeError("result must be CanonicalClosedLoopResult")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be a non-empty string")
    if isinstance(chunk_samples, bool) or not isinstance(chunk_samples, int) or chunk_samples < 1:
        raise ValueError("chunk_samples must be a positive integer")
    receipts = _strict_json_copy(dict(runtime_receipts or {}), "runtime receipts")
    bundle = canonical_array_bundle(result)
    scientific_content_sha256 = _canonical_result_content_sha256(result, bundle)
    checkpoint_receipt = _checkpoint_receipt(checkpoint, result)
    run_id = canonical_run_id(
        result, scenario_id=scenario_id, checkpoint=checkpoint
    )
    web_projection_receipt = None
    if web_replay is not None:
        replay_copy = _strict_json_copy(dict(web_replay), "canonical web replay")
        if replay_copy.get("schema_version") != WEB_REPLAY_SCHEMA_VERSION:
            raise ValueError("canonical web replay schema is unsupported")
        if replay_copy.get("source_run_id") != run_id:
            raise ValueError("canonical web replay source_run_id does not match run identity")
        circular = [
            field for field in _WEB_PROJECTION_EXCLUDED_FIELDS if field in replay_copy
        ]
        if circular:
            raise ValueError(
                "artifact-reference fields are added only after immutable commit: %s"
                % ", ".join(circular)
            )
        web_projection_receipt = {
            "schema_version": "1.0.0",
            "sha256": canonical_web_projection_sha256(replay_copy),
            "canonicalization": CANONICAL_WEB_PROJECTION_CANONICALIZATION,
            "excluded_top_level_fields": list(_WEB_PROJECTION_EXCLUDED_FIELDS),
        }
    output_dir = Path(output_dir)
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError("canonical artifact target must be an empty directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=str(output_dir.parent), prefix=".%s.staging-" % output_dir.name)
    )
    try:
        descriptors: Dict[str, Any] = {}
        for name, (values, unit, provenance) in bundle.arrays.items():
            descriptor = _write_chunked_array(
                staging,
                name,
                np.asarray(values),
                unit,
                chunk_samples,
                provenance,
            )
            if name in bundle.axis_metadata:
                descriptor = dict(descriptor, **bundle.axis_metadata[name])
            descriptors[name] = descriptor

        event_path = staging / "motor_events.json"
        _json_dump(event_path, {"schema_version": "1.0.0", "events": list(bundle.event_records)})
        rate_path = staging / "rate_samples.json"
        _json_dump(
            rate_path,
            {"schema_version": "1.0.0", "samples": list(bundle.rate_records)},
        )
        circuit_path = staging / "circuit_observations.json"
        _json_dump(
            circuit_path,
            {
                "schema_version": "1.0.0",
                "observations": list(bundle.circuit_records),
            },
        )
        intervention_path = staging / "intervention_activity.json"
        _json_dump(
            intervention_path,
            {
                "schema_version": "1.0.0",
                "records": list(bundle.intervention_records),
                "contract_notice": (
                    "Active-ID receipts are complete; immutable definitions are in the "
                    "checkpoint receipt when an endpoint checkpoint was supplied."
                ),
            },
        )
        checkpoint_descriptor = None
        if checkpoint is not None:
            checkpoint_path = staging / "checkpoint.json"
            _json_dump(checkpoint_path, checkpoint.to_dict())
            checkpoint_descriptor = {
                **dict(checkpoint_receipt or {}),
                "path": checkpoint_path.name,
                "file_sha256": sha256_file(checkpoint_path),
            }

        manifest: Dict[str, Any] = {
            "schema_version": CANONICAL_ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "scenario_id": scenario_id,
            "created_at_utc": created_at_utc or _utc_now(),
            "episode_status": "complete",
            "validation_status": result.validation_status,
            "scientific_content_sha256": scientific_content_sha256,
            "source_kind": CANONICAL_ARTIFACT_SOURCE_KIND,
            "authority_notice": (
                "Exact canonical fly-FGS/bridge/mechanics/physics execution; neural-to-muscle "
                "and hinge parameters remain exploratory and anatomy/laterality unresolved."
            ),
            "configuration": dict(result.config.to_dict()),
            "clocks": {
                "circuit_dt_s": result.config.circuit_dt_s,
                "bridge_dt_s": result.config.bridge_dt_s,
                "physics_dt_s": result.config.physics_dt_s,
                "circuit_to_bridge_ratio": 10,
                "bridge_to_physics_ratio": 5,
            },
            "runtime": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "canonical_schema_version": CANONICAL_CLOSED_LOOP_SCHEMA_VERSION,
                "canonical_runtime_version": CANONICAL_CLOSED_LOOP_RUNTIME_VERSION,
                "physics_backend": result.backend_name,
                "aerodynamic_owner": result.aerodynamic_owner,
                "receipts": receipts,
            },
            "effector_mapping": dict(result.effector_mapping_receipt.to_dict()),
            "signed_behavior_claim_policy": result.config.effector_signed_behavior_claim_policy,
            "source_limitations": list(result.source_limitations),
            "units_policy": "SI internally; every array descriptor declares its unit",
            "arrays": descriptors,
            "tables": {
                "motor_events": {
                    "path": event_path.name,
                    "sha256": sha256_file(event_path),
                    "count": len(bundle.event_records),
                    "timing_semantics": "discrete; never interpolate",
                },
                "rate_samples": {
                    "path": rate_path.name,
                    "sha256": sha256_file(rate_path),
                    "count": len(bundle.rate_records),
                    "timing_semantics": "separate measurement and causal availability clocks",
                },
                "circuit_observations": {
                    "path": circuit_path.name,
                    "sha256": sha256_file(circuit_path),
                    "count": len(bundle.circuit_records),
                },
                "intervention_activity": {
                    "path": intervention_path.name,
                    "sha256": sha256_file(intervention_path),
                    "count": len(bundle.intervention_records),
                    "timing_semantics": "exact half-open component intervals",
                    "definitions": (
                        None
                        if checkpoint_receipt is None
                        else checkpoint_receipt["intervention_contracts"]
                    ),
                },
            },
            "checkpoint": checkpoint_descriptor,
            "web_replay_projection": web_projection_receipt,
        }
        manifest_path = staging / "manifest.json"
        _json_dump(manifest_path, manifest)
        manifest_digest = sha256_file(manifest_path)
        (staging / "manifest.sha256").write_text(
            "%s  manifest.json\n" % manifest_digest, encoding="ascii"
        )
        if output_dir.exists() and (
            not output_dir.is_dir() or any(output_dir.iterdir())
        ):
            raise FileExistsError("canonical artifact target became non-empty")
        os.replace(str(staging), str(output_dir))
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def bind_canonical_web_replay_to_artifact(
    replay: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
) -> Mapping[str, Any]:
    """Attach immutable artifact references without changing projection hash."""

    replay_copy = _strict_json_copy(dict(replay), "canonical web replay")
    if replay_copy.get("source_run_id") != manifest.get("run_id"):
        raise ValueError("web replay and artifact run identities do not match")
    if (
        not isinstance(manifest_file_sha256, str)
        or len(manifest_file_sha256) != 64
        or any(character not in "0123456789abcdef" for character in manifest_file_sha256)
    ):
        raise ValueError("manifest_file_sha256 must be a SHA-256 digest")
    manifest_copy = _strict_json_copy(dict(manifest), "canonical artifact manifest")
    encoded_manifest = (
        json.dumps(
            manifest_copy,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            separators=(",", ": "),
        ).encode("utf-8")
        + b"\n"
    )
    if hashlib.sha256(encoded_manifest).hexdigest() != manifest_file_sha256:
        raise ValueError("manifest_file_sha256 does not bind the supplied manifest")
    if manifest_copy.get("schema_version") != CANONICAL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("canonical artifact manifest schema is unsupported")
    receipt = manifest.get("web_replay_projection")
    if not isinstance(receipt, Mapping):
        raise ValueError("artifact does not contain a web replay projection receipt")
    if canonical_web_projection_sha256(replay_copy) != receipt.get("sha256"):
        raise ValueError("web replay projection does not match the artifact receipt")
    replay_copy["source_artifact_manifest_sha256"] = manifest_file_sha256
    replay_copy["source_artifact_schema_version"] = manifest.get("schema_version")
    if canonical_web_projection_sha256(replay_copy) != receipt.get("sha256"):
        raise RuntimeError("artifact reference changed the excluded web projection")
    return _strict_json_copy(replay_copy, "bound canonical web replay")


def _latest_observation_by_time(
    result: CanonicalClosedLoopResult, times: np.ndarray
) -> Tuple[Any, ...]:
    observations = _circuit_observations(result)
    output: List[Any] = []
    cursor = 0
    for time_s in times:
        while (
            cursor + 1 < len(observations)
            and observations[cursor + 1].sample.availability_time_s
            <= time_s + _TIME_TOLERANCE_S
        ):
            cursor += 1
        output.append(observations[cursor])
    return tuple(output)


def _rate_lookup(samples: Iterable[Any]) -> Mapping[str, float]:
    return {
        "%s:%s" % ("raw_app_" + sample.raw_app_side.value, sample.target_name): float(
            sample.rate_hz
        )
        for sample in samples
    }


def _normalize_display(raw: Sequence[Optional[float]], scale: float) -> np.ndarray:
    values = np.asarray([0.0 if value is None else float(value) for value in raw])
    return np.clip(np.abs(values) / max(scale, 1.0e-30), 0.0, 1.0)


def _online_circuit_replay_attachment(
    result: CanonicalClosedLoopResult,
    bundle: CanonicalArrayBundle,
    observations: Sequence[Any],
) -> Optional[Mapping[str, Any]]:
    """Build the rich display-only attachment for registered online state.

    This deliberately has a different source kind and field name from the
    frozen open-loop ``circuit_replay`` attachment.  It contains values that
    were actually sampled during this closed-loop episode.  The only signals
    eligible to leave the circuit for motor generation remain the four exact
    NOD1 voltages already recorded by the bridge.
    """

    if "full_circuit_voltage_v" not in bundle.arrays:
        return None
    full_axis = bundle.axis_metadata.get("full_circuit_voltage_v", {})
    if (
        full_axis.get("axis_registration")
        != "registered_fly_fgs_circuit_bundle_order"
    ):
        return None
    contract = _registered_fly_fgs_display_contract()
    cell_ids = list(contract["cell_ids"])
    if full_axis.get("axis_labels") != cell_ids:
        raise ValueError("online circuit replay full-state axis is not registered")

    full_voltage = np.asarray(bundle.arrays["full_circuit_voltage_v"][0], dtype=float)
    full_activity = np.asarray(
        bundle.arrays["full_circuit_activity"][0], dtype=float
    )
    if full_voltage.shape != full_activity.shape or full_voltage.shape != (
        len(observations),
        len(cell_ids),
    ):
        raise ValueError("online circuit replay full-state matrices are inconsistent")

    retinal_values: Optional[np.ndarray] = None
    retinal_axis: Optional[Mapping[str, Any]] = None
    if "retinal_input_luminance" in bundle.arrays:
        retinal_axis = bundle.axis_metadata.get("retinal_input_luminance", {})
        if (
            retinal_axis.get("axis_registration")
            != "registered_fly_fgs_circuit_bundle_order"
        ):
            raise ValueError("online circuit replay retinal axis is not registered")
        retinal_values = np.asarray(
            bundle.arrays["retinal_input_luminance"][0], dtype=float
        )
        if retinal_values.shape != (len(observations), 1441):
            raise ValueError("online circuit replay retinal matrix is inconsistent")

    measurement_time_s = np.asarray(
        bundle.arrays["circuit_measurement_time_s"][0], dtype=float
    )
    availability_time_s = np.asarray(
        bundle.arrays["circuit_availability_time_s"][0], dtype=float
    )
    causal_samples = []
    for index, observation in enumerate(observations):
        sample = observation.sample
        causal_samples.append(
            {
                "sample_index": sample.sample_index,
                "measurement_time_s": sample.measurement_time_s,
                "availability_time_s": sample.availability_time_s,
                "initialization_mode": observation.initialization_mode,
                "requested_control": dict(observation.requested_control.to_dict()),
                "applied_circuit_control": dict(sample.last_control.to_dict()),
                "control_semantics": (
                    "captured static-scene pre-roll gauge at reset"
                    if index == 0
                    else "body/scene control causally applied for this 5 ms circuit step"
                ),
                "wrapped_world_yaw_rad": observation.wrapped_world_yaw_rad,
                "unwrapped_world_yaw_rad": observation.unwrapped_world_yaw_rad,
                "world_z_yaw_rate_rad_s": observation.world_z_yaw_rate_rad_s,
                "nod1_voltage_v": dict(sample.nod1_voltage_v),
                "pooled_readout": _strict_json_copy(
                    dict(sample.pooled_readout), "online circuit pooled readout"
                ),
            }
        )

    topology = _strict_json_copy(
        contract["circuit_topology"], "online circuit topology"
    )
    unsigned: Dict[str, Any] = {
        "schema_version": ONLINE_CIRCUIT_REPLAY_SCHEMA_VERSION,
        "source_kind": "registered_fly_fgs_online_circuit_replay",
        "snapshot_id": contract["snapshot_id"],
        "display_only": True,
        "eligible_motor_input": False,
        "source_receipts": {
            "source_manifest_sha256": contract["source_manifest_sha256"],
            "source_manifest_uri": contract["source_manifest_uri"],
            "asset_receipts": _strict_json_copy(
                contract["asset_receipts"], "online circuit asset receipts"
            ),
        },
        "dataset": _strict_json_copy(contract["dataset"], "online circuit dataset"),
        "cell_inventory": _strict_json_copy(
            contract["cell_inventory"], "online circuit inventory"
        ),
        "circuit_topology": topology,
        "axis_registration": {
            "cell_axis_sha256": contract["cell_axis_sha256"],
            "cell_ids_sha256": contract["cell_ids_sha256"],
            "retinal_axis_sha256": contract["retinal_axis_sha256"],
            "cell_axis_source": "registered content-addressed fly-FGS circuit bundle",
            "full_state_axis_labels": cell_ids,
            "retinal_axis_labels": list(
                topology["retinotopic_t4a"]["cell_ids"]
            ),
        },
        "clocks": {
            "circuit_dt_s": result.config.circuit_dt_s,
            "sample_count": len(observations),
            "measurement_time_s": measurement_time_s.tolist(),
            "availability_time_s": availability_time_s.tolist(),
            "episode_interval_end_s": result.config.duration_s,
            "sample_semantics": (
                "sample 0 is the registered static pre-roll gauge; sample i>=1 is the "
                "exact online state after the causal body/scene control; the last sample "
                "is held to the episode boundary"
            ),
        },
        "causal_samples": causal_samples,
        "retinal_input": (
            None
            if retinal_values is None
            else {
                "cell_indices": list(
                    topology["retinotopic_t4a"]["cell_indices"]
                ),
                "cell_ids": list(topology["retinotopic_t4a"]["cell_ids"]),
                "azimuth_deg": list(
                    topology["retinotopic_t4a"]["azimuth_deg"]
                ),
                "elevation_deg": list(
                    topology["retinotopic_t4a"]["elevation_deg"]
                ),
                "luminance": retinal_values.tolist(),
                "unit": "1",
                "sampling_semantics": (
                    "online analytic endpoint luminance in exact registered T4a order; "
                    "not calibrated compound-eye optics or radiometry"
                ),
                "eligible_motor_input": False,
            }
        ),
        "full_cell_state": {
            "cell_ids": cell_ids,
            "voltage_v": full_voltage.tolist(),
            "activity": full_activity.tolist(),
            "eligible_motor_input": False,
        },
        "motor_boundary": {
            "eligible_signal": "nod1_voltage_v",
            "eligible_root_ids": list(FLY_FGS_NOD1_ROOT_IDS),
            "retinal_input_eligible": False,
            "pooled_readout_eligible": False,
            "full_cell_state_eligible": False,
            "notice": (
                "This attachment is visualization/audit data. It cannot be consumed by "
                "the motor bridge; only the four exact NOD1 voltage channels cross that boundary."
            ),
        },
        "relation_to_frozen_replay": {
            "is_frozen_registered_capture": False,
            "notice": (
                "These are exact states from this online closed-loop run, not rows from "
                "the separately registered fixed-stimulus circuit_replay capture."
            ),
        },
        "rejected_downstream_fields": list(contract["rejected_downstream_fields"]),
        "claim_scope": _strict_json_copy(
            contract["claim_scope"], "online circuit claim scope"
        ),
        "content_canonicalization": (
            "SHA-256 of UTF-8 Python json.dumps(sort_keys=True,"
            "separators=(',',':'),allow_nan=False,ensure_ascii=True); "
            "attachment_sha256 excluded; browser integrity relies on the enclosing "
            "artifact web-projection receipt because JavaScript number spelling can differ"
        ),
    }
    attachment = dict(unsigned)
    attachment["attachment_sha256"] = _json_content_sha256(unsigned)
    return _strict_json_copy(attachment, "online circuit replay attachment")


def canonical_closed_loop_to_web_replay(
    result: CanonicalClosedLoopResult,
    *,
    scenario_id: str = "canonical_closed_loop",
    label: str = "Canonical fly-FGS closed loop",
    checkpoint: Optional[CanonicalClosedLoopCheckpoint] = None,
    source_run_id: Optional[str] = None,
    source_artifact_manifest_sha256: Optional[str] = None,
) -> Mapping[str, Any]:
    """Build an exact-clock browser projection from a complete canonical run."""

    if result.initial_bridge_tick_index != 0 or not result.completed:
        raise ValueError("web replay publication requires a complete run starting at t=0")
    bundle = canonical_array_bundle(result)
    transitions = _flatten_transitions(result)
    time_s = np.asarray(bundle.arrays["physics_time_s"][0], dtype=float)
    body_position = np.asarray(bundle.arrays["body_position_world_m"][0], dtype=float)
    body_velocity = np.asarray(bundle.arrays["body_velocity_world_m_s"][0], dtype=float)
    body_quaternion = np.asarray(
        bundle.arrays["body_quaternion_body_to_world"][0], dtype=float
    )
    body_orientation = _quaternion_to_euler(body_quaternion)
    force_body = np.asarray(bundle.arrays["aerodynamic_force_body_n"][0], dtype=float)
    torque_body = np.asarray(
        bundle.arrays["aerodynamic_torque_body_n_m"][0], dtype=float
    )
    force_world = np.asarray(
        [quaternion_to_matrix(q).dot(value) for q, value in zip(body_quaternion, force_body)]
    )
    torque_world = np.asarray(
        [quaternion_to_matrix(q).dot(value) for q, value in zip(body_quaternion, torque_body)]
    )
    phase = np.asarray(bundle.arrays["wing_phase_unwrapped_rad"][0], dtype=float)
    frequency = np.asarray(bundle.arrays["wing_frequency_hz"][0], dtype=float)
    physical_stroke_transition = np.asarray(
        bundle.arrays["physical_actuation_stroke_rad"][0], dtype=float
    )
    physical_stroke = np.vstack(
        (physical_stroke_transition[0], physical_stroke_transition)
    )
    envelope = _rolling_wing_envelope(
        physical_stroke, time_s, float(np.mean(frequency))
    )

    effective_activation = np.asarray(
        bundle.arrays["muscle_effective_activation"][0], dtype=float
    )
    effective_force = np.asarray(
        bundle.arrays["muscle_effective_force_n"][0], dtype=float
    )
    effective_phase = np.asarray(
        bundle.arrays["muscle_effective_phase_effect"][0], dtype=float
    )
    natural_activation_endpoint = np.asarray(
        bundle.arrays["muscle_natural_activation"][0], dtype=float
    )
    muscle_ids = tuple(
        bundle.axis_metadata["muscle_effective_activation"]["axis_labels"]
    )
    index_by_muscle = {name: index for index, name in enumerate(muscle_ids)}

    observations = _latest_observation_by_time(result, time_s)
    # A replay frame is a physics *boundary*.  Bridge-held rates and the
    # outgoing intervention set take effect at the left boundary of their
    # half-open interval.  Consequently frame t=i*bridge_dt belongs to bridge
    # interval i, while the final episode endpoint is clamped to the last
    # interval.  Prefixing the transition-owned sequence with interval zero
    # would instead leave every exact bridge boundary one physics tick stale.
    interval_by_left_boundary: List[Any] = []
    for interval in result.intervals:
        interval_by_left_boundary.extend([interval] * len(interval.physics))
    frame_intervals = interval_by_left_boundary + [result.intervals[-1]]
    if len(frame_intervals) != len(time_s):
        raise ValueError(
            "bridge interval-to-frame projection does not cover every physics boundary"
        )

    neural_rows: List[Mapping[str, float]] = []
    raw_pathway: List[List[Optional[float]]] = []
    for frame_index, (observation, interval) in enumerate(
        zip(observations, frame_intervals)
    ):
        sample = observation.sample
        signals: Dict[str, float] = {
            "retina:mean_luminance": float(
                np.mean(sample.retinal_input_luminance)
                if sample.retinal_input_luminance is not None
                else 0.0
            )
        }
        for key, value in sample.pooled_readout.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                signals["circuit:%s" % key] = float(value)
        for root_id, value in sample.nod1_voltage_v.items():
            signals["nod1:%s" % root_id] = float(value)
        dn = _rate_lookup(interval.bridge.held_dn_rates)
        mn = _rate_lookup(interval.bridge.held_motor_rates)
        signals.update({"dn:%s" % key: value for key, value in dn.items()})
        signals.update({"mn:%s" % key: value for key, value in mn.items()})
        signals.update(
            {
                "muscle:%s:activation" % muscle_id: float(
                    effective_activation[frame_index, muscle_index]
                )
                for muscle_index, muscle_id in enumerate(muscle_ids)
            }
        )
        neural_rows.append(signals)
        retinal_value: Optional[float] = signals["retina:mean_luminance"]
        vch_dch_values = [
            signals[key]
            for key in (
                "circuit:vchL",
                "circuit:vchR",
                "circuit:dchL",
                "circuit:dchR",
            )
            if key in signals
        ]
        llpc_values = [
            signals[key]
            for key in ("circuit:llpcAct", "circuit:llpcL", "circuit:llpcR")
            if key in signals
        ]
        nod_values = tuple(sample.nod1_voltage_v.values())
        dn_values = tuple(dn.values())
        mn_values = tuple(mn.values())
        steering_indices = [
            index
            for name, index in index_by_muscle.items()
            if name.endswith(tuple(":" + muscle for muscle in _MUSCLE_ORDER))
        ]
        raw_pathway.append(
            [
                retinal_value,
                None if not vch_dch_values else float(np.mean(vch_dch_values)),
                None if not llpc_values else float(np.mean(llpc_values)),
                float(np.mean(nod_values)),
                None if not dn_values else float(np.mean(dn_values)),
                None if not mn_values else float(np.mean(mn_values)),
                float(np.mean(effective_activation[frame_index, steering_indices])),
            ]
        )

    normalized = np.column_stack(
        (
            _normalize_display([row[0] for row in raw_pathway], 1.0),
            _normalize_display([row[1] for row in raw_pathway], 1.0),
            _normalize_display([row[2] for row in raw_pathway], 1.0),
            np.clip(
                np.abs(
                    np.asarray([float(row[3]) for row in raw_pathway]) + 0.060
                )
                / 0.005,
                0.0,
                1.0,
            ),
            _normalize_display([row[4] for row in raw_pathway], 200.0),
            _normalize_display([row[5] for row in raw_pathway], 200.0),
            _normalize_display([row[6] for row in raw_pathway], 1.5),
        )
    )

    telemetry_rows = [result.initial_articulated_physics_telemetry] + [
        transition.articulated_telemetry_after for transition in transitions
    ]
    has_telemetry = all(item is not None for item in telemetry_rows)
    if any(item is not None for item in telemetry_rows) and not has_telemetry:
        raise ValueError("articulated telemetry cannot be partially populated")

    generated_event_phase_by_frame: Dict[Tuple[int, str], float] = {}
    for event in bundle.event_records:
        causal_index = int(
            np.searchsorted(time_s, float(event["availability_time_s"]), side="left")
        )
        event_id = "%s:%s" % (
            "raw_app_" + str(event["raw_app_side"]),
            event["muscle"],
        )
        generated_event_phase_by_frame[(causal_index, event_id)] = float(
            event["wingbeat_phase_rad"]
        )

    frames: List[Mapping[str, Any]] = []
    for index, time_value in enumerate(time_s):
        individual: Dict[str, Any] = {}
        for muscle_index, muscle_id in enumerate(muscle_ids):
            state: Dict[str, Any] = {
                "activation": float(effective_activation[index, muscle_index]),
                "force_n": float(effective_force[index, muscle_index]),
                "phase_effect": float(effective_phase[index, muscle_index]),
            }
            if index > 0:
                state["natural_activation"] = float(
                    natural_activation_endpoint[index - 1, muscle_index]
                )
            event_phase = generated_event_phase_by_frame.get((index, muscle_id))
            if event_phase is not None:
                state["event_phase_rad"] = event_phase
            individual[muscle_id] = state

        power_l = np.mean(
            [
                effective_activation[index, index_by_muscle[name]]
                for name in ("raw_app_L:DLM", "raw_app_L:DVM")
            ]
        )
        power_r = np.mean(
            [
                effective_activation[index, index_by_muscle[name]]
                for name in ("raw_app_R:DLM", "raw_app_R:DVM")
            ]
        )
        steering_l = np.mean(
            [
                effective_activation[index, index_by_muscle["raw_app_L:" + muscle]]
                for muscle in _MUSCLE_ORDER
            ]
        )
        steering_r = np.mean(
            [
                effective_activation[index, index_by_muscle["raw_app_R:" + muscle]]
                for muscle in _MUSCLE_ORDER
            ]
        )
        tension = 0.5 * (
            effective_activation[index, index_by_muscle["raw_app_L:tp1"]]
            + effective_activation[index, index_by_muscle["raw_app_R:tp1"]]
        )
        outgoing_mechanics = (
            transitions[index].mechanics if index < len(transitions) else None
        )
        completed_mechanics = (
            transitions[index - 1].mechanics if index > 0 else None
        )
        frame: Dict[str, Any] = {
            "t": float(time_value),
            "wing_phase_unwrapped_rad": float(phase[index]),
            "position_m": [float(value) for value in body_position[index]],
            "orientation_rad": [float(value) for value in body_orientation[index]],
            "velocity_m_s": [float(value) for value in body_velocity[index]],
            "wing_envelope_deg": [float(value) for value in envelope[index]],
            "force_world_n": [float(value) for value in force_world[index]],
            "moment_world_n_m": [float(value) for value in torque_world[index]],
            "circuit": [float(value) for value in normalized[index]],
            "muscles": [
                float(np.clip(power_l, 0.0, 1.0)),
                float(np.clip(power_r, 0.0, 1.0)),
                float(np.clip(steering_l, 0.0, 1.0)),
                float(np.clip(steering_r, 0.0, 1.0)),
                float(np.clip(tension, 0.0, 1.0)),
            ],
            "evidence_incompleteness": 1.0,
            "individual_muscles": individual,
            "neural_signals": dict(neural_rows[index]),
            "pathway_values": raw_pathway[index],
            "desired_wing_stroke_rad": [float(value) for value in physical_stroke[index]],
            "active_intervention_ids": (
                []
                if outgoing_mechanics is None
                else list(outgoing_mechanics.active_intervention_ids)
            ),
            "applied_motor_event_ids": (
                []
                if completed_mechanics is None
                else list(completed_mechanics.applied_event_ids)
            ),
            "suppressed_motor_event_ids": (
                []
                if completed_mechanics is None
                else list(completed_mechanics.suppressed_event_ids)
            ),
        }
        if has_telemetry:
            telemetry = telemetry_rows[index]
            assert telemetry is not None
            frame["position_m"] = [
                float(value) for value in telemetry.whole_fly_com_position_world_m
            ]
            frame["root_position_m"] = [
                float(value) for value in body_position[index]
            ]
            frame["ground_contact_count"] = telemetry.ground_contact_count
            frame["measured_wing_joint_angle_rad"] = [
                float(value) for value in telemetry.measured_wing_position_rad
            ]
            frame["measured_wing_joint_velocity_rad_s"] = [
                float(value) for value in telemetry.measured_wing_velocity_rad_s
            ]
        frames.append(frame)

    neural_channels: List[Mapping[str, Any]] = [
        {
            "id": "retina:mean_luminance",
            "label": "Analytic retinal/T4a luminance mean",
            "stage": "retina",
            "entity_id": "fly_fgs_t4a_input",
            "cell_type": "T4a analytic input",
            "side": "raw_app_bilateral",
            "signal_kind": "sampled_luminance",
            "unit": "1",
            "origin": "simulated",
            "confidence": "low",
            "provenance": "canonical fly-FGS analytic T4a input; not calibrated irradiance",
            "causal_role": "visual_input",
        }
    ]
    for root_id in tuple(observations[0].sample.nod1_voltage_v):
        neural_channels.append(
            {
                "id": "nod1:%s" % root_id,
                "label": "NOD1 %s" % root_id,
                "stage": "circuit",
                "entity_id": root_id,
                "cell_type": "NOD1",
                "side": "raw_app_label_only",
                "signal_kind": "membrane_voltage",
                "unit": "V",
                "origin": "simulated",
                "confidence": "medium",
                "provenance": "exact motor-eligible canonical fly-FGS output",
                "causal_role": "circuit_output",
            }
        )
    pooled_channel_ids: Dict[str, str] = {}
    for key, value in observations[0].sample.pooled_readout.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        channel_id = "circuit:%s" % key
        pooled_channel_ids[key] = channel_id
        neural_channels.append(
            {
                "id": channel_id,
                "label": "fly-FGS pooled %s" % key,
                "stage": "circuit",
                "entity_id": key,
                "cell_type": "pooled circuit readout",
                "side": "raw_app_label_only",
                "signal_kind": "pooled_activity",
                "unit": "1",
                "origin": "simulated",
                "confidence": "medium",
                "provenance": "exact pooled readout exported by the canonical fly-FGS runtime",
                "causal_role": "circuit_output",
            }
        )
    seen_rate_ids: set[str] = set()
    for row in neural_rows:
        for channel_id in sorted(key for key in row if key.startswith(("dn:", "mn:"))):
            if channel_id in seen_rate_ids:
                continue
            seen_rate_ids.add(channel_id)
            stage = "descending" if channel_id.startswith("dn:") else "motor"
            neural_channels.append(
                {
                    "id": channel_id,
                    "label": channel_id.replace(":", " · "),
                    "stage": stage,
                    "entity_id": channel_id.split(":", 1)[1],
                    "cell_type": "DNp26" if stage == "descending" else "wing MN",
                    "side": "raw_app_label_only",
                    "signal_kind": "inferred_rate",
                    "unit": "Hz",
                    "origin": "inferred",
                    "confidence": "low",
                    "provenance": "uncalibrated causal streaming bridge",
                    "causal_role": "circuit_driven",
                }
            )

    muscle_channels = []
    for muscle_id in muscle_ids:
        muscle = muscle_id.split(":", 1)[1]
        muscle_class = (
            "power"
            if muscle in ("DLM", "DVM")
            else "tension"
            if muscle == "tp1"
            else "steering"
        )
        muscle_channels.append(
            {
                "id": muscle_id,
                "label": muscle_id.replace(":", " · "),
                "side": "unknown",
                "muscle_class": muscle_class,
                "motor_neuron": None
                if muscle_class != "steering"
                else "MN-" + muscle,
                "timing_semantics": (
                    "seeded_synthetic_spikes"
                    if muscle_class == "steering"
                    else "unavailable"
                ),
                "origin": "simulated",
                "confidence": "low",
                "provenance": (
                    "raw app lane only; virtual mechanics state; anatomical side unknown"
                ),
            }
        )
        neural_channels.append(
            {
                "id": "muscle:%s:activation" % muscle_id,
                "label": "%s effective activation" % muscle_id,
                "stage": "muscle",
                "entity_id": muscle_id,
                "cell_type": "%s flight muscle" % muscle_class,
                "side": "raw_app_label_only",
                "signal_kind": "modeled_effective_activation",
                "unit": "1",
                "origin": "simulated",
                "confidence": "low",
                "provenance": "virtual mechanics state; anatomical side unknown",
                "causal_role": "circuit_driven",
            }
        )

    pathway_channels = [
        {
            "index": index,
            "label": label_value,
            "status": "attached" if source_ids else "unavailable",
            "source_channel_ids": source_ids,
            "aggregation": aggregation,
            "raw_unit": unit,
            "display_normalization": normalization,
        }
        for index, (label_value, source_ids, aggregation, unit, normalization) in enumerate(
            (
                (
                    "Retinal analytic input",
                    ["retina:mean_luminance"],
                    "mean",
                    "1",
                    "absolute value / 1",
                ),
                (
                    "vCH / DCH",
                    [
                        pooled_channel_ids[key]
                        for key in ("vchL", "vchR", "dchL", "dchR")
                        if key in pooled_channel_ids
                    ],
                    "pooled fly-FGS readout",
                    "1",
                    "absolute value / 1",
                ),
                (
                    "LLPC1",
                    [
                        pooled_channel_ids[key]
                        for key in ("llpcAct", "llpcL", "llpcR")
                        if key in pooled_channel_ids
                    ],
                    "pooled fly-FGS readout",
                    "1",
                    "absolute value / 1",
                ),
                (
                    "NOD1",
                    ["nod1:%s" % root_id for root_id in observations[0].sample.nod1_voltage_v],
                    "mean voltage",
                    "V",
                    "absolute depolarization from -60 mV / 5 mV",
                ),
                (
                    "DNp26",
                    sorted(channel for channel in seen_rate_ids if channel.startswith("dn:")),
                    "mean available inferred rate",
                    "Hz",
                    "absolute value / 200 Hz",
                ),
                (
                    "Wing MNs",
                    sorted(channel for channel in seen_rate_ids if channel.startswith("mn:")),
                    "mean available inferred rate",
                    "Hz",
                    "absolute value / 200 Hz",
                ),
                (
                    "Wing muscles",
                    ["muscle:%s:activation" % muscle_id for muscle_id in muscle_ids],
                    "mean effective steering activation",
                    "1",
                    "absolute value / 1.5",
                ),
            )
        )
    ]
    checkpoint_receipt = _checkpoint_receipt(checkpoint, result)
    online_trace = {
        "schema_version": CANONICAL_WEB_TRACE_SCHEMA_VERSION,
        "mode": "causal_online_fly_fgs_to_streaming_muscle_to_external_physics",
        "clocks": {
            "circuit_dt_s": result.config.circuit_dt_s,
            "bridge_dt_s": result.config.bridge_dt_s,
            "physics_dt_s": result.config.physics_dt_s,
            "circuit_sample_count": len(bundle.circuit_records),
            "bridge_interval_count": len(result.intervals),
            "physics_transition_count": len(transitions),
        },
        "circuit_samples": list(bundle.circuit_records),
        "rate_samples": list(bundle.rate_records),
        "motor_events": list(bundle.event_records),
        "intervention_activity": list(bundle.intervention_records),
        "event_timing_semantics": "discrete generation and NMJ availability; never interpolate",
        "frame_event_receipt_semantics": (
            "applied/suppressed IDs are reported at the completed physics right endpoint; "
            "active intervention IDs govern the outgoing half-open transition"
        ),
        "wing_phase_semantics": "exported model-owned unwrapped oscillator endpoints; not measured",
        "effector_mapping": dict(result.effector_mapping_receipt.to_dict()),
        "anatomical_laterality": "unknown",
        "signed_behavior_claim_policy": result.config.effector_signed_behavior_claim_policy,
        "checkpoint_receipt": checkpoint_receipt,
        "source_limitations": list(result.source_limitations),
    }

    replay: Dict[str, Any] = {
        "schema_version": WEB_REPLAY_SCHEMA_VERSION,
        "id": scenario_id,
        "label": label,
        "status": "exploratory",
        "source_kind": CANONICAL_WEB_SOURCE_KIND,
        "authority_notice": (
            "Canonical online fly-FGS circuit and exact causal multi-rate execution. "
            "Motor encoding, muscle mechanics, hinge parameters, and effector anatomy remain exploratory."
        ),
        "duration_s": result.config.duration_s,
        "envelope_sample_rate_hz": 1.0 / result.config.physics_dt_s,
        "physics_timestep_s": result.config.physics_dt_s,
        "wingbeat_hz": float(np.mean(frequency)),
        "wing_phase_source": "exported_model_owned_oscillator_endpoints",
        "stimulus": "online figure world azimuth minus current body yaw; analytic T4a fly-FGS input",
        "perturbation": "canonical configured bridge and mechanics intervention schedules",
        "circuit_activity_status": (
            "exact canonical fly-FGS samples at 5 ms; held causally between samples"
        ),
        "evidence_incompleteness_status": (
            "set to 1.0 because no calibrated trajectory posterior exists"
        ),
        "physics_backend": result.backend_name,
        "body_state_reference": (
            "whole-fly articulated COM when complete telemetry exists; otherwise external physics root"
        ),
        "contact_telemetry_status": (
            "exact_mujoco_preintegration_transition_contact_points"
            if has_telemetry
            else "unavailable"
        ),
        "wing_kinematics_source": "explicit physical mapping of raw-app virtual-hinge command",
        "measured_wing_joint_order": (
            list(telemetry_rows[0].wing_joint_order) if has_telemetry else []
        ),
        "signal_semantics": {
            "neural_origin": "simulated",
            "motor_timing": "seeded_synthetic_spikes",
            "mechanics_origin": "simulated",
            "confidence": "low",
            "provenance": (
                "Exact circuit state and causal availability clocks; downstream rates/events, "
                "muscles, virtual hinge, and unresolved effector mapping remain inferred/modelled."
            ),
        },
        "individual_muscle_channels": muscle_channels,
        "neural_trace_channels": neural_channels,
        "pathway_channels": pathway_channels,
        "neural_model_scope": {
            "kind": "online_registered_fly_fgs_closed_loop",
            "label": "Canonical online fly-FGS fixed-step runtime",
            "full_circuit_executed": True,
            "circuit_cell_count": (
                len(observations[0].sample.full_cell_voltage_v)
                if observations[0].sample.full_cell_voltage_v is not None
                else 1684
            ),
            "exported_circuit_channel_count": 4,
            "notice": (
                "The full captured circuit executes online; only four exact NOD1 voltage channels cross the motor boundary."
            ),
        },
        "online_closed_loop": online_trace,
        "units": {
            "position_m": "m",
            "root_position_m": "m",
            "ground_contact_count": "1",
            "orientation_rad": "rad",
            "velocity_m_s": "m s^-1",
            "wing_envelope_deg": "degree",
            "desired_wing_stroke_rad": "rad",
            "measured_wing_joint_angle_rad": "rad",
            "measured_wing_joint_velocity_rad_s": "rad s^-1",
            "force_world_n": "N",
            "moment_world_n_m": "N m",
            "circuit": "normalized display envelope derived from attached raw signals (1)",
            "muscles": "normalized modeled state (1)",
            "evidence_incompleteness": "fraction (1)",
            "wing_phase_unwrapped_rad": "rad",
        },
        "frames": frames,
    }
    online_circuit_replay = _online_circuit_replay_attachment(
        result,
        bundle,
        _circuit_observations(result),
    )
    if online_circuit_replay is not None:
        replay["online_circuit_replay"] = online_circuit_replay
    if has_telemetry:
        contact_counts = [int(row.ground_contact_count) for row in telemetry_rows if row]
        first_index = next((index for index, value in enumerate(contact_counts) if value > 0), None)
        replay["ground_contact_summary"] = {
            "telemetry": "exact_mujoco_preintegration_transition_contact_points",
            "sample_semantics": "sample 0 is the reset-state mj_forward contact count; sample i>=1 is the pre-integration mjContact list used for the MuJoCo transition ending at that sample time, not a collision query at the displayed post-step pose and not a contact-force measurement",
            "initial_count": contact_counts[0],
            "occurred": max(contact_counts) > 0,
            "first_transition_start_s": (
                None
                if first_index is None
                else 0.0
                if first_index == 0
                else float(time_s[first_index - 1])
            ),
            "first_transition_end_s": (
                None if first_index is None else float(time_s[first_index])
            ),
            "transition_count": sum(value > 0 for value in contact_counts[1:]),
            "maximum_count": max(contact_counts),
        }
    if source_run_id is not None:
        if not source_run_id:
            raise ValueError("source_run_id must be non-empty")
        replay["source_run_id"] = source_run_id
    if source_artifact_manifest_sha256 is not None:
        if (
            source_run_id is None
            or not isinstance(source_artifact_manifest_sha256, str)
            or len(source_artifact_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in source_artifact_manifest_sha256
            )
        ):
            raise ValueError("artifact digest requires source_run_id and SHA-256")
        replay["source_artifact_manifest_sha256"] = source_artifact_manifest_sha256
        replay["source_artifact_schema_version"] = CANONICAL_ARTIFACT_SCHEMA_VERSION
    return _strict_json_copy(replay, "canonical web replay")


__all__ = [
    "CANONICAL_ARTIFACT_SCHEMA_VERSION",
    "CANONICAL_ARTIFACT_SOURCE_KIND",
    "CANONICAL_WEB_SOURCE_KIND",
    "CANONICAL_WEB_TRACE_SCHEMA_VERSION",
    "CanonicalArrayBundle",
    "CANONICAL_WEB_PROJECTION_CANONICALIZATION",
    "bind_canonical_web_replay_to_artifact",
    "canonical_array_bundle",
    "canonical_closed_loop_to_web_replay",
    "canonical_run_id",
    "canonical_web_projection_sha256",
    "write_canonical_closed_loop_artifact",
]
