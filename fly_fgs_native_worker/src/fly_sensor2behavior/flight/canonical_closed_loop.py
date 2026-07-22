"""Causal canonical fly-FGS-to-flight closed-loop orchestration.

This module joins the captured fly-FGS circuit runtime to the streaming
NOD1-to-wing-motor bridge, the streaming muscle/virtual-hinge mechanics, and
an injected external flight-physics adapter.  It owns clocks and causality;
it does not add another neural, muscle, aerodynamic, or rigid-body model.

The authoritative interval is the half-open 0.5 ms bridge interval.  At its
left boundary the runner samples the *current* rigid body when a 5 ms circuit
sample is due, projects the mechanics-owned carrier phase without mutation,
advances the bridge, lets mechanics consume only NMJ-available events, and
then executes exactly five 0.1 ms physics transitions.  A minimum one-bridge
NMJ delay is required, so an event inferred from the projected phase can only
affect a later interval and can never be applied retroactively.

Scientific scope is intentionally narrow.  The source runtime is the
SHA-locked analytic T4a input of the captured application.  It has no
photoreceptor, lamina, or T5 model; no translation/parallax rendering; no
calibrated FlyBody/camera coordinate crosswalk; and no anatomical meaning for
the source application's L/R labels.  The bilateral carrier, DNp26 bridge,
muscle gains, virtual hinge, and resulting flight control remain exploratory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..fly_fgs_runtime import (
    FlyFGSCircuitCheckpoint,
    FlyFGSCircuitSample,
    FlyFGSSceneBodyInput,
    NodeFlyFGSCircuitRuntime,
)
from .rigid_body import ExternalFlightPhysicsAdapter, quaternion_to_matrix
from .effector_mapping import (
    EFFECTOR_MAPPING_PROVENANCE,
    RAW_APP_LATERALITY_STATUS,
    SIGNED_BEHAVIOR_CLAIM_POLICY,
    EffectorLateralityHypothesis,
    EffectorMappingReceipt,
    effector_mapping_receipt,
    map_raw_app_wing_kinematics_to_physical,
)
from .streaming_bridge import (
    STREAMING_BRIDGE_DT_S,
    StreamingBridgeCheckpoint,
    StreamingBridgeFrame,
    StreamingBridgeIntervalStart,
    StreamingNOD1MotorBridge,
)
from .streaming_mechanics import (
    STREAMING_MECHANICS_DT_S,
    STREAMING_MECHANICS_STEPS_PER_BRIDGE,
    StreamingMechanicsCheckpoint,
    StreamingMechanicsFrame,
    StreamingMuscleWingStepper,
    WingPhaseSource,
)
from .types import AerodynamicWrench, RigidBodyState, WingKinematics


CANONICAL_CLOSED_LOOP_SCHEMA_VERSION = "1.2.0"
CANONICAL_CLOSED_LOOP_RUNTIME_VERSION = "1.2.0"
CANONICAL_CIRCUIT_DT_S = 0.005
CANONICAL_BRIDGE_DT_S = STREAMING_BRIDGE_DT_S
CANONICAL_PHYSICS_DT_S = STREAMING_MECHANICS_DT_S
CANONICAL_BRIDGE_STEPS_PER_CIRCUIT = 10
CANONICAL_PHYSICS_STEPS_PER_BRIDGE = 5
CANONICAL_MAX_DURATION_S = 0.5
CANONICAL_MAX_CIRCUIT_SAMPLES = 100
_TIME_TOLERANCE_S = 1.0e-12

CANONICAL_SOURCE_LIMITATIONS: Tuple[str, ...] = (
    "SHA-locked captured fly-FGS analytic T4a input; no calibrated compound-eye irradiance",
    "no explicit photoreceptor, lamina, or T5 model in the captured runtime",
    "no translation, depth, occlusion, or motion-parallax input",
    "FlyBody root, world, visual, and application coordinate frames are not calibrated",
    "fly-FGS application L/R labels retain unknown anatomical laterality",
    "signed yaw and roll claims are prohibited until the raw app-lane to physical-wing mapping is resolved",
    "sample 0 is a fixed static-scene pre-roll state; only its relative scene angle can be matched at reset",
    "DNp26 functional gains, seeded phase-gated events, muscle parameters, and virtual hinge are exploratory",
    "bilateral power/tension carrier is uncalibrated and is not evidence of biological stable flight",
)


class CanonicalClosedLoopError(RuntimeError):
    """Raised when a component violates the closed-loop causal contract."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("%s must be a finite number" % label)
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a finite number" % label) from exc
    if not math.isfinite(converted):
        raise ValueError("%s must be a finite number" % label)
    return converted


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a non-negative integer" % label)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _exact_mapping(value: Any, expected: Sequence[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError("%s fields do not match the schema" % label)
    return value


def _checkpoint_dict(value: Any, label: str) -> Mapping[str, Any]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        result = value.to_dict()
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise TypeError("%s checkpoint must be a mapping or expose to_dict" % label)
    if not isinstance(result, Mapping):
        raise TypeError("%s checkpoint to_dict must return a mapping" % label)
    # Detach and fail on NaN/Infinity at the composite boundary.
    return json.loads(_canonical_json(result))


def _body_to_dict(state: RigidBodyState) -> Mapping[str, Any]:
    state.validate()
    return {
        "position_world_m": state.position_world_m.tolist(),
        "velocity_world_m_s": state.velocity_world_m_s.tolist(),
        "quaternion_body_to_world": state.quaternion_body_to_world.tolist(),
        "angular_velocity_body_rad_s": state.angular_velocity_body_rad_s.tolist(),
    }


def _body_from_dict(value: Any) -> RigidBodyState:
    mapping = _exact_mapping(
        value,
        (
            "position_world_m",
            "velocity_world_m_s",
            "quaternion_body_to_world",
            "angular_velocity_body_rad_s",
        ),
        "body state",
    )
    return RigidBodyState(
        position_world_m=np.asarray(mapping["position_world_m"], dtype=float),
        velocity_world_m_s=np.asarray(mapping["velocity_world_m_s"], dtype=float),
        quaternion_body_to_world=np.asarray(
            mapping["quaternion_body_to_world"], dtype=float
        ),
        angular_velocity_body_rad_s=np.asarray(
            mapping["angular_velocity_body_rad_s"], dtype=float
        ),
    )


def _wrench_copy(wrench: AerodynamicWrench) -> AerodynamicWrench:
    if not isinstance(wrench, AerodynamicWrench):
        raise CanonicalClosedLoopError(
            "physics aerodynamic_wrench must return AerodynamicWrench"
        )
    return AerodynamicWrench(
        force_body_n=wrench.force_body_n.copy(),
        torque_body_n_m=wrench.torque_body_n_m.copy(),
        left_force_body_n=wrench.left_force_body_n.copy(),
        right_force_body_n=wrench.right_force_body_n.copy(),
        mechanical_power_w=float(wrench.mechanical_power_w),
    )


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


def _angle_difference(first: float, second: float) -> float:
    return math.atan2(math.sin(first - second), math.cos(first - second))


@dataclass(frozen=True)
class CanonicalClosedLoopConfig:
    """Clock, scene, logging, and source-display configuration.

    ``figure_initial_world_azimuth_rad`` is unwrapped and evolves at the
    explicitly supplied constant world velocity.  ``ground_velocity_rad_s``
    is the captured application's angular panorama velocity; it is not body
    translation or optic flow derived from a 3-D world.
    """

    effector_laterality_hypothesis: EffectorLateralityHypothesis
    duration_s: float = CANONICAL_MAX_DURATION_S
    circuit_dt_s: float = CANONICAL_CIRCUIT_DT_S
    bridge_dt_s: float = CANONICAL_BRIDGE_DT_S
    physics_dt_s: float = CANONICAL_PHYSICS_DT_S
    figure_initial_world_azimuth_rad: float = 0.0
    figure_velocity_rad_s: float = 0.0
    ground_velocity_rad_s: float = 0.0
    include_retinal_input: bool = True
    include_full_cell_state: bool = False
    effector_anatomical_status: str = RAW_APP_LATERALITY_STATUS
    effector_mapping_provenance: Tuple[str, ...] = EFFECTOR_MAPPING_PROVENANCE
    effector_signed_behavior_claim_policy: str = SIGNED_BEHAVIOR_CLAIM_POLICY

    def __post_init__(self) -> None:
        try:
            hypothesis = EffectorLateralityHypothesis(
                self.effector_laterality_hypothesis
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "effector_laterality_hypothesis must explicitly select a raw-lane mapping"
            ) from exc
        object.__setattr__(self, "effector_laterality_hypothesis", hypothesis)
        if self.effector_anatomical_status != RAW_APP_LATERALITY_STATUS:
            raise ValueError("raw fly-FGS app-lane anatomical status must remain unknown")
        provenance = tuple(self.effector_mapping_provenance)
        if provenance != EFFECTOR_MAPPING_PROVENANCE:
            raise ValueError("effector mapping provenance does not match the locked receipt")
        if self.effector_signed_behavior_claim_policy != SIGNED_BEHAVIOR_CLAIM_POLICY:
            raise ValueError("effector signed-behavior claim policy is not locked")
        object.__setattr__(self, "effector_mapping_provenance", provenance)
        duration = _finite(self.duration_s, "duration_s")
        if duration <= 0.0 or duration > CANONICAL_MAX_DURATION_S + _TIME_TOLERANCE_S:
            raise ValueError("duration_s must be in (0, 0.5] seconds")
        for name, expected in (
            ("circuit_dt_s", CANONICAL_CIRCUIT_DT_S),
            ("bridge_dt_s", CANONICAL_BRIDGE_DT_S),
            ("physics_dt_s", CANONICAL_PHYSICS_DT_S),
        ):
            observed = _finite(getattr(self, name), name)
            if abs(observed - expected) > 1.0e-15:
                raise ValueError("%s must equal %.7g" % (name, expected))
            object.__setattr__(self, name, observed)
        interval_count = round(duration / self.bridge_dt_s)
        if (
            interval_count < 1
            or abs(interval_count * self.bridge_dt_s - duration)
            > _TIME_TOLERANCE_S
        ):
            raise ValueError("duration_s must be an integer number of 0.5 ms intervals")
        if round(self.circuit_dt_s / self.bridge_dt_s) != CANONICAL_BRIDGE_STEPS_PER_CIRCUIT:
            raise ValueError("circuit/bridge clock ratio must be exactly 10:1")
        if round(self.bridge_dt_s / self.physics_dt_s) != CANONICAL_PHYSICS_STEPS_PER_BRIDGE:
            raise ValueError("bridge/physics clock ratio must be exactly 5:1")
        if CANONICAL_PHYSICS_STEPS_PER_BRIDGE != STREAMING_MECHANICS_STEPS_PER_BRIDGE:
            raise ValueError("mechanics and orchestrator physics clocks disagree")
        for name in (
            "figure_initial_world_azimuth_rad",
            "figure_velocity_rad_s",
            "ground_velocity_rad_s",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if not isinstance(self.include_retinal_input, bool):
            raise TypeError("include_retinal_input must be boolean")
        if not isinstance(self.include_full_cell_state, bool):
            raise TypeError("include_full_cell_state must be boolean")
        object.__setattr__(self, "duration_s", interval_count * self.bridge_dt_s)

    @property
    def bridge_interval_count(self) -> int:
        return round(self.duration_s / self.bridge_dt_s)

    @property
    def circuit_sample_count(self) -> int:
        return (self.bridge_interval_count - 1) // CANONICAL_BRIDGE_STEPS_PER_CIRCUIT + 1

    def to_dict(self) -> Mapping[str, Any]:
        result = {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }
        result["effector_laterality_hypothesis"] = (
            self.effector_laterality_hypothesis.value
        )
        result["effector_mapping_provenance"] = list(
            self.effector_mapping_provenance
        )
        return result

    @property
    def effector_mapping_receipt(self) -> EffectorMappingReceipt:
        return effector_mapping_receipt(self.effector_laterality_hypothesis)

    @classmethod
    def from_dict(cls, value: Any) -> "CanonicalClosedLoopConfig":
        mapping = _exact_mapping(
            value,
            tuple(field.name for field in dataclasses.fields(cls)),
            "closed-loop config",
        )
        return cls(**dict(mapping))


@dataclass(frozen=True)
class CanonicalCircuitObservation:
    """One circuit-boundary observation taken at a bridge left endpoint."""

    bridge_tick_index: int
    observation_time_s: float
    body_state: RigidBodyState
    wrapped_world_yaw_rad: float
    unwrapped_world_yaw_rad: float
    world_z_yaw_rate_rad_s: float
    requested_control: FlyFGSSceneBodyInput
    sample: FlyFGSCircuitSample
    initialization_mode: str


@dataclass(frozen=True)
class CanonicalPhysicsTelemetry:
    """Exact optional articulated-physics telemetry after one transition.

    This record is populated only when the adapter exposes the complete
    reviewed capability group.  Missing fields are never reconstructed from
    desired virtual-hinge kinematics or rigid-body state.
    """

    wing_joint_order: Tuple[str, ...]
    measured_wing_position_rad: np.ndarray
    measured_wing_velocity_rad_s: np.ndarray
    actuator_torque_n_m: np.ndarray
    whole_fly_com_position_world_m: np.ndarray
    ground_contact_count: int

    def __post_init__(self) -> None:
        order = tuple(self.wing_joint_order)
        if len(order) < 1 or len(set(order)) != len(order) or any(
            not isinstance(name, str) or not name for name in order
        ):
            raise ValueError("wing_joint_order must contain unique non-empty names")
        for name in (
            "measured_wing_position_rad",
            "measured_wing_velocity_rad_s",
            "actuator_torque_n_m",
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (len(order),) or not np.all(np.isfinite(values)):
                raise ValueError("%s must match the finite wing-joint inventory" % name)
            object.__setattr__(self, name, values.copy())
        com = np.asarray(self.whole_fly_com_position_world_m, dtype=float)
        if com.shape != (3,) or not np.all(np.isfinite(com)):
            raise ValueError("whole-fly COM must contain three finite SI values")
        contact_count = _nonnegative_int(
            self.ground_contact_count, "ground_contact_count"
        )
        object.__setattr__(self, "wing_joint_order", order)
        object.__setattr__(self, "whole_fly_com_position_world_m", com.copy())
        object.__setattr__(self, "ground_contact_count", contact_count)


@dataclass(frozen=True)
class CanonicalPhysicsTransition:
    """One mechanics/physics transition on the exact 0.1 ms grid."""

    physics_tick_index: int
    interval_start_s: float
    interval_end_s: float
    mechanics: StreamingMechanicsFrame
    physical_actuation_wing_kinematics: WingKinematics
    body_state_before: RigidBodyState
    body_state_after: RigidBodyState
    aerodynamic_wrench_after: AerodynamicWrench
    articulated_telemetry_after: Optional[CanonicalPhysicsTelemetry]


@dataclass(frozen=True)
class CanonicalClosedLoopInterval:
    """Complete audit record for one half-open 0.5 ms interval."""

    bridge_tick_index: int
    interval_start_s: float
    interval_end_s: float
    circuit_observation: Optional[CanonicalCircuitObservation]
    projected_model_phase_end_unwrapped_rad: float
    phase_projection_semantics: str
    bridge_start: StreamingBridgeIntervalStart
    bridge: StreamingBridgeFrame
    mechanics: Tuple[StreamingMechanicsFrame, ...]
    physics: Tuple[CanonicalPhysicsTransition, ...]


@dataclass(frozen=True)
class CanonicalClosedLoopResult:
    """One complete or partial run segment, retaining visualization telemetry."""

    config: CanonicalClosedLoopConfig
    initial_bridge_tick_index: int
    final_bridge_tick_index: int
    initial_body_state: RigidBodyState
    initial_aerodynamic_wrench: AerodynamicWrench
    initial_articulated_physics_telemetry: Optional[CanonicalPhysicsTelemetry]
    final_body_state: RigidBodyState
    intervals: Tuple[CanonicalClosedLoopInterval, ...]
    backend_name: str
    aerodynamic_owner: str
    circuit_sample_count: int
    articulated_physics_telemetry_available: bool
    effector_mapping_receipt: EffectorMappingReceipt
    source_limitations: Tuple[str, ...] = CANONICAL_SOURCE_LIMITATIONS
    validation_status: str = "exploratory"

    @property
    def completed(self) -> bool:
        return self.final_bridge_tick_index == self.config.bridge_interval_count


@dataclass(frozen=True)
class CanonicalClosedLoopCheckpoint:
    """Hash-protected complete stack state at a completed bridge boundary."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        copied = json.loads(_canonical_json(self.payload))
        mapping = _exact_mapping(
            copied,
            (
                "schema_version",
                "runtime_version",
                "config",
                "components",
                "state",
                "receipts",
                "payload_sha256",
            ),
            "canonical closed-loop checkpoint",
        )
        if mapping["schema_version"] != CANONICAL_CLOSED_LOOP_SCHEMA_VERSION:
            raise ValueError("closed-loop checkpoint schema version mismatch")
        if mapping["runtime_version"] != CANONICAL_CLOSED_LOOP_RUNTIME_VERSION:
            raise ValueError("closed-loop checkpoint runtime version mismatch")
        digest = mapping["payload_sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("closed-loop checkpoint SHA-256 is invalid")
        unsigned = {key: child for key, child in mapping.items() if key != "payload_sha256"}
        if _sha256(unsigned) != digest:
            raise ValueError("closed-loop checkpoint payload SHA-256 mismatch")
        validated_config = CanonicalClosedLoopConfig.from_dict(mapping["config"])
        _exact_mapping(mapping["components"], ("circuit", "bridge", "mechanics", "physics"), "component checkpoints")
        state = _exact_mapping(
            mapping["state"],
            (
                "bridge_tick_index",
                "physics_tick_index",
                "circuit_sample_count",
                "last_circuit_sample_index",
                "body_state",
                "yaw_wrapped_rad",
                "yaw_unwrapped_rad",
                "figure_world_azimuth_rad",
                "figure_velocity_rad_s",
                "ground_velocity_rad_s",
            ),
            "orchestrator state",
        )
        _nonnegative_int(state["bridge_tick_index"], "bridge_tick_index")
        _nonnegative_int(state["physics_tick_index"], "physics_tick_index")
        _nonnegative_int(state["circuit_sample_count"], "circuit_sample_count")
        _nonnegative_int(state["last_circuit_sample_index"], "last_circuit_sample_index")
        _body_from_dict(state["body_state"])
        for name in (
            "yaw_wrapped_rad",
            "yaw_unwrapped_rad",
            "figure_world_azimuth_rad",
            "figure_velocity_rad_s",
            "ground_velocity_rad_s",
        ):
            _finite(state[name], name)
        _exact_mapping(
            mapping["receipts"],
            (
                "backend_name",
                "aerodynamic_owner",
                "articulated_physics_telemetry_available",
                "effector_mapping",
                "source_limitations",
            ),
            "checkpoint receipts",
        )
        if not isinstance(
            mapping["receipts"]["articulated_physics_telemetry_available"],
            bool,
        ):
            raise TypeError(
                "articulated_physics_telemetry_available must be boolean"
            )
        mapping_receipt = EffectorMappingReceipt.from_dict(
            mapping["receipts"]["effector_mapping"]
        )
        if mapping_receipt != validated_config.effector_mapping_receipt:
            raise ValueError(
                "checkpoint effector mapping receipt does not match its configuration"
            )
        object.__setattr__(self, "payload", MappingProxyType(copied))

    def to_dict(self) -> Mapping[str, Any]:
        return json.loads(_canonical_json(dict(self.payload)))


class CanonicalClosedLoopSimulator:
    """Own and advance one re-entrant canonical visual-to-flight stack."""

    def __init__(
        self,
        config: CanonicalClosedLoopConfig,
        *,
        circuit_runtime: NodeFlyFGSCircuitRuntime,
        bridge: StreamingNOD1MotorBridge,
        mechanics: StreamingMuscleWingStepper,
        physics_adapter: ExternalFlightPhysicsAdapter,
        initial_body_state: Optional[RigidBodyState] = None,
    ) -> None:
        if not isinstance(config, CanonicalClosedLoopConfig):
            raise TypeError("config must be CanonicalClosedLoopConfig")
        self.config = config
        self.circuit_runtime = circuit_runtime
        self.bridge = bridge
        self.mechanics = mechanics
        self.physics_adapter = physics_adapter
        self._has_articulated_physics_telemetry = self._validate_components()
        default_state = (
            physics_adapter.default_initial_state()
            if initial_body_state is None
            else initial_body_state
        )
        if not isinstance(default_state, RigidBodyState):
            raise TypeError("initial body state must be RigidBodyState")
        self._initial_body_state = default_state.copy()
        self._body_state = default_state.copy()
        self._segment_initial_body_state = default_state.copy()
        self._segment_initial_aerodynamic_wrench: Optional[AerodynamicWrench] = None
        self._segment_initial_articulated_telemetry: Optional[
            CanonicalPhysicsTelemetry
        ] = None
        self._bridge_tick_index = 0
        self._physics_tick_index = 0
        self._circuit_sample_count = 0
        self._last_circuit_sample_index: Optional[int] = None
        self._figure_world_azimuth_rad = config.figure_initial_world_azimuth_rad
        self._yaw_wrapped_rad: Optional[float] = None
        self._yaw_unwrapped_rad: Optional[float] = None
        self._initialized = False
        self._pending_initial_sample: Optional[FlyFGSCircuitSample] = None
        self._records: list[CanonicalClosedLoopInterval] = []
        self._failed = False
        self._failure_reason: Optional[str] = None

    def _validate_components(self) -> bool:
        for name in ("backend_name", "aerodynamic_owner"):
            if not isinstance(getattr(self.physics_adapter, name, None), str) or not getattr(
                self.physics_adapter, name
            ):
                raise ValueError("physics adapter must declare %s" % name)
        if abs(self.bridge.config.base_dt_s - self.config.bridge_dt_s) > 1.0e-15:
            raise ValueError("bridge clock does not match the orchestrator")
        if self.bridge.config.nmj_delay_s + _TIME_TOLERANCE_S < self.config.bridge_dt_s:
            raise ValueError("canonical closed loop requires NMJ delay >= one bridge interval")
        if self.mechanics.config.phase_source is not WingPhaseSource.MODEL_OWNED_OSCILLATOR:
            raise ValueError("canonical closed loop requires model-owned oscillator phase")
        if abs(self.mechanics.config.bridge_dt_s - self.config.bridge_dt_s) > 1.0e-15:
            raise ValueError("mechanics bridge clock does not match the orchestrator")
        if abs(self.mechanics.config.physics_dt_s - self.config.physics_dt_s) > 1.0e-15:
            raise ValueError("mechanics physics clock does not match the orchestrator")
        if self.bridge.tick_index != 0 or self.mechanics.tick_index != 0:
            raise ValueError("fresh construction requires fresh bridge and mechanics")
        for method in (
            "default_initial_state",
            "reset",
            "step",
            "aerodynamic_wrench",
            "checkpoint",
            "restore_checkpoint",
        ):
            if not callable(getattr(self.physics_adapter, method, None)):
                raise TypeError("physics adapter must implement %s" % method)
        telemetry_capabilities = {
            "wing_joint_state": callable(
                getattr(self.physics_adapter, "wing_joint_state", None)
            ),
            "wing_joint_order": hasattr(self.physics_adapter, "wing_joint_order"),
            "last_actuator_torque_n_m": hasattr(
                self.physics_adapter, "last_actuator_torque_n_m"
            ),
            "whole_fly_com_position_m": callable(
                getattr(self.physics_adapter, "whole_fly_com_position_m", None)
            ),
            "ground_contact_count": callable(
                getattr(self.physics_adapter, "ground_contact_count", None)
            ),
        }
        if any(telemetry_capabilities.values()) and not all(
            telemetry_capabilities.values()
        ):
            missing = sorted(
                name for name, available in telemetry_capabilities.items() if not available
            )
            raise TypeError(
                "articulated physics telemetry is all-or-none; missing: %s"
                % ", ".join(missing)
            )
        return all(telemetry_capabilities.values())

    def _articulated_physics_telemetry(
        self,
    ) -> Optional[CanonicalPhysicsTelemetry]:
        if not self._has_articulated_physics_telemetry:
            return None
        joint_state = self.physics_adapter.wing_joint_state()
        if not isinstance(joint_state, (tuple, list)) or len(joint_state) != 2:
            raise CanonicalClosedLoopError(
                "physics wing_joint_state must return position and velocity"
            )
        return CanonicalPhysicsTelemetry(
            wing_joint_order=tuple(self.physics_adapter.wing_joint_order),
            measured_wing_position_rad=np.asarray(joint_state[0], dtype=float),
            measured_wing_velocity_rad_s=np.asarray(joint_state[1], dtype=float),
            actuator_torque_n_m=np.asarray(
                self.physics_adapter.last_actuator_torque_n_m, dtype=float
            ),
            whole_fly_com_position_world_m=np.asarray(
                self.physics_adapter.whole_fly_com_position_m(), dtype=float
            ),
            ground_contact_count=self.physics_adapter.ground_contact_count(),
        )

    @property
    def bridge_tick_index(self) -> int:
        return self._bridge_tick_index

    @property
    def current_time_s(self) -> float:
        return self._bridge_tick_index * self.config.bridge_dt_s

    @property
    def body_state(self) -> RigidBodyState:
        return self._body_state.copy()

    @property
    def failed(self) -> bool:
        """Whether an interval failed after the stack may have mutated."""

        return self._failed

    @property
    def failure_reason(self) -> Optional[str]:
        return self._failure_reason

    @staticmethod
    def _start_runtime(runtime: Any) -> None:
        start = getattr(runtime, "start", None)
        if callable(start):
            start()

    def _update_yaw(self, body_state: RigidBodyState) -> Tuple[float, float, float]:
        rotation = quaternion_to_matrix(body_state.quaternion_body_to_world)
        wrapped = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        if self._yaw_wrapped_rad is None:
            unwrapped = wrapped
        else:
            assert self._yaw_unwrapped_rad is not None
            unwrapped = self._yaw_unwrapped_rad + _angle_difference(
                wrapped, self._yaw_wrapped_rad
            )
        omega_world = rotation.dot(body_state.angular_velocity_body_rad_s)
        yaw_rate = _finite(omega_world[2], "world-z yaw rate")
        self._yaw_wrapped_rad = wrapped
        self._yaw_unwrapped_rad = unwrapped
        return wrapped, unwrapped, yaw_rate

    def _current_control(self) -> Tuple[FlyFGSSceneBodyInput, float, float, float]:
        wrapped, unwrapped, yaw_rate = self._update_yaw(self._body_state)
        return (
            FlyFGSSceneBodyInput(
                heading_rad=unwrapped,
                heading_velocity_rad_s=yaw_rate,
                figure_world_azimuth_rad=self._figure_world_azimuth_rad,
                figure_velocity_rad_s=self.config.figure_velocity_rad_s,
                ground_velocity_rad_s=self.config.ground_velocity_rad_s,
            ),
            wrapped,
            unwrapped,
            yaw_rate,
        )

    def _validate_initial_static_scene(
        self, sample: FlyFGSCircuitSample, control: FlyFGSSceneBodyInput
    ) -> None:
        # Sample zero follows a fixed static pre-roll and cannot accept a new
        # control through the captured engine API.  A common world-angle gauge
        # shift is harmless; a different figure-on-retina angle is not.
        expected_relative = (
            sample.last_control.figure_world_azimuth_rad
            - sample.last_control.heading_rad
        )
        observed_relative = (
            control.figure_world_azimuth_rad - control.heading_rad
        )
        if abs(_angle_difference(observed_relative, expected_relative)) > _TIME_TOLERANCE_S:
            raise CanonicalClosedLoopError(
                "initial body/figure angle is incompatible with the captured static pre-roll"
            )

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self.physics_adapter.reset(self._initial_body_state.copy())
        self._body_state = self._initial_body_state.copy()
        self._segment_initial_body_state = self._body_state.copy()
        # Capture reset-boundary observables before the first transition.  The
        # artifact layer must not back-fill t=0 from a future physics step.
        self._segment_initial_aerodynamic_wrench = _wrench_copy(
            self.physics_adapter.aerodynamic_wrench()
        )
        self._segment_initial_articulated_telemetry = (
            self._articulated_physics_telemetry()
        )
        self._start_runtime(self.circuit_runtime)
        sample = self.circuit_runtime.initialize(
            include_retinal_input=self.config.include_retinal_input,
            include_full_cell_state=self.config.include_full_cell_state,
        )
        control, _wrapped, _unwrapped, _rate = self._current_control()
        if sample.sample_index != 0 or sample.measurement_time_s != 0.0 or sample.availability_time_s != 0.0:
            raise CanonicalClosedLoopError("circuit initialization must produce sample 0 at t=0")
        self._validate_initial_static_scene(sample, control)
        self._pending_initial_sample = sample
        self._initialized = True

    def _obtain_circuit_sample(self) -> Optional[CanonicalCircuitObservation]:
        if self._bridge_tick_index % CANONICAL_BRIDGE_STEPS_PER_CIRCUIT != 0:
            return None
        control, wrapped, unwrapped, yaw_rate = self._current_control()
        expected_sample_index = self._bridge_tick_index // CANONICAL_BRIDGE_STEPS_PER_CIRCUIT
        if expected_sample_index == 0:
            sample = self._pending_initial_sample
            if sample is None:
                raise CanonicalClosedLoopError("initial circuit sample is unavailable")
            self._pending_initial_sample = None
            mode = "captured_static_preroll_gauge_matched"
        else:
            sample = self.circuit_runtime.advance(
                control,
                include_retinal_input=self.config.include_retinal_input,
                include_full_cell_state=self.config.include_full_cell_state,
            )
            mode = "body_scene_control_applied"
        if not isinstance(sample, FlyFGSCircuitSample):
            raise CanonicalClosedLoopError("circuit runtime returned an invalid sample type")
        if (
            sample.sample_index != expected_sample_index
            or abs(sample.measurement_time_s - self.current_time_s) > _TIME_TOLERANCE_S
            or abs(sample.availability_time_s - self.current_time_s) > _TIME_TOLERANCE_S
        ):
            raise CanonicalClosedLoopError("circuit sample index/time does not match the 5 ms clock")
        if expected_sample_index > 0 and sample.last_control != control:
            raise CanonicalClosedLoopError("circuit runtime did not apply the requested SI control")
        self._last_circuit_sample_index = sample.sample_index
        self._circuit_sample_count += 1
        return CanonicalCircuitObservation(
            bridge_tick_index=self._bridge_tick_index,
            observation_time_s=self.current_time_s,
            body_state=self._body_state.copy(),
            wrapped_world_yaw_rad=wrapped,
            unwrapped_world_yaw_rad=unwrapped,
            world_z_yaw_rate_rad_s=yaw_rate,
            requested_control=control,
            sample=sample,
            initialization_mode=mode,
        )

    def _advance_one_interval_unchecked(self) -> CanonicalClosedLoopInterval:
        if self._bridge_tick_index >= self.config.bridge_interval_count:
            raise CanonicalClosedLoopError("configured episode interval is exhausted")
        start_s = self.current_time_s
        observation = self._obtain_circuit_sample()
        projected_phase = self.mechanics.project_phase_end_unwrapped_rad(
            self.config.bridge_dt_s
        )
        bridge_start = self.bridge.begin_interval(
            circuit_sample=None if observation is None else observation.sample,
        )
        if (
            bridge_start.tick_index != self._bridge_tick_index
            or abs(bridge_start.interval_start_s - start_s) > _TIME_TOLERANCE_S
        ):
            raise CanonicalClosedLoopError("bridge clock diverged from the orchestrator")
        mechanics_frames = self.mechanics.advance_bridge_interval(
            bridge_start
        )
        if len(mechanics_frames) != CANONICAL_PHYSICS_STEPS_PER_BRIDGE:
            raise CanonicalClosedLoopError("mechanics did not emit exactly five frames")
        if abs(mechanics_frames[-1].phase_end_unwrapped_rad - projected_phase) > _TIME_TOLERANCE_S:
            raise CanonicalClosedLoopError("mechanics endpoint differs from exact projection")

        transitions = []
        zero = np.zeros(3, dtype=float)
        for mechanics_frame in mechanics_frames:
            expected_start = self._physics_tick_index * self.config.physics_dt_s
            if (
                mechanics_frame.tick_index != self._physics_tick_index
                or abs(mechanics_frame.interval_start_s - expected_start)
                > _TIME_TOLERANCE_S
            ):
                raise CanonicalClosedLoopError("mechanics/physics clocks diverged")
            before = self._body_state.copy()
            physical_wing_command = map_raw_app_wing_kinematics_to_physical(
                mechanics_frame.actuation_wing_kinematics,
                self.config.effector_laterality_hypothesis,
            )
            after = self.physics_adapter.step(
                physical_wing_command,
                zero,
                zero,
                self.config.physics_dt_s,
            )
            if not isinstance(after, RigidBodyState):
                raise CanonicalClosedLoopError("physics adapter returned invalid body state")
            after.validate()
            self._body_state = after.copy()
            # Update at physics cadence so quaternion wrap crossings cannot be
            # missed between 5 ms visual samples.
            self._update_yaw(self._body_state)
            wrench = _wrench_copy(self.physics_adapter.aerodynamic_wrench())
            articulated_telemetry = self._articulated_physics_telemetry()
            transitions.append(
                CanonicalPhysicsTransition(
                    physics_tick_index=self._physics_tick_index,
                    interval_start_s=expected_start,
                    interval_end_s=expected_start + self.config.physics_dt_s,
                    mechanics=mechanics_frame,
                    physical_actuation_wing_kinematics=physical_wing_command,
                    body_state_before=before,
                    body_state_after=self._body_state.copy(),
                    aerodynamic_wrench_after=wrench,
                    articulated_telemetry_after=articulated_telemetry,
                )
            )
            self._physics_tick_index += 1

        bridge_frame = self.bridge.end_interval(
            (
                bridge_start.wing_phase_start_unwrapped_rad,
                *(frame.phase_end_unwrapped_rad for frame in mechanics_frames),
            )
        )
        if bridge_frame.delivered_events != bridge_start.delivered_events:
            raise CanonicalClosedLoopError(
                "bridge altered the start-of-interval delivered-event set"
            )

        self._figure_world_azimuth_rad += (
            self.config.figure_velocity_rad_s * self.config.bridge_dt_s
        )
        interval = CanonicalClosedLoopInterval(
            bridge_tick_index=self._bridge_tick_index,
            interval_start_s=start_s,
            interval_end_s=start_s + self.config.bridge_dt_s,
            circuit_observation=observation,
            projected_model_phase_end_unwrapped_rad=projected_phase,
            phase_projection_semantics=(
                "exact non-mutating model-owned carrier projection; not observed or measured"
            ),
            bridge_start=bridge_start,
            bridge=bridge_frame,
            mechanics=mechanics_frames,
            physics=tuple(transitions),
        )
        self._bridge_tick_index += 1
        self._records.append(interval)
        return interval

    def _advance_one_interval(self) -> CanonicalClosedLoopInterval:
        """Advance atomically from the caller's perspective or poison the stack.

        The component APIs do not implement a distributed rollback.  If any
        exception occurs after one component has advanced, continuing could
        duplicate or skip an event.  The instance therefore becomes fail-stop
        and rejects all later advance/checkpoint requests; callers must restore
        a prior composite checkpoint into a fresh stack.
        """

        if self._failed:
            raise CanonicalClosedLoopError(
                "simulator is fail-stop after an incomplete interval; restore a prior checkpoint"
            )
        try:
            return self._advance_one_interval_unchecked()
        except Exception as exc:
            self._failed = True
            self._failure_reason = "%s: %s" % (type(exc).__name__, exc)
            raise

    def advance_intervals(self, count: int) -> Tuple[CanonicalClosedLoopInterval, ...]:
        """Advance ``count`` bridge intervals and return only the new records."""

        count = _nonnegative_int(count, "count")
        if self._failed:
            raise CanonicalClosedLoopError(
                "simulator is fail-stop after an incomplete interval; restore a prior checkpoint"
            )
        self._ensure_initialized()
        if self._bridge_tick_index + count > self.config.bridge_interval_count:
            raise ValueError("requested intervals exceed the configured episode")
        return tuple(self._advance_one_interval() for _ in range(count))

    def run(self) -> CanonicalClosedLoopResult:
        """Run to the configured half-open endpoint and return full local telemetry."""

        self._ensure_initialized()
        remaining = self.config.bridge_interval_count - self._bridge_tick_index
        self.advance_intervals(remaining)
        return self.result()

    def result(self) -> CanonicalClosedLoopResult:
        if self._failed:
            raise CanonicalClosedLoopError(
                "cannot publish a result from a fail-stop stack after an incomplete interval"
            )
        if not self._initialized:
            raise CanonicalClosedLoopError("the simulator has not been initialized")
        if self._segment_initial_aerodynamic_wrench is None:
            raise CanonicalClosedLoopError(
                "initial physics telemetry was not captured at the segment boundary"
            )
        initial_tick = (
            self._records[0].bridge_tick_index
            if self._records
            else self._bridge_tick_index
        )
        return CanonicalClosedLoopResult(
            config=self.config,
            initial_bridge_tick_index=initial_tick,
            final_bridge_tick_index=self._bridge_tick_index,
            initial_body_state=self._segment_initial_body_state.copy(),
            initial_aerodynamic_wrench=_wrench_copy(
                self._segment_initial_aerodynamic_wrench
            ),
            initial_articulated_physics_telemetry=(
                self._segment_initial_articulated_telemetry
            ),
            final_body_state=self._body_state.copy(),
            intervals=tuple(self._records),
            backend_name=self.physics_adapter.backend_name,
            aerodynamic_owner=self.physics_adapter.aerodynamic_owner,
            circuit_sample_count=self._circuit_sample_count,
            articulated_physics_telemetry_available=(
                self._has_articulated_physics_telemetry
            ),
            effector_mapping_receipt=self.config.effector_mapping_receipt,
        )

    def checkpoint(self) -> CanonicalClosedLoopCheckpoint:
        """Capture a complete stack checkpoint at a completed bridge boundary."""

        if self._failed:
            raise CanonicalClosedLoopError(
                "cannot checkpoint a fail-stop stack after an incomplete interval"
            )
        self._ensure_initialized()
        if self._bridge_tick_index == 0 or self._pending_initial_sample is not None:
            raise CanonicalClosedLoopError(
                "checkpoint requires at least one completed bridge interval"
            )
        if self.bridge.tick_index != self._bridge_tick_index:
            raise CanonicalClosedLoopError("bridge is not at the completed boundary")
        if self.mechanics.tick_index != self._physics_tick_index:
            raise CanonicalClosedLoopError("mechanics is not at the completed boundary")
        if self._physics_tick_index != self._bridge_tick_index * CANONICAL_PHYSICS_STEPS_PER_BRIDGE:
            raise CanonicalClosedLoopError("orchestrator clock counters are inconsistent")
        if self._last_circuit_sample_index is None:
            raise CanonicalClosedLoopError("no circuit sample has crossed the boundary")
        bridge_delivered_ids = tuple(
            event.event_id for event in self.bridge.delivered_events
        )
        mechanics_terminal_ids = (
            self.mechanics.applied_event_ids + self.mechanics.suppressed_event_ids
        )
        if (
            self.mechanics.pending_event_count != 0
            or len(mechanics_terminal_ids) != len(set(mechanics_terminal_ids))
            or set(mechanics_terminal_ids) != set(bridge_delivered_ids)
        ):
            raise CanonicalClosedLoopError(
                "bridge and mechanics event ledgers disagree at the checkpoint boundary"
            )
        circuit_checkpoint = self.circuit_runtime.checkpoint()
        bridge_checkpoint = self.bridge.checkpoint()
        mechanics_checkpoint = self.mechanics.checkpoint()
        physics_checkpoint = self.physics_adapter.checkpoint()
        unsigned: Dict[str, Any] = {
            "schema_version": CANONICAL_CLOSED_LOOP_SCHEMA_VERSION,
            "runtime_version": CANONICAL_CLOSED_LOOP_RUNTIME_VERSION,
            "config": dict(self.config.to_dict()),
            "components": {
                "circuit": _checkpoint_dict(circuit_checkpoint, "circuit"),
                "bridge": _checkpoint_dict(bridge_checkpoint, "bridge"),
                "mechanics": _checkpoint_dict(mechanics_checkpoint, "mechanics"),
                "physics": _checkpoint_dict(physics_checkpoint, "physics"),
            },
            "state": {
                "bridge_tick_index": self._bridge_tick_index,
                "physics_tick_index": self._physics_tick_index,
                "circuit_sample_count": self._circuit_sample_count,
                "last_circuit_sample_index": self._last_circuit_sample_index,
                "body_state": _body_to_dict(self._body_state),
                "yaw_wrapped_rad": self._yaw_wrapped_rad,
                "yaw_unwrapped_rad": self._yaw_unwrapped_rad,
                "figure_world_azimuth_rad": self._figure_world_azimuth_rad,
                "figure_velocity_rad_s": self.config.figure_velocity_rad_s,
                "ground_velocity_rad_s": self.config.ground_velocity_rad_s,
            },
            "receipts": {
                "backend_name": self.physics_adapter.backend_name,
                "aerodynamic_owner": self.physics_adapter.aerodynamic_owner,
                "articulated_physics_telemetry_available": (
                    self._has_articulated_physics_telemetry
                ),
                "effector_mapping": dict(
                    self.config.effector_mapping_receipt.to_dict()
                ),
                "source_limitations": list(CANONICAL_SOURCE_LIMITATIONS),
            },
        }
        return CanonicalClosedLoopCheckpoint(
            {**unsigned, "payload_sha256": _sha256(unsigned)}
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: CanonicalClosedLoopCheckpoint,
        config: CanonicalClosedLoopConfig,
        *,
        circuit_factory: Callable[[], NodeFlyFGSCircuitRuntime],
        physics_factory: Callable[[], ExternalFlightPhysicsAdapter],
        bridge_factory: Callable[[], StreamingNOD1MotorBridge] = StreamingNOD1MotorBridge,
        mechanics_factory: Callable[[], StreamingMuscleWingStepper] = StreamingMuscleWingStepper,
        circuit_checkpoint_loader: Callable[[Mapping[str, Any]], Any] = FlyFGSCircuitCheckpoint,
        physics_checkpoint_loader: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    ) -> "CanonicalClosedLoopSimulator":
        """Build a fresh stack and restore it transactionally.

        No caller-owned live component is mutated.  The composite hash and the
        trusted expected ``config`` are checked before any factory runs.  Each
        component then performs its own content/config/model validation on the
        fresh candidate.
        """

        if not isinstance(checkpoint, CanonicalClosedLoopCheckpoint):
            raise TypeError("checkpoint must be CanonicalClosedLoopCheckpoint")
        if not isinstance(config, CanonicalClosedLoopConfig):
            raise TypeError("config must be CanonicalClosedLoopConfig")
        payload = CanonicalClosedLoopCheckpoint(checkpoint.to_dict()).to_dict()
        if payload["config"] != config.to_dict():
            raise ValueError("closed-loop checkpoint configuration mismatch")
        components = payload["components"]
        state_payload = payload["state"]
        receipts = payload["receipts"]
        if (
            state_payload["figure_velocity_rad_s"]
            != config.figure_velocity_rad_s
            or state_payload["ground_velocity_rad_s"]
            != config.ground_velocity_rad_s
        ):
            raise ValueError("scene velocity checkpoint mismatch")

        circuit = circuit_factory()
        physics = None
        try:
            cls._start_runtime(circuit)
            circuit_checkpoint = circuit_checkpoint_loader(components["circuit"])
            restored_sample = circuit.restore(
                circuit_checkpoint,
                include_retinal_input=config.include_retinal_input,
                include_full_cell_state=config.include_full_cell_state,
            )
            bridge = bridge_factory()
            bridge.restore(StreamingBridgeCheckpoint(components["bridge"]))
            mechanics = mechanics_factory()
            mechanics.restore(StreamingMechanicsCheckpoint(components["mechanics"]))
            physics = physics_factory()
            if physics_checkpoint_loader is None:
                if getattr(physics, "backend_name", None) == "flybody":
                    from ..flybody_adapter import FlyBodyPhysicsCheckpoint

                    physics_checkpoint = FlyBodyPhysicsCheckpoint(components["physics"])
                else:
                    physics_checkpoint = components["physics"]
            else:
                physics_checkpoint = physics_checkpoint_loader(components["physics"])
            restored_body = physics.restore_checkpoint(physics_checkpoint)
            expected_body = _body_from_dict(state_payload["body_state"])
            if not isinstance(restored_body, RigidBodyState) or not _body_equal(
                restored_body, expected_body
            ):
                raise ValueError("physics checkpoint body state mismatch")

            instance = cls.__new__(cls)
            instance.config = config
            instance.circuit_runtime = circuit
            instance.bridge = bridge
            instance.mechanics = mechanics
            instance.physics_adapter = physics
            instance._has_articulated_physics_telemetry = instance._validate_components_for_restored_stack()
            instance._initial_body_state = expected_body.copy()
            instance._body_state = expected_body.copy()
            instance._segment_initial_body_state = expected_body.copy()
            instance._segment_initial_aerodynamic_wrench = _wrench_copy(
                physics.aerodynamic_wrench()
            )
            instance._segment_initial_articulated_telemetry = (
                instance._articulated_physics_telemetry()
            )
            instance._bridge_tick_index = _nonnegative_int(
                state_payload["bridge_tick_index"], "bridge_tick_index"
            )
            instance._physics_tick_index = _nonnegative_int(
                state_payload["physics_tick_index"], "physics_tick_index"
            )
            instance._circuit_sample_count = _nonnegative_int(
                state_payload["circuit_sample_count"], "circuit_sample_count"
            )
            instance._last_circuit_sample_index = _nonnegative_int(
                state_payload["last_circuit_sample_index"],
                "last_circuit_sample_index",
            )
            instance._figure_world_azimuth_rad = _finite(
                state_payload["figure_world_azimuth_rad"],
                "figure_world_azimuth_rad",
            )
            instance._yaw_wrapped_rad = _finite(
                state_payload["yaw_wrapped_rad"], "yaw_wrapped_rad"
            )
            instance._yaw_unwrapped_rad = _finite(
                state_payload["yaw_unwrapped_rad"], "yaw_unwrapped_rad"
            )
            instance._initialized = True
            instance._pending_initial_sample = None
            instance._records = []
            instance._failed = False
            instance._failure_reason = None

            if receipts["backend_name"] != physics.backend_name or receipts[
                "aerodynamic_owner"
            ] != physics.aerodynamic_owner:
                raise ValueError("physics checkpoint backend receipt mismatch")
            if receipts[
                "articulated_physics_telemetry_available"
            ] is not instance._has_articulated_physics_telemetry:
                raise ValueError("physics telemetry capability receipt mismatch")
            if tuple(receipts["source_limitations"]) != CANONICAL_SOURCE_LIMITATIONS:
                raise ValueError("source limitation receipt mismatch")
            if (
                EffectorMappingReceipt.from_dict(receipts["effector_mapping"])
                != config.effector_mapping_receipt
            ):
                raise ValueError("effector mapping checkpoint receipt mismatch")
            if instance._bridge_tick_index > config.bridge_interval_count:
                raise ValueError("checkpoint lies beyond configured duration")
            if instance._physics_tick_index != instance._bridge_tick_index * CANONICAL_PHYSICS_STEPS_PER_BRIDGE:
                raise ValueError("checkpoint clock ratio mismatch")
            if bridge.tick_index != instance._bridge_tick_index:
                raise ValueError("bridge checkpoint clock mismatch")
            if mechanics.tick_index != instance._physics_tick_index:
                raise ValueError("mechanics checkpoint clock mismatch")
            bridge_delivered_ids = tuple(
                event.event_id for event in bridge.delivered_events
            )
            mechanics_terminal_ids = (
                mechanics.applied_event_ids + mechanics.suppressed_event_ids
            )
            if (
                mechanics.pending_event_count != 0
                or len(mechanics_terminal_ids) != len(set(mechanics_terminal_ids))
                or set(mechanics_terminal_ids) != set(bridge_delivered_ids)
            ):
                raise ValueError("bridge/mechanics checkpoint event ledger mismatch")
            expected_samples = (
                (instance._bridge_tick_index - 1)
                // CANONICAL_BRIDGE_STEPS_PER_CIRCUIT
                + 1
            )
            if (
                instance._circuit_sample_count != expected_samples
                or instance._last_circuit_sample_index != expected_samples - 1
                or restored_sample.sample_index != expected_samples - 1
            ):
                raise ValueError("circuit checkpoint clock mismatch")
            expected_figure = (
                config.figure_initial_world_azimuth_rad
                + config.figure_velocity_rad_s * instance.current_time_s
            )
            if abs(instance._figure_world_azimuth_rad - expected_figure) > _TIME_TOLERANCE_S:
                raise ValueError("scene checkpoint clock mismatch")
            rotation = quaternion_to_matrix(expected_body.quaternion_body_to_world)
            wrapped = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
            if abs(_angle_difference(wrapped, instance._yaw_wrapped_rad)) > _TIME_TOLERANCE_S:
                raise ValueError("yaw checkpoint does not match the rigid body")
            return instance
        except Exception:
            close = getattr(circuit, "close", None)
            if callable(close):
                close()
            if physics is not None:
                close_physics = getattr(physics, "close", None)
                if callable(close_physics):
                    close_physics()
            raise

    def close(self) -> None:
        close = getattr(self.circuit_runtime, "close", None)
        if callable(close):
            close()

    def _validate_components_for_restored_stack(self) -> bool:
        """Validate optional physics telemetry without requiring fresh clocks."""

        telemetry_capabilities = {
            "wing_joint_state": callable(
                getattr(self.physics_adapter, "wing_joint_state", None)
            ),
            "wing_joint_order": hasattr(self.physics_adapter, "wing_joint_order"),
            "last_actuator_torque_n_m": hasattr(
                self.physics_adapter, "last_actuator_torque_n_m"
            ),
            "whole_fly_com_position_m": callable(
                getattr(self.physics_adapter, "whole_fly_com_position_m", None)
            ),
            "ground_contact_count": callable(
                getattr(self.physics_adapter, "ground_contact_count", None)
            ),
        }
        if any(telemetry_capabilities.values()) and not all(
            telemetry_capabilities.values()
        ):
            raise TypeError("articulated physics telemetry is all-or-none")
        return all(telemetry_capabilities.values())

    def __enter__(self) -> "CanonicalClosedLoopSimulator":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "CANONICAL_BRIDGE_DT_S",
    "CANONICAL_BRIDGE_STEPS_PER_CIRCUIT",
    "CANONICAL_CIRCUIT_DT_S",
    "CANONICAL_CLOSED_LOOP_RUNTIME_VERSION",
    "CANONICAL_CLOSED_LOOP_SCHEMA_VERSION",
    "CANONICAL_MAX_CIRCUIT_SAMPLES",
    "CANONICAL_MAX_DURATION_S",
    "CANONICAL_PHYSICS_DT_S",
    "CANONICAL_PHYSICS_STEPS_PER_BRIDGE",
    "CANONICAL_SOURCE_LIMITATIONS",
    "CanonicalCircuitObservation",
    "CanonicalClosedLoopCheckpoint",
    "CanonicalClosedLoopConfig",
    "CanonicalClosedLoopError",
    "CanonicalClosedLoopInterval",
    "CanonicalClosedLoopResult",
    "CanonicalClosedLoopSimulator",
    "CanonicalPhysicsTransition",
    "CanonicalPhysicsTelemetry",
]
