"""Open-loop Reichardt 1983 stimulus-to-tether orchestration."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from ..paper_fgs import (
    PaperFGSProtocol,
    load_nod1_laterality_receipt,
    sample_protocol,
)
from ..paper_fgs_runtime import NodePaperFGSCircuitRuntime, PaperFGSCircuitSample
from ..paper_calibration import create_torque_calibration_receipt
from ..paper_signal import named_torque_products, synchronize_raw_torque
from ..paper_tether import (
    PAPER_TORQUE_METER_MODES,
    PaperComparisonFlyBodyAdapter,
    PaperFixedLoadCellFlyBodyAdapter,
    PaperTetheredFlyBodyAdapter,
    make_paper_torque_meter,
)
from .effector_mapping import (
    EffectorLateralityHypothesis,
    map_raw_app_wing_kinematics_to_physical,
)
from .rigid_body import quaternion_to_matrix
from .streaming_bridge import StreamingNOD1MotorBridge
from .streaming_mechanics import StreamingMechanicsConfig, StreamingMuscleWingStepper


PAPER_OPEN_LOOP_SCHEMA_VERSION = "3.0.0"
PAPER_BRIDGE_DT_S = 0.0005
PAPER_PHYSICS_DT_S = 0.0001
PAPER_CIRCUIT_DT_S = 0.0025
PAPER_TORQUE_LOG_DT_S = 0.001
_TIME_TOLERANCE_S = 1.0e-12


class PaperOpenLoopError(RuntimeError):
    pass


class PaperStreamingNOD1MotorBridge(StreamingNOD1MotorBridge):
    """Long-duration paper source while preserving the frozen legacy limit."""

    _MAXIMUM_SOURCE_SAMPLE_INDEX = None


@dataclass(frozen=True)
class PaperOpenLoopConfig:
    protocol: PaperFGSProtocol
    repetitions: int = 100
    run_seed: int = 73
    texture_relationship_mode: str = "registered_copy"
    texture_seed: int = 123456
    pre_roll_duration_s: float = 0.4
    post_roll_duration_s: float = 0.4
    circuit_rate_hz: int = 400
    physics_rate_hz: int = 10000
    torque_meter_mode: str = "fixed-load-cell"
    wingbeat_phase_mode: str = "uniformly_stratified"
    effector_laterality_hypothesis: EffectorLateralityHypothesis = (
        EffectorLateralityHypothesis.RAW_L_TO_PHYSICAL_RIGHT
    )

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, PaperFGSProtocol):
            raise TypeError("protocol must be PaperFGSProtocol")
        if (
            isinstance(self.repetitions, bool)
            or not isinstance(self.repetitions, int)
            or self.repetitions <= 0
        ):
            raise ValueError("repetitions must be positive")
        if not isinstance(self.run_seed, int) or isinstance(self.run_seed, bool):
            raise ValueError("run_seed must be an integer")
        if self.circuit_rate_hz not in (400, 800):
            raise ValueError("circuit_rate_hz must be 400 or 800")
        if self.physics_rate_hz not in (10000, 20000):
            raise ValueError("physics_rate_hz must be 10000 or 20000")
        if self.texture_relationship_mode not in (
            "registered_copy",
            "independent_matched_statistics",
        ):
            raise ValueError("unsupported texture relationship mode")
        if (
            isinstance(self.texture_seed, bool)
            or not isinstance(self.texture_seed, int)
            or self.texture_seed < 0
            or self.texture_seed > 0xFFFFFFFF
        ):
            raise ValueError("texture_seed must be uint32")
        if not math.isclose(
            self.pre_roll_duration_s, 0.4, rel_tol=0.0, abs_tol=_TIME_TOLERANCE_S
        ):
            raise ValueError("canonical paper pre-roll is one 0.4 s period")
        if not math.isclose(
            self.post_roll_duration_s, 0.4, rel_tol=0.0, abs_tol=_TIME_TOLERANCE_S
        ):
            raise ValueError("canonical paper post-roll is one 0.4 s period")
        if self.torque_meter_mode not in PAPER_TORQUE_METER_MODES:
            raise ValueError("unsupported torque meter mode")
        if self.wingbeat_phase_mode not in (
            "uniformly_stratified",
            "fixed",
        ):
            raise ValueError("unsupported wingbeat phase mode")
        hypothesis = EffectorLateralityHypothesis(
            self.effector_laterality_hypothesis
        )
        object.__setattr__(self, "effector_laterality_hypothesis", hypothesis)
        for value, dt, label in (
            (self.pre_roll_duration_s, self.circuit_dt_s, "pre-roll/circuit"),
            (self.pre_roll_duration_s, PAPER_BRIDGE_DT_S, "pre-roll/bridge"),
            (self.post_roll_duration_s, self.circuit_dt_s, "post-roll/circuit"),
            (self.post_roll_duration_s, PAPER_BRIDGE_DT_S, "post-roll/bridge"),
            (self.protocol.trial_duration_s, PAPER_TORQUE_LOG_DT_S, "trial/log"),
            (self.protocol.trial_duration_s, self.physics_dt_s, "trial/physics"),
        ):
            if not math.isclose(value / dt, round(value / dt), abs_tol=1e-9):
                raise ValueError("{} clocks do not divide exactly".format(label))

    @property
    def circuit_dt_s(self) -> float:
        return 1.0 / self.circuit_rate_hz

    @property
    def physics_dt_s(self) -> float:
        return 1.0 / self.physics_rate_hz

    def trial_seed(self, trial_id: int) -> int:
        payload = "{}|{}|{}".format(
            self.run_seed, self.protocol.protocol_id, trial_id
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")

    def wingbeat_phase_radians(self, trial_id: int) -> float:
        """Return a deterministic phase stratum, permuted by the run seed."""

        if not 0 <= trial_id < self.repetitions:
            raise ValueError("trial_id is outside the repetition range")
        if self.wingbeat_phase_mode == "fixed":
            return 0.0
        ranked = sorted(
            range(self.repetitions),
            key=lambda item: hashlib.sha256(
                "{}|{}|wingbeat|{}".format(
                    self.run_seed, self.protocol.protocol_id, item
                ).encode("utf-8")
            ).digest(),
        )
        stratum = ranked.index(trial_id)
        return 2.0 * math.pi * (stratum + 0.5) / self.repetitions


def _quaternion_to_euler_deg(quaternion: np.ndarray) -> np.ndarray:
    rotation = quaternion_to_matrix(np.asarray(quaternion, dtype=float))
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return np.rad2deg([roll, pitch, yaw])


class PaperOpenLoopSimulator:
    """Execute phase-locked trials with prescribed stimulus and fixed body."""

    def __init__(
        self,
        config: PaperOpenLoopConfig,
        *,
        physics_adapter: Optional[Any] = None,
        circuit_factory: Optional[Callable[[], NodePaperFGSCircuitRuntime]] = None,
        bridge_factory: Callable[..., StreamingNOD1MotorBridge] = (
            PaperStreamingNOD1MotorBridge
        ),
        mechanics_factory: Callable[[], StreamingMuscleWingStepper] = (
            StreamingMuscleWingStepper
        ),
    ) -> None:
        self.config = config
        self.physics = physics_adapter or make_paper_torque_meter(
            config.torque_meter_mode
        )
        self.circuit_factory = circuit_factory or (
            lambda: NodePaperFGSCircuitRuntime(circuit_dt_s=config.circuit_dt_s)
        )
        self.bridge_factory = bridge_factory
        self.mechanics_factory = mechanics_factory
        bundle = getattr(self.physics, "bundle", None)
        if isinstance(bundle, Mapping):
            worker_dt = float(
                bundle.get(
                    "paper_physics_sample_timestep_s",
                    bundle["config"].timestep_s,
                )
            )
            if not math.isclose(
                worker_dt, config.physics_dt_s, rel_tol=0.0, abs_tol=1.0e-15
            ):
                raise PaperOpenLoopError(
                    "FlyBody timestep does not match paper physics_rate_hz"
                )
        self.sign_calibration = self.physics.calibrate_paper_yaw_sign()
        if self.sign_calibration.get("passed") is not True:
            raise PaperOpenLoopError("paper torque sign is not calibrated")
        self.torque_calibration = None
        if isinstance(
            self.physics,
            (PaperFixedLoadCellFlyBodyAdapter, PaperComparisonFlyBodyAdapter),
        ):
            self.torque_calibration = create_torque_calibration_receipt(
                self.physics
            )

    def _record_boundary(
        self,
        trial: Dict[str, list],
        relative_time_s: float,
        wingbeat_phase_rad: float,
    ) -> None:
        torque = self.physics.tether_torque_sample()
        body = self.physics.body_state()
        wing_position, _wing_velocity = self.physics.wing_joint_state()
        trial["time_s"].append(relative_time_s)
        trial["support_on_fly_yaw_reaction_engine_Nm"].append(
            torque.support_moment_engine_n_m[2]
        )
        trial["support_load_cell_force_engine_N"].append(
            torque.support_load_cell_force_engine_n.copy()
        )
        trial["support_load_cell_moment_engine_Nm"].append(
            torque.support_load_cell_moment_engine_n_m.copy()
        )
        trial["attempted_fly_yaw_moment_from_support_engine_Nm"].append(
            torque.attempted_fly_yaw_moment_from_support_engine_n_m
        )
        trial["equality_only_yaw_reaction_engine_Nm"].append(
            torque.equality_only_yaw_reaction_engine_n_m
        )
        trial["fly_generated_yaw_moment_engine_Nm"].append(
            torque.fly_generated_moment_engine_n_m[2]
        )
        trial["root_fluid_yaw_moment_engine_Nm"].append(
            torque.root_fluid_moment_engine_n_m[2]
        )
        trial["reported_yaw_torque_Nm"].append(torque.reported_yaw_torque_n_m)
        trial["reported_yaw_torque_dyne_cm"].append(
            torque.reported_yaw_torque_dyne_cm
        )
        trial["body_position_m"].append(body.position_world_m.copy())
        trial["body_euler_deg"].append(
            _quaternion_to_euler_deg(body.quaternion_body_to_world)
        )
        # Retain MuJoCo's instantaneous generalized velocity as a solver
        # diagnostic.  The public body-yaw-rate channel is derived below from
        # the realized pose at the declared 1 kHz synchronized cadence.  A
        # stiff equality can contain sub-sample generalized-velocity ringing
        # even though the measured body pose remains stationary.
        trial["solver_generalized_yaw_rate_deg_s"].append(
            math.degrees(float(body.angular_velocity_body_rad_s[2]))
        )
        trial["wing_joint_position_rad"].append(wing_position.copy())
        trial["head_transform_body"].append(
            self.physics.head_transform_body().copy()
        )
        trial["yaw_equation_residual_Nm"].append(
            self.physics.yaw_equation_residual_n_m()
        )
        trial["yaw_balance_residual_Nm"].append(torque.yaw_balance_residual_n_m)
        trial["wingbeat_phase_rad"].append(float(wingbeat_phase_rad % (2.0 * math.pi)))
        trial["actuator_torque_Nm"].append(torque.actuator_torque_n_m.copy())
        trial["actuator_clipped"].append(float(torque.actuator_clipped))
        trial["wing_load_cell_position_engine_m"].append(
            torque.wing_load_cell_position_engine_m.copy()
        )
        trial["wing_load_cell_orientation_site_to_engine"].append(
            torque.wing_load_cell_orientation_site_to_engine.copy()
        )
        trial["wing_parent_on_child_force_engine_N"].append(
            torque.wing_parent_on_child_force_engine_n.copy()
        )
        trial["wing_parent_on_child_moment_at_hinge_engine_Nm"].append(
            torque.wing_parent_on_child_moment_at_hinge_engine_n_m.copy()
        )
        trial["wing_on_thorax_force_engine_N"].append(
            torque.wing_on_thorax_force_engine_n.copy()
        )
        trial["wing_on_thorax_moment_at_hinge_engine_Nm"].append(
            torque.wing_on_thorax_moment_at_hinge_engine_n_m.copy()
        )
        trial["wing_on_thorax_moment_at_tether_engine_Nm"].append(
            torque.wing_on_thorax_moment_at_tether_engine_n_m.copy()
        )
        trial["wing_reported_yaw_torque_Nm"].append(
            torque.wing_reported_yaw_torque_n_m.copy()
        )
        trial["wing_sum_reported_yaw_torque_Nm"].append(
            torque.wing_sum_reported_yaw_torque_n_m
        )
        trial["nonwing_reported_yaw_torque_residual_Nm"].append(
            torque.nonwing_reported_yaw_torque_residual_n_m
        )
        trial["wing_aerodynamic_moment_at_tether_engine_Nm"].append(
            torque.wing_aerodynamic_moment_at_tether_engine_n_m.copy()
        )
        trial["wing_aerodynamic_reported_yaw_torque_Nm"].append(
            torque.wing_aerodynamic_reported_yaw_torque_n_m.copy()
        )
        trial["full_aerodynamic_reported_yaw_torque_Nm"].append(
            torque.full_aerodynamic_reported_yaw_torque_n_m
        )
        trial["background_aerodynamic_reported_yaw_torque_Nm"].append(
            torque.background_aerodynamic_reported_yaw_torque_n_m
        )
        trial["aerodynamic_closure_residual_Nm"].append(
            torque.aerodynamic_closure_residual_n_m
        )

    def _run_trial(self, trial_id: int) -> Tuple[Mapping[str, np.ndarray], Mapping[str, Any]]:
        circuit = self.circuit_factory()
        bridge = self.bridge_factory(seed=self.config.trial_seed(trial_id))
        initial_wingbeat_phase = self.config.wingbeat_phase_radians(trial_id)
        if self.mechanics_factory is StreamingMuscleWingStepper:
            mechanics = self.mechanics_factory(
                StreamingMechanicsConfig(
                    initial_phase_unwrapped_rad=initial_wingbeat_phase
                )
            )
        else:
            mechanics = self.mechanics_factory()
        self.physics.reset(self.physics.default_initial_state())
        duration = (
            self.config.pre_roll_duration_s
            + self.config.protocol.trial_duration_s
            + self.config.post_roll_duration_s
        )
        bridge_count = int(round(duration / PAPER_BRIDGE_DT_S))
        circuit_samples_per_motor_boundary = int(
            round(0.005 / self.config.circuit_dt_s)
        )
        physics_substeps_per_mechanics_frame = int(
            round(PAPER_PHYSICS_DT_S / self.config.physics_dt_s)
        )
        pre_roll_bridge_count = int(
            round(self.config.pre_roll_duration_s / PAPER_BRIDGE_DT_S)
        )
        trial: Dict[str, list] = {
            name: []
            for name in (
                "time_s",
                "support_on_fly_yaw_reaction_engine_Nm",
                "support_load_cell_force_engine_N",
                "support_load_cell_moment_engine_Nm",
                "attempted_fly_yaw_moment_from_support_engine_Nm",
                "equality_only_yaw_reaction_engine_Nm",
                "fly_generated_yaw_moment_engine_Nm",
                "root_fluid_yaw_moment_engine_Nm",
                "reported_yaw_torque_Nm",
                "reported_yaw_torque_dyne_cm",
                "body_position_m",
                "body_euler_deg",
                "solver_generalized_yaw_rate_deg_s",
                "wing_joint_position_rad",
                "head_transform_body",
                "yaw_equation_residual_Nm",
                "yaw_balance_residual_Nm",
                "wingbeat_phase_rad",
                "actuator_torque_Nm",
                "actuator_clipped",
                "wing_load_cell_position_engine_m",
                "wing_load_cell_orientation_site_to_engine",
                "wing_parent_on_child_force_engine_N",
                "wing_parent_on_child_moment_at_hinge_engine_Nm",
                "wing_on_thorax_force_engine_N",
                "wing_on_thorax_moment_at_hinge_engine_Nm",
                "wing_on_thorax_moment_at_tether_engine_Nm",
                "wing_reported_yaw_torque_Nm",
                "wing_sum_reported_yaw_torque_Nm",
                "nonwing_reported_yaw_torque_residual_Nm",
                "wing_aerodynamic_moment_at_tether_engine_Nm",
                "wing_aerodynamic_reported_yaw_torque_Nm",
                "full_aerodynamic_reported_yaw_torque_Nm",
                "background_aerodynamic_reported_yaw_torque_Nm",
                "aerodynamic_closure_residual_Nm",
            )
        }
        raw_time = []
        raw_support = []
        raw_reported = []
        raw_residual = []
        raw_padded_time = []
        raw_padded_channels: Dict[str, list] = {
            name: []
            for name in (
                "support_load_cell_force_engine_N",
                "support_load_cell_moment_engine_Nm",
                "support_on_fly_yaw_reaction_engine_Nm",
                "attempted_fly_yaw_moment_from_support_engine_Nm",
                "root_fluid_yaw_moment_engine_Nm",
                "equality_only_yaw_reaction_engine_Nm",
                "yaw_balance_residual_Nm",
                "reported_yaw_torque_Nm",
                "reported_yaw_torque_dyne_cm",
                "actuator_torque_Nm",
                "actuator_clipped",
                "wing_load_cell_position_engine_m",
                "wing_load_cell_orientation_site_to_engine",
                "wing_parent_on_child_force_engine_N",
                "wing_parent_on_child_moment_at_hinge_engine_Nm",
                "wing_on_thorax_force_engine_N",
                "wing_on_thorax_moment_at_hinge_engine_Nm",
                "wing_on_thorax_moment_at_tether_engine_Nm",
                "wing_reported_yaw_torque_Nm",
                "wing_sum_reported_yaw_torque_Nm",
                "nonwing_reported_yaw_torque_residual_Nm",
                "wing_aerodynamic_moment_at_tether_engine_Nm",
                "wing_aerodynamic_reported_yaw_torque_Nm",
                "full_aerodynamic_reported_yaw_torque_Nm",
                "background_aerodynamic_reported_yaw_torque_Nm",
                "aerodynamic_closure_residual_Nm",
            )
        }
        internal_samples_per_published_step: Optional[int] = None
        current_circuit: Optional[PaperFGSCircuitSample] = None
        runtime_receipt: Optional[Mapping[str, Any]] = None
        zero = np.zeros(3, dtype=float)
        maximum_translation_drift_m = 0.0
        maximum_rotation_drift_rad = 0.0
        maximum_abs_yaw_rate_rad_s = 0.0
        maximum_head_transform_drift = 0.0
        maximum_ground_contact_count = 0
        reference_head_transform = self.physics.head_transform_body()
        try:
            current_circuit = circuit.initialize(
                self.config.protocol.protocol_id,
                texture_relationship_mode=self.config.texture_relationship_mode,
                texture_seed=self.config.texture_seed,
                stimulus_time_s=-self.config.pre_roll_duration_s,
            )
            runtime_receipt = dict(circuit.ready_receipt)
            for bridge_tick in range(bridge_count):
                source_sample = None
                if bridge_tick % 10 == 0:
                    assert current_circuit is not None
                    bridge_sample_index = bridge_tick // 10
                    target_circuit_index = (
                        bridge_sample_index * circuit_samples_per_motor_boundary
                    )
                    while current_circuit.sample_index < target_circuit_index:
                        next_index = current_circuit.sample_index + 1
                        stimulus_time = (
                            -self.config.pre_roll_duration_s
                            + next_index * self.config.circuit_dt_s
                        )
                        current_circuit = circuit.advance(stimulus_time)
                    if current_circuit.sample_index != target_circuit_index:
                        raise PaperOpenLoopError("paper circuit clock diverged")
                    source_sample = current_circuit.to_bridge_sample(
                        bridge_sample_index
                    )

                if bridge_tick >= pre_roll_bridge_count:
                    relative_bridge_tick = bridge_tick - pre_roll_bridge_count
                    relative_boundary_s = (
                        relative_bridge_tick * PAPER_BRIDGE_DT_S
                    )
                    if (
                        relative_bridge_tick % 2 == 0
                        and relative_boundary_s
                        < self.config.protocol.trial_duration_s
                        - _TIME_TOLERANCE_S
                    ):
                        self._record_boundary(
                            trial,
                            relative_boundary_s,
                            mechanics.phase_unwrapped_rad,
                        )

                bridge_start = bridge.begin_interval(circuit_sample=source_sample)
                mechanics_frames = mechanics.advance_bridge_interval(bridge_start)
                if len(mechanics_frames) != 5:
                    raise PaperOpenLoopError("mechanics did not emit five 0.1 ms steps")
                for frame in mechanics_frames:
                    physical_wings = map_raw_app_wing_kinematics_to_physical(
                        frame.actuation_wing_kinematics,
                        self.config.effector_laterality_hypothesis,
                    )
                    for physics_substep in range(
                        physics_substeps_per_mechanics_frame
                    ):
                        self.physics.step(
                            physical_wings,
                            zero,
                            zero,
                            self.config.physics_dt_s,
                        )
                        physics_end_s = (
                            frame.tick_index * PAPER_PHYSICS_DT_S
                            + (physics_substep + 1) * self.config.physics_dt_s
                        )
                        relative_end_s = (
                            physics_end_s - self.config.pre_roll_duration_s
                        )
                        consume = getattr(
                            self.physics,
                            "consume_last_internal_torque_samples",
                            None,
                        )
                        internal_samples = (
                            tuple(consume()) if callable(consume) else ()
                        )
                        if not internal_samples:
                            internal_samples = (self.physics.tether_torque_sample(),)
                        if internal_samples_per_published_step is None:
                            internal_samples_per_published_step = len(internal_samples)
                        elif internal_samples_per_published_step != len(internal_samples):
                            raise PaperOpenLoopError("raw internal sample count changed")
                        raw_internal_dt = (
                            self.config.physics_dt_s / len(internal_samples)
                        )
                        raw_interval_start = relative_end_s - self.config.physics_dt_s
                        for raw_index, sample in enumerate(internal_samples):
                            raw_relative_time = (
                                raw_interval_start
                                + (raw_index + 1) * raw_internal_dt
                            )
                            raw_padded_time.append(raw_relative_time)
                            raw_padded_channels[
                                "support_load_cell_force_engine_N"
                            ].append(sample.support_load_cell_force_engine_n.copy())
                            raw_padded_channels[
                                "support_load_cell_moment_engine_Nm"
                            ].append(sample.support_load_cell_moment_engine_n_m.copy())
                            raw_padded_channels[
                                "support_on_fly_yaw_reaction_engine_Nm"
                            ].append(sample.support_moment_engine_n_m[2])
                            raw_padded_channels[
                                "attempted_fly_yaw_moment_from_support_engine_Nm"
                            ].append(-sample.support_moment_engine_n_m[2])
                            raw_padded_channels[
                                "root_fluid_yaw_moment_engine_Nm"
                            ].append(sample.root_fluid_moment_engine_n_m[2])
                            raw_padded_channels[
                                "equality_only_yaw_reaction_engine_Nm"
                            ].append(sample.equality_only_yaw_reaction_engine_n_m)
                            raw_padded_channels["yaw_balance_residual_Nm"].append(
                                sample.yaw_balance_residual_n_m
                            )
                            raw_padded_channels["reported_yaw_torque_Nm"].append(
                                sample.reported_yaw_torque_n_m
                            )
                            raw_padded_channels[
                                "reported_yaw_torque_dyne_cm"
                            ].append(sample.reported_yaw_torque_dyne_cm)
                            raw_padded_channels["actuator_torque_Nm"].append(
                                sample.actuator_torque_n_m.copy()
                            )
                            raw_padded_channels["actuator_clipped"].append(
                                float(sample.actuator_clipped)
                            )
                            raw_padded_channels[
                                "wing_load_cell_position_engine_m"
                            ].append(sample.wing_load_cell_position_engine_m.copy())
                            raw_padded_channels[
                                "wing_load_cell_orientation_site_to_engine"
                            ].append(
                                sample.wing_load_cell_orientation_site_to_engine.copy()
                            )
                            raw_padded_channels[
                                "wing_parent_on_child_force_engine_N"
                            ].append(sample.wing_parent_on_child_force_engine_n.copy())
                            raw_padded_channels[
                                "wing_parent_on_child_moment_at_hinge_engine_Nm"
                            ].append(
                                sample.wing_parent_on_child_moment_at_hinge_engine_n_m.copy()
                            )
                            raw_padded_channels[
                                "wing_on_thorax_force_engine_N"
                            ].append(sample.wing_on_thorax_force_engine_n.copy())
                            raw_padded_channels[
                                "wing_on_thorax_moment_at_hinge_engine_Nm"
                            ].append(
                                sample.wing_on_thorax_moment_at_hinge_engine_n_m.copy()
                            )
                            raw_padded_channels[
                                "wing_on_thorax_moment_at_tether_engine_Nm"
                            ].append(
                                sample.wing_on_thorax_moment_at_tether_engine_n_m.copy()
                            )
                            raw_padded_channels[
                                "wing_reported_yaw_torque_Nm"
                            ].append(sample.wing_reported_yaw_torque_n_m.copy())
                            raw_padded_channels[
                                "wing_sum_reported_yaw_torque_Nm"
                            ].append(sample.wing_sum_reported_yaw_torque_n_m)
                            raw_padded_channels[
                                "nonwing_reported_yaw_torque_residual_Nm"
                            ].append(
                                sample.nonwing_reported_yaw_torque_residual_n_m
                            )
                            raw_padded_channels[
                                "wing_aerodynamic_moment_at_tether_engine_Nm"
                            ].append(
                                sample.wing_aerodynamic_moment_at_tether_engine_n_m.copy()
                            )
                            raw_padded_channels[
                                "wing_aerodynamic_reported_yaw_torque_Nm"
                            ].append(
                                sample.wing_aerodynamic_reported_yaw_torque_n_m.copy()
                            )
                            raw_padded_channels[
                                "full_aerodynamic_reported_yaw_torque_Nm"
                            ].append(
                                sample.full_aerodynamic_reported_yaw_torque_n_m
                            )
                            raw_padded_channels[
                                "background_aerodynamic_reported_yaw_torque_Nm"
                            ].append(
                                sample.background_aerodynamic_reported_yaw_torque_n_m
                            )
                            raw_padded_channels[
                                "aerodynamic_closure_residual_Nm"
                            ].append(sample.aerodynamic_closure_residual_n_m)
                        if (
                            relative_end_s > 0
                            and relative_end_s
                            <= self.config.protocol.trial_duration_s
                            + _TIME_TOLERANCE_S
                        ):
                            sample = internal_samples[-1]
                            raw_time.append(relative_end_s)
                            raw_support.append(
                                sample.support_moment_engine_n_m[2]
                            )
                            raw_reported.append(sample.reported_yaw_torque_n_m)
                            residual = self.physics.yaw_equation_residual_n_m()
                            raw_residual.append(residual)
                            constraint = self.physics.body_constraint_diagnostics()
                            maximum_translation_drift_m = max(
                                maximum_translation_drift_m,
                                abs(
                                    float(
                                        constraint["body_translation_drift_m"]
                                    )
                                ),
                            )
                            maximum_rotation_drift_rad = max(
                                maximum_rotation_drift_rad,
                                abs(float(constraint["body_rotation_drift_rad"])),
                            )
                            maximum_abs_yaw_rate_rad_s = max(
                                maximum_abs_yaw_rate_rad_s,
                                abs(float(constraint["body_yaw_rate_rad_s"])),
                            )
                            maximum_head_transform_drift = max(
                                maximum_head_transform_drift,
                                float(
                                    np.max(
                                        np.abs(
                                            self.physics.head_transform_body()
                                            - reference_head_transform
                                        )
                                    )
                                ),
                            )
                            maximum_ground_contact_count = max(
                                maximum_ground_contact_count,
                                int(self.physics.ground_contact_count()),
                            )
                bridge.end_interval(
                    (
                        bridge_start.wing_phase_start_unwrapped_rad,
                        *(frame.phase_end_unwrapped_rad for frame in mechanics_frames),
                    )
                )
        finally:
            circuit.close()

        converted = {
            name: np.asarray(values, dtype=float) for name, values in trial.items()
        }
        yaw_rad = np.unwrap(np.deg2rad(converted["body_euler_deg"][:, 2]))
        converted["body_yaw_rate_deg_s"] = np.rad2deg(
            np.gradient(yaw_rad, PAPER_TORQUE_LOG_DT_S, edge_order=1)
        )
        converted["raw_physics_time_s"] = np.asarray(raw_time, dtype=float)
        converted["raw_support_on_fly_yaw_reaction_engine_Nm"] = np.asarray(
            raw_support, dtype=float
        )
        converted["raw_reported_yaw_torque_Nm"] = np.asarray(
            raw_reported, dtype=float
        )
        converted["raw_yaw_equation_residual_Nm"] = np.asarray(
            raw_residual, dtype=float
        )
        converted["raw_internal_padded_time_s"] = np.asarray(
            raw_padded_time, dtype=float
        )
        for name, values in raw_padded_channels.items():
            converted["raw_internal_padded_" + name] = np.asarray(
                values, dtype=float
            )
        peak_torque = float(
            np.max(np.abs(converted["raw_reported_yaw_torque_Nm"]), initial=0.0)
        )
        finite_residual = converted[
            "raw_internal_padded_yaw_balance_residual_Nm"
        ]
        finite_residual = finite_residual[np.isfinite(finite_residual)]
        maximum_balance_residual = float(
            np.max(np.abs(finite_residual), initial=0.0)
        )
        rms_balance_residual = float(
            np.sqrt(np.mean(finite_residual ** 2))
            if finite_residual.size
            else 0.0
        )
        balance_tolerance = max(1.0e-3 * peak_torque, 1.0e-10)
        maximum_abs_logged_yaw_rate_rad_s = float(
            np.max(
                np.abs(np.deg2rad(converted["body_yaw_rate_deg_s"])),
                initial=0.0,
            )
        )
        maximum_abs_synchronized_solver_yaw_rate_rad_s = float(
            np.max(
                np.abs(
                    np.deg2rad(
                        converted["solver_generalized_yaw_rate_deg_s"]
                    )
                ),
                initial=0.0,
            )
        )
        diagnostics = dict(self.physics.body_constraint_diagnostics())
        diagnostics.update(
            {
                "trial_id": trial_id,
                "trial_seed": self.config.trial_seed(trial_id),
                "initial_wingbeat_phase_rad": initial_wingbeat_phase,
                "raw_internal_samples_per_published_step": (
                    internal_samples_per_published_step
                ),
                "maximum_abs_raw_reported_torque_n_m": peak_torque,
                "maximum_body_translation_drift_m": maximum_translation_drift_m,
                "maximum_body_rotation_drift_rad": maximum_rotation_drift_rad,
                "maximum_abs_body_yaw_rate_rad_s": maximum_abs_yaw_rate_rad_s,
                "maximum_abs_logged_body_yaw_rate_rad_s": (
                    maximum_abs_logged_yaw_rate_rad_s
                ),
                "maximum_abs_synchronized_solver_generalized_yaw_rate_rad_s": (
                    maximum_abs_synchronized_solver_yaw_rate_rad_s
                ),
                "maximum_head_transform_element_drift": maximum_head_transform_drift,
                "maximum_ground_contact_count": maximum_ground_contact_count,
                "maximum_yaw_equation_residual_n_m": maximum_balance_residual,
                "rms_yaw_balance_residual_n_m": rms_balance_residual,
                "yaw_equation_residual_tolerance_n_m": balance_tolerance,
            }
        )
        solver_diagnostics = getattr(self.physics, "solver_diagnostics", None)
        if callable(solver_diagnostics):
            diagnostics["solver"] = solver_diagnostics()
        raw_time_array = converted["raw_internal_padded_time_s"]
        validation_mask = (
            (raw_time_array > 0.0)
            & (raw_time_array <= self.config.protocol.trial_duration_s)
        )
        fixed_meter = converted[
            "raw_internal_padded_support_on_fly_yaw_reaction_engine_Nm"
        ][validation_mask]
        equality_meter = converted[
            "raw_internal_padded_equality_only_yaw_reaction_engine_Nm"
        ][validation_mask]
        dual_meter = {
            "available": bool(
                fixed_meter.size
                and np.all(np.isfinite(fixed_meter))
                and np.all(np.isfinite(equality_meter))
            )
        }
        if dual_meter["available"]:
            difference = fixed_meter - equality_meter
            theta = 2.0 * math.pi * self.config.protocol.frequency_hz * raw_time_array[
                validation_mask
            ]

            def harmonic(signal: np.ndarray) -> Tuple[float, float]:
                centered = signal - np.mean(signal)
                sine = 2.0 / len(signal) * float(np.sum(centered * np.sin(theta)))
                cosine = 2.0 / len(signal) * float(np.sum(centered * np.cos(theta)))
                return float(np.hypot(sine, cosine)), math.degrees(
                    math.atan2(cosine, sine)
                )

            fixed_harmonic, fixed_phase = harmonic(fixed_meter)
            equality_harmonic, equality_phase = harmonic(equality_meter)
            scale = max(
                float(np.ptp(fixed_meter)), fixed_harmonic, 1.0e-30
            )
            mean_scale = max(
                abs(float(np.mean(fixed_meter))),
                float(np.max(np.abs(fixed_meter), initial=0.0)),
                1.0e-30,
            )
            mean_error_percent = (
                100.0
                * abs(float(np.mean(fixed_meter) - np.mean(equality_meter)))
                / mean_scale
            )
            amplitude_error_percent = (
                100.0
                * abs(fixed_harmonic - equality_harmonic)
                / max(fixed_harmonic, 1.0e-30)
            )
            phase_error_deg = abs(
                (fixed_phase - equality_phase + 180.0) % 360.0 - 180.0
            )
            waveform_nrmse_percent = (
                100.0 * float(np.sqrt(np.mean(difference ** 2))) / scale
            )
            dual_meter.update(
                {
                    "mean_error_percent": mean_error_percent,
                    "harmonic_amplitude_error_percent": amplitude_error_percent,
                    "harmonic_phase_error_deg": phase_error_deg,
                    "waveform_nrmse_percent": waveform_nrmse_percent,
                    "passed": bool(
                        mean_error_percent <= 1.0
                        and amplitude_error_percent <= 1.0
                        and phase_error_deg <= 1.0
                        and waveform_nrmse_percent <= 2.0
                    ),
                }
            )
        else:
            dual_meter["passed"] = False
        diagnostics["dual_meter_validation"] = dual_meter
        raw_wing = converted[
            "raw_internal_padded_wing_reported_yaw_torque_Nm"
        ][validation_mask]
        raw_wing_sum = converted[
            "raw_internal_padded_wing_sum_reported_yaw_torque_Nm"
        ][validation_mask]
        raw_nonwing = converted[
            "raw_internal_padded_nonwing_reported_yaw_torque_residual_Nm"
        ][validation_mask]
        raw_total = converted[
            "raw_internal_padded_reported_yaw_torque_Nm"
        ][validation_mask]
        wing_load_cells_available = bool(
            raw_wing.size
            and raw_wing.shape[-1] == 2
            and np.all(np.isfinite(raw_wing))
            and np.all(np.isfinite(raw_wing_sum))
            and np.all(np.isfinite(raw_nonwing))
        )
        wing_identity_error = float("inf")
        if wing_load_cells_available:
            wing_identity_error = float(
                np.max(
                    np.abs(raw_wing_sum + raw_nonwing - raw_total),
                    initial=0.0,
                )
            )
        wing_identity_tolerance = max(1.0e-12 * peak_torque, 1.0e-18)

        raw_aerodynamic = converted[
            "raw_internal_padded_wing_aerodynamic_reported_yaw_torque_Nm"
        ][validation_mask]
        raw_full_aerodynamic = converted[
            "raw_internal_padded_full_aerodynamic_reported_yaw_torque_Nm"
        ][validation_mask]
        raw_aerodynamic_closure = converted[
            "raw_internal_padded_aerodynamic_closure_residual_Nm"
        ][validation_mask]
        aerodynamic_probe_available = bool(
            raw_aerodynamic.size
            and raw_aerodynamic.shape[-1] == 2
            and np.all(np.isfinite(raw_aerodynamic))
            and np.all(np.isfinite(raw_full_aerodynamic))
            and np.all(np.isfinite(raw_aerodynamic_closure))
        )
        aerodynamic_peak = (
            float(np.max(np.abs(raw_full_aerodynamic), initial=0.0))
            if aerodynamic_probe_available
            else 0.0
        )
        aerodynamic_closure_maximum = (
            float(np.max(np.abs(raw_aerodynamic_closure), initial=0.0))
            if aerodynamic_probe_available
            else float("inf")
        )
        aerodynamic_closure_rms = (
            float(np.sqrt(np.mean(raw_aerodynamic_closure ** 2)))
            if aerodynamic_probe_available
            else float("inf")
        )
        aerodynamic_closure_tolerance = max(
            1.0e-3 * aerodynamic_peak, 1.0e-10
        )
        native_wing_load_cells_required = isinstance(
            self.physics,
            (PaperFixedLoadCellFlyBodyAdapter, PaperComparisonFlyBodyAdapter),
        )
        aerodynamic_probe_required = isinstance(
            self.physics, PaperComparisonFlyBodyAdapter
        )
        diagnostics["per_wing_metrology"] = {
            "wing_load_cells_required": native_wing_load_cells_required,
            "wing_load_cells_available": wing_load_cells_available,
            "wing_sum_and_residual_identity_maximum_error_Nm": wing_identity_error,
            "wing_sum_and_residual_identity_tolerance_Nm": wing_identity_tolerance,
            "aerodynamic_probe_required": aerodynamic_probe_required,
            "aerodynamic_probe_available": aerodynamic_probe_available,
            "aerodynamic_closure_maximum_error_Nm": aerodynamic_closure_maximum,
            "aerodynamic_closure_rms_error_Nm": aerodynamic_closure_rms,
            "aerodynamic_closure_tolerance_Nm": aerodynamic_closure_tolerance,
            "passed": bool(
                (not native_wing_load_cells_required or wing_load_cells_available)
                and (
                    not wing_load_cells_available
                    or wing_identity_error <= wing_identity_tolerance
                )
                and (
                    not aerodynamic_probe_required
                    or (
                        aerodynamic_probe_available
                        and aerodynamic_closure_maximum
                        <= aerodynamic_closure_tolerance
                    )
                )
            ),
        }
        failures = []
        if maximum_translation_drift_m >= 1.0e-6:
            failures.append("body translation drift")
        if maximum_rotation_drift_rad >= 1.0e-4:
            failures.append("body rotation drift")
        # Gate the pose-derived 1 kHz observable.  Keep both the physics-step
        # and synchronized generalized-velocity maxima as numerical solver
        # diagnostics; neither is relabelled as physical body motion.
        if maximum_abs_logged_yaw_rate_rad_s >= 1.0e-3:
            failures.append("body yaw rate")
        if maximum_head_transform_drift >= 1.0e-10:
            failures.append("head transform drift")
        if maximum_ground_contact_count != 0:
            failures.append("ground contact")
        if maximum_balance_residual > balance_tolerance:
            failures.append("yaw equation balance")
        if native_wing_load_cells_required and not wing_load_cells_available:
            failures.append("wing-root load-cell samples")
        if wing_load_cells_available and wing_identity_error > wing_identity_tolerance:
            failures.append("wing decomposition identity")
        if aerodynamic_probe_required and not aerodynamic_probe_available:
            failures.append("aerodynamic wing probes")
        if (
            aerodynamic_probe_required
            and aerodynamic_probe_available
            and aerodynamic_closure_maximum > aerodynamic_closure_tolerance
        ):
            failures.append("aerodynamic probe closure")
        diagnostics["apparatus_validation_passed"] = not failures
        diagnostics["apparatus_validation_failures"] = failures
        if failures:
            raise PaperOpenLoopError(
                "paper apparatus validation failed: {}; diagnostics={!r}".format(
                    ", ".join(failures), diagnostics
                )
            )
        return converted, {
            "diagnostics": diagnostics,
            "circuit_runtime": runtime_receipt,
        }

    def run(self) -> Tuple[Mapping[str, np.ndarray], Mapping[str, Any]]:
        trials = []
        receipts = []
        for trial_id in range(self.config.repetitions):
            trial, receipt = self._run_trial(trial_id)
            trials.append(trial)
            receipts.append(receipt)
        return self.finalize_trials(trials, receipts)

    def run_trial(
        self, trial_id: int
    ) -> Tuple[Mapping[str, np.ndarray], Mapping[str, Any]]:
        """Run one deterministic trial for the resumable scientific worker."""

        if not 0 <= trial_id < self.config.repetitions:
            raise ValueError("trial_id is outside the repetition range")
        return self._run_trial(trial_id)

    def finalize_trials(
        self,
        trials: Any,
        receipts: Any,
    ) -> Tuple[Mapping[str, np.ndarray], Mapping[str, Any]]:
        """Assemble and analyze trial views, including disk-backed views.

        ``trials`` may be an ordinary sequence of trial mappings or expose a
        ``stacked_array(name)`` method.  The latter lets the scientific release
        runner retain raw 40/80 kHz channels as NumPy memmaps without stacking
        all repetitions in resident memory.
        """

        if len(trials) != self.config.repetitions:
            raise PaperOpenLoopError("trial count does not match configuration")
        if len(receipts) != self.config.repetitions:
            raise PaperOpenLoopError("trial receipt count does not match configuration")
        first_time = trials[0]["time_s"]
        if any(not np.array_equal(trial["time_s"], first_time) for trial in trials[1:]):
            raise PaperOpenLoopError("phase-locked trial grids diverged")
        stacked_keys = (
            "support_on_fly_yaw_reaction_engine_Nm",
            "support_load_cell_force_engine_N",
            "support_load_cell_moment_engine_Nm",
            "attempted_fly_yaw_moment_from_support_engine_Nm",
            "equality_only_yaw_reaction_engine_Nm",
            "fly_generated_yaw_moment_engine_Nm",
            "root_fluid_yaw_moment_engine_Nm",
            "reported_yaw_torque_Nm",
            "reported_yaw_torque_dyne_cm",
            "body_position_m",
            "body_euler_deg",
            "body_yaw_rate_deg_s",
            "solver_generalized_yaw_rate_deg_s",
            "wing_joint_position_rad",
            "head_transform_body",
            "yaw_equation_residual_Nm",
            "yaw_balance_residual_Nm",
            "wingbeat_phase_rad",
            "actuator_torque_Nm",
            "actuator_clipped",
            "raw_support_on_fly_yaw_reaction_engine_Nm",
            "raw_reported_yaw_torque_Nm",
            "raw_yaw_equation_residual_Nm",
            "raw_internal_padded_support_load_cell_force_engine_N",
            "raw_internal_padded_support_load_cell_moment_engine_Nm",
            "raw_internal_padded_support_on_fly_yaw_reaction_engine_Nm",
            "raw_internal_padded_attempted_fly_yaw_moment_from_support_engine_Nm",
            "raw_internal_padded_root_fluid_yaw_moment_engine_Nm",
            "raw_internal_padded_equality_only_yaw_reaction_engine_Nm",
            "raw_internal_padded_yaw_balance_residual_Nm",
            "raw_internal_padded_reported_yaw_torque_Nm",
            "raw_internal_padded_reported_yaw_torque_dyne_cm",
            "raw_internal_padded_actuator_torque_Nm",
            "raw_internal_padded_actuator_clipped",
            "raw_internal_padded_wing_load_cell_position_engine_m",
            "raw_internal_padded_wing_load_cell_orientation_site_to_engine",
            "raw_internal_padded_wing_parent_on_child_force_engine_N",
            "raw_internal_padded_wing_parent_on_child_moment_at_hinge_engine_Nm",
            "raw_internal_padded_wing_on_thorax_force_engine_N",
            "raw_internal_padded_wing_on_thorax_moment_at_hinge_engine_Nm",
            "raw_internal_padded_wing_on_thorax_moment_at_tether_engine_Nm",
            "wing_load_cell_position_engine_m",
            "wing_load_cell_orientation_site_to_engine",
            "wing_parent_on_child_force_engine_N",
            "wing_parent_on_child_moment_at_hinge_engine_Nm",
            "wing_on_thorax_force_engine_N",
            "wing_on_thorax_moment_at_hinge_engine_Nm",
            "wing_on_thorax_moment_at_tether_engine_Nm",
            "wing_reported_yaw_torque_Nm",
            "wing_sum_reported_yaw_torque_Nm",
            "nonwing_reported_yaw_torque_residual_Nm",
            "wing_aerodynamic_moment_at_tether_engine_Nm",
            "wing_aerodynamic_reported_yaw_torque_Nm",
            "full_aerodynamic_reported_yaw_torque_Nm",
            "background_aerodynamic_reported_yaw_torque_Nm",
            "aerodynamic_closure_residual_Nm",
            "raw_internal_padded_wing_reported_yaw_torque_Nm",
            "raw_internal_padded_wing_sum_reported_yaw_torque_Nm",
            "raw_internal_padded_nonwing_reported_yaw_torque_residual_Nm",
            "raw_internal_padded_wing_aerodynamic_moment_at_tether_engine_Nm",
            "raw_internal_padded_wing_aerodynamic_reported_yaw_torque_Nm",
            "raw_internal_padded_full_aerodynamic_reported_yaw_torque_Nm",
            "raw_internal_padded_background_aerodynamic_reported_yaw_torque_Nm",
            "raw_internal_padded_aerodynamic_closure_residual_Nm",
        )
        stimulus = sample_protocol(self.config.protocol, first_time)
        arrays: Dict[str, np.ndarray] = {
            "time_s": first_time,
            "raw_physics_time_s": trials[0]["raw_physics_time_s"],
            "raw_internal_padded_time_s": trials[0][
                "raw_internal_padded_time_s"
            ],
            **{
                key: value
                for key, value in stimulus.items()
                if np.asarray(value).dtype.kind != "U"
            },
        }
        stacked_array = getattr(trials, "stacked_array", None)
        if callable(stacked_array):
            arrays.update({key: stacked_array(key) for key in stacked_keys})
        else:
            arrays.update(
                {
                    key: np.stack([trial[key] for trial in trials], axis=0)
                    for key in stacked_keys
                }
            )
        internal_count = int(
            receipts[0]["diagnostics"][
                "raw_internal_samples_per_published_step"
            ]
        )
        raw_rate_hz = self.config.physics_rate_hz * internal_count
        synchronized_time, synchronized_torque, sampling_receipt = (
            synchronize_raw_torque(
                arrays["raw_internal_padded_time_s"],
                arrays["raw_internal_padded_reported_yaw_torque_Nm"],
                trial_duration_s=self.config.protocol.trial_duration_s,
                input_rate_hz=raw_rate_hz,
                output_rate_hz=int(round(1.0 / PAPER_TORQUE_LOG_DT_S)),
            )
        )
        if not np.allclose(
            synchronized_time, first_time, rtol=0.0, atol=1.0e-12
        ):
            raise PaperOpenLoopError("FIR sample centers diverged from trial grid")
        arrays["reported_yaw_torque_Nm"] = synchronized_torque
        arrays["reported_yaw_torque_dyne_cm"] = synchronized_torque * 1.0e7
        products = named_torque_products(synchronized_torque)
        for product_name, values in products.items():
            arrays["torque_product_{}_Nm".format(product_name)] = values

        def synchronize_raw_channel(raw_name: str) -> Optional[np.ndarray]:
            raw = np.asarray(arrays[raw_name], dtype=float)
            if not np.all(np.isfinite(raw)):
                return None
            if raw.ndim == 2:
                _time, values, _receipt = synchronize_raw_torque(
                    arrays["raw_internal_padded_time_s"],
                    raw,
                    trial_duration_s=self.config.protocol.trial_duration_s,
                    input_rate_hz=raw_rate_hz,
                    output_rate_hz=int(round(1.0 / PAPER_TORQUE_LOG_DT_S)),
                )
                return values
            if raw.ndim >= 3:
                trailing_shape = raw.shape[2:]
                flattened = raw.reshape(raw.shape[0], raw.shape[1], -1)
                components = []
                for component in range(flattened.shape[-1]):
                    _time, values, _receipt = synchronize_raw_torque(
                        arrays["raw_internal_padded_time_s"],
                        flattened[:, :, component],
                        trial_duration_s=self.config.protocol.trial_duration_s,
                        input_rate_hz=raw_rate_hz,
                        output_rate_hz=int(round(1.0 / PAPER_TORQUE_LOG_DT_S)),
                    )
                    components.append(values)
                synchronized = np.stack(components, axis=-1)
                return synchronized.reshape(
                    synchronized.shape[:2] + trailing_shape
                )
            raise PaperOpenLoopError(
                "unsupported raw torque channel rank for {}".format(raw_name)
            )

        synchronized_wing = synchronize_raw_channel(
            "raw_internal_padded_wing_reported_yaw_torque_Nm"
        )
        if synchronized_wing is None:
            # Equality-only validation meters and lightweight test adapters do
            # not have physical wing-root sensors.  Preserve a shaped fallback
            # for schema compatibility while authority remains explicitly false.
            synchronized_wing = np.zeros(
                synchronized_torque.shape + (2,), dtype=float
            )
        arrays["wing_reported_yaw_torque_Nm"] = synchronized_wing
        arrays["wing_sum_reported_yaw_torque_Nm"] = np.sum(
            synchronized_wing, axis=-1
        )
        arrays["nonwing_reported_yaw_torque_residual_Nm"] = (
            synchronized_torque - arrays["wing_sum_reported_yaw_torque_Nm"]
        )
        for detailed_channel in (
            "wing_load_cell_position_engine_m",
            "wing_load_cell_orientation_site_to_engine",
            "wing_parent_on_child_force_engine_N",
            "wing_parent_on_child_moment_at_hinge_engine_Nm",
            "wing_on_thorax_force_engine_N",
            "wing_on_thorax_moment_at_hinge_engine_Nm",
            "wing_on_thorax_moment_at_tether_engine_Nm",
            "wing_aerodynamic_moment_at_tether_engine_Nm",
        ):
            synchronized_detail = synchronize_raw_channel(
                "raw_internal_padded_" + detailed_channel
            )
            if synchronized_detail is not None:
                arrays[detailed_channel] = synchronized_detail

        per_wing_product_sources = {
            "left_wing": synchronized_wing[:, :, 0],
            "right_wing": synchronized_wing[:, :, 1],
            "wing_sum": arrays["wing_sum_reported_yaw_torque_Nm"],
            "nonwing_residual": arrays[
                "nonwing_reported_yaw_torque_residual_Nm"
            ],
        }
        synchronized_aerodynamic = synchronize_raw_channel(
            "raw_internal_padded_wing_aerodynamic_reported_yaw_torque_Nm"
        )
        if synchronized_aerodynamic is not None:
            arrays[
                "wing_aerodynamic_reported_yaw_torque_Nm"
            ] = synchronized_aerodynamic
            full_aerodynamic = synchronize_raw_channel(
                "raw_internal_padded_full_aerodynamic_reported_yaw_torque_Nm"
            )
            if full_aerodynamic is None:
                raise PaperOpenLoopError(
                    "full aerodynamic probe channel is unavailable"
                )
            arrays["full_aerodynamic_reported_yaw_torque_Nm"] = full_aerodynamic
            background = synchronize_raw_channel(
                "raw_internal_padded_background_aerodynamic_reported_yaw_torque_Nm"
            )
            if background is not None:
                arrays[
                    "background_aerodynamic_reported_yaw_torque_Nm"
                ] = background
            arrays["aerodynamic_closure_residual_Nm"] = synchronize_raw_channel(
                "raw_internal_padded_aerodynamic_closure_residual_Nm"
            )
            per_wing_product_sources.update(
                {
                    "left_wing_aerodynamic": synchronized_aerodynamic[:, :, 0],
                    "right_wing_aerodynamic": synchronized_aerodynamic[:, :, 1],
                }
            )
        for prefix, source in per_wing_product_sources.items():
            for product_name, values in named_torque_products(source).items():
                arrays[
                    "{}_torque_product_{}_Nm".format(prefix, product_name)
                ] = values

        filterable_channels = (
            "support_on_fly_yaw_reaction_engine_Nm",
            "attempted_fly_yaw_moment_from_support_engine_Nm",
        )
        for name in filterable_channels:
            _time, values, _receipt = synchronize_raw_torque(
                arrays["raw_internal_padded_time_s"],
                arrays["raw_internal_padded_" + name],
                trial_duration_s=self.config.protocol.trial_duration_s,
                input_rate_hz=raw_rate_hz,
                output_rate_hz=int(round(1.0 / PAPER_TORQUE_LOG_DT_S)),
            )
            arrays[name] = values
        arrays["fly_generated_yaw_moment_engine_Nm"] = -arrays[
            "support_on_fly_yaw_reaction_engine_Nm"
        ]
        for name in (
            "support_load_cell_force_engine_N",
            "support_load_cell_moment_engine_Nm",
        ):
            raw = arrays["raw_internal_padded_" + name]
            if np.all(np.isfinite(raw)):
                components = []
                for component in range(3):
                    _time, values, _receipt = synchronize_raw_torque(
                        arrays["raw_internal_padded_time_s"],
                        raw[:, :, component],
                        trial_duration_s=self.config.protocol.trial_duration_s,
                        input_rate_hz=raw_rate_hz,
                        output_rate_hz=int(round(1.0 / PAPER_TORQUE_LOG_DT_S)),
                    )
                    components.append(values)
                arrays[name] = np.stack(components, axis=-1)
        for name in (
            "root_fluid_yaw_moment_engine_Nm",
            "equality_only_yaw_reaction_engine_Nm",
            "yaw_balance_residual_Nm",
        ):
            raw_name = "raw_internal_padded_" + name
            raw = arrays[raw_name]
            if np.all(np.isfinite(raw)):
                _time, values, _receipt = synchronize_raw_torque(
                    arrays["raw_internal_padded_time_s"],
                    raw,
                    trial_duration_s=self.config.protocol.trial_duration_s,
                    input_rate_hz=raw_rate_hz,
                    output_rate_hz=int(round(1.0 / PAPER_TORQUE_LOG_DT_S)),
                )
                arrays[name] = values
        aggregate_dual_meter: Dict[str, Any] = {"available": False, "passed": False}
        equality = arrays["equality_only_yaw_reaction_engine_Nm"]
        fixed = arrays["support_on_fly_yaw_reaction_engine_Nm"]
        if np.all(np.isfinite(fixed)) and np.all(np.isfinite(equality)):
            fixed_mean = np.mean(fixed, axis=0)
            equality_mean = np.mean(equality, axis=0)
            post = synchronized_time >= self.config.protocol.transition_end_s
            theta = 2.0 * math.pi * self.config.protocol.frequency_hz * synchronized_time[post]

            def aggregate_harmonic(signal: np.ndarray) -> Tuple[float, float]:
                selected = signal[post] - np.mean(signal[post])
                sine = 2.0 / len(selected) * float(
                    np.sum(selected * np.sin(theta))
                )
                cosine = 2.0 / len(selected) * float(
                    np.sum(selected * np.cos(theta))
                )
                return float(np.hypot(sine, cosine)), math.degrees(
                    math.atan2(cosine, sine)
                )

            fixed_amplitude, fixed_phase = aggregate_harmonic(fixed_mean)
            equality_amplitude, equality_phase = aggregate_harmonic(equality_mean)
            scale = max(float(np.ptp(fixed_mean)), fixed_amplitude, 1.0e-30)
            mean_scale = max(
                abs(float(np.mean(fixed_mean))),
                float(np.max(np.abs(fixed_mean), initial=0.0)),
                1.0e-30,
            )
            mean_error = (
                100.0
                * abs(float(np.mean(fixed_mean) - np.mean(equality_mean)))
                / mean_scale
            )
            amplitude_error = (
                100.0
                * abs(fixed_amplitude - equality_amplitude)
                / max(fixed_amplitude, 1.0e-30)
            )
            phase_error = abs(
                (fixed_phase - equality_phase + 180.0) % 360.0 - 180.0
            )
            waveform_nrmse = (
                100.0
                * float(np.sqrt(np.mean((fixed_mean - equality_mean) ** 2)))
                / scale
            )
            aggregate_dual_meter = {
                "available": True,
                "mean_error_percent": mean_error,
                "harmonic_amplitude_error_percent": amplitude_error,
                "harmonic_phase_error_deg": phase_error,
                "waveform_nrmse_percent": waveform_nrmse,
                "passed": bool(
                    mean_error <= 1.0
                    and amplitude_error <= 1.0
                    and phase_error <= 1.0
                    and waveform_nrmse <= 2.0
                ),
            }
        offset_sensitivity_method = getattr(
            self.physics, "attachment_offset_sensitivity", None
        )
        offset_sensitivity = (
            offset_sensitivity_method(0.0005)
            if callable(offset_sensitivity_method)
            else None
        )
        per_wing_receipts = [
            receipt.get("diagnostics", {}).get("per_wing_metrology", {})
            for receipt in receipts
        ]
        wing_load_cells_available = bool(per_wing_receipts) and all(
            item.get("wing_load_cells_available") is True
            for item in per_wing_receipts
        )
        aerodynamic_probes_available = bool(per_wing_receipts) and all(
            item.get("aerodynamic_probe_available") is True
            for item in per_wing_receipts
        )
        per_wing_metrology_passed = bool(per_wing_receipts) and all(
            item.get("passed") is True for item in per_wing_receipts
        )
        metadata = {
            "schema_version": PAPER_OPEN_LOOP_SCHEMA_VERSION,
            "config": {
                **asdict(self.config),
                "protocol": self.config.protocol.to_dict(),
                "effector_laterality_hypothesis": self.config.effector_laterality_hypothesis.value,
            },
            "sign_calibration": dict(self.sign_calibration),
            "torque_calibration": self.torque_calibration,
            "sampling": {
                **dict(sampling_receipt),
                "raw_internal_rate_hz": raw_rate_hz,
                "raw_internal_samples_retained": True,
                "pre_roll_duration_s": self.config.pre_roll_duration_s,
                "post_roll_duration_s": self.config.post_roll_duration_s,
                "named_products": list(products),
            },
            "per_wing_metrology": {
                "wing_load_cells_available": wing_load_cells_available,
                "aerodynamic_probes_available": aerodynamic_probes_available,
                "passed": per_wing_metrology_passed,
                "physical_side_order": ["left", "right"],
            },
            "channel_authority": {
                "reported_yaw_torque_Nm": "authoritative_native_torque",
                "wing_reported_yaw_torque_Nm": "calibrated_simulation_decomposition",
                "wing_aerodynamic_reported_yaw_torque_Nm": "validated_fluid_diagnostic",
                "paper_figure3_trace": "digitized_reference_no_reported_sem",
            },
            "dual_meter_validation": aggregate_dual_meter,
            "tether_attachment_offset_sensitivity": offset_sensitivity,
            "physics_provenance": dict(self.physics.provenance_metadata()),
            "nod1_laterality": load_nod1_laterality_receipt(),
            "trial_receipts": receipts,
            "dataset_boundaries": {
                "brain_circuit": "FlyWire FAFB materialization 783",
                "downstream_motor_evidence": "MANC male-cns:v1.0 separate identifier space",
                "raw_application_side_labels_are_anatomical": False,
                "structural_synapse_counts_are_physiological_weights": False,
            },
            "torque_field_semantics": {
                "support_on_fly_yaw_reaction_engine_Nm": "authoritative parent-on-child load-cell reaction in fixed/comparison mode",
                "attempted_fly_yaw_moment_from_support_engine_Nm": "negative support reaction",
                "fly_generated_yaw_moment_engine_Nm": "compatibility alias for negative support reaction; not independent",
                "root_fluid_yaw_moment_engine_Nm": "equality-validator generalized fluid diagnostic",
                "equality_only_yaw_reaction_engine_Nm": "six tether-weld equality rows reconstructed with J^T f",
                "reported_yaw_torque_Nm": "paper-positive right-turn channel from support reaction",
                "wing_reported_yaw_torque_Nm": "physical left/right complete hinge-transmitted wrench transported to the tether origin; simulation-only decomposition",
                "wing_sum_reported_yaw_torque_Nm": "left plus right after synchronized filtering; never substituted for authoritative total",
                "nonwing_reported_yaw_torque_residual_Nm": "authoritative total minus synchronized wing sum",
                "wing_aerodynamic_reported_yaw_torque_Nm": "non-integrating left/right aerodynamic probe diagnostic",
            },
            "scientific_scope": {
                "species": "Drosophila model under Musca-derived apparatus protocol",
                "apparatus_validation_separate_from_behavior": True,
                "stimulus_feedback_from_fly": False,
                "tether_model": self.config.torque_meter_mode,
            },
        }
        return arrays, metadata


__all__ = [
    "PAPER_BRIDGE_DT_S",
    "PAPER_CIRCUIT_DT_S",
    "PAPER_OPEN_LOOP_SCHEMA_VERSION",
    "PAPER_PHYSICS_DT_S",
    "PAPER_TORQUE_LOG_DT_S",
    "PaperOpenLoopConfig",
    "PaperOpenLoopError",
    "PaperOpenLoopSimulator",
    "PaperStreamingNOD1MotorBridge",
]
