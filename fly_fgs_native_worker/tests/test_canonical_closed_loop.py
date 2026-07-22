from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fly_sensor2behavior.fly_fgs import (
    FLY_FGS_NOD1_RAW_APP_SIDES,
    FLY_FGS_NOD1_ROOT_IDS,
)
from fly_sensor2behavior.fly_fgs_runtime import (
    FlyFGSCircuitSample,
    FlyFGSSceneBodyInput,
    NodeFlyFGSCircuitRuntime,
    default_fly_fgs_runtime_script_path,
)
from fly_sensor2behavior.flight.canonical_closed_loop import (
    CANONICAL_SOURCE_LIMITATIONS,
    CanonicalClosedLoopCheckpoint,
    CanonicalClosedLoopConfig,
    CanonicalClosedLoopError,
    CanonicalClosedLoopSimulator,
)
from fly_sensor2behavior.flight.effector_mapping import (
    EFFECTOR_MAPPING_PROVENANCE,
    RAW_APP_LATERALITY_STATUS,
    RAW_L_TO_PHYSICAL_LEFT,
    RAW_L_TO_PHYSICAL_RIGHT,
    SIGNED_BEHAVIOR_CLAIM_POLICY,
)
from fly_sensor2behavior.flight.streaming_bridge import (
    StreamingBridgeConfig,
    StreamingNOD1MotorBridge,
    default_streaming_motor_pathways,
)
from fly_sensor2behavior.flight.streaming_mechanics import (
    StreamingMechanicsConfig,
    StreamingMuscleWingStepper,
    WingPhaseSource,
)
from fly_sensor2behavior.flight.types import (
    AerodynamicWrench,
    RigidBodyState,
    WingKinematics,
)


_TWO_PI = 2.0 * math.pi
_DEFAULT_EFFECTOR_HYPOTHESIS = RAW_L_TO_PHYSICAL_LEFT


def _digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _yaw_quaternion(yaw_rad: float) -> np.ndarray:
    return np.array(
        [math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad)],
        dtype=float,
    )


def _control_to_dict(control: FlyFGSSceneBodyInput) -> dict:
    return dict(control.to_dict())


class FakeCircuitRuntime:
    """Stateful canonical-cadence circuit double with a hashed checkpoint."""

    def __init__(self, physics_step_probe=None) -> None:
        self.sample_index = None
        self.last_control = None
        self.started = False
        self.closed = False
        self.advance_controls = []
        self.advance_physics_steps = []
        self.physics_step_probe = physics_step_probe

    def start(self):
        self.started = True
        return self

    @staticmethod
    def _initial_control() -> FlyFGSSceneBodyInput:
        return FlyFGSSceneBodyInput(
            heading_rad=0.0,
            heading_velocity_rad_s=0.0,
            figure_world_azimuth_rad=0.0,
            figure_velocity_rad_s=0.0,
            ground_velocity_rad_s=0.0,
        )

    @staticmethod
    def _relative_drive(control: FlyFGSSceneBodyInput) -> float:
        angle = control.figure_world_azimuth_rad - control.heading_rad
        velocity = control.figure_velocity_rad_s - control.heading_velocity_rad_s
        return 0.0045 + 0.0004 * math.cos(angle) + 0.00005 * math.tanh(velocity)

    def _sample(self, include_retinal_input=False, include_full_cell_state=False):
        assert self.sample_index is not None
        assert self.last_control is not None
        drive = self._relative_drive(self.last_control)
        nod1 = {
            root_id: -0.060
            + drive
            * (1.0 if FLY_FGS_NOD1_RAW_APP_SIDES[root_id] == "L" else 0.9)
            for root_id in FLY_FGS_NOD1_ROOT_IDS
        }
        return FlyFGSCircuitSample(
            sample_index=self.sample_index,
            measurement_time_s=self.sample_index * 0.005,
            availability_time_s=self.sample_index * 0.005,
            nod1_voltage_v=nod1,
            pooled_readout={"relative_drive": drive},
            last_control=self.last_control,
            retinal_input_luminance=(drive, 1.0 - drive)
            if include_retinal_input
            else None,
            full_cell_voltage_v=tuple(nod1.values())
            if include_full_cell_state
            else None,
            full_cell_activity=(drive,) * 4 if include_full_cell_state else None,
            full_cell_state_eligible_motor_input=False,
        )

    def initialize(self, *, include_retinal_input=False, include_full_cell_state=False):
        assert self.started
        self.sample_index = 0
        self.last_control = self._initial_control()
        self.advance_controls = []
        self.advance_physics_steps = []
        return self._sample(include_retinal_input, include_full_cell_state)

    def advance(
        self,
        control,
        *,
        include_retinal_input=False,
        include_full_cell_state=False,
    ):
        assert isinstance(control, FlyFGSSceneBodyInput)
        assert self.sample_index is not None
        self.sample_index += 1
        self.last_control = control
        self.advance_controls.append(control)
        self.advance_physics_steps.append(
            None if self.physics_step_probe is None else self.physics_step_probe()
        )
        return self._sample(include_retinal_input, include_full_cell_state)

    def checkpoint(self):
        unsigned = {
            "sample_index": self.sample_index,
            "last_control": _control_to_dict(self.last_control),
            "advance_controls": [
                _control_to_dict(control) for control in self.advance_controls
            ],
        }
        return {**unsigned, "payload_sha256": _digest(unsigned)}

    def restore(
        self,
        checkpoint,
        *,
        include_retinal_input=False,
        include_full_cell_state=False,
    ):
        payload = json.loads(json.dumps(checkpoint))
        supplied = payload.pop("payload_sha256")
        if supplied != _digest(payload):
            raise ValueError("fake circuit checkpoint digest mismatch")
        self.sample_index = payload["sample_index"]
        self.last_control = FlyFGSSceneBodyInput(**payload["last_control"])
        self.advance_controls = [
            FlyFGSSceneBodyInput(**item) for item in payload["advance_controls"]
        ]
        self.advance_physics_steps = []
        return self._sample(include_retinal_input, include_full_cell_state)

    def close(self):
        self.closed = True


class FakeFlightPhysics:
    backend_name = "deterministic_fake_flight"
    aerodynamic_owner = "fake_external_physics"
    wing_joint_order = (
        "left_stroke",
        "left_deviation",
        "left_pitch",
        "right_stroke",
        "right_deviation",
        "right_pitch",
    )

    def __init__(self, initial_state: RigidBodyState) -> None:
        self._default = initial_state.copy()
        self.state = initial_state.copy()
        self.step_count = 0
        self.yaw_unwrapped_rad = self._wrapped_yaw(initial_state)
        self.last_wrench = self._zero_wrench()
        self.wing_phase_history = []
        self._measured_wing_position_rad = np.zeros(6)
        self._measured_wing_velocity_rad_s = np.zeros(6)
        self.last_actuator_torque_n_m = np.zeros(6)

    @staticmethod
    def _wrapped_yaw(state: RigidBodyState) -> float:
        q = state.quaternion_body_to_world
        return math.atan2(
            2.0 * (q[0] * q[3] + q[1] * q[2]),
            1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
        )

    @staticmethod
    def _zero_wrench() -> AerodynamicWrench:
        return AerodynamicWrench(
            force_body_n=np.zeros(3),
            torque_body_n_m=np.zeros(3),
            left_force_body_n=np.zeros(3),
            right_force_body_n=np.zeros(3),
            mechanical_power_w=0.0,
        )

    def default_initial_state(self):
        return self._default.copy()

    def reset(self, initial_state):
        self.state = initial_state.copy()
        self.step_count = 0
        self.yaw_unwrapped_rad = self._wrapped_yaw(initial_state)
        self.last_wrench = self._zero_wrench()
        self.wing_phase_history = []
        self._measured_wing_position_rad = np.zeros(6)
        self._measured_wing_velocity_rad_s = np.zeros(6)
        self.last_actuator_torque_n_m = np.zeros(6)

    def step(self, wings, force_body_n, torque_body_n_m, dt_s):
        assert isinstance(wings, WingKinematics)
        np.testing.assert_array_equal(force_body_n, np.zeros(3))
        np.testing.assert_array_equal(torque_body_n_m, np.zeros(3))
        assert dt_s == 0.0001
        yaw_rate = float(self.state.angular_velocity_body_rad_s[2])
        self.yaw_unwrapped_rad += yaw_rate * dt_s
        self.state.quaternion_body_to_world = _yaw_quaternion(
            self.yaw_unwrapped_rad
        )
        # A tiny deterministic translation makes all rigid-body channels live.
        self.state.position_world_m += self.state.velocity_world_m_s * dt_s
        self.step_count += 1
        self.wing_phase_history.append(float(wings.phase_rad))
        self._measured_wing_position_rad = np.array(
            [
                wings.stroke_rad[0],
                wings.deviation_rad[0],
                wings.angle_of_attack_rad[0],
                wings.stroke_rad[1],
                wings.deviation_rad[1],
                wings.angle_of_attack_rad[1],
            ]
        )
        self._measured_wing_velocity_rad_s = np.array(
            [
                wings.stroke_velocity_rad_s[0],
                0.0,
                0.0,
                wings.stroke_velocity_rad_s[1],
                0.0,
                0.0,
            ]
        )
        self.last_actuator_torque_n_m = np.asarray(
            wings.wing_axis_torque_n_m, dtype=float
        ).copy()
        force = np.array([0.0, 0.0, 1.0e-6 + wings.stroke_rad.mean() * 1.0e-9])
        torque = np.array([0.0, 0.0, np.ptp(wings.stroke_rad) * 1.0e-12])
        self.last_wrench = AerodynamicWrench(
            force_body_n=force,
            torque_body_n_m=torque,
            left_force_body_n=0.5 * force,
            right_force_body_n=0.5 * force,
            mechanical_power_w=abs(float(wings.generalized_torque_n_m.sum())),
        )
        return self.state.copy()

    def wing_joint_state(self):
        return (
            self._measured_wing_position_rad.copy(),
            self._measured_wing_velocity_rad_s.copy(),
        )

    def whole_fly_com_position_m(self):
        return self.state.position_world_m + np.array([0.0, 0.0, 0.0001])

    def ground_contact_count(self):
        return 0

    def aerodynamic_wrench(self):
        value = self.last_wrench
        return AerodynamicWrench(
            value.force_body_n.copy(),
            value.torque_body_n_m.copy(),
            value.left_force_body_n.copy(),
            value.right_force_body_n.copy(),
            value.mechanical_power_w,
        )

    def checkpoint(self):
        unsigned = {
            "state": {
                "position_world_m": self.state.position_world_m.tolist(),
                "velocity_world_m_s": self.state.velocity_world_m_s.tolist(),
                "quaternion_body_to_world": self.state.quaternion_body_to_world.tolist(),
                "angular_velocity_body_rad_s": self.state.angular_velocity_body_rad_s.tolist(),
            },
            "step_count": self.step_count,
            "yaw_unwrapped_rad": self.yaw_unwrapped_rad,
            "wing_phase_history": self.wing_phase_history,
            "measured_wing_position_rad": self._measured_wing_position_rad.tolist(),
            "measured_wing_velocity_rad_s": self._measured_wing_velocity_rad_s.tolist(),
            "last_actuator_torque_n_m": self.last_actuator_torque_n_m.tolist(),
            "wrench": {
                "force": self.last_wrench.force_body_n.tolist(),
                "torque": self.last_wrench.torque_body_n_m.tolist(),
                "left": self.last_wrench.left_force_body_n.tolist(),
                "right": self.last_wrench.right_force_body_n.tolist(),
                "power": self.last_wrench.mechanical_power_w,
            },
        }
        return {**unsigned, "payload_sha256": _digest(unsigned)}

    def restore_checkpoint(self, checkpoint):
        payload = json.loads(json.dumps(checkpoint))
        supplied = payload.pop("payload_sha256")
        if supplied != _digest(payload):
            raise ValueError("fake physics checkpoint digest mismatch")
        state = payload["state"]
        self.state = RigidBodyState(
            np.asarray(state["position_world_m"]),
            np.asarray(state["velocity_world_m_s"]),
            np.asarray(state["quaternion_body_to_world"]),
            np.asarray(state["angular_velocity_body_rad_s"]),
        )
        self.step_count = payload["step_count"]
        self.yaw_unwrapped_rad = payload["yaw_unwrapped_rad"]
        self.wing_phase_history = list(payload["wing_phase_history"])
        self._measured_wing_position_rad = np.asarray(
            payload["measured_wing_position_rad"]
        )
        self._measured_wing_velocity_rad_s = np.asarray(
            payload["measured_wing_velocity_rad_s"]
        )
        self.last_actuator_torque_n_m = np.asarray(
            payload["last_actuator_torque_n_m"]
        )
        wrench = payload["wrench"]
        self.last_wrench = AerodynamicWrench(
            np.asarray(wrench["force"]),
            np.asarray(wrench["torque"]),
            np.asarray(wrench["left"]),
            np.asarray(wrench["right"]),
            wrench["power"],
        )
        return self.state.copy()


def _initial_body(yaw=0.0, yaw_rate=2.0) -> RigidBodyState:
    return RigidBodyState(
        position_world_m=np.array([0.0, 0.0, 0.01]),
        velocity_world_m_s=np.array([0.02, -0.01, 0.0]),
        quaternion_body_to_world=_yaw_quaternion(yaw),
        angular_velocity_body_rad_s=np.array([0.0, 0.0, yaw_rate]),
    )


def _high_gain_bridge(seed=31):
    pathways = tuple(
        replace(pathway, functional_rate_gain=1.0)
        for pathway in default_streaming_motor_pathways()
    )
    config = StreamingBridgeConfig(
        encoder_delay_s=0.0,
        vnc_delay_s=0.0,
        nmj_delay_s=0.0005,
        dn_functional_gain_hz=200.0,
        maximum_dn_rate_hz=200.0,
        maximum_motor_rate_hz=200.0,
        pathways=pathways,
    )
    return StreamingNOD1MotorBridge(config, seed=seed)


def _runner(
    config: CanonicalClosedLoopConfig,
    *,
    yaw=0.0,
    yaw_rate=2.0,
    bridge=None,
    circuit=None,
):
    body = _initial_body(yaw, yaw_rate)
    physics = FakeFlightPhysics(body)
    if circuit is None:
        circuit = FakeCircuitRuntime(lambda: physics.step_count)
    simulator = CanonicalClosedLoopSimulator(
        config,
        circuit_runtime=circuit,
        bridge=StreamingNOD1MotorBridge(seed=31) if bridge is None else bridge,
        mechanics=StreamingMuscleWingStepper(),
        physics_adapter=physics,
        initial_body_state=body,
    )
    return simulator, circuit, physics


def test_exact_10_to_1_and_5_to_1_clocks_and_body_feedback_causality():
    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=0.006,
        figure_velocity_rad_s=0.3,
        ground_velocity_rad_s=0.1,
    )
    simulator, circuit, physics = _runner(config, yaw_rate=2.0)
    result = simulator.run()

    assert len(result.intervals) == 12
    assert physics.step_count == 60
    assert sum(len(interval.physics) for interval in result.intervals) == 60
    observations = [
        interval.circuit_observation
        for interval in result.intervals
        if interval.circuit_observation is not None
    ]
    assert [item.bridge_tick_index for item in observations] == [0, 10]
    assert [item.sample.sample_index for item in observations] == [0, 1]
    assert circuit.advance_physics_steps == [50]
    assert circuit.advance_controls[0].heading_rad == pytest.approx(0.010)
    assert circuit.advance_controls[0].heading_velocity_rad_s == pytest.approx(2.0)
    assert circuit.advance_controls[0].figure_world_azimuth_rad == pytest.approx(
        0.0015
    )
    assert circuit.advance_controls[0].figure_velocity_rad_s == pytest.approx(0.3)
    assert circuit.advance_controls[0].ground_velocity_rad_s == pytest.approx(0.1)
    assert observations[1].body_state.quaternion_body_to_world.tolist() == pytest.approx(
        result.intervals[9].physics[-1].body_state_after.quaternion_body_to_world
    )
    assert all(len(interval.mechanics) == 5 for interval in result.intervals)
    assert all(len(interval.physics) == 5 for interval in result.intervals)
    assert all(
        interval.phase_projection_semantics
        == "exact non-mutating model-owned carrier projection; not observed or measured"
        for interval in result.intervals
    )
    assert result.articulated_physics_telemetry_available is True
    telemetry = result.intervals[0].physics[0].articulated_telemetry_after
    assert telemetry is not None
    assert telemetry.wing_joint_order == FakeFlightPhysics.wing_joint_order
    assert telemetry.measured_wing_position_rad.shape == (6,)
    assert telemetry.measured_wing_velocity_rad_s.shape == (6,)
    assert telemetry.actuator_torque_n_m.shape == (6,)
    assert telemetry.whole_fly_com_position_world_m.shape == (3,)
    assert telemetry.ground_contact_count == 0


def test_world_z_yaw_rate_is_rotated_from_body_angular_velocity():
    half = math.sqrt(0.5)
    body = RigidBodyState(
        position_world_m=np.array([0.0, 0.0, 0.01]),
        velocity_world_m_s=np.zeros(3),
        quaternion_body_to_world=np.array([half, half, 0.0, 0.0]),
        angular_velocity_body_rad_s=np.array([0.0, 1.25, 0.0]),
    )
    physics = FakeFlightPhysics(body)
    simulator = CanonicalClosedLoopSimulator(
        CanonicalClosedLoopConfig(
            effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
            duration_s=0.0005,
        ),
        circuit_runtime=FakeCircuitRuntime(),
        bridge=StreamingNOD1MotorBridge(),
        mechanics=StreamingMuscleWingStepper(),
        physics_adapter=physics,
        initial_body_state=body,
    )
    interval = simulator.advance_intervals(1)[0]
    observation = interval.circuit_observation
    assert observation.wrapped_world_yaw_rad == pytest.approx(0.0)
    assert observation.world_z_yaw_rate_rad_s == pytest.approx(1.25)


def test_world_yaw_is_unwrapped_across_quaternion_branch_cut():
    initial_yaw = math.pi - 0.01
    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=0.006,
        figure_initial_world_azimuth_rad=initial_yaw,
    )
    simulator, circuit, _ = _runner(
        config,
        yaw=initial_yaw,
        yaw_rate=100.0,
    )
    simulator.run()
    assert circuit.advance_controls[0].heading_rad == pytest.approx(
        initial_yaw + 0.5
    )
    assert circuit.advance_controls[0].heading_rad > math.pi


def test_common_world_angle_gauge_shift_preserves_visual_and_neural_drive():
    shift = 1.1
    base_config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=0.006,
        figure_initial_world_azimuth_rad=0.0,
    )
    shifted_config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=0.006,
        figure_initial_world_azimuth_rad=shift,
    )
    first, _, _ = _runner(base_config, yaw=0.0, yaw_rate=0.7)
    second, _, _ = _runner(shifted_config, yaw=shift, yaw_rate=0.7)
    first_result = first.run()
    second_result = second.run()

    first_observations = [
        item.circuit_observation
        for item in first_result.intervals
        if item.circuit_observation is not None
    ]
    second_observations = [
        item.circuit_observation
        for item in second_result.intervals
        if item.circuit_observation is not None
    ]
    for left, right in zip(first_observations, second_observations):
        left_relative = (
            left.requested_control.figure_world_azimuth_rad
            - left.requested_control.heading_rad
        )
        right_relative = (
            right.requested_control.figure_world_azimuth_rad
            - right.requested_control.heading_rad
        )
        assert right_relative == pytest.approx(left_relative, abs=2e-15)
        assert right.sample.nod1_voltage_v == pytest.approx(
            left.sample.nod1_voltage_v, abs=1e-18
        )
    for left, right in zip(first_result.intervals, second_result.intervals):
        assert [event.event_id for event in right.bridge.generated_events] == [
            event.event_id for event in left.bridge.generated_events
        ]


def test_incompatible_initial_retinal_angle_is_rejected_not_silently_relabelled():
    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=0.001,
        figure_initial_world_azimuth_rad=0.3,
    )
    simulator, _, _ = _runner(config, yaw=0.0)
    with pytest.raises(CanonicalClosedLoopError, match="static pre-roll"):
        simulator.advance_intervals(1)


def test_events_are_generated_for_future_intervals_and_applied_at_nmj_availability():
    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=0.020,
    )
    simulator, _, _ = _runner(config, bridge=_high_gain_bridge())
    result = simulator.run()

    generated = {
        event.event_id: (interval.bridge_tick_index, event)
        for interval in result.intervals
        for event in interval.bridge.generated_events
    }
    assert generated
    applied = {}
    for interval in result.intervals:
        start_delivered = {event.event_id for event in interval.bridge_start.delivered_events}
        assert start_delivered == {
            event.event_id for event in interval.bridge.delivered_events
        }
        for frame in interval.mechanics:
            for event_id in frame.applied_event_ids:
                applied[event_id] = frame

    assert applied
    for event_id, frame in applied.items():
        generated_tick, event = generated[event_id]
        assert event.availability_time_s >= (
            generated_tick + 1
        ) * config.bridge_dt_s - 1e-12
        assert frame.interval_start_s - 1e-12 <= event.availability_time_s
        assert event.availability_time_s < frame.interval_end_s - 1e-12
        assert frame.tick_index >= (generated_tick + 1) * 5


def _assert_tail_exact(expected, actual):
    assert len(expected) == len(actual)
    for left, right in zip(expected, actual):
        assert right.bridge == left.bridge
        assert right.bridge_start == left.bridge_start
        assert right.projected_model_phase_end_unwrapped_rad == left.projected_model_phase_end_unwrapped_rad
        if left.circuit_observation is None:
            assert right.circuit_observation is None
        else:
            assert right.circuit_observation.sample == left.circuit_observation.sample
            assert right.circuit_observation.requested_control == left.circuit_observation.requested_control
        for left_step, right_step in zip(left.physics, right.physics):
            np.testing.assert_array_equal(
                right_step.body_state_after.position_world_m,
                left_step.body_state_after.position_world_m,
            )
            np.testing.assert_array_equal(
                right_step.body_state_after.quaternion_body_to_world,
                left_step.body_state_after.quaternion_body_to_world,
            )
            np.testing.assert_array_equal(
                right_step.mechanics.wing_kinematics.stroke_rad,
                left_step.mechanics.wing_kinematics.stroke_rad,
            )
            assert right_step.mechanics.applied_event_ids == left_step.mechanics.applied_event_ids


def test_composite_checkpoint_has_exact_fresh_instance_tail_and_final_digest():
    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=0.015,
        figure_velocity_rad_s=0.4,
    )
    source, _, _ = _runner(config, bridge=_high_gain_bridge(seed=12))
    source.advance_intervals(13)
    checkpoint = source.checkpoint()
    expected_tail = source.advance_intervals(17)
    expected_final = source.checkpoint().to_dict()

    body = _initial_body()
    resumed = CanonicalClosedLoopSimulator.from_checkpoint(
        checkpoint,
        config,
        circuit_factory=FakeCircuitRuntime,
        circuit_checkpoint_loader=lambda value: value,
        bridge_factory=lambda: _high_gain_bridge(seed=12),
        mechanics_factory=StreamingMuscleWingStepper,
        physics_factory=lambda: FakeFlightPhysics(body),
        physics_checkpoint_loader=lambda value: value,
    )
    actual_tail = resumed.advance_intervals(17)

    _assert_tail_exact(expected_tail, actual_tail)
    assert resumed.checkpoint().to_dict() == expected_final
    assert resumed.result().initial_bridge_tick_index == 13
    assert resumed.result().final_bridge_tick_index == 30


def test_composite_tampering_and_cross_config_restore_fail_before_factories():
    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=0.010,
    )
    simulator, _, _ = _runner(config)
    simulator.advance_intervals(4)
    checkpoint = simulator.checkpoint()

    tampered = checkpoint.to_dict()
    tampered["state"]["figure_world_azimuth_rad"] += 1.0
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        CanonicalClosedLoopCheckpoint(tampered)

    cross = checkpoint.to_dict()
    cross["config"]["duration_s"] = 0.0095
    unsigned = {key: value for key, value in cross.items() if key != "payload_sha256"}
    cross["payload_sha256"] = _digest(unsigned)
    cross_checkpoint = CanonicalClosedLoopCheckpoint(cross)
    factory_calls = []
    with pytest.raises(ValueError, match="configuration mismatch"):
        CanonicalClosedLoopSimulator.from_checkpoint(
            cross_checkpoint,
            config,
            circuit_factory=lambda: factory_calls.append("circuit"),
            physics_factory=lambda: factory_calls.append("physics"),
        )
    assert factory_calls == []

    cross_hypothesis = replace(
        config,
        effector_laterality_hypothesis=RAW_L_TO_PHYSICAL_RIGHT,
    )
    factory_calls = []
    with pytest.raises(ValueError, match="configuration mismatch"):
        CanonicalClosedLoopSimulator.from_checkpoint(
            checkpoint,
            cross_hypothesis,
            circuit_factory=lambda: factory_calls.append("circuit"),
            physics_factory=lambda: factory_calls.append("physics"),
        )
    assert factory_calls == []

    for field_name in ("figure_velocity_rad_s", "ground_velocity_rad_s"):
        velocity_tamper = checkpoint.to_dict()
        velocity_tamper["state"][field_name] += 0.25
        unsigned = {
            key: value
            for key, value in velocity_tamper.items()
            if key != "payload_sha256"
        }
        velocity_tamper["payload_sha256"] = _digest(unsigned)
        rehashed = CanonicalClosedLoopCheckpoint(velocity_tamper)
        factory_calls = []
        with pytest.raises(ValueError, match="scene velocity checkpoint mismatch"):
            CanonicalClosedLoopSimulator.from_checkpoint(
                rehashed,
                config,
                circuit_factory=lambda: factory_calls.append("circuit"),
                physics_factory=lambda: factory_calls.append("physics"),
            )
        assert factory_calls == []


def test_mid_interval_failure_is_fail_stop_and_cannot_be_checkpointed_or_resumed():
    class FailingPhysics(FakeFlightPhysics):
        def step(self, wings, force_body_n, torque_body_n_m, dt_s):
            raise RuntimeError("injected transition failure")

    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=0.001,
    )
    body = _initial_body()
    simulator = CanonicalClosedLoopSimulator(
        config,
        circuit_runtime=FakeCircuitRuntime(),
        bridge=StreamingNOD1MotorBridge(),
        mechanics=StreamingMuscleWingStepper(),
        physics_adapter=FailingPhysics(body),
        initial_body_state=body,
    )
    with pytest.raises(RuntimeError, match="injected transition failure"):
        simulator.advance_intervals(1)
    assert simulator.failed is True
    assert "injected transition failure" in simulator.failure_reason
    with pytest.raises(CanonicalClosedLoopError, match="fail-stop"):
        simulator.advance_intervals(1)
    with pytest.raises(CanonicalClosedLoopError, match="fail-stop"):
        simulator.checkpoint()


def test_restore_failure_closes_every_constructed_external_resource():
    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=0.002,
    )
    simulator, _, _ = _runner(config)
    simulator.advance_intervals(2)
    payload = simulator.checkpoint().to_dict()
    payload["components"]["physics"]["step_count"] += 1
    unsigned = {
        key: value for key, value in payload.items() if key != "payload_sha256"
    }
    payload["payload_sha256"] = _digest(unsigned)
    internally_corrupt = CanonicalClosedLoopCheckpoint(payload)

    circuits = []
    physics_adapters = []

    class CloseablePhysics(FakeFlightPhysics):
        def __init__(self, body):
            super().__init__(body)
            self.closed = False

        def close(self):
            self.closed = True

    def circuit_factory():
        value = FakeCircuitRuntime()
        circuits.append(value)
        return value

    def physics_factory():
        value = CloseablePhysics(_initial_body())
        physics_adapters.append(value)
        return value

    with pytest.raises(ValueError, match="fake physics checkpoint digest mismatch"):
        CanonicalClosedLoopSimulator.from_checkpoint(
            internally_corrupt,
            config,
            circuit_factory=circuit_factory,
            circuit_checkpoint_loader=lambda value: value,
            bridge_factory=lambda: StreamingNOD1MotorBridge(seed=31),
            physics_factory=physics_factory,
            physics_checkpoint_loader=lambda value: value,
        )
    assert len(circuits) == len(physics_adapters) == 1
    assert circuits[0].closed is True
    assert physics_adapters[0].closed is True


def test_articulated_physics_telemetry_capabilities_are_strictly_all_or_none():
    class PartialTelemetryPhysics(FakeFlightPhysics):
        wing_joint_state = None

    body = _initial_body()
    with pytest.raises(TypeError, match="all-or-none"):
        CanonicalClosedLoopSimulator(
            CanonicalClosedLoopConfig(
                effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
                duration_s=0.0005,
            ),
            circuit_runtime=FakeCircuitRuntime(),
            bridge=StreamingNOD1MotorBridge(),
            mechanics=StreamingMuscleWingStepper(),
            physics_adapter=PartialTelemetryPhysics(body),
            initial_body_state=body,
        )

    class CoreOnlyPhysics:
        backend_name = "core_only_fake"
        aerodynamic_owner = "core_only_fake"

        def __init__(self, state):
            self.inner = FakeFlightPhysics(state)

        def default_initial_state(self):
            return self.inner.default_initial_state()

        def reset(self, state):
            return self.inner.reset(state)

        def step(self, wings, force_body_n, torque_body_n_m, dt_s):
            return self.inner.step(wings, force_body_n, torque_body_n_m, dt_s)

        def aerodynamic_wrench(self):
            return self.inner.aerodynamic_wrench()

        def checkpoint(self):
            return self.inner.checkpoint()

        def restore_checkpoint(self, checkpoint):
            return self.inner.restore_checkpoint(checkpoint)

    core_only = CanonicalClosedLoopSimulator(
        CanonicalClosedLoopConfig(
            effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
            duration_s=0.0005,
        ),
        circuit_runtime=FakeCircuitRuntime(),
        bridge=StreamingNOD1MotorBridge(),
        mechanics=StreamingMuscleWingStepper(),
        physics_adapter=CoreOnlyPhysics(body),
        initial_body_state=body,
    ).run()
    assert core_only.articulated_physics_telemetry_available is False
    assert core_only.intervals[0].physics[0].articulated_telemetry_after is None


def test_measured_phase_is_rejected_and_config_enforces_bounded_integer_clocks():
    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=0.001,
    )
    body = _initial_body()
    with pytest.raises(ValueError, match="model-owned"):
        CanonicalClosedLoopSimulator(
            config,
            circuit_runtime=FakeCircuitRuntime(),
            bridge=StreamingNOD1MotorBridge(),
            mechanics=StreamingMuscleWingStepper(
                StreamingMechanicsConfig(
                    phase_source=WingPhaseSource.MEASURED_UNWRAPPED
                )
            ),
            physics_adapter=FakeFlightPhysics(body),
            initial_body_state=body,
        )
    with pytest.raises(ValueError, match=r"\(0, 0.5\]"):
        CanonicalClosedLoopConfig(
            effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
            duration_s=0.5005,
        )
    with pytest.raises(ValueError, match="integer number"):
        CanonicalClosedLoopConfig(
            effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
            duration_s=0.0011,
        )
    with pytest.raises(ValueError, match="circuit_dt_s"):
        CanonicalClosedLoopConfig(
            effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
            duration_s=0.001,
            circuit_dt_s=0.004,
        )
    maximum = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=0.5,
    )
    assert maximum.bridge_interval_count == 1000
    assert maximum.circuit_sample_count == 100


def test_effector_hypothesis_is_required_and_mirrors_only_the_physics_boundary():
    with pytest.raises(TypeError, match="effector_laterality_hypothesis"):
        CanonicalClosedLoopConfig(duration_s=0.001)

    left_config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=RAW_L_TO_PHYSICAL_LEFT,
        duration_s=0.020,
    )
    right_config = replace(
        left_config,
        effector_laterality_hypothesis=RAW_L_TO_PHYSICAL_RIGHT,
    )
    left_simulator, _, _ = _runner(
        left_config,
        bridge=_high_gain_bridge(seed=919),
    )
    right_simulator, _, _ = _runner(
        right_config,
        bridge=_high_gain_bridge(seed=919),
    )
    left_result = left_simulator.run()
    right_result = right_simulator.run()

    two_lane_fields = (
        "stroke_rad",
        "stroke_velocity_rad_s",
        "stroke_acceleration_rad_s2",
        "angle_of_attack_rad",
        "deviation_rad",
        "generalized_torque_n_m",
    )
    saw_asymmetric_raw_command = False
    for left_interval, right_interval in zip(
        left_result.intervals, right_result.intervals
    ):
        for left_step, right_step in zip(
            left_interval.physics, right_interval.physics
        ):
            left_raw = left_step.mechanics.actuation_wing_kinematics
            right_raw = right_step.mechanics.actuation_wing_kinematics
            left_physical = left_step.physical_actuation_wing_kinematics
            right_physical = right_step.physical_actuation_wing_kinematics
            for name in two_lane_fields:
                np.testing.assert_array_equal(
                    getattr(right_raw, name), getattr(left_raw, name)
                )
                np.testing.assert_array_equal(
                    getattr(right_physical, name), getattr(left_physical, name)[::-1]
                )
                np.testing.assert_array_equal(
                    np.sort(np.abs(getattr(right_physical, name))),
                    np.sort(np.abs(getattr(left_physical, name))),
                )
            np.testing.assert_array_equal(
                right_physical.wing_axis_torque_n_m[:3],
                left_physical.wing_axis_torque_n_m[3:],
            )
            np.testing.assert_array_equal(
                right_physical.wing_axis_torque_n_m[3:],
                left_physical.wing_axis_torque_n_m[:3],
            )
            if np.ptp(left_raw.stroke_rad) > 0.0:
                saw_asymmetric_raw_command = True
            np.testing.assert_array_equal(
                right_step.aerodynamic_wrench_after.force_body_n,
                left_step.aerodynamic_wrench_after.force_body_n,
            )
            np.testing.assert_array_equal(
                right_step.aerodynamic_wrench_after.torque_body_n_m,
                left_step.aerodynamic_wrench_after.torque_body_n_m,
            )
    assert saw_asymmetric_raw_command
    np.testing.assert_array_equal(
        right_result.final_body_state.position_world_m,
        left_result.final_body_state.position_world_m,
    )
    np.testing.assert_array_equal(
        right_result.final_body_state.quaternion_body_to_world,
        left_result.final_body_state.quaternion_body_to_world,
    )


def test_visualization_records_are_finite_and_disclose_scientific_limits():
    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=0.003,
        include_retinal_input=True,
        include_full_cell_state=True,
    )
    simulator, _, _ = _runner(config)
    result = simulator.run()

    assert result.validation_status == "exploratory"
    assert result.source_limitations == CANONICAL_SOURCE_LIMITATIONS
    assert any("no explicit photoreceptor" in item for item in result.source_limitations)
    assert any("translation" in item for item in result.source_limitations)
    assert any("unknown anatomical laterality" in item for item in result.source_limitations)
    assert any("signed yaw and roll claims are prohibited" in item for item in result.source_limitations)
    assert config.effector_anatomical_status == RAW_APP_LATERALITY_STATUS
    assert config.effector_mapping_provenance == EFFECTOR_MAPPING_PROVENANCE
    assert config.effector_signed_behavior_claim_policy == SIGNED_BEHAVIOR_CLAIM_POLICY
    assert result.effector_mapping_receipt == config.effector_mapping_receipt
    assert result.effector_mapping_receipt.anatomical_status == RAW_APP_LATERALITY_STATUS
    assert "prohibited" in result.effector_mapping_receipt.signed_behavior_claim_policy
    checkpoint_receipt = simulator.checkpoint().to_dict()["receipts"][
        "effector_mapping"
    ]
    assert checkpoint_receipt == dict(config.effector_mapping_receipt.to_dict())
    values = []
    for interval in result.intervals:
        values.extend(
            [
                interval.interval_start_s,
                interval.interval_end_s,
                interval.projected_model_phase_end_unwrapped_rad,
            ]
        )
        for transition in interval.physics:
            values.extend(transition.body_state_after.position_world_m)
            values.extend(transition.body_state_after.velocity_world_m_s)
            values.extend(transition.body_state_after.quaternion_body_to_world)
            values.extend(transition.body_state_after.angular_velocity_body_rad_s)
            values.extend(transition.aerodynamic_wrench_after.force_body_n)
            values.extend(transition.aerodynamic_wrench_after.torque_body_n_m)
            values.extend(transition.mechanics.wing_kinematics.stroke_rad)
    assert np.all(np.isfinite(np.asarray(values, dtype=float)))
    first_sample = result.intervals[0].circuit_observation.sample
    assert first_sample.retinal_input_luminance is not None
    assert first_sample.full_cell_voltage_v is not None
    assert first_sample.full_cell_state_eligible_motor_input is False


NODE_AVAILABLE = shutil.which("node") is not None and Path(
    default_fly_fgs_runtime_script_path()
).is_file()


@pytest.mark.skipif(not NODE_AVAILABLE, reason="Node fly-FGS runtime is unavailable")
def test_real_node_runtime_short_closed_loop_smoke_without_flybody():
    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=0.010,
        include_retinal_input=True,
        include_full_cell_state=False,
    )
    body = _initial_body(yaw=0.0, yaw_rate=0.0)
    physics = FakeFlightPhysics(body)
    with CanonicalClosedLoopSimulator(
        config,
        circuit_runtime=NodeFlyFGSCircuitRuntime(request_timeout_s=180.0),
        bridge=StreamingNOD1MotorBridge(seed=4),
        mechanics=StreamingMuscleWingStepper(),
        physics_adapter=physics,
        initial_body_state=body,
    ) as simulator:
        result = simulator.run()

    observations = [
        interval.circuit_observation
        for interval in result.intervals
        if interval.circuit_observation is not None
    ]
    assert [item.sample.sample_index for item in observations] == [0, 1]
    assert all(len(item.sample.nod1_voltage_v) == 4 for item in observations)
    assert all(len(item.sample.retinal_input_luminance) == 1441 for item in observations)
    assert all(item.sample.full_cell_voltage_v is None for item in observations)
    assert physics.step_count == 100
