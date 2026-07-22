import numpy as np
import pytest

from fly_sensor2behavior.flight import (
    AerodynamicWrench,
    FlightEpisodeRunner,
    FlightSimulationConfig,
    IndividualMuscleState,
    MotorCommand,
    MotorSignalKind,
    MuscleClass,
    MuscleSnapshot,
    Perturbation,
    PerturbationMode,
    PerturbationTarget,
    QuasiSteadyAerodynamics,
    RigidBodyState,
    Side,
    VirtualWingHinge,
)
from fly_sensor2behavior.flybody_adapter import WingAxisTorqueMap


class _MockExternalPhysics:
    backend_name = "mock_external"
    aerodynamic_owner = "mock_external"

    def __init__(self):
        self.calls = 0
        self.state = RigidBodyState()
        self.last_actuator_torque_n_m = np.full(6, 2.0e-9)

    def default_initial_state(self):
        state = RigidBodyState()
        state.position_world_m[2] = 0.123
        return state

    def reset(self, initial_state):
        self.state = initial_state.copy()

    def step(self, wings, force_body_n, torque_body_n_m, dt_s):
        del wings
        np.testing.assert_array_equal(force_body_n, np.zeros(3))
        np.testing.assert_array_equal(torque_body_n_m, np.zeros(3))
        self.calls += 1
        self.state.position_world_m[0] += dt_s
        return self.state.copy()

    def aerodynamic_wrench(self):
        return AerodynamicWrench(
            force_body_n=np.array([1.0e-6, 0.0, 0.0]),
            torque_body_n_m=np.array([0.0, 2.0e-9, 0.0]),
            left_force_body_n=np.zeros(3),
            right_force_body_n=np.zeros(3),
            mechanical_power_w=3.0e-6,
        )


class _MockTelemetryPhysics(_MockExternalPhysics):
    backend_name = "flybody"
    aerodynamic_owner = "flybody"
    body_state_reference = "mock FlyBody root/thorax frame; not whole-fly COM"
    wing_joint_order = (
        "c_thorax-l_wing-yaw",
        "c_thorax-l_wing-roll",
        "c_thorax-l_wing-pitch",
        "c_thorax-r_wing-yaw",
        "c_thorax-r_wing-roll",
        "c_thorax-r_wing-pitch",
    )

    def wing_joint_state(self):
        offset = 0.01 * self.calls
        return np.arange(6, dtype=float) + offset, -np.arange(6, dtype=float) - offset

    def whole_fly_com_position_m(self):
        return self.state.position_world_m + np.array([2.0e-4, -3.0e-4, 4.0e-4])


class _NonfiniteExternalPhysics(_MockExternalPhysics):
    def step(self, wings, force_body_n, torque_body_n_m, dt_s):
        state = super().step(wings, force_body_n, torque_body_n_m, dt_s)
        state.position_world_m[0] = np.nan
        return state


def test_neural_updates_occur_only_on_declared_clock():
    result = FlightEpisodeRunner().run(
        FlightSimulationConfig(
            duration_s=0.020,
            physics_dt_s=0.0001,
            neural_dt_s=0.005,
            logging_dt_s=0.001,
        )
    )
    assert result.diagnostics.metrics["neural_update_count"] == 4.0
    assert result.diagnostics.physics_steps == 200


def test_episode_preserves_individual_muscle_force_phase_and_work():
    command = MotorCommand(
        neuron_id="MN-iv2-left-exact",
        muscle="iv2",
        side=Side.LEFT,
        muscle_class=MuscleClass.STEERING,
        signal_kind=MotorSignalKind.EXACT_SPIKES,
        spike_times_s=(0.001, 0.006),
        preferred_phase_rad=0.0,
        provenance="manufactured exact-spike fixture",
    )
    result = FlightEpisodeRunner().run(
        FlightSimulationConfig(
            duration_s=0.010,
            physics_dt_s=0.0001,
            neural_dt_s=0.005,
            logging_dt_s=0.001,
            motor_commands=(command,),
        )
    )
    key = "left:iv2"
    assert key in result.muscle_activation
    assert key in result.muscle_force_n
    assert key in result.muscle_phase_effect
    assert key in result.muscle_work_j
    assert result.muscle_force_n[key].shape == result.time_s.shape
    assert np.max(result.muscle_force_n[key]) > 0.0
    assert np.max(np.abs(result.muscle_phase_effect[key])) > 0.0
    assert np.all(np.isfinite(result.muscle_work_j[key]))


def test_muscle_scale_is_applied_once_to_steering_force_not_phase_twice():
    command = MotorCommand(
        neuron_id="MN-iv2-left-exact-scale",
        muscle="iv2",
        side=Side.LEFT,
        muscle_class=MuscleClass.STEERING,
        signal_kind=MotorSignalKind.EXACT_SPIKES,
        spike_times_s=(0.001,),
        preferred_phase_rad=0.0,
        provenance="manufactured exact-spike scaling fixture",
    )
    common = dict(
        duration_s=0.006,
        physics_dt_s=0.0001,
        neural_dt_s=0.001,
        logging_dt_s=0.001,
        motor_commands=(command,),
    )
    baseline = FlightEpisodeRunner().run(FlightSimulationConfig(**common))
    scaled = FlightEpisodeRunner().run(
        FlightSimulationConfig(
            **common,
            perturbations=(
                Perturbation(
                    target_type=PerturbationTarget.MUSCLE,
                    target="left:iv2",
                    mode=PerturbationMode.SCALE,
                    start_s=0.0,
                    end_s=0.006,
                    magnitude=0.5,
                ),
            ),
        )
    )
    key = "left:iv2"
    np.testing.assert_allclose(
        scaled.muscle_phase_effect[key], baseline.muscle_phase_effect[key]
    )
    np.testing.assert_allclose(
        scaled.muscle_force_n[key], 0.5 * baseline.muscle_force_n[key]
    )
    baseline_drive = baseline.muscle_force_n[key] * baseline.muscle_phase_effect[key]
    scaled_drive = scaled.muscle_force_n[key] * scaled.muscle_phase_effect[key]
    np.testing.assert_allclose(scaled_drive, 0.5 * baseline_drive)


def test_exact_event_mn_scaling_is_rejected_instead_of_silently_ignored():
    command = MotorCommand(
        neuron_id="MN-iv2-left-exact-immutable",
        muscle="iv2",
        side=Side.LEFT,
        muscle_class=MuscleClass.STEERING,
        signal_kind=MotorSignalKind.EXACT_SPIKES,
        spike_times_s=(0.001,),
    )
    scale = Perturbation(
        target_type=PerturbationTarget.MN,
        target=command.neuron_id,
        mode=PerturbationMode.SCALE,
        start_s=0.0,
        end_s=0.003,
        magnitude=0.5,
    )
    with pytest.raises(ValueError, match="support silencing only"):
        FlightEpisodeRunner().run(
            FlightSimulationConfig(
                duration_s=0.003,
                physics_dt_s=0.0001,
                neural_dt_s=0.001,
                logging_dt_s=0.001,
                motor_commands=(command,),
                perturbations=(scale,),
            )
        )


def test_invalid_or_unknown_perturbations_fail_closed():
    with pytest.raises(ValueError, match="non-negative"):
        Perturbation(
            target_type=PerturbationTarget.MUSCLE,
            target="left:iv2",
            mode=PerturbationMode.SCALE,
            start_s=0.0,
            end_s=0.001,
            magnitude=-0.1,
        )
    with pytest.raises(ValueError, match="must be paired"):
        Perturbation(
            target_type=PerturbationTarget.MN,
            target="MN-iv2-left",
            mode=PerturbationMode.GUST,
            start_s=0.0,
            end_s=0.001,
        )
    with pytest.raises(ValueError, match="unknown motor command"):
        FlightEpisodeRunner().run(
            FlightSimulationConfig(
                duration_s=0.001,
                physics_dt_s=0.0001,
                neural_dt_s=0.001,
                logging_dt_s=0.001,
                dn_motor_rate_gains_hz={"DNa04": {"missing-MN": 1.0}},
            )
        )
    unknown_muscle = Perturbation(
        target_type=PerturbationTarget.MUSCLE,
        target="left:not-a-muscle",
        mode=PerturbationMode.SILENCE,
        start_s=0.0,
        end_s=0.001,
    )
    with pytest.raises(ValueError, match="unknown muscle"):
        FlightEpisodeRunner().run(
            FlightSimulationConfig(
                duration_s=0.001,
                physics_dt_s=0.0001,
                neural_dt_s=0.001,
                logging_dt_s=0.001,
                perturbations=(unknown_muscle,),
            )
        )


def test_empty_motor_command_tuple_means_no_drive_and_endpoint_spikes_fail():
    silent = FlightEpisodeRunner().run(
        FlightSimulationConfig(
            duration_s=0.002,
            physics_dt_s=0.0001,
            neural_dt_s=0.001,
            logging_dt_s=0.001,
            motor_commands=(),
        )
    )
    assert silent.muscle_activation == {}
    assert silent.motor_event_times_s == {}

    endpoint = MotorCommand(
        neuron_id="MN-endpoint",
        muscle="iv2",
        side=Side.LEFT,
        muscle_class=MuscleClass.STEERING,
        signal_kind=MotorSignalKind.EXACT_SPIKES,
        spike_times_s=(0.002,),
    )
    with pytest.raises(ValueError, match=r"\[0, duration_s\)"):
        FlightEpisodeRunner().run(
            FlightSimulationConfig(
                duration_s=0.002,
                physics_dt_s=0.0001,
                neural_dt_s=0.001,
                logging_dt_s=0.001,
                motor_commands=(endpoint,),
            )
        )


def test_state_and_wrench_contracts_reject_nonfinite_or_nonunit_values():
    with pytest.raises(ValueError, match="three finite"):
        RigidBodyState(position_world_m=np.array([0.0, np.nan, 0.0]))
    with pytest.raises(ValueError, match="unit normalized"):
        RigidBodyState(quaternion_body_to_world=np.zeros(4))
    with pytest.raises(ValueError, match="three finite"):
        AerodynamicWrench(
            force_body_n=np.array([np.inf, 0.0, 0.0]),
            torque_body_n_m=np.zeros(3),
            left_force_body_n=np.zeros(3),
            right_force_body_n=np.zeros(3),
            mechanical_power_w=0.0,
        )


def test_virtual_hinge_emits_named_six_axis_torque_for_flybody():
    state = IndividualMuscleState(
        muscle="iv2",
        side=Side.LEFT,
        muscle_class=MuscleClass.STEERING,
        activation=0.5,
        force_n=10.0e-6,
        phase_effect=0.8,
        virtual_moment_arm_m=45.0e-6,
    )
    snapshot = MuscleSnapshot(
        power_left=0.62,
        power_right=0.62,
        steering_left=0.8,
        steering_right=0.0,
        tension_left=1.0,
        tension_right=1.0,
        individual={"left:iv2": state},
    )
    wings = VirtualWingHinge().evaluate(0.2, 200.0, snapshot)
    assert wings.wing_axis_torque_n_m is not None
    assert wings.wing_axis_torque_n_m.shape == (6,)
    assert np.linalg.norm(wings.wing_axis_torque_n_m[:3]) > 0.0
    np.testing.assert_array_equal(
        WingAxisTorqueMap().map_torque(wings), wings.wing_axis_torque_n_m
    )


def test_external_backend_is_exclusive_and_called_once_per_physics_tick():
    adapter = _MockExternalPhysics()
    runner = FlightEpisodeRunner(physics_adapter=adapter)
    result = runner.run(
        FlightSimulationConfig(
            duration_s=0.005,
            physics_dt_s=0.0001,
            neural_dt_s=0.005,
            logging_dt_s=0.001,
        )
    )
    assert adapter.calls == 50
    assert result.diagnostics.physics_backend == "mock_external"
    assert result.diagnostics.aerodynamic_owner == "mock_external"
    assert result.position_world_m[-1, 0] == 0.005
    assert result.position_world_m[0, 2] == 0.123
    assert result.position_world_m[-1, 2] == 0.123
    assert result.aerodynamic_force_body_n[-1, 0] == 1.0e-6
    assert result.diagnostics.integrated_aerodynamic_impulse_n_s[0] == pytest.approx(
        5.0e-9
    )
    assert result.diagnostics.metrics[
        "maximum_external_actuator_torque_n_m"
    ] == pytest.approx(2.0e-9)
    assert result.physics_time_s.shape == (51,)
    assert result.external_actuator_torque_physics_n_m.shape == (51, 6)
    assert result.aerodynamic_force_body_physics_n.shape == (51, 3)
    assert result.aerodynamic_torque_body_physics_n_m.shape == (51, 3)
    np.testing.assert_array_equal(
        result.aerodynamic_force_body_physics_n[10],
        result.aerodynamic_force_body_n[1],
    )
    with np.testing.assert_raises_regex(ValueError, "owns aerodynamics"):
        FlightEpisodeRunner(
            aerodynamics=QuasiSteadyAerodynamics(), physics_adapter=adapter
        )


def test_external_backend_state_is_validated_after_every_step():
    with pytest.raises(ValueError, match="three finite"):
        FlightEpisodeRunner(physics_adapter=_NonfiniteExternalPhysics()).run(
            FlightSimulationConfig(
                duration_s=0.001,
                physics_dt_s=0.0001,
                neural_dt_s=0.001,
                logging_dt_s=0.001,
                motor_commands=(),
            )
        )


def test_external_backend_rejects_nonzero_wind_without_an_ambient_air_api():
    adapter = _MockExternalPhysics()
    gust = Perturbation(
        target_type=PerturbationTarget.ENVIRONMENT,
        target="wind",
        mode=PerturbationMode.GUST,
        start_s=0.0,
        end_s=0.001,
        magnitude=1.0,
        vector=(0.2, -0.1, 0.0),
    )

    with pytest.raises(ValueError, match="does not implement ambient wind"):
        FlightEpisodeRunner(physics_adapter=adapter).run(
            FlightSimulationConfig(
                duration_s=0.001,
                physics_dt_s=0.0001,
                neural_dt_s=0.001,
                logging_dt_s=0.001,
                perturbations=(gust,),
            )
        )
    assert adapter.calls == 0


def test_external_measured_wing_and_whole_fly_com_are_logged_separately():
    adapter = _MockTelemetryPhysics()
    result = FlightEpisodeRunner(physics_adapter=adapter).run(
        FlightSimulationConfig(
            duration_s=0.005,
            physics_dt_s=0.0001,
            neural_dt_s=0.005,
            logging_dt_s=0.001,
        )
    )

    assert result.measured_wing_joint_angle_rad is not None
    assert result.measured_wing_joint_velocity_rad_s is not None
    assert result.whole_fly_com_position_world_m is not None
    assert result.measured_wing_joint_angle_rad.shape == (6, 6)
    assert result.measured_wing_joint_velocity_rad_s.shape == (6, 6)
    assert result.measured_wing_joint_angle_physics_rad.shape == (51, 6)
    assert result.measured_wing_joint_velocity_physics_rad_s.shape == (51, 6)
    assert result.whole_fly_com_position_world_m.shape == (6, 3)
    assert result.measured_wing_joint_order == adapter.wing_joint_order
    np.testing.assert_array_equal(
        result.measured_wing_joint_angle_rad[0], np.arange(6, dtype=float)
    )
    np.testing.assert_allclose(
        result.measured_wing_joint_angle_rad[-1], np.arange(6, dtype=float) + 0.5
    )
    np.testing.assert_allclose(
        result.whole_fly_com_position_world_m[0], [2.0e-4, -3.0e-4, 0.1234]
    )
    np.testing.assert_allclose(
        result.whole_fly_com_position_world_m[-1], [0.0052, -3.0e-4, 0.1234]
    )
    # Root/thorax and articulated COM remain distinct channels.
    assert result.position_world_m[-1, 0] == pytest.approx(0.005)
    assert result.diagnostics.metrics["final_whole_fly_com_altitude_m"] == pytest.approx(
        0.1234
    )
    assert "desired_measured_stroke_rms_rad" in result.diagnostics.metrics
    assert any("Body-state reference" in item for item in result.diagnostics.warnings)
