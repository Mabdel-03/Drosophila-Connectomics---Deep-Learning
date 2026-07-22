from dataclasses import replace

import numpy as np
import pytest

from fly_sensor2behavior.flight.closed_loop import (
    ReducedClosedLoopYawSimulator,
    ReducedYawLoopConfig,
    YawTorquePulse,
)
from fly_sensor2behavior.flight.types import ValidationStatus
from fly_sensor2behavior.vision import UniformScene


def _config(**overrides):
    values = {
        "duration_s": 0.250,
        "physics_dt_s": 0.0002,
        "retinal_frame_interval_s": 0.002,
        "retinal_exposure_s": 0.002,
        "initial_yaw_rate_rad_s": 3.0,
    }
    values.update(overrides)
    return ReducedYawLoopConfig(**values)


def test_closed_loop_is_explicitly_exploratory_and_causally_auditable():
    config = _config(duration_s=0.080)
    result = ReducedClosedLoopYawSimulator().run(config)

    assert result.status is ValidationStatus.EXPLORATORY
    assert result.backend == "reduced_analytic_yaw_only"
    assert result.validation_scope == "self_supervised_software_invariants_only"
    assert any("not biological validation" in warning.lower() for warning in result.warnings)
    assert result.feedback_events
    assert result.causal_timing_violations() == ()

    first_event = result.feedback_events[0]
    assert first_event.motion_availability_time_s == pytest.approx(
        first_event.retinal_availability_time_s
        + config.motion_processing_latency_s
    )
    assert first_event.muscle_torque_availability_time_s == pytest.approx(
        first_event.motion_availability_time_s
        + config.descending_latency_s
        + config.motor_latency_s
        + config.muscle_activation_latency_s
    )


def test_feedback_reduces_yaw_rate_and_integrated_rotation_vs_open_loop():
    simulator = ReducedClosedLoopYawSimulator()
    closed = simulator.run(_config(feedback_enabled=True))
    opened = simulator.run(_config(feedback_enabled=False))

    assert closed.causal_timing_violations() == ()
    assert opened.causal_timing_violations() == ()
    assert abs(closed.yaw_rate_rad_s[-1]) < 0.25 * abs(opened.yaw_rate_rad_s[-1])
    assert (
        closed.integrated_absolute_yaw_rate_rad()
        < 0.55 * opened.integrated_absolute_yaw_rate_rad()
    )


def test_mirrored_yaw_perturbations_produce_mirrored_feedback():
    simulator = ReducedClosedLoopYawSimulator()
    positive = simulator.run(_config(initial_yaw_rate_rad_s=3.0))
    negative = simulator.run(_config(initial_yaw_rate_rad_s=-3.0))

    np.testing.assert_allclose(
        negative.yaw_rate_rad_s,
        -positive.yaw_rate_rad_s,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        negative.control_yaw_torque_n_m,
        -positive.control_yaw_torque_n_m,
        rtol=1e-10,
        atol=1e-18,
    )


def test_uniform_scene_cannot_manufacture_visual_control_torque():
    result = ReducedClosedLoopYawSimulator(UniformScene()).run(_config())

    np.testing.assert_array_equal(result.optic_flow_response, 0.0)
    np.testing.assert_array_equal(result.control_yaw_torque_n_m, 0.0)
    assert result.yaw_rate_rad_s[-1] == pytest.approx(3.0)


def test_torque_pulse_uses_half_open_interval_and_si_torque():
    pulse = YawTorquePulse(start_s=0.010, end_s=0.020, torque_n_m=9.0e-11)
    result = ReducedClosedLoopYawSimulator(UniformScene()).run(
        _config(
            duration_s=0.040,
            initial_yaw_rate_rad_s=0.0,
            feedback_enabled=False,
            torque_pulses=(pulse,),
        )
    )
    active = (result.time_s >= pulse.start_s) & (result.time_s < pulse.end_s)
    np.testing.assert_array_equal(
        result.disturbance_yaw_torque_n_m[active], pulse.torque_n_m
    )
    np.testing.assert_array_equal(
        result.disturbance_yaw_torque_n_m[~active], 0.0
    )
    assert result.yaw_rate_rad_s[-1] > 0.0


def test_loop_is_deterministic_and_rejects_nonintegral_clock_ratios():
    config = _config(duration_s=0.080)
    simulator = ReducedClosedLoopYawSimulator()
    first = simulator.run(config)
    second = simulator.run(replace(config))
    np.testing.assert_array_equal(first.yaw_world_rad, second.yaw_world_rad)
    np.testing.assert_array_equal(
        first.control_yaw_torque_n_m, second.control_yaw_torque_n_m
    )
    assert first.feedback_events == second.feedback_events

    with pytest.raises(ValueError, match="integer multiple"):
        _config(duration_s=0.0801)
