import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from fly_sensor2behavior.flybody_adapter import (
    PINNED_FLYGYM_VERSION,
    FlyBodyPhysicsAdapter,
    FlyBodyPhysicsCheckpoint,
    FlyBodyWorkerConfig,
    WingAxisTorqueMap,
    dependency_versions,
)
from fly_sensor2behavior.flight.types import AerodynamicWrench, WingKinematics
from fly_sensor2behavior.schema import REVIEWED_FLYBODY_WING_AXIS_ORDER


def _fake_adapter(monkeypatch):
    qpos = np.array([1.0, 2.0, 30.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
    qvel = np.array([4.0, 5.0, 6.0, 0.1, 0.2, 0.3], dtype=float)
    integration_state = np.concatenate(([0.125], qpos, qvel, [7.0, 8.0]))
    data = SimpleNamespace(
        qpos=qpos.copy(),
        qvel=qvel.copy(),
        integration_state=integration_state.copy(),
        forward_count=0,
    )
    simulation = SimpleNamespace(mj_model=object(), mj_data=data)

    def state_size(_model, _spec):
        return int(data.integration_state.size)

    def get_state(_model, _data, destination, _spec):
        destination[:] = data.integration_state

    def set_state(_model, _data, source, _spec):
        data.integration_state = np.asarray(source, dtype=float).copy()
        data.qpos[:] = source[1:8]
        data.qvel[:] = source[8:14]

    def forward(_model, _data):
        data.forward_count += 1

    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(
            mjtState=SimpleNamespace(mjSTATE_INTEGRATION=0x7FFF),
            mj_stateSize=state_size,
            mj_getState=get_state,
            mj_setState=set_state,
            mj_forward=forward,
        ),
    )
    adapter = FlyBodyPhysicsAdapter.__new__(FlyBodyPhysicsAdapter)
    adapter.bundle = {
        "simulation": simulation,
        "compiled_model_fingerprint": {"sha256": "sha256:" + "a1" * 32},
        "versions": {"flygym": "2.1.0", "mujoco": "3.9.0"},
        "config": SimpleNamespace(timestep_s=1.0e-4),
        "wing_dofs": [
            SimpleNamespace(name=name) for name in REVIEWED_FLYBODY_WING_AXIS_ORDER
        ],
        "root_qpos_adr": 0,
        "root_dof_adr": 0,
    }
    adapter._last_actuator_torque_n_m = np.arange(6, dtype=float) * 1.0e-10
    adapter._last_wrench = AerodynamicWrench(
        force_body_n=np.array([1.0e-6, 2.0e-6, 3.0e-6]),
        torque_body_n_m=np.array([4.0e-9, 5.0e-9, 6.0e-9]),
        left_force_body_n=np.zeros(3),
        right_force_body_n=np.zeros(3),
        mechanical_power_w=7.0e-6,
    )
    return adapter, data


def test_flybody_checkpoint_round_trip_restores_integration_and_telemetry(monkeypatch):
    adapter, data = _fake_adapter(monkeypatch)
    expected_state = data.integration_state.copy()
    expected_torque = adapter.last_actuator_torque_n_m
    expected_wrench = adapter.aerodynamic_wrench()

    checkpoint = adapter.checkpoint()
    restored_wrapper = FlyBodyPhysicsCheckpoint(checkpoint.to_dict())

    data.integration_state[:] = -99.0
    data.qpos[:] = -99.0
    data.qvel[:] = -99.0
    adapter._last_actuator_torque_n_m[:] = -99.0
    adapter._last_wrench = AerodynamicWrench(
        force_body_n=np.full(3, -99.0),
        torque_body_n_m=np.full(3, -99.0),
        left_force_body_n=np.full(3, -99.0),
        right_force_body_n=np.full(3, -99.0),
        mechanical_power_w=-99.0,
    )

    restored_state = adapter.restore_checkpoint(restored_wrapper)

    np.testing.assert_array_equal(data.integration_state, expected_state)
    np.testing.assert_array_equal(adapter.last_actuator_torque_n_m, expected_torque)
    observed_wrench = adapter.aerodynamic_wrench()
    np.testing.assert_array_equal(observed_wrench.force_body_n, expected_wrench.force_body_n)
    np.testing.assert_array_equal(
        observed_wrench.torque_body_n_m, expected_wrench.torque_body_n_m
    )
    assert observed_wrench.mechanical_power_w == expected_wrench.mechanical_power_w
    np.testing.assert_array_equal(restored_state.position_world_m, [0.001, 0.002, 0.03])
    np.testing.assert_array_equal(restored_state.velocity_world_m_s, [0.004, 0.005, 0.006])
    assert data.forward_count == 1


def test_flybody_checkpoint_rejects_tampering_and_cross_model_before_mutation(
    monkeypatch,
):
    adapter, data = _fake_adapter(monkeypatch)
    checkpoint = adapter.checkpoint()

    checkpoint.payload["simulation_state_b64"] = "AAAA"
    before = data.integration_state.copy()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        adapter.restore_checkpoint(checkpoint)
    np.testing.assert_array_equal(data.integration_state, before)
    assert data.forward_count == 0

    valid = adapter.checkpoint()
    other, other_data = _fake_adapter(monkeypatch)
    other.bundle["compiled_model_fingerprint"] = {
        "sha256": "sha256:" + "b2" * 32
    }
    other_before = other_data.integration_state.copy()
    with pytest.raises(ValueError, match="compiled model mismatch"):
        other.restore_checkpoint(valid)
    np.testing.assert_array_equal(other_data.integration_state, other_before)
    assert other_data.forward_count == 0


def test_flybody_checkpoint_constructor_rejects_corrupt_digest(monkeypatch):
    adapter, _ = _fake_adapter(monkeypatch)
    payload = copy.deepcopy(adapter.checkpoint().to_dict())
    payload["physics_timestep_s"] = 5.0e-5

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        FlyBodyPhysicsCheckpoint(payload)


def _native_wing_command(step):
    torque = np.array(
        [
            1.0e-8 * np.sin(0.31 * step),
            -0.7e-8 * np.cos(0.23 * step),
            0.4e-8 * np.sin(0.17 * step),
            -0.9e-8 * np.sin(0.29 * step),
            0.6e-8 * np.cos(0.19 * step),
            -0.3e-8 * np.sin(0.13 * step),
        ]
    )
    return WingKinematics(
        phase_rad=0.1 * step,
        frequency_hz=200.0,
        stroke_rad=np.zeros(2),
        stroke_velocity_rad_s=np.zeros(2),
        stroke_acceleration_rad_s2=np.zeros(2),
        angle_of_attack_rad=np.zeros(2),
        deviation_rad=np.zeros(2),
        generalized_torque_n_m=np.zeros(2),
        wing_axis_torque_n_m=torque,
    )


def _native_observation(adapter, state):
    angles, velocities = adapter.wing_joint_state()
    wrench = adapter.aerodynamic_wrench()
    return np.concatenate(
        (
            state.position_world_m,
            state.velocity_world_m_s,
            state.quaternion_body_to_world,
            state.angular_velocity_body_rad_s,
            angles,
            velocities,
            adapter.last_actuator_torque_n_m,
            wrench.force_body_n,
            wrench.torque_body_n_m,
            [wrench.mechanical_power_w],
        )
    )


@pytest.mark.skipif(
    dependency_versions()["flygym"] != PINNED_FLYGYM_VERSION,
    reason="pinned FlyBody worker dependencies are not installed",
)
def test_compiled_flybody_checkpoint_reentry_is_bit_exact():
    """A fresh compiled adapter must reproduce the post-checkpoint tail."""

    config = FlyBodyWorkerConfig(timestep_s=1.0e-4)
    first = FlyBodyPhysicsAdapter(WingAxisTorqueMap(), config)
    first.reset(first.default_initial_state())
    state = first.default_initial_state()
    for step in range(5):
        state = first.step(
            _native_wing_command(step), np.zeros(3), np.zeros(3), config.timestep_s
        )
    checkpoint_state = _native_observation(first, state)
    checkpoint = first.checkpoint()
    expected_tail = []
    for step in range(5, 11):
        state = first.step(
            _native_wing_command(step), np.zeros(3), np.zeros(3), config.timestep_s
        )
        expected_tail.append(_native_observation(first, state))

    restored = FlyBodyPhysicsAdapter(WingAxisTorqueMap(), config)
    state = restored.restore_checkpoint(checkpoint)
    np.testing.assert_array_equal(
        _native_observation(restored, state), checkpoint_state
    )
    observed_tail = []
    for step in range(5, 11):
        state = restored.step(
            _native_wing_command(step), np.zeros(3), np.zeros(3), config.timestep_s
        )
        observed_tail.append(_native_observation(restored, state))

    np.testing.assert_array_equal(np.asarray(observed_tail), np.asarray(expected_tail))
