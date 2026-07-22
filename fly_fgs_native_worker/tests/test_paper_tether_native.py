import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")
pytest.importorskip("flygym")

from fly_sensor2behavior.flight.types import WingKinematics
from fly_sensor2behavior.flybody_adapter import build_flybody_simulation
from fly_sensor2behavior.paper_calibration import create_torque_calibration_receipt
from fly_sensor2behavior.paper_tether import (
    PaperComparisonFlyBodyAdapter,
    PaperFixedLoadCellFlyBodyAdapter,
    PaperTetheredFlyBodyAdapter,
)


def _zero_wings():
    zero = np.zeros(2)
    return WingKinematics(
        phase_rad=0.0,
        frequency_hz=200.0,
        stroke_rad=zero,
        stroke_velocity_rad_s=zero,
        stroke_acceleration_rad_s2=zero,
        angle_of_attack_rad=zero,
        deviation_rad=zero,
        generalized_torque_n_m=zero,
    )


def test_native_fixed_topology_and_full_calibration_receipt():
    meter = PaperFixedLoadCellFlyBodyAdapter()
    model = meter.bundle["simulation"].mj_model
    assert not np.any(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    assert model.neq == 0
    assert model.nsensor == 6
    assert meter.bundle["load_cell_force_sensor_id"] is not None
    assert meter.bundle["load_cell_torque_sensor_id"] is not None
    symmetric_sample = meter.tether_torque_sample()
    symmetric_scale = max(
        float(np.max(np.abs(symmetric_sample.wing_reported_yaw_torque_n_m))),
        1.0e-30,
    )
    assert abs(symmetric_sample.wing_sum_reported_yaw_torque_n_m) <= 1.0e-3 * symmetric_scale
    for side in ("left", "right"):
        body_id = int(meter.bundle["wing_body_ids"][side])
        site_id = int(meter.bundle["wing_load_cell_site_ids"][side])
        joint_ids = np.flatnonzero(model.jnt_bodyid == body_id)
        assert joint_ids.size == 3
        np.testing.assert_array_equal(
            model.jnt_pos[joint_ids],
            np.repeat(model.site_pos[site_id][None, :], 3, axis=0),
        )
    unmonitored = build_flybody_simulation(
        meter.bundle["config"],
        paper_tether_mode="fixed-load-cell",
        paper_wing_load_cells=False,
    )
    unmonitored_model = unmonitored["simulation"].mj_model
    assert (model.nq, model.nv, model.nu) == (
        unmonitored_model.nq,
        unmonitored_model.nv,
        unmonitored_model.nu,
    )
    np.testing.assert_array_equal(model.body_mass, unmonitored_model.body_mass)
    np.testing.assert_array_equal(model.body_inertia, unmonitored_model.body_inertia)
    monitored_data = meter.bundle["simulation"].mj_data
    unmonitored_data = unmonitored["simulation"].mj_data
    np.testing.assert_array_equal(monitored_data.qpos, unmonitored_data.qpos)
    np.testing.assert_array_equal(monitored_data.qvel, unmonitored_data.qvel)
    for _ in range(4):
        meter.bundle["simulation"].step()
        unmonitored["simulation"].step()
    np.testing.assert_array_equal(monitored_data.qpos, unmonitored_data.qpos)
    np.testing.assert_array_equal(monitored_data.qvel, unmonitored_data.qvel)
    meter.reset(meter.default_initial_state())
    assert meter.calibrate_paper_yaw_sign()["passed"] is True
    receipt = create_torque_calibration_receipt(meter)
    assert receipt["passed"] is True
    assert receipt["checks"] == {
        "gain_error_le_0_05_percent": True,
        "zero_yaw_offset_le_1e_12_Nm": True,
        "cross_axis_leakage_le_0_1_percent": True,
        "lever_arm_error_le_0_1_percent": True,
        "2_5hz_phase_error_le_0_1_deg": True,
        "unit_conversion_roundtrip_exact": True,
        "bilateral_wing_load_cell_frames_and_transport": True,
    }
    assert receipt["bilateral_wing_load_cells"]["passed"] is True
    assert receipt["bilateral_wing_load_cells"]["physical_side_order"] == [
        "left",
        "right",
    ]
    meter.step(_zero_wings(), np.zeros(3), np.zeros(3), 1.0e-4)
    raw = meter.consume_last_internal_torque_samples()
    assert len(raw) == 4
    np.testing.assert_allclose(
        [sample.timestamp_s for sample in raw],
        [2.5e-5, 5.0e-5, 7.5e-5, 1.0e-4],
        rtol=0.0,
        atol=1.0e-15,
    )
    sample = raw[-1]
    assert sample.wing_reported_yaw_torque_n_m.shape == (2,)
    assert np.all(np.isfinite(sample.wing_reported_yaw_torque_n_m))
    assert (
        sample.wing_sum_reported_yaw_torque_n_m
        + sample.nonwing_reported_yaw_torque_residual_n_m
        == sample.reported_yaw_torque_n_m
    )


def test_native_per_wing_aerodynamic_probe_is_non_integrating_and_closes():
    meter = PaperComparisonFlyBodyAdapter()
    meter.step(_zero_wings(), np.zeros(3), np.zeros(3), 1.0e-4)
    fixed_data = meter.fixed.bundle["simulation"].mj_data
    qpos_before = fixed_data.qpos.copy()
    qvel_before = fixed_data.qvel.copy()
    time_before = float(fixed_data.time)
    sample = meter.tether_torque_sample()
    np.testing.assert_array_equal(fixed_data.qpos, qpos_before)
    np.testing.assert_array_equal(fixed_data.qvel, qvel_before)
    assert float(fixed_data.time) == time_before
    assert np.all(np.isfinite(sample.wing_aerodynamic_reported_yaw_torque_n_m))
    tolerance = max(
        1.0e-3 * abs(sample.full_aerodynamic_reported_yaw_torque_n_m),
        1.0e-10,
    )
    assert abs(sample.aerodynamic_closure_residual_n_m) <= tolerance


def test_native_equality_validator_uses_only_six_tether_rows():
    meter = PaperTetheredFlyBodyAdapter()
    assert meter.bundle["simulation"].mj_model.neq == 1
    assert meter.calibrate_paper_yaw_sign()["passed"] is True
    generalized = meter._equality_only_generalized_reaction()
    assert generalized.shape == (meter.bundle["simulation"].mj_model.nv,)
    assert np.all(np.isfinite(generalized))
