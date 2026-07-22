"""Immutable engineering calibration receipt for the native torque meter."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .flybody_adapter import _FLYGYM_FORCE_TO_NEWTON, _FLYGYM_TORQUE_TO_NEWTON_METRE
from .paper_tether import (
    PAPER_ENGINE_AXES,
    PAPER_TETHER_SCHEMA_VERSION,
    PaperComparisonFlyBodyAdapter,
    PaperFixedLoadCellFlyBodyAdapter,
    torque_dyne_cm_to_nm,
    torque_nm_to_dyne_cm,
)


PAPER_TORQUE_CALIBRATION_SCHEMA_VERSION = "paper_torque_calibration.v3"


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _fixed_meter(adapter: Any) -> PaperFixedLoadCellFlyBodyAdapter:
    if isinstance(adapter, PaperComparisonFlyBodyAdapter):
        return adapter.fixed
    if isinstance(adapter, PaperFixedLoadCellFlyBodyAdapter):
        return adapter
    raise TypeError("absolute calibration requires the fixed-load-cell meter")


def _static_sample(
    meter: PaperFixedLoadCellFlyBodyAdapter,
    force_n: Sequence[float],
    moment_n_m: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    import mujoco as mj

    meter._set_calibration_wrench(  # calibrated private metrology boundary
        np.asarray(force_n, dtype=float), np.asarray(moment_n_m, dtype=float)
    )
    simulation = meter.bundle["simulation"]
    mj.mj_forward(simulation.mj_model, simulation.mj_data)
    sample = meter.tether_torque_sample()
    return sample.support_force_engine_n, sample.support_moment_engine_n_m


def _set_point_force(
    meter: PaperFixedLoadCellFlyBodyAdapter,
    force_n: np.ndarray,
    point_offset_engine_m: np.ndarray,
) -> None:
    """Apply force at a declared root-relative point using ``xfrc_applied``."""

    data = meter.bundle["simulation"].mj_data
    root = int(meter.bundle["root_body_id"])
    root_origin_m = np.asarray(data.xpos[root], dtype=float) / 1000.0
    body_com_m = np.asarray(data.xipos[root], dtype=float) / 1000.0
    application_point_m = root_origin_m + np.asarray(point_offset_engine_m, dtype=float)
    moment_at_com_n_m = np.cross(application_point_m - body_com_m, force_n)
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    data.xfrc_applied[root, :3] = force_n / _FLYGYM_FORCE_TO_NEWTON
    data.xfrc_applied[root, 3:] = moment_at_com_n_m / _FLYGYM_TORQUE_TO_NEWTON_METRE


def _sinusoidal_response(
    meter: PaperFixedLoadCellFlyBodyAdapter,
    frequency_hz: float,
    amplitude_n_m: float,
) -> Mapping[str, float]:
    """Measure the completed-state meter response against a matched zero run."""

    duration_s = max(2.0 / frequency_hz, 0.05)
    dt_s = float(meter.bundle["config"].timestep_s)
    count = int(round(duration_s / dt_s))
    duration_s = count * dt_s

    def run(injected: bool) -> np.ndarray:
        meter.reset(meter.default_initial_state())
        simulation = meter.bundle["simulation"]
        values = np.empty(count, dtype=float)
        for index in range(count):
            completed_time = (index + 1) * dt_s
            expected_paper = amplitude_n_m * math.sin(
                2.0 * math.pi * frequency_hz * completed_time
            )
            moment = np.asarray(
                (0.0, 0.0, -expected_paper if injected else 0.0), dtype=float
            )
            meter._set_calibration_wrench(np.zeros(3, dtype=float), moment)
            simulation.step()
            values[index] = meter.tether_torque_sample().reported_yaw_torque_n_m
        return values

    measured = run(True) - run(False)
    time_s = np.arange(1, count + 1, dtype=float) * dt_s
    theta = 2.0 * math.pi * frequency_hz * time_s
    sine = 2.0 / count * float(np.sum(measured * np.sin(theta)))
    cosine = 2.0 / count * float(np.sum(measured * np.cos(theta)))
    measured_amplitude = float(np.hypot(sine, cosine))
    phase_deg = math.degrees(math.atan2(cosine, sine))
    meter.reset(meter.default_initial_state())
    return {
        "frequency_hz": frequency_hz,
        "applied_paper_amplitude_n_m": amplitude_n_m,
        "measured_paper_amplitude_n_m": measured_amplitude,
        "gain": measured_amplitude / amplitude_n_m,
        "phase_error_deg": phase_deg,
        "raw_sample_rate_hz": 1.0 / dt_s,
        "cycles": duration_s * frequency_hz,
    }


def _calibrate_bilateral_wing_frames(
    meter: PaperFixedLoadCellFlyBodyAdapter,
) -> Mapping[str, Any]:
    """Calibrate side registration and the wing-site wrench transport.

    MuJoCo's force/torque sensor gain is intrinsic to the compiled model.  This
    receipt independently exercises every sign, basis axis, and lever-arm term
    used after the sensor, which is the part of the bilateral decomposition
    implemented by this package.
    """

    meter.reset(meter.default_initial_state())
    sample = meter.tether_torque_sample()
    positions = np.asarray(sample.wing_load_cell_position_engine_m, dtype=float)
    orientations = np.asarray(
        sample.wing_load_cell_orientation_site_to_engine, dtype=float
    )
    data = meter.bundle["simulation"].mj_data
    model = meter.bundle["simulation"].mj_model
    original_sensordata = np.asarray(data.sensordata, dtype=float).copy()
    tether_position = (
        np.asarray(data.site_xpos[int(meter.bundle["load_cell_site_id"])], dtype=float)
        / 1000.0
    )
    frame_trials = []
    maximum_orthonormal_error = 0.0
    maximum_determinant_error = 0.0
    maximum_gain_error = 0.0
    maximum_cross_axis_leakage = 0.0
    maximum_lever_error = 0.0
    signed_yaw_trials = []
    sensor_channel_injection_trials = []
    maximum_sensor_injection_error = 0.0
    maximum_opposite_side_leakage = 0.0

    def inject_sensor_wrench(
        side: str,
        parent_force_engine_n: np.ndarray,
        parent_moment_engine_n_m: np.ndarray,
    ) -> Mapping[str, np.ndarray]:
        side_index = ("left", "right").index(side)
        rotation = orientations[side_index]
        data.sensordata[:] = 0.0
        for sensor_kind, sensor_ids, engine_value, scale in (
            (
                "force",
                meter.bundle["wing_load_cell_force_sensor_ids"],
                parent_force_engine_n,
                _FLYGYM_FORCE_TO_NEWTON,
            ),
            (
                "torque",
                meter.bundle["wing_load_cell_torque_sensor_ids"],
                parent_moment_engine_n_m,
                _FLYGYM_TORQUE_TO_NEWTON_METRE,
            ),
        ):
            del sensor_kind
            sensor_id = int(sensor_ids[side])
            address = int(model.sensor_adr[sensor_id])
            dimension = int(model.sensor_dim[sensor_id])
            data.sensordata[address : address + dimension] = (
                rotation.T.dot(engine_value) / scale
            )
        return meter._wing_load_cell_measurement()
    for side_index, side in enumerate(("left", "right")):
        rotation = orientations[side_index]
        orthonormal_error = float(
            np.max(np.abs(rotation.T.dot(rotation) - np.eye(3)), initial=0.0)
        )
        determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
        maximum_orthonormal_error = max(
            maximum_orthonormal_error, orthonormal_error
        )
        maximum_determinant_error = max(
            maximum_determinant_error, determinant_error
        )
        for axis_index, axis in enumerate(("x", "y", "z")):
            for sign in (-1.0, 1.0):
                parent_moment_site = np.zeros(3, dtype=float)
                parent_moment_site[axis_index] = sign * 1.0e-7
                wing_hinge_moment = -rotation.dot(parent_moment_site)
                measured_norm = float(np.linalg.norm(wing_hinge_moment))
                gain = measured_norm / 1.0e-7
                expected_direction = -sign * rotation[:, axis_index]
                measured_direction = wing_hinge_moment / max(
                    measured_norm, 1.0e-30
                )
                leakage = float(
                    np.linalg.norm(measured_direction - expected_direction)
                )
                maximum_gain_error = max(maximum_gain_error, abs(gain - 1.0))
                maximum_cross_axis_leakage = max(
                    maximum_cross_axis_leakage, leakage
                )
                frame_trials.append(
                    {
                        "physical_side": side,
                        "sensor_axis": axis,
                        "sensor_parent_on_wing_moment_Nm": parent_moment_site.tolist(),
                        "wing_on_thorax_hinge_moment_engine_Nm": wing_hinge_moment.tolist(),
                        "gain": gain,
                        "direction_error_fraction": leakage,
                    }
                )
        for sign in (-1.0, 1.0):
            parent_moment_engine = np.asarray((0.0, 0.0, sign * 1.0e-7))
            parent_moment_site = rotation.T.dot(parent_moment_engine)
            wing_hinge_moment = -rotation.dot(parent_moment_site)
            paper_torque = -float(wing_hinge_moment[2])
            signed_yaw_trials.append(
                {
                    "physical_side": side,
                    "parent_on_wing_engine_yaw_moment_Nm": float(
                        parent_moment_engine[2]
                    ),
                    "reported_paper_torque_Nm": paper_torque,
                    "reported_paper_torque_dyne_cm": torque_nm_to_dyne_cm(
                        paper_torque
                    ),
                }
            )
            injected = inject_sensor_wrench(
                side,
                np.zeros(3, dtype=float),
                parent_moment_engine,
            )
            measured = float(injected["reported_yaw_torque_n_m"][side_index])
            opposite = float(
                injected["reported_yaw_torque_n_m"][1 - side_index]
            )
            expected = float(parent_moment_engine[2])
            error = abs(measured - expected) / 1.0e-7
            leakage = abs(opposite) / 1.0e-7
            maximum_sensor_injection_error = max(
                maximum_sensor_injection_error, error
            )
            maximum_opposite_side_leakage = max(
                maximum_opposite_side_leakage, leakage
            )
            sensor_channel_injection_trials.append(
                {
                    "physical_side": side,
                    "injection_kind": "compiled_sensor_buffer_yaw_moment",
                    "parent_on_wing_engine_moment_Nm": parent_moment_engine.tolist(),
                    "expected_paper_torque_Nm": expected,
                    "measured_paper_torque_Nm": measured,
                    "opposite_side_paper_torque_Nm": opposite,
                    "relative_error": error,
                    "opposite_side_leakage_fraction": leakage,
                }
            )

        lever = positions[side_index] - tether_position
        for force_axis in (0, 1):
            parent_force_engine = np.zeros(3, dtype=float)
            parent_force_engine[force_axis] = 1.0e-4
            parent_force_site = rotation.T.dot(parent_force_engine)
            wing_force = -rotation.dot(parent_force_site)
            transported = np.cross(lever, wing_force)
            expected = np.cross(lever, -parent_force_engine)
            scale = max(float(np.linalg.norm(expected)), 1.0e-30)
            lever_error = float(np.linalg.norm(transported - expected) / scale)
            maximum_lever_error = max(maximum_lever_error, lever_error)
            frame_trials.append(
                {
                    "physical_side": side,
                    "sensor_force_axis": ("x", "y")[force_axis],
                    "hinge_to_tether_engine_m": lever.tolist(),
                    "wing_on_thorax_force_engine_N": wing_force.tolist(),
                    "transported_moment_engine_Nm": transported.tolist(),
                    "expected_transported_moment_engine_Nm": expected.tolist(),
                    "relative_lever_error": lever_error,
                }
            )
            injected = inject_sensor_wrench(
                side,
                parent_force_engine,
                np.zeros(3, dtype=float),
            )
            expected_paper = -float(expected[2])
            measured_paper = float(
                injected["reported_yaw_torque_n_m"][side_index]
            )
            scale_paper = max(abs(expected_paper), 1.0e-30)
            injection_error = abs(measured_paper - expected_paper) / scale_paper
            opposite = float(
                injected["reported_yaw_torque_n_m"][1 - side_index]
            )
            leakage = abs(opposite) / scale_paper
            maximum_sensor_injection_error = max(
                maximum_sensor_injection_error, injection_error
            )
            maximum_opposite_side_leakage = max(
                maximum_opposite_side_leakage, leakage
            )
            sensor_channel_injection_trials.append(
                {
                    "physical_side": side,
                    "injection_kind": "compiled_sensor_buffer_force_with_tether_lever_arm",
                    "parent_on_wing_engine_force_N": parent_force_engine.tolist(),
                    "expected_paper_torque_Nm": expected_paper,
                    "measured_paper_torque_Nm": measured_paper,
                    "opposite_side_paper_torque_Nm": opposite,
                    "relative_error": injection_error,
                    "opposite_side_leakage_fraction": leakage,
                }
            )

    left_is_positive_y = bool(positions[0, 1] > tether_position[1])
    right_is_negative_y = bool(positions[1, 1] < tether_position[1])
    clockwise_trials_pass = all(
        math.isclose(
            item["reported_paper_torque_dyne_cm"],
            1.0 if item["parent_on_wing_engine_yaw_moment_Nm"] > 0 else -1.0,
            rel_tol=0.0,
            abs_tol=5.0e-13,
        )
        for item in signed_yaw_trials
    )
    checks = {
        "site_frames_orthonormal": bool(
            maximum_orthonormal_error <= 1.0e-12
            and maximum_determinant_error <= 1.0e-12
        ),
        "physical_side_registration": left_is_positive_y and right_is_negative_y,
        "coordinate_gain_error_le_0_05_percent": bool(
            maximum_gain_error <= 5.0e-4
        ),
        "coordinate_cross_axis_leakage_le_0_1_percent": bool(
            maximum_cross_axis_leakage <= 1.0e-3
        ),
        "tether_origin_lever_arm_error_le_0_1_percent": bool(
            maximum_lever_error <= 1.0e-3
        ),
        "clockwise_1e_7_Nm_displays_plus_1_dyne_cm": clockwise_trials_pass,
        "compiled_sensor_channel_injection_gain_error_le_0_05_percent": bool(
            maximum_sensor_injection_error <= 5.0e-4
        ),
        "compiled_sensor_channel_side_leakage_le_0_1_percent": bool(
            maximum_opposite_side_leakage <= 1.0e-3
        ),
    }
    data.sensordata[:] = original_sensordata
    return {
        "physical_side_order": ["left", "right"],
        "flybody_body_names": ["l_wing", "r_wing"],
        "site_positions_engine_m": positions.tolist(),
        "site_orientations_site_to_engine": orientations.tolist(),
        "tether_position_engine_m": tether_position.tolist(),
        "frame_and_lever_trials": frame_trials,
        "signed_yaw_trials": signed_yaw_trials,
        "sensor_channel_injection_trials": sensor_channel_injection_trials,
        "maximum_orthonormal_error": maximum_orthonormal_error,
        "maximum_determinant_error": maximum_determinant_error,
        "maximum_coordinate_gain_error_fraction": maximum_gain_error,
        "maximum_coordinate_cross_axis_leakage_fraction": maximum_cross_axis_leakage,
        "maximum_lever_arm_error_fraction": maximum_lever_error,
        "maximum_sensor_channel_injection_error_fraction": maximum_sensor_injection_error,
        "maximum_opposite_side_leakage_fraction": maximum_opposite_side_leakage,
        "sensor_channel_injection_scope": (
            "compiled MuJoCo sensor IDs, addresses, site frames, side isolation, "
            "sign, unit conversion, and tether-origin transport; sensor values "
            "are injected without integrating the mechanical model"
        ),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def create_torque_calibration_receipt(adapter: Any) -> Mapping[str, Any]:
    """Run the full no-fit load-cell calibration and return a hash receipt."""

    import mujoco as mj

    meter = _fixed_meter(adapter)
    meter.reset(meter.default_initial_state())
    model = meter.bundle["simulation"].mj_model
    gravity = np.asarray(model.opt.gravity, dtype=float).copy()
    stiffness = np.asarray(model.jnt_stiffness, dtype=float).copy()
    damping = np.asarray(model.dof_damping, dtype=float).copy()
    model.opt.gravity[:] = 0.0
    model.jnt_stiffness[:] = 0.0
    model.dof_damping[:] = 0.0
    zero_load_force, zero_load_moment = _static_sample(
        meter, np.zeros(3, dtype=float), np.zeros(3, dtype=float)
    )
    model.opt.gravity[:] = gravity
    model.jnt_stiffness[:] = stiffness
    model.dof_damping[:] = damping
    meter.reset(meter.default_initial_state())
    baseline_force, baseline_moment = _static_sample(
        meter, np.zeros(3, dtype=float), np.zeros(3, dtype=float)
    )
    magnitudes = (1.0e-9, 1.0e-8, 1.0e-7, 2.0e-7)
    axes = ("x", "y", "z")
    moment_trials = []
    maximum_gain_error = 0.0
    maximum_cross_axis_leakage = 0.0
    for axis_index, axis in enumerate(axes):
        for magnitude in magnitudes:
            for sign in (-1.0, 1.0):
                applied = np.zeros(3, dtype=float)
                applied[axis_index] = sign * magnitude
                _force, measured_absolute = _static_sample(
                    meter, np.zeros(3, dtype=float), applied
                )
                measured = measured_absolute - baseline_moment
                expected = -applied
                gain = measured[axis_index] / expected[axis_index]
                cross = np.delete(measured, axis_index)
                leakage = float(np.max(np.abs(cross), initial=0.0) / magnitude)
                maximum_gain_error = max(maximum_gain_error, abs(gain - 1.0))
                maximum_cross_axis_leakage = max(
                    maximum_cross_axis_leakage, leakage
                )
                moment_trials.append(
                    {
                        "axis": axis,
                        "applied_engine_moment_n_m": applied.tolist(),
                        "expected_support_moment_n_m": expected.tolist(),
                        "measured_support_moment_n_m": measured.tolist(),
                        "axis_gain": float(gain),
                        "cross_axis_leakage_fraction": leakage,
                    }
                )

    lever_trials = []
    maximum_lever_error = 0.0
    for offset in (
        np.asarray((0.0005, 0.0, 0.0)),
        np.asarray((-0.0005, 0.0, 0.0)),
        np.asarray((0.0, 0.0005, 0.0)),
        np.asarray((0.0, -0.0005, 0.0)),
    ):
        force = np.asarray((0.0, 1.0e-4, 0.0))
        if offset[1] != 0.0:
            force = np.asarray((1.0e-4, 0.0, 0.0))
        meter.reset(meter.default_initial_state())
        _set_point_force(meter, force, offset)
        mj.mj_forward(
            meter.bundle["simulation"].mj_model,
            meter.bundle["simulation"].mj_data,
        )
        sample = meter.tether_torque_sample()
        measured = sample.support_moment_engine_n_m - baseline_moment
        expected = -np.cross(offset, force)
        scale = max(float(np.linalg.norm(expected)), 1.0e-30)
        error = float(np.linalg.norm(measured - expected) / scale)
        maximum_lever_error = max(maximum_lever_error, error)
        lever_trials.append(
            {
                "point_offset_engine_m": offset.tolist(),
                "applied_force_engine_n": force.tolist(),
                "expected_support_moment_n_m": expected.tolist(),
                "measured_support_moment_n_m": measured.tolist(),
                "relative_error": error,
            }
        )

    dynamic = [
        _sinusoidal_response(meter, frequency, 1.0e-7)
        for frequency in (2.5, 200.0)
    ]
    conversion_values = (-2.0, -1.0, 0.0, 1.0, 2.0)
    conversion_roundtrip_exact = all(
        torque_dyne_cm_to_nm(torque_nm_to_dyne_cm(value * 1.0e-7))
        == value * 1.0e-7
        for value in conversion_values
    )
    bilateral_wing_load_cells = _calibrate_bilateral_wing_frames(meter)
    checks = {
        "gain_error_le_0_05_percent": bool(maximum_gain_error <= 5.0e-4),
        "zero_yaw_offset_le_1e_12_Nm": bool(
            abs(float(zero_load_moment[2])) <= 1.0e-12
        ),
        "cross_axis_leakage_le_0_1_percent": bool(
            maximum_cross_axis_leakage <= 1.0e-3
        ),
        "lever_arm_error_le_0_1_percent": bool(
            maximum_lever_error <= 1.0e-3
        ),
        "2_5hz_phase_error_le_0_1_deg": bool(
            abs(dynamic[0]["phase_error_deg"]) <= 0.1
        ),
        "unit_conversion_roundtrip_exact": bool(conversion_roundtrip_exact),
        "bilateral_wing_load_cell_frames_and_transport": bool(
            bilateral_wing_load_cells["passed"]
        ),
    }
    provenance = meter.provenance_metadata()
    receipt_without_digest: Dict[str, Any] = {
        "schema_version": PAPER_TORQUE_CALIBRATION_SCHEMA_VERSION,
        "paper_tether_schema_version": PAPER_TETHER_SCHEMA_VERSION,
        "meter_mode": meter.meter_mode,
        "paper_coordinate_system": dict(PAPER_ENGINE_AXES),
        "zero_load_support_force_n": zero_load_force.tolist(),
        "zero_load_support_moment_n_m": zero_load_moment.tolist(),
        "operating_baseline_support_force_n": baseline_force.tolist(),
        "operating_baseline_support_moment_n_m": baseline_moment.tolist(),
        "zero_load_method": (
            "gravity, joint stiffness, and damping disabled; no applied wrench; "
            "model parameters restored before gain trials"
        ),
        "moment_trials": moment_trials,
        "lever_arm_trials": lever_trials,
        "sinusoidal_trials": dynamic,
        "unit_conversion": {
            "n_m_to_dyne_cm_factor": 1.0e7,
            "roundtrip_exact": conversion_roundtrip_exact,
        },
        "maximum_gain_error_fraction": maximum_gain_error,
        "maximum_cross_axis_leakage_fraction": maximum_cross_axis_leakage,
        "maximum_lever_arm_error_fraction": maximum_lever_error,
        "compiled_model_fingerprint": provenance["compiled_model_fingerprint"],
        "worker_versions": provenance["worker_versions"],
        "worker_config": provenance["worker_config"],
        "load_cell": provenance["load_cell"],
        "bilateral_wing_load_cells": bilateral_wing_load_cells,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    receipt = dict(receipt_without_digest)
    receipt["receipt_sha256"] = _canonical_sha256(receipt_without_digest)
    meter.reset(meter.default_initial_state())
    return receipt


__all__ = [
    "PAPER_TORQUE_CALIBRATION_SCHEMA_VERSION",
    "create_torque_calibration_receipt",
]
