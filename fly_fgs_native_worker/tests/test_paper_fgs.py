import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fly_sensor2behavior.paper_fgs import (
    analyze_phase_locked_trials,
    compare_to_paper_figure3,
    compare_convergence,
    load_protocol,
    load_figure3_reference,
    sample_protocol,
    load_nod1_laterality_receipt,
    write_paper_artifact,
)


def test_protocols_match_figure_3_schedule():
    fig3a = load_protocol("R83_Fig3a_0_to_90")
    samples = sample_protocol(fig3a, [0.1, 0.4, 0.6, 0.8])
    assert np.isclose(samples["ground_angle_realized_deg"][0], 5.0)
    np.testing.assert_allclose(
        samples["relative_phase_realized_deg"], [0.0, 0.0, 45.0, 90.0]
    )
    assert list(samples["stimulus_interval"]) == [
        "synchronous",
        "phase_transition",
        "phase_transition",
        "relative_motion",
    ]

    fig3b = load_protocol("R83_Fig3b_0_to_270")
    assert sample_protocol(fig3b, [0.8])["relative_phase_realized_deg"][0] == -90

    fig3c = load_protocol("R83_Fig3c_0_to_180")
    assert fig3c.trial_duration_s == 4.0
    assert sample_protocol(fig3c, [1.6])["relative_phase_realized_deg"][0] == 180
    assert fig3a.raw_yaw_reaction_logging_rate_hz == 10000
    assert fig3a.synchronized_channel_logging_rate_hz == 1000


def test_nod1_soma_coordinates_resolve_the_paper_lane_swap():
    receipt = load_nod1_laterality_receipt()
    assert receipt["paper_effector_assignment"] == "raw_l_to_physical_right"
    for cell in receipt["cells"]:
        expected = "right" if cell["soma_nm"][0] < 533000 else "left"
        assert cell["resolved_anatomical_side"] == expected


def test_analysis_recovers_known_mean_amplitude_and_phase():
    protocol = load_protocol("R83_Fig3a_0_to_90")
    time = np.arange(0.0, protocol.trial_duration_s, 0.001)
    # Work in N m but use the specification's numeric waveform in dyne cm.
    torque = (0.3 + 0.2 * np.sin(2 * np.pi * 2.5 * time + np.deg2rad(30))) * 1e-7
    result = analyze_phase_locked_trials(protocol, time, np.stack([torque, torque]))
    assert np.isclose(result["post_transition_mean_torque_n_m"], 0.3e-7, atol=1e-15)
    assert np.isclose(result["first_harmonic_amplitude_n_m"], 0.2e-7, atol=1e-15)
    assert np.isclose(
        result["first_harmonic_phase_relative_ground_position_deg"], 30, atol=1e-10
    )
    assert np.isclose(
        result["first_harmonic_phase_relative_ground_velocity_deg"], -60, atol=1e-10
    )
    convergence = compare_convergence(result, result)
    assert convergence["passed"] is True
    assert convergence["phase_binned_waveform_nrmse_percent"] == 0


def test_figure3_reference_identity_uncertainty_and_no_fit_comparison():
    reference = load_figure3_reference()
    assert reference["source"]["pdf_sha256"] == (
        "9159ec24548e1ee0eaf4edca0d4790d7656890d2fb611dd0b7d27cc85a6f05d7"
    )
    assert reference["paper_curve_semantics"]["sweeps_per_curve"] == 100
    assert reference["paper_curve_semantics"]["paper_reported_sem"] is False
    protocol = load_protocol("R83_Fig3a_0_to_90")
    panel = reference["panels"][protocol.protocol_id]
    paper_time = np.asarray(panel["time_s"])
    paper_torque_nm = np.asarray(panel["torque_dyne_cm"]) * 1e-7
    time = np.arange(0.0, protocol.trial_duration_s, 0.001)
    reconstructed = np.interp(time, paper_time, paper_torque_nm)
    comparison = compare_to_paper_figure3(
        protocol, time, np.stack((reconstructed, reconstructed))
    )
    assert comparison["amplitude_fit_applied"] is False
    assert comparison["offset_fit_applied"] is False
    assert comparison["time_shift_applied"] is False
    assert comparison["paper_trace_has_reported_sem"] is False
    assert np.all(np.asarray(comparison["paper_digitization_uncertainty_dyne_cm"]) > 0)


def test_checked_in_native_torque_calibration_receipt_is_content_addressed():
    path = (
        Path(__file__).parents[1]
        / "data/reference/paper_fgs/torque_calibration.fixed_load_cell.v2.json"
    )
    receipt = json.loads(path.read_text("utf-8"))
    expected_digest = receipt.pop("receipt_sha256")
    actual_digest = hashlib.sha256(
        json.dumps(
            receipt,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert actual_digest == expected_digest
    assert receipt["passed"] is True
    assert all(receipt["checks"].values())
    assert receipt["worker_versions"] == {"flygym": "2.1.0", "mujoco": "3.9.0"}


def test_immutable_artifact_and_browser_projection(tmp_path):
    protocol = load_protocol("R83_Fig3a_0_to_90")
    time = np.arange(0.0, protocol.trial_duration_s, 0.001)
    stimulus = sample_protocol(protocol, time)
    torque = np.zeros((2, len(time)))
    analysis = analyze_phase_locked_trials(protocol, time, torque)
    zeros = np.zeros_like(torque)
    arrays = {
        "time_s": time,
        **{key: value for key, value in stimulus.items() if value.dtype.kind != "U"},
        "support_on_fly_yaw_reaction_engine_Nm": zeros,
        "fly_generated_yaw_moment_engine_Nm": zeros,
        "root_fluid_yaw_moment_engine_Nm": zeros,
        "reported_yaw_torque_Nm": zeros,
        "reported_yaw_torque_dyne_cm": zeros,
        "body_position_m": np.zeros((2, len(time), 3)),
        "body_euler_deg": np.zeros((2, len(time), 3)),
        "body_yaw_rate_deg_s": zeros,
        "wing_joint_position_rad": np.zeros((2, len(time), 6)),
        "head_transform_body": np.zeros((2, len(time), 4, 4)),
        "yaw_equation_residual_Nm": zeros,
        "wing_reported_yaw_torque_Nm": np.zeros((2, len(time), 2)),
        "wing_sum_reported_yaw_torque_Nm": zeros,
        "nonwing_reported_yaw_torque_residual_Nm": zeros,
        "wing_aerodynamic_reported_yaw_torque_Nm": np.zeros(
            (2, len(time), 2)
        ),
    }
    for product in (
        "paper_comparison",
        "lowpass_10hz",
        "lowpass_25hz",
        "lowpass_50hz",
        "wingbeat_averaged",
    ):
        arrays["torque_product_{}_Nm".format(product)] = zeros
        for prefix in (
            "left_wing",
            "right_wing",
            "wing_sum",
            "nonwing_residual",
            "left_wing_aerodynamic",
            "right_wing_aerodynamic",
        ):
            arrays["{}_torque_product_{}_Nm".format(prefix, product)] = zeros
    output = tmp_path / "artifact"
    web_replay = tmp_path / "paper_replay.json"
    manifest = write_paper_artifact(
        output,
        protocol,
        arrays,
        analysis,
        run_metadata={"test": True},
        authoritative=False,
        web_replay_output=web_replay,
    )
    assert manifest["scientific_status"] == "software_integration_only"
    replay = json.loads((output / "web_replay.json").read_text("utf-8"))
    assert replay["protocol_id"] == protocol.protocol_id
    assert replay["scientific_status"] == "software_integration_only"
    assert replay["schema_version"] == "paper_fgs_web_replay.v3"
    assert len(replay["frames"]) == protocol.trial_duration_s * protocol.web_replay_rate_hz
    assert replay["trial_count"] == 2
    assert len(replay["frames"][0]["raw_trial_wing_joint_position_rad"]) == 2
    assert set(replay["frames"][0]["torque_channels_Nm"]) == {
        "authoritative_total",
        "left_wing",
        "right_wing",
        "wing_sum",
        "nonwing_residual",
        "left_wing_aerodynamic",
        "right_wing_aerodynamic",
    }
    assert (output / "multichannel_phase_locked_average.csv").is_file()
    assert web_replay.read_bytes() == (output / "web_replay.json").read_bytes()
    registered_manifest = json.loads(
        (tmp_path / "paper_replay.json.manifest.json").read_text("utf-8")
    )
    assert registered_manifest["run_id"] == manifest["run_id"]
    with pytest.raises(ValueError, match="passing apparatus receipts"):
        write_paper_artifact(
            tmp_path / "invalid-authoritative",
            protocol,
            arrays,
            analysis,
            run_metadata={"test": True},
            authoritative=True,
        )
