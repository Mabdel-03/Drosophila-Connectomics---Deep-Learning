"""Paper-defined figure/ground schedules, analysis, and immutable artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sysconfig
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .paper_tether import torque_nm_to_dyne_cm


PAPER_FGS_SCHEMA_VERSION = "1.0.0"
PAPER_FGS_ARTIFACT_SCHEMA_VERSION = "3.0.0"
PAPER_FGS_WEB_REPLAY_SCHEMA_VERSION = "paper_fgs_web_replay.v3"
PAPER_FGS_PROTOCOL_IDS = (
    "R83_Fig3a_0_to_90",
    "R83_Fig3b_0_to_270",
    "R83_Fig3c_0_to_180",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_protocol_path() -> Path:
    checkout = (
        Path(__file__).resolve().parents[2]
        / "data/reference/paper_fgs/paper_protocols.v1.json"
    )
    installed = (
        Path(sysconfig.get_path("data"))
        / "share/fly-sensor2behavior/reference/paper_fgs/paper_protocols.v1.json"
    )
    for candidate in (checkout, installed):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("paper figure-ground protocol definitions are unavailable")


def default_laterality_path() -> Path:
    checkout = (
        Path(__file__).resolve().parents[2]
        / "data/reference/paper_fgs/nod1_laterality.v1.json"
    )
    installed = (
        Path(sysconfig.get_path("data"))
        / "share/fly-sensor2behavior/reference/paper_fgs/nod1_laterality.v1.json"
    )
    for candidate in (checkout, installed):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("paper NOD1 laterality receipt is unavailable")


def default_release_matrix_path() -> Path:
    checkout = (
        Path(__file__).resolve().parents[2]
        / "data/reference/paper_fgs/scientific_release_matrix.v1.json"
    )
    installed = (
        Path(sysconfig.get_path("data"))
        / "share/fly-sensor2behavior/reference/paper_fgs/scientific_release_matrix.v1.json"
    )
    for candidate in (checkout, installed):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("paper scientific-release matrix is unavailable")


def default_figure3_reference_path() -> Path:
    checkout = (
        Path(__file__).resolve().parents[2]
        / "data/reference/paper_fgs/figure3_reference.v1.json"
    )
    installed = (
        Path(sysconfig.get_path("data"))
        / "share/fly-sensor2behavior/reference/paper_fgs/figure3_reference.v1.json"
    )
    for candidate in (checkout, installed):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("digitized Figure 3 reference is unavailable")


def load_figure3_reference() -> Mapping[str, Any]:
    path = default_figure3_reference_path()
    payload = json.loads(path.read_text("utf-8"))
    if payload.get("schema_version") != "reichardt_1983_figure3_digitization.v1":
        raise ValueError("Figure 3 reference schema changed")
    if payload.get("source", {}).get("pdf_sha256") != (
        "9159ec24548e1ee0eaf4edca0d4790d7656890d2fb611dd0b7d27cc85a6f05d7"
    ):
        raise ValueError("Figure 3 source identity changed")
    return {
        **payload,
        "reference_path": str(path.resolve()),
        "reference_sha256": _sha256_file(path),
    }


def load_nod1_laterality_receipt(
    path: Optional[Path] = None,
) -> Mapping[str, Any]:
    source = Path(path or default_laterality_path())
    document = json.loads(source.read_text("utf-8"))
    if document.get("schema_version") != "1.0.0":
        raise ValueError("paper NOD1 laterality schema version mismatch")
    cells = document.get("cells")
    if not isinstance(cells, list) or len(cells) != 4:
        raise ValueError("paper NOD1 laterality receipt must contain four cells")
    observed = {
        (cell.get("raw_application_side"), cell.get("resolved_anatomical_side"))
        for cell in cells
    }
    if observed != {("L", "right"), ("R", "left")}:
        raise ValueError("paper NOD1 laterality receipt does not resolve the lane swap")
    if document.get("paper_effector_assignment") != "raw_l_to_physical_right":
        raise ValueError("paper NOD1 effector assignment mismatch")
    return {
        **document,
        "receipt_path": str(source.resolve()),
        "receipt_sha256": _sha256_file(source),
    }


@dataclass(frozen=True)
class PaperFGSProtocol:
    protocol_id: str
    label: str
    trial_duration_s: float
    figure_mean_azimuth_deg: float
    figure_width_deg: float
    figure_amplitude_deg: float
    ground_amplitude_deg: float
    frequency_hz: float
    initial_relative_phase_deg: float
    target_relative_phase_deg: float
    signed_phase_delta_deg: float
    phase_transition_start_s: float
    phase_transition_duration_s: float
    phase_transition_mode: str
    repetitions: int
    stimulus_eye_update_rate_hz: int
    raw_yaw_reaction_logging_rate_hz: int
    synchronized_channel_logging_rate_hz: int
    physics_rate_hz: int
    web_replay_rate_hz: int

    def __post_init__(self) -> None:
        if self.protocol_id not in PAPER_FGS_PROTOCOL_IDS:
            raise ValueError("unsupported paper protocol_id")
        for field in (
            "trial_duration_s",
            "figure_mean_azimuth_deg",
            "figure_width_deg",
            "figure_amplitude_deg",
            "ground_amplitude_deg",
            "frequency_hz",
            "initial_relative_phase_deg",
            "target_relative_phase_deg",
            "signed_phase_delta_deg",
            "phase_transition_start_s",
            "phase_transition_duration_s",
        ):
            if not math.isfinite(float(getattr(self, field))):
                raise ValueError("{} must be finite".format(field))
        if self.phase_transition_mode != "one_period_phase_ramp":
            raise ValueError("canonical protocols require one_period_phase_ramp")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if self.physics_rate_hz < 2000:
            raise ValueError("physics rate is below the apparatus minimum")
        if self.stimulus_eye_update_rate_hz < 400:
            raise ValueError("stimulus/eye rate is below the apparatus minimum")
        if self.raw_yaw_reaction_logging_rate_hz < 10000:
            raise ValueError("raw yaw-reaction logging rate is below 10 kHz")
        if self.synchronized_channel_logging_rate_hz < 1000:
            raise ValueError("synchronized logging rate is below 1 kHz")

    @property
    def transition_end_s(self) -> float:
        return self.phase_transition_start_s + self.phase_transition_duration_s

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def load_protocol(
    protocol_id: str,
    path: Optional[Path] = None,
) -> PaperFGSProtocol:
    document = json.loads(Path(path or default_protocol_path()).read_text("utf-8"))
    if document.get("schema_version") != PAPER_FGS_SCHEMA_VERSION:
        raise ValueError("paper protocol schema version mismatch")
    try:
        selected = document["protocols"][protocol_id]
    except KeyError as exc:
        raise ValueError("unknown paper protocol_id: {}".format(protocol_id)) from exc
    merged = {**document["defaults"], **selected, "protocol_id": protocol_id}
    return PaperFGSProtocol(**merged)


def sample_protocol(
    protocol: PaperFGSProtocol,
    time_s: Sequence[float],
) -> Mapping[str, np.ndarray]:
    """Evaluate exact commanded and realized angles without integration."""

    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1 or not np.all(np.isfinite(time)):
        raise ValueError("time_s must be a finite one-dimensional array")
    start = protocol.phase_transition_start_s
    duration = protocol.phase_transition_duration_s
    fraction = np.clip((time - start) / duration, 0.0, 1.0)
    phase = protocol.signed_phase_delta_deg * fraction
    phase_velocity_deg_s = np.where(
        (time >= start) & (time < start + duration),
        protocol.signed_phase_delta_deg / duration,
        0.0,
    )
    interval = np.full(time.shape, "relative_motion", dtype="<U20")
    interval[time < start] = "synchronous"
    interval[(time >= start) & (time < start + duration)] = "phase_transition"
    omega = 2.0 * np.pi * protocol.frequency_hz
    ground_theta = omega * time
    figure_theta = ground_theta + np.deg2rad(phase)
    ground_angle = protocol.ground_amplitude_deg * np.sin(ground_theta)
    figure_angle = protocol.figure_amplitude_deg * np.sin(figure_theta)
    ground_velocity = protocol.ground_amplitude_deg * omega * np.cos(ground_theta)
    figure_velocity = protocol.figure_amplitude_deg * (
        omega + np.deg2rad(phase_velocity_deg_s)
    ) * np.cos(figure_theta)
    return {
        "time_s": time.copy(),
        "stimulus_interval": interval,
        "figure_angle_command_deg": figure_angle,
        "figure_angle_realized_deg": figure_angle.copy(),
        "figure_velocity_command_deg_s": figure_velocity,
        "ground_angle_command_deg": ground_angle,
        "ground_angle_realized_deg": ground_angle.copy(),
        "ground_velocity_command_deg_s": ground_velocity,
        "relative_displacement_realized_deg": figure_angle - ground_angle,
        "relative_phase_command_deg": phase,
        "relative_phase_realized_deg": phase.copy(),
    }


def _wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _harmonic(
    time_s: np.ndarray,
    signal: np.ndarray,
    frequency_hz: float,
) -> Tuple[float, float, float]:
    theta = 2.0 * np.pi * frequency_hz * time_s
    centered = signal - np.mean(signal)
    sine = 2.0 / len(signal) * float(np.sum(centered * np.sin(theta)))
    cosine = 2.0 / len(signal) * float(np.sum(centered * np.cos(theta)))
    amplitude = float(np.hypot(sine, cosine))
    position_phase = math.degrees(math.atan2(cosine, sine))
    return amplitude, _wrap_degrees(position_phase), _wrap_degrees(position_phase - 90.0)


def analyze_phase_locked_trials(
    protocol: PaperFGSProtocol,
    time_s: Sequence[float],
    reported_yaw_torque_n_m: np.ndarray,
    *,
    phase_bin_count: int = 100,
) -> Mapping[str, Any]:
    time = np.asarray(time_s, dtype=float)
    torque = np.asarray(reported_yaw_torque_n_m, dtype=float)
    if time.ndim != 1 or torque.ndim != 2 or torque.shape[1] != len(time):
        raise ValueError("torque must have shape (trial, time)")
    if torque.shape[0] < 1 or not np.all(np.isfinite(torque)):
        raise ValueError("torque trials must be finite and non-empty")
    mean = np.mean(torque, axis=0)
    sem = (
        np.std(torque, axis=0, ddof=1) / math.sqrt(torque.shape[0])
        if torque.shape[0] > 1
        else np.zeros_like(mean)
    )
    pre = time < protocol.phase_transition_start_s
    post = time >= protocol.transition_end_s
    if not np.any(pre) or not np.any(post):
        raise ValueError("time grid does not cover pre- and post-transition windows")
    harmonic = _harmonic(time[post], mean[post], protocol.frequency_hz)
    phase = np.mod(time[post] * protocol.frequency_hz, 1.0)
    bins = np.linspace(0.0, 1.0, phase_bin_count + 1)
    indices = np.minimum(np.digitize(phase, bins) - 1, phase_bin_count - 1)
    waveform = np.full(phase_bin_count, np.nan, dtype=float)
    for index in range(phase_bin_count):
        selected = mean[post][indices == index]
        if selected.size:
            waveform[index] = float(np.mean(selected))
    return {
        "trial_count": int(torque.shape[0]),
        "phase_locked_mean_torque_n_m": mean,
        "phase_locked_sem_torque_n_m": sem,
        "pre_transition_mean_torque_n_m": float(np.mean(mean[pre])),
        "post_transition_mean_torque_n_m": float(np.mean(mean[post])),
        "delta_mean_torque_n_m": float(np.mean(mean[post]) - np.mean(mean[pre])),
        "first_harmonic_amplitude_n_m": harmonic[0],
        "first_harmonic_phase_relative_ground_position_deg": harmonic[1],
        "first_harmonic_phase_relative_ground_velocity_deg": harmonic[2],
        "phase_bin_centers_cycles": 0.5 * (bins[:-1] + bins[1:]),
        "phase_binned_mean_torque_n_m": waveform,
    }


def compare_to_paper_figure3(
    protocol: PaperFGSProtocol,
    simulation_time_s: Sequence[float],
    simulation_torque_trials_n_m: np.ndarray,
    *,
    signal_product: str = "paper_comparison",
) -> Mapping[str, Any]:
    """Compare absolute physical torque to Figure 3 with no fitted transform."""

    time = np.asarray(simulation_time_s, dtype=float)
    trials = np.asarray(simulation_torque_trials_n_m, dtype=float)
    if time.ndim != 1 or trials.ndim != 2 or trials.shape[1] != len(time):
        raise ValueError("simulation torque must have shape (trial, time)")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(trials)):
        raise ValueError("simulation comparison inputs must be finite")
    reference = load_figure3_reference()
    panel = reference["panels"].get(protocol.protocol_id)
    if panel is None:
        raise ValueError("protocol has no digitized Figure 3 panel")
    paper_time = np.asarray(panel["time_s"], dtype=float)
    paper_torque = np.asarray(panel["torque_dyne_cm"], dtype=float)
    paper_uncertainty = np.asarray(
        panel["digitization_uncertainty_dyne_cm"], dtype=float
    )
    simulation_mean_dyne_cm = np.mean(trials, axis=0) * 1.0e7
    simulation_sem_dyne_cm = (
        np.std(trials, axis=0, ddof=1) / math.sqrt(trials.shape[0]) * 1.0e7
        if trials.shape[0] > 1
        else np.zeros_like(simulation_mean_dyne_cm)
    )
    # The primary comparison uses only interpolation to the paper sample times;
    # there is no time shift, warp, gain, offset, or transfer-function fit.
    simulation_on_paper = np.interp(paper_time, time, simulation_mean_dyne_cm)
    simulation_sem_on_paper = np.interp(
        paper_time, time, simulation_sem_dyne_cm
    )
    residual = simulation_on_paper - paper_torque
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    paper_range = max(float(np.ptp(paper_torque)), 1.0e-30)
    correlation = (
        float(np.corrcoef(simulation_on_paper, paper_torque)[0, 1])
        if np.std(simulation_on_paper) > 0.0 and np.std(paper_torque) > 0.0
        else 0.0
    )
    pre = paper_time < protocol.phase_transition_start_s
    post = paper_time >= protocol.transition_end_s
    simulation_harmonic = _harmonic(
        paper_time[post], simulation_on_paper[post], protocol.frequency_hz
    )
    paper_harmonic = _harmonic(
        paper_time[post], paper_torque[post], protocol.frequency_hz
    )

    centered_simulation = simulation_on_paper - np.mean(simulation_on_paper)
    centered_paper = paper_torque - np.mean(paper_torque)
    cross = np.correlate(centered_simulation, centered_paper, mode="full")
    lag_samples = int(np.argmax(cross) - (len(paper_time) - 1))
    lag_seconds = lag_samples * float(np.median(np.diff(paper_time)))
    extraction_checks = {
        "time_landmarks_roundtrip_within_0_25px": bool(
            panel["time_axis"]["landmark_roundtrip_within_0_25px"]
        ),
        "torque_landmarks_roundtrip_within_0_25px": bool(
            panel["torque_axis_dyne_cm"]["landmark_roundtrip_within_0_25px"]
        ),
        "two_independent_traces_recorded": bool(
            len(panel["independent_trace_1_dyne_cm"]) == len(paper_time)
            and len(panel["independent_trace_2_dyne_cm"]) == len(paper_time)
        ),
        "per_sample_uncertainty_positive": bool(
            np.all(np.isfinite(paper_uncertainty))
            and np.all(paper_uncertainty > 0.0)
        ),
    }
    return {
        "schema_version": "paper_figure3_comparison.v1",
        "protocol_id": protocol.protocol_id,
        "paper_panel": panel["panel"],
        "signal_product": signal_product,
        "absolute_physical_units": True,
        "amplitude_fit_applied": False,
        "offset_fit_applied": False,
        "time_shift_applied": False,
        "time_warp_applied": False,
        "paper_trace_has_reported_sem": False,
        "paper_reference_sha256": reference["reference_sha256"],
        "paper_pdf_sha256": reference["source"]["pdf_sha256"],
        "paper_extraction_validation": {
            "checks": extraction_checks,
            "passed": all(extraction_checks.values()),
        },
        "paper_time_s": paper_time,
        "paper_torque_dyne_cm": paper_torque,
        "paper_digitization_uncertainty_dyne_cm": paper_uncertainty,
        "simulation_mean_dyne_cm_at_paper_times": simulation_on_paper,
        "simulation_sem_dyne_cm_at_paper_times": simulation_sem_on_paper,
        "residual_dyne_cm": residual,
        "pre_transition": {
            "paper_mean_dyne_cm": float(np.mean(paper_torque[pre])),
            "simulation_mean_dyne_cm": float(np.mean(simulation_on_paper[pre])),
        },
        "post_transition": {
            "paper_mean_dyne_cm": float(np.mean(paper_torque[post])),
            "simulation_mean_dyne_cm": float(np.mean(simulation_on_paper[post])),
        },
        "delta_mean": {
            "paper_dyne_cm": float(
                np.mean(paper_torque[post]) - np.mean(paper_torque[pre])
            ),
            "simulation_dyne_cm": float(
                np.mean(simulation_on_paper[post])
                - np.mean(simulation_on_paper[pre])
            ),
        },
        "first_harmonic": {
            "paper_amplitude_dyne_cm": paper_harmonic[0],
            "paper_phase_relative_ground_position_deg": paper_harmonic[1],
            "paper_phase_relative_ground_velocity_deg": paper_harmonic[2],
            "simulation_amplitude_dyne_cm": simulation_harmonic[0],
            "simulation_phase_relative_ground_position_deg": simulation_harmonic[1],
            "simulation_phase_relative_ground_velocity_deg": simulation_harmonic[2],
        },
        "scores": {
            "rmse_dyne_cm": rmse,
            "normalized_rmse_percent_of_paper_range": 100.0 * rmse / paper_range,
            "mae_dyne_cm": float(np.mean(np.abs(residual))),
            "mean_bias_dyne_cm": float(np.mean(residual)),
            "waveform_correlation": correlation,
        },
        "diagnostic_cross_correlation": {
            "lag_samples": lag_samples,
            "lag_s": lag_seconds,
            "applied_to_primary_score": False,
        },
    }


def compare_convergence(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    normalized_scalar_tolerance_percent: float = 2.0,
    phase_tolerance_deg: float = 2.0,
    waveform_nrmse_tolerance_percent: float = 5.0,
) -> Mapping[str, Any]:
    """Compare 10/20 kHz or 400/800 Hz phase-locked analyses.

    The plan's degree symbol on mean/amplitude thresholds is dimensionally
    incompatible with torque, so scalar changes are implemented as normalized
    percentages; phase remains degrees and waveform error is percent NRMSE.
    """

    reference_waveform = np.asarray(
        reference["phase_binned_mean_torque_n_m"], dtype=float
    )
    candidate_waveform = np.asarray(
        candidate["phase_binned_mean_torque_n_m"], dtype=float
    )
    if reference_waveform.shape != candidate_waveform.shape:
        raise ValueError("convergence waveforms must have identical bins")
    valid = np.isfinite(reference_waveform) & np.isfinite(candidate_waveform)
    if not np.any(valid):
        raise ValueError("convergence waveforms contain no shared finite bins")
    reference_amplitude = abs(float(reference["first_harmonic_amplitude_n_m"]))
    candidate_amplitude = abs(float(candidate["first_harmonic_amplitude_n_m"]))
    waveform_scale = max(
        float(np.ptp(reference_waveform[valid])),
        reference_amplitude,
        1.0e-30,
    )
    mean_difference = abs(
        float(candidate["post_transition_mean_torque_n_m"])
        - float(reference["post_transition_mean_torque_n_m"])
    )
    mean_change_percent = 100.0 * mean_difference / waveform_scale
    amplitude_change_percent = (
        100.0
        * abs(candidate_amplitude - reference_amplitude)
        / max(reference_amplitude, 1.0e-30)
    )
    phase_difference_deg = abs(
        _wrap_degrees(
            float(candidate["first_harmonic_phase_relative_ground_position_deg"])
            - float(reference["first_harmonic_phase_relative_ground_position_deg"])
        )
    )
    waveform_nrmse_percent = (
        100.0
        * float(
            np.sqrt(
                np.mean(
                    (
                        candidate_waveform[valid]
                        - reference_waveform[valid]
                    )
                    ** 2
                )
            )
        )
        / waveform_scale
    )
    checks = {
        "post_mean_normalized_change_percent": mean_change_percent
        <= normalized_scalar_tolerance_percent,
        "harmonic_amplitude_change_percent": amplitude_change_percent
        <= normalized_scalar_tolerance_percent,
        "harmonic_phase_change_deg": phase_difference_deg
        <= phase_tolerance_deg,
        "phase_binned_waveform_nrmse_percent": waveform_nrmse_percent
        <= waveform_nrmse_tolerance_percent,
    }
    return {
        "schema_version": "1.0.0",
        "threshold_interpretation": (
            "torque mean/amplitude use normalized percent; phase uses degrees"
        ),
        "post_mean_normalized_change_percent": mean_change_percent,
        "harmonic_amplitude_change_percent": amplitude_change_percent,
        "harmonic_phase_change_deg": phase_difference_deg,
        "phase_binned_waveform_nrmse_percent": waveform_nrmse_percent,
        "tolerances": {
            "normalized_scalar_percent": normalized_scalar_tolerance_percent,
            "phase_deg": phase_tolerance_deg,
            "waveform_nrmse_percent": waveform_nrmse_tolerance_percent,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


def write_paper_artifact(
    output: Path,
    protocol: PaperFGSProtocol,
    arrays: Mapping[str, np.ndarray],
    analysis: Mapping[str, Any],
    *,
    run_metadata: Mapping[str, Any],
    authoritative: bool,
    web_replay_output: Optional[Path] = None,
) -> Mapping[str, Any]:
    """Atomically create one immutable native run plus browser projection."""

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("artifact output already exists: {}".format(output))
    replay_destination = (
        None if web_replay_output is None else Path(web_replay_output).resolve()
    )
    replay_manifest_destination = (
        None
        if replay_destination is None
        else replay_destination.with_name(replay_destination.name + ".manifest.json")
    )
    for destination in (replay_destination, replay_manifest_destination):
        if destination is not None and destination.exists():
            raise FileExistsError(
                "browser replay registration already exists: {}".format(destination)
            )
    required = {
        "time_s",
        "figure_angle_command_deg",
        "figure_angle_realized_deg",
        "ground_angle_command_deg",
        "ground_angle_realized_deg",
        "relative_displacement_realized_deg",
        "relative_phase_command_deg",
        "relative_phase_realized_deg",
        "support_on_fly_yaw_reaction_engine_Nm",
        "fly_generated_yaw_moment_engine_Nm",
        "root_fluid_yaw_moment_engine_Nm",
        "reported_yaw_torque_Nm",
        "reported_yaw_torque_dyne_cm",
        "body_position_m",
        "body_euler_deg",
        "body_yaw_rate_deg_s",
        "wing_joint_position_rad",
        "head_transform_body",
        "yaw_equation_residual_Nm",
    }
    if run_metadata.get("torque_calibration") is not None:
        required.update(
            {
                "support_load_cell_force_engine_N",
                "support_load_cell_moment_engine_Nm",
                "attempted_fly_yaw_moment_from_support_engine_Nm",
                "equality_only_yaw_reaction_engine_Nm",
                "yaw_balance_residual_Nm",
                "wingbeat_phase_rad",
                "actuator_torque_Nm",
                "actuator_clipped",
                "raw_internal_padded_time_s",
                "raw_internal_padded_reported_yaw_torque_Nm",
                "torque_product_paper_comparison_Nm",
                "torque_product_lowpass_10hz_Nm",
                "torque_product_lowpass_25hz_Nm",
                "torque_product_lowpass_50hz_Nm",
                "torque_product_wingbeat_averaged_Nm",
                "wing_reported_yaw_torque_Nm",
                "wing_sum_reported_yaw_torque_Nm",
                "nonwing_reported_yaw_torque_residual_Nm",
                "raw_internal_padded_wing_reported_yaw_torque_Nm",
                "raw_internal_padded_wing_load_cell_position_engine_m",
                "raw_internal_padded_wing_load_cell_orientation_site_to_engine",
                "raw_internal_padded_wing_parent_on_child_force_engine_N",
                "raw_internal_padded_wing_parent_on_child_moment_at_hinge_engine_Nm",
                "raw_internal_padded_wing_on_thorax_force_engine_N",
                "raw_internal_padded_wing_on_thorax_moment_at_hinge_engine_Nm",
                "raw_internal_padded_wing_on_thorax_moment_at_tether_engine_Nm",
                "left_wing_torque_product_paper_comparison_Nm",
                "right_wing_torque_product_paper_comparison_Nm",
                "wing_sum_torque_product_paper_comparison_Nm",
                "nonwing_residual_torque_product_paper_comparison_Nm",
            }
        )
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError("paper artifact arrays missing: {}".format(", ".join(missing)))
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=str(parent)))
    try:
        arrays_path = temporary / "scientific_arrays.npz"
        np.savez_compressed(
            arrays_path,
            **{key: np.asarray(value) for key, value in arrays.items()},
        )
        summary_path = temporary / "analysis.json"
        summary_path.write_text(
            json.dumps(_jsonable(analysis), allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        time = np.asarray(arrays["time_s"], dtype=float)
        mean = np.asarray(analysis["phase_locked_mean_torque_n_m"], dtype=float)
        sem = np.asarray(analysis["phase_locked_sem_torque_n_m"], dtype=float)
        average_csv = temporary / "phase_locked_average.csv"
        with average_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["time_s", "mean_torque_Nm", "sem_torque_Nm", "mean_torque_dyne_cm"]
            )
            for values in zip(time, mean, sem):
                writer.writerow([values[0], values[1], values[2], torque_nm_to_dyne_cm(values[1])])

        mechanical_channel_arrays = {
            "authoritative_total": np.asarray(
                arrays["reported_yaw_torque_Nm"], dtype=float
            ),
        }
        if "wing_reported_yaw_torque_Nm" in arrays:
            wing_channels = np.asarray(
                arrays["wing_reported_yaw_torque_Nm"], dtype=float
            )
            if wing_channels.ndim == 3 and wing_channels.shape[-1] == 2:
                mechanical_channel_arrays.update(
                    {
                        "left_wing": wing_channels[:, :, 0],
                        "right_wing": wing_channels[:, :, 1],
                    }
                )
        for channel, array_name in (
            ("wing_sum", "wing_sum_reported_yaw_torque_Nm"),
            ("nonwing_residual", "nonwing_reported_yaw_torque_residual_Nm"),
        ):
            if array_name in arrays:
                mechanical_channel_arrays[channel] = np.asarray(
                    arrays[array_name], dtype=float
                )
        if "wing_aerodynamic_reported_yaw_torque_Nm" in arrays:
            aerodynamic_channels = np.asarray(
                arrays["wing_aerodynamic_reported_yaw_torque_Nm"], dtype=float
            )
            if (
                aerodynamic_channels.ndim == 3
                and aerodynamic_channels.shape[-1] == 2
                and np.all(np.isfinite(aerodynamic_channels))
            ):
                mechanical_channel_arrays.update(
                    {
                        "left_wing_aerodynamic": aerodynamic_channels[:, :, 0],
                        "right_wing_aerodynamic": aerodynamic_channels[:, :, 1],
                    }
                )

        multichannel_csv = temporary / "multichannel_phase_locked_average.csv"
        multichannel_statistics = {}
        for channel, values in mechanical_channel_arrays.items():
            multichannel_statistics[channel] = {
                "mean": np.mean(values, axis=0),
                "sem": (
                    np.std(values, axis=0, ddof=1) / math.sqrt(values.shape[0])
                    if values.shape[0] > 1
                    else np.zeros(values.shape[1], dtype=float)
                ),
            }
        with multichannel_csv.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            header = ["time_s"]
            for channel in multichannel_statistics:
                header.extend(
                    [
                        channel + "_mean_Nm",
                        channel + "_sem_Nm",
                        channel + "_mean_dyne_cm",
                    ]
                )
            writer.writerow(header)
            for index, sample_time in enumerate(time):
                row = [sample_time]
                for statistics in multichannel_statistics.values():
                    channel_mean = float(statistics["mean"][index])
                    row.extend(
                        [
                            channel_mean,
                            float(statistics["sem"][index]),
                            torque_nm_to_dyne_cm(channel_mean),
                        ]
                    )
                writer.writerow(row)

        trial_metrics_csv = temporary / "trial_metrics.csv"
        torque_trials = np.asarray(arrays["reported_yaw_torque_Nm"], dtype=float)
        transition_start = protocol.phase_transition_start_s
        transition_end = protocol.transition_end_s
        pre_mask = time < transition_start
        post_mask = time >= transition_end
        with trial_metrics_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "trial_id",
                    "pre_mean_torque_Nm",
                    "post_mean_torque_Nm",
                    "delta_mean_torque_Nm",
                    "pre_mean_torque_dyne_cm",
                    "post_mean_torque_dyne_cm",
                    "delta_mean_torque_dyne_cm",
                ]
            )
            for trial_id, trial_torque in enumerate(torque_trials):
                pre_mean = float(np.mean(trial_torque[pre_mask]))
                post_mean = float(np.mean(trial_torque[post_mask]))
                delta_mean = post_mean - pre_mean
                writer.writerow(
                    [
                        trial_id,
                        pre_mean,
                        post_mean,
                        delta_mean,
                        torque_nm_to_dyne_cm(pre_mean),
                        torque_nm_to_dyne_cm(post_mean),
                        torque_nm_to_dyne_cm(delta_mean),
                    ]
                )

        paper_comparisons = analysis.get("paper_figure3_comparisons", {})
        primary_paper_comparison = paper_comparisons.get("paper_comparison")
        reference_path = None
        if primary_paper_comparison is not None:
            reference_path = temporary / "figure3_reference_for_protocol.json"
            reference_path.write_text(
                json.dumps(
                    _jsonable(primary_paper_comparison),
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            overlay_csv = temporary / "paper_overlay_and_residual.csv"
            with overlay_csv.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "time_s",
                        "paper_torque_dyne_cm",
                        "paper_digitization_uncertainty_dyne_cm",
                        "simulation_total_mean_dyne_cm",
                        "simulation_total_sem_dyne_cm",
                        "simulation_minus_paper_residual_dyne_cm",
                    ]
                )
                for values in zip(
                    primary_paper_comparison["paper_time_s"],
                    primary_paper_comparison["paper_torque_dyne_cm"],
                    primary_paper_comparison[
                        "paper_digitization_uncertainty_dyne_cm"
                    ],
                    primary_paper_comparison[
                        "simulation_mean_dyne_cm_at_paper_times"
                    ],
                    primary_paper_comparison[
                        "simulation_sem_dyne_cm_at_paper_times"
                    ],
                    primary_paper_comparison["residual_dyne_cm"],
                ):
                    writer.writerow(values)

        replay_stride = max(
            1,
            int(
                round(
                    protocol.synchronized_channel_logging_rate_hz
                    / protocol.web_replay_rate_hz
                )
            ),
        )
        stimulus = sample_protocol(protocol, time)
        frames = []
        wing_trials = np.asarray(arrays["wing_joint_position_rad"], dtype=float)
        product_arrays = {
            name: np.asarray(
                arrays["torque_product_{}_Nm".format(name)], dtype=float
            )
            for name in (
                "paper_comparison",
                "lowpass_10hz",
                "lowpass_25hz",
                "lowpass_50hz",
                "wingbeat_averaged",
            )
            if "torque_product_{}_Nm".format(name) in arrays
        }
        channel_product_prefixes = {
            "authoritative_total": "torque_product_{}_Nm",
            "left_wing": "left_wing_torque_product_{}_Nm",
            "right_wing": "right_wing_torque_product_{}_Nm",
            "wing_sum": "wing_sum_torque_product_{}_Nm",
            "nonwing_residual": "nonwing_residual_torque_product_{}_Nm",
            "left_wing_aerodynamic": "left_wing_aerodynamic_torque_product_{}_Nm",
            "right_wing_aerodynamic": "right_wing_aerodynamic_torque_product_{}_Nm",
        }
        channel_products = {}
        for channel, template in channel_product_prefixes.items():
            available = {
                product: np.asarray(arrays[template.format(product)], dtype=float)
                for product in product_arrays
                if template.format(product) in arrays
            }
            if available:
                channel_products[channel] = available
        for index in range(0, len(time), replay_stride):
            frame = {
                    "time_s": float(time[index]),
                    "figure_angle_realized_deg": float(stimulus["figure_angle_realized_deg"][index]),
                    "ground_angle_realized_deg": float(stimulus["ground_angle_realized_deg"][index]),
                    "relative_phase_realized_deg": float(stimulus["relative_phase_realized_deg"][index]),
                    "reported_yaw_torque_Nm": float(mean[index]),
                    "reported_yaw_torque_sem_Nm": float(sem[index]),
                    "raw_trial_reported_yaw_torque_Nm": torque_trials[:, index].tolist(),
                    "raw_trial_wing_joint_position_rad": wing_trials[:, index, :].tolist(),
                    "torque_products_Nm": {
                        name: {
                            "mean": float(np.mean(values[:, index])),
                            "sem": float(
                                np.std(values[:, index], ddof=1)
                                / math.sqrt(values.shape[0])
                                if values.shape[0] > 1
                                else 0.0
                            ),
                        }
                        for name, values in product_arrays.items()
                    },
                }
            frame["torque_channels_Nm"] = {
                channel: {
                    "raw_trials": mechanical_channel_arrays[channel][:, index].tolist(),
                    "products": {
                        product: {
                            "mean": float(np.mean(values[:, index])),
                            "sem": float(
                                np.std(values[:, index], ddof=1)
                                / math.sqrt(values.shape[0])
                                if values.shape[0] > 1
                                else 0.0
                            ),
                        }
                        for product, values in channel_products.get(
                            channel, {}
                        ).items()
                    },
                }
                for channel in mechanical_channel_arrays
                if channel in channel_products
            }
            frames.append(frame)
        trial_receipts = run_metadata.get("trial_receipts", [])
        apparatus_passed = bool(trial_receipts) and all(
            receipt.get("diagnostics", {}).get("apparatus_validation_passed") is True
            for receipt in trial_receipts
        )
        sign_calibrated = (
            run_metadata.get("sign_calibration", {}).get("passed") is True
        )
        physics_provenance = run_metadata.get("physics_provenance", {})
        meter_mode = physics_provenance.get("torque_meter_mode")
        if meter_mode is None and "authoritative_meter" in physics_provenance:
            meter_mode = "comparison"
        native_load_cell = meter_mode in ("fixed-load-cell", "comparison")
        dual_meter_passed = (
            run_metadata.get("dual_meter_validation", {}).get("passed") is True
        )
        calibration_passed = (
            (run_metadata.get("torque_calibration") or {}).get("passed") is True
        )
        bilateral_calibration_passed = (
            (run_metadata.get("torque_calibration") or {})
            .get("bilateral_wing_load_cells", {})
            .get("passed")
            is True
        )
        per_wing_metrology = run_metadata.get("per_wing_metrology", {})
        per_wing_metrology_passed = (
            per_wing_metrology.get("passed") is True
            and per_wing_metrology.get("wing_load_cells_available") is True
        )
        aerodynamic_probe_passed = (
            per_wing_metrology_passed
            and per_wing_metrology.get("aerodynamic_probes_available") is True
        )
        sampling = run_metadata.get("sampling", {})
        response = sampling.get("measured_response", {})
        sampling_passed = (
            response.get("passband_ripple_db", float("inf")) <= 0.01
            and response.get("stopband_attenuation_db", float("-inf")) >= 100.0
            and sampling.get("raw_internal_samples_retained") is True
        )
        convergence_passed = (
            analysis.get("convergence_against_reference", {}).get("passed") is True
        )
        paper_extraction_passed = bool(primary_paper_comparison) and (
            primary_paper_comparison.get("paper_extraction_validation", {}).get(
                "passed"
            )
            is True
        )
        repetition_count_passed = int(torque_trials.shape[0]) == int(
            protocol.repetitions
        )
        if authoritative and not (
            apparatus_passed
            and sign_calibrated
            and native_load_cell
            and meter_mode == "comparison"
            and dual_meter_passed
            and calibration_passed
            and bilateral_calibration_passed
            and per_wing_metrology_passed
            and aerodynamic_probe_passed
            and sampling_passed
            and convergence_passed
            and paper_extraction_passed
            and repetition_count_passed
        ):
            raise ValueError(
                "authoritative paper replay requires passing apparatus receipts, "
                "native total/per-wing calibration, aerodynamic closure, 100 trials, "
                "FIR metrology, and multichannel convergence"
            )
        replay = {
            "schema_version": PAPER_FGS_WEB_REPLAY_SCHEMA_VERSION,
            "scientific_status": (
                "authoritative_native_torque" if authoritative else "software_integration_only"
            ),
            "protocol_id": protocol.protocol_id,
            "replay_rate_hz": protocol.web_replay_rate_hz,
            "trial_count": int(torque_trials.shape[0]),
            "torque_meter_mode": meter_mode or "unavailable",
            "default_signal_product": "paper_comparison",
            "available_signal_products": list(product_arrays),
            "metrology": {
                "absolute_physical_units": bool(authoritative),
                "no_amplitude_fit": True,
                "no_offset_fit": True,
                "no_time_shift": True,
                "calibration_passed": calibration_passed,
                "bilateral_wing_calibration_passed": bilateral_calibration_passed,
                "per_wing_metrology_passed": per_wing_metrology_passed,
                "aerodynamic_probe_passed": aerodynamic_probe_passed,
                "sampling_passed": sampling_passed,
                "convergence_passed": convergence_passed,
                "dual_meter_passed": dual_meter_passed,
                "paper_trace_has_reported_sem": False,
                "paper_extraction_passed": paper_extraction_passed,
            },
            "channel_authority": dict(
                run_metadata.get(
                    "channel_authority",
                    {
                        "reported_yaw_torque_Nm": "authoritative_native_torque",
                        "wing_reported_yaw_torque_Nm": "unavailable",
                        "wing_aerodynamic_reported_yaw_torque_Nm": "unavailable",
                        "paper_figure3_trace": "digitized_reference_no_reported_sem",
                    },
                )
            ),
            "paper_reference": (
                None
                if primary_paper_comparison is None
                else {
                    "panel": primary_paper_comparison["paper_panel"],
                    "time_s": _jsonable(primary_paper_comparison["paper_time_s"]),
                    "torque_dyne_cm": _jsonable(
                        primary_paper_comparison["paper_torque_dyne_cm"]
                    ),
                    "digitization_uncertainty_dyne_cm": _jsonable(
                        primary_paper_comparison[
                            "paper_digitization_uncertainty_dyne_cm"
                        ]
                    ),
                    "paper_reported_sem": False,
                    "pdf_sha256": primary_paper_comparison["paper_pdf_sha256"],
                    "reference_sha256": primary_paper_comparison[
                        "paper_reference_sha256"
                    ],
                }
            ),
            "frames": frames,
            "validation": {
                "apparatus_passed": apparatus_passed,
                "apparatus_validation_is_not_behavioral_validation": True,
            },
            "analysis": _jsonable(
                {
                    key: value
                    for key, value in analysis.items()
                    if not isinstance(value, np.ndarray)
                }
            ),
        }
        replay_path = temporary / "web_replay.json"
        replay_path.write_text(
            json.dumps(replay, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_files = [
            arrays_path,
            summary_path,
            average_csv,
            trial_metrics_csv,
            multichannel_csv,
            replay_path,
        ]
        if reference_path is not None:
            artifact_files.append(reference_path)
            artifact_files.append(overlay_csv)
        files = {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in artifact_files
        }
        identity_payload = {
            "schema_version": PAPER_FGS_ARTIFACT_SCHEMA_VERSION,
            "protocol": protocol.to_dict(),
            "metadata": _jsonable(run_metadata),
            "files": files,
        }
        run_id = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        manifest = {
            **identity_payload,
            "run_id": run_id,
            "scientific_status": replay["scientific_status"],
            "apparatus_validation_separate_from_behavior": True,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary), str(output))
        if replay_destination is not None:
            assert replay_manifest_destination is not None
            replay_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output / "web_replay.json", replay_destination)
            shutil.copyfile(output / "manifest.json", replay_manifest_destination)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "PAPER_FGS_ARTIFACT_SCHEMA_VERSION",
    "PAPER_FGS_PROTOCOL_IDS",
    "PAPER_FGS_SCHEMA_VERSION",
    "PAPER_FGS_WEB_REPLAY_SCHEMA_VERSION",
    "PaperFGSProtocol",
    "analyze_phase_locked_trials",
    "compare_convergence",
    "compare_to_paper_figure3",
    "default_figure3_reference_path",
    "default_protocol_path",
    "default_laterality_path",
    "default_release_matrix_path",
    "load_protocol",
    "load_figure3_reference",
    "load_nod1_laterality_receipt",
    "sample_protocol",
    "write_paper_artifact",
]
