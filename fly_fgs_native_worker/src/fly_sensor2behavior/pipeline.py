"""Operational visual/NOD1-circuit-to-flight vertical slices.

This module connects the registered fixed-step fly-FGS visual circuit, a saved
result from the historical NOD1 browser simulator, or causal ``RetinalFrame``
samples passed through the reduced NOD1 surrogate to the exploratory DN/VNC
bridge and flight runtime.  These source contracts remain distinct.  In
particular, fly-FGS contributes visual/circuit state only: its muscle scaling,
prescribed wing kinematics, and algebraic body motion never enter this path.

The bridge gains, phase model, muscle parameters, and reduced mechanics remain
uncalibrated.  This is an executable software-validation vertical slice, not a
claim of quantitatively accurate fly behavior.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .banc_fanc_evidence import default_banc_fanc_evidence_path
from .artifacts import (
    episode_run_id,
    episode_to_web_replay,
    sha256_file,
    write_episode_artifact,
    write_web_replay_episode,
)
from .flight import (
    BridgeIntervention,
    CausalBridgeResult,
    CausalNeuralBridge,
    FlightEpisodeOutput,
    FlightEpisodeRunner,
    FlightSimulationConfig,
    MuscleClass,
    default_motor_commands,
)
from .fly_fgs import (
    FLY_FGS_FIXTURE_KIND,
    RegisteredFlyFGSFixture,
    load_registered_fly_fgs_fixture,
)
from .nod1 import legacy_result_to_circuit_trace, load_legacy_nod1_manifest
from .nod1_parity import load_registered_nod1_browser_fixture
from .schema import CircuitOutputTrace, CircuitSignalKind, RetinalFrame


@dataclass(frozen=True)
class NOD1FlightPipelineConfig:
    """Clock and stochastic configuration for one open-loop vertical slice."""

    physics_dt_s: float = 0.0001
    neural_dt_s: float = 0.005
    logging_dt_s: float = 0.001
    seed: int = 0
    include_airborne_flight_state_drive: bool = True


@dataclass(frozen=True)
class NOD1FlightRun:
    """All source-of-record stages in one NOD1-to-flight execution."""

    circuit: CircuitOutputTrace
    bridge: CausalBridgeResult
    flight_config: FlightSimulationConfig
    flight: FlightEpisodeOutput
    source_metadata: Mapping[str, Any]
    retinal_frames: Tuple[RetinalFrame, ...] = ()
    circuit_replay: Mapping[str, Any] = field(default_factory=dict)

    @property
    def circuit_trace_sha256(self) -> str:
        encoded = self.circuit.to_json().encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _json_safe_mapping(
    value: Optional[Mapping[str, Any]],
    *,
    label: str,
) -> Mapping[str, Any]:
    """Return a detached JSON-safe mapping or fail at the source boundary."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("%s must be a mapping" % label)
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must contain only finite JSON values" % label) from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, Mapping):  # Defensive: the input check already ensures this.
        raise ValueError("%s must encode as a JSON object" % label)
    return decoded


def _is_registered_fly_fgs_source(source_metadata: Mapping[str, Any]) -> bool:
    return source_metadata.get("input_mode") in {
        "registered_fly_fgs_fixed_step_circuit",
        "registered_fly_fgs_fixed_step_fixture",  # pre-release compatibility
    }


def nod1_run_identity_context(run: NOD1FlightRun) -> Mapping[str, Any]:
    """Return the exact source/runtime receipt used in the content run ID."""

    encoded_physics = json.dumps(
        dict(run.flight.diagnostics.physics_provenance),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    context = {
        "banc_fanc_wing_pathway_evidence_sha256": sha256_file(
            default_banc_fanc_evidence_path()
        ),
        "circuit_trace_sha256": run.circuit_trace_sha256,
        "physics_provenance_sha256": hashlib.sha256(encoded_physics).hexdigest(),
    }
    if _is_registered_fly_fgs_source(run.source_metadata):
        context.update(
            {
                "fly_fgs_source_manifest_sha256": str(
                    run.source_metadata["source_manifest_sha256"]
                ),
                "fly_fgs_registered_capture_sha256": str(
                    run.source_metadata["capture_sha256"]
                ),
                "fly_fgs_circuit_replay_sha256": hashlib.sha256(
                    json.dumps(
                        run.circuit_replay,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
    return context


def run_nod1_flight_pipeline(
    legacy_result: Mapping[str, Any],
    *,
    config: Optional[NOD1FlightPipelineConfig] = None,
    interventions: Sequence[BridgeIntervention] = (),
    bridge: Optional[CausalNeuralBridge] = None,
    runner: Optional[FlightEpisodeRunner] = None,
) -> NOD1FlightRun:
    """Run a saved historical NOD1 voltage trace through muscles and mechanics.

    Only individual circuit voltage channels are imported.  The legacy
    ``steering`` population mean and forward-looking phase metric are excluded
    from the motor path by :func:`legacy_result_to_circuit_trace`.
    """

    settings = config if config is not None else NOD1FlightPipelineConfig()
    circuit = legacy_result_to_circuit_trace(legacy_result)
    manifest = load_legacy_nod1_manifest()
    source_metadata = (
        dict(legacy_result.get("meta", {}))
        if isinstance(legacy_result.get("meta", {}), Mapping)
        else {"legacy_meta_value": legacy_result.get("meta")}
    )
    source_metadata.update(
        {
            "input_mode": "saved_legacy_nod1_voltage",
            "circuit_model_scope": "frozen_legacy_1208_cell_cable_export",
            "circuit_snapshot_id": str(manifest["snapshot_id"]),
            "circuit_cell_count": int(manifest["circuit_inventory"]["cells"]),
            "exported_circuit_channels": len(circuit.signals),
        }
    )
    return _run_circuit_flight(
        circuit,
        settings=settings,
        interventions=interventions,
        bridge=bridge,
        runner=runner,
        source_metadata=source_metadata,
    )


def _run_circuit_flight(
    circuit: CircuitOutputTrace,
    *,
    settings: NOD1FlightPipelineConfig,
    interventions: Sequence[BridgeIntervention],
    bridge: Optional[CausalNeuralBridge],
    runner: Optional[FlightEpisodeRunner],
    source_metadata: Mapping[str, Any],
    retinal_frames: Sequence[RetinalFrame] = (),
    circuit_replay: Optional[Mapping[str, Any]] = None,
    episode_duration_s: Optional[float] = None,
) -> NOD1FlightRun:
    safe_circuit_replay = _json_safe_mapping(
        circuit_replay,
        label="circuit_replay",
    )
    neural_bridge = bridge if bridge is not None else CausalNeuralBridge()
    bridge_result = neural_bridge.run(
        circuit,
        seed=settings.seed,
        interventions=tuple(interventions),
    )

    commands = list(bridge_result.wing_motor.to_flight_motor_commands())
    if settings.include_airborne_flight_state_drive:
        # These are explicitly model-provided flight-state inputs.  They keep
        # the already-airborne animal aloft while the NOD1 path supplies only
        # steering activity.  They are not inferred from the visual circuit.
        commands[:0] = [
            command
            for command in default_motor_commands()
            if command.muscle_class is not MuscleClass.STEERING
        ]

    duration_s = (
        float(circuit.sample_times_s[-1])
        if episode_duration_s is None
        else float(episode_duration_s)
    )
    if duration_s + 1.0e-12 < float(circuit.sample_times_s[-1]):
        raise ValueError("episode duration cannot end before the circuit trace")
    flight_config = FlightSimulationConfig(
        duration_s=duration_s,
        physics_dt_s=settings.physics_dt_s,
        neural_dt_s=settings.neural_dt_s,
        logging_dt_s=settings.logging_dt_s,
        seed=settings.seed,
        motor_commands=tuple(commands),
    )
    physics_runner = runner if runner is not None else FlightEpisodeRunner()
    flight = physics_runner.run(flight_config)
    return NOD1FlightRun(
        circuit=circuit,
        bridge=bridge_result,
        flight_config=flight_config,
        flight=flight,
        source_metadata=dict(source_metadata),
        retinal_frames=tuple(retinal_frames),
        circuit_replay=safe_circuit_replay,
    )


def run_registered_nod1_browser_flight_pipeline(
    *,
    registry_path: Optional[Path] = None,
    fixture_path: Optional[Path] = None,
    config: Optional[NOD1FlightPipelineConfig] = None,
    interventions: Sequence[BridgeIntervention] = (),
    bridge: Optional[CausalNeuralBridge] = None,
    runner: Optional[FlightEpisodeRunner] = None,
) -> NOD1FlightRun:
    """Run the historical registered Chromium voltage fixture to behavior.

    The registry resolves and hashes the compressed fixture before parsing.
    Only its four ``browser_voltage_v`` channels enter the bridge.  The 100
    samples occupy the exact half-open interval ``[0, 0.5 s)`` while the
    physical episode retains the registered full 0.5 s duration.
    """

    fixture = load_registered_nod1_browser_fixture(
        registry_path=registry_path,
        fixture_path=fixture_path,
    )
    settings = config if config is not None else NOD1FlightPipelineConfig(
        neural_dt_s=fixture.dt_s
    )
    return _run_circuit_flight(
        fixture.circuit_trace,
        settings=settings,
        interventions=interventions,
        bridge=bridge,
        runner=runner,
        source_metadata=fixture.pipeline_source_metadata(),
        retinal_frames=(),
        episode_duration_s=fixture.duration_s,
    )


def _fly_fgs_pipeline_source_metadata(
    fixture: RegisteredFlyFGSFixture,
) -> Mapping[str, Any]:
    metadata = dict(
        _json_safe_mapping(
            fixture.pipeline_source_metadata(),
            label="fly-FGS pipeline_source_metadata",
        )
    )
    inventory = _json_safe_mapping(
        fixture.circuit_inventory,
        label="fly-FGS circuit_inventory",
    )
    metadata.update(
        {
            "fixture_kind": FLY_FGS_FIXTURE_KIND,
            "snapshot_id": fixture.snapshot_id,
            "circuit_inventory": inventory,
            "exported_circuit_channels": len(fixture.circuit_trace.signals),
            "motor_input_scope": "four_registered_nod1_voltage_channels_only",
            "circuit_replay_present": True,
            "full_cell_state_eligible_motor_input": False,
            "fly_fgs_downstream_mechanics_imported": False,
        }
    )
    return _json_safe_mapping(metadata, label="fly-FGS pipeline metadata")


def run_registered_fly_fgs_flight_pipeline(
    *,
    manifest_path: Optional[Path] = None,
    capture_path: Optional[Path] = None,
    config: Optional[NOD1FlightPipelineConfig] = None,
    interventions: Sequence[BridgeIntervention] = (),
    bridge: Optional[CausalNeuralBridge] = None,
    runner: Optional[FlightEpisodeRunner] = None,
) -> NOD1FlightRun:
    """Run the canonical fixed-step fly-FGS circuit through project mechanics.

    Only the fixture's four registered NOD1 voltage channels enter the existing
    evidence-qualified DNp26/VNC/MN/muscle bridge.  The analytic retinal display,
    pooled circuit readouts, and full-cell matrices are attached for synchronized
    visualization only.  No fly-FGS DN weighting, muscle scaling, prescribed wing
    kinematics, yaw equation, or toy body motion is imported.
    """

    fixture = load_registered_fly_fgs_fixture(
        manifest_path=manifest_path,
        capture_path=capture_path,
    )
    dt_s = fixture.dt_s
    duration_s = fixture.duration_s
    settings = config if config is not None else NOD1FlightPipelineConfig(
        neural_dt_s=dt_s
    )
    return _run_circuit_flight(
        fixture.circuit_trace,
        settings=settings,
        interventions=interventions,
        bridge=bridge,
        runner=runner,
        source_metadata=_fly_fgs_pipeline_source_metadata(fixture),
        retinal_frames=fixture.retinal_frames,
        circuit_replay=fixture.circuit_replay_attachment(),
        episode_duration_s=duration_s,
    )


def run_retinal_flight_pipeline(
    retinal_frames: Sequence[RetinalFrame],
    *,
    config: Optional[NOD1FlightPipelineConfig] = None,
    interventions: Sequence[BridgeIntervention] = (),
    circuit_model: Optional[Any] = None,
    bridge: Optional[CausalNeuralBridge] = None,
    runner: Optional[FlightEpisodeRunner] = None,
) -> NOD1FlightRun:
    """Run causal retinal samples through the reduced NOD1 software surrogate.

    This mode exercises the complete image-sample-to-body software path.  It is
    intentionally kept separate from the frozen browser-compatibility circuit
    and remains blocked from neural/empirical promotion.
    """

    from .vision import ReducedNOD1MotionCircuit

    frames = tuple(retinal_frames)
    model = circuit_model if circuit_model is not None else ReducedNOD1MotionCircuit()
    if not hasattr(model, "run"):
        raise TypeError("circuit_model must provide run(retinal_frames)")
    circuit = model.run(frames)
    settings = config if config is not None else NOD1FlightPipelineConfig(
        neural_dt_s=float(circuit.sample_times_s[1] - circuit.sample_times_s[0])
    )
    return _run_circuit_flight(
        circuit,
        settings=settings,
        interventions=interventions,
        bridge=bridge,
        runner=runner,
        source_metadata={
            "input_mode": "causal_image_derived_reduced_nod1",
            "retinal_frame_count": len(frames),
            "circuit_model": type(model).__name__,
            "circuit_model_scope": "reduced_nod1_surrogate_not_full_1208_cell",
            "circuit_cell_count": None,
            "exported_circuit_channels": len(circuit.signals),
        },
        retinal_frames=frames,
    )


def _pipeline_arrays(run: NOD1FlightRun) -> Tuple[Tuple[str, np.ndarray, str, str], ...]:
    arrays = []
    registered_fly_fgs_fixture = _is_registered_fly_fgs_source(
        run.source_metadata
    )
    frozen_browser_fixture = (
        run.source_metadata.get("input_mode")
        == "registered_frozen_browser_nod1_parity"
    )
    if registered_fly_fgs_fixture:
        circuit_spike_provenance = "registered_fly_fgs_fixed_step_circuit_output"
        circuit_trace_provenance = "registered_fly_fgs_fixed_step_voltage_v"
        circuit_timing_provenance = "registered_fly_fgs_fixed_step_timing"
        circuit_timebase_provenance = "registered_fly_fgs_exact_timebase"
    elif run.retinal_frames:
        circuit_spike_provenance = "reduced_nod1_simulated_spike_output"
        circuit_trace_provenance = "reduced_nod1_simulated_trace"
        circuit_timing_provenance = "reduced_nod1_trace_timing"
        circuit_timebase_provenance = "reduced_nod1_exact_timebase"
    elif frozen_browser_fixture:
        circuit_spike_provenance = "registered_frozen_chromium_circuit_output"
        circuit_trace_provenance = "registered_frozen_chromium_browser_voltage_v"
        circuit_timing_provenance = "registered_frozen_chromium_readout_timing"
        circuit_timebase_provenance = "registered_frozen_chromium_exact_timebase"
    else:
        circuit_spike_provenance = "legacy_nod1_exact_export"
        circuit_trace_provenance = "legacy_nod1_simulated_trace"
        circuit_timing_provenance = "legacy_nod1_trace_timing"
        circuit_timebase_provenance = "legacy_nod1_exact_timebase"
    if run.retinal_frames:
        arrays.extend(
            (
                (
                    "retinal_normalized_luminance",
                    np.asarray([frame.samples for frame in run.retinal_frames]),
                    "1",
                    "causal_retinal_frame",
                ),
                (
                    "retinal_measurement_time_s",
                    np.asarray(
                        [frame.measurement_time_s for frame in run.retinal_frames]
                    ),
                    "s",
                    "causal_retinal_frame_timing",
                ),
                (
                    "retinal_availability_time_s",
                    np.asarray(
                        [frame.availability_time_s for frame in run.retinal_frames]
                    ),
                    "s",
                    "causal_retinal_frame_timing",
                ),
                (
                    "retinal_exposure_interval_s",
                    np.asarray(
                        [
                            (frame.exposure_start_s, frame.exposure_end_s)
                            for frame in run.retinal_frames
                        ]
                    ),
                    "s",
                    "causal_retinal_frame_timing",
                ),
            )
        )
    if registered_fly_fgs_fixture:
        arrays.extend(_fly_fgs_replay_scientific_arrays(run.circuit_replay))
    for signal in run.circuit.signals:
        if signal.signal_kind is CircuitSignalKind.SPIKE_EVENTS:
            availability_times_s = (
                signal.availability_times_s or signal.spike_times_s
            )
            arrays.extend(
                (
                    (
                        "circuit_spike_times_s/%s" % signal.neuron.entity_id,
                        np.asarray(signal.spike_times_s, dtype=float),
                        "s",
                        circuit_spike_provenance,
                    ),
                    (
                        "circuit_spike_availability_time_s/%s"
                        % signal.neuron.entity_id,
                        np.asarray(availability_times_s, dtype=float),
                        "s",
                        circuit_timing_provenance,
                    ),
                )
            )
        else:
            availability_times_s = (
                signal.availability_times_s or run.circuit.sample_times_s
            )
            arrays.append(
                (
                    "circuit_%s/%s" % (signal.signal_kind.value, signal.neuron.entity_id),
                    np.asarray(signal.values, dtype=float),
                    signal.unit,
                    circuit_trace_provenance,
                )
            )
            arrays.append(
                (
                    "circuit_availability_time_s/%s" % signal.neuron.entity_id,
                    np.asarray(availability_times_s, dtype=float),
                    "s",
                    circuit_timing_provenance,
                )
            )

    arrays.append(
        (
            "circuit_sample_time_s",
            np.asarray(run.circuit.sample_times_s, dtype=float),
            "s",
            circuit_timebase_provenance,
        )
    )
    for channel in run.bridge.descending.channels:
        key = "%s/%s" % (channel.dn_type, channel.side.value)
        arrays.extend(
            (
                (
                    "descending_rate_hz/%s" % key,
                    np.asarray([sample.rate_hz for sample in channel.samples]),
                    "Hz",
                    "exploratory_circuit_to_dn_model_inference",
                ),
                (
                    "descending_measurement_time_s/%s" % key,
                    np.asarray([sample.measurement_time_s for sample in channel.samples]),
                    "s",
                    "causal_bridge_timing",
                ),
                (
                    "descending_availability_time_s/%s" % key,
                    np.asarray([sample.availability_time_s for sample in channel.samples]),
                    "s",
                    "causal_bridge_timing",
                ),
            )
        )
    for channel in run.bridge.wing_motor.channels:
        key = "%s/%s" % (channel.motor_neuron, channel.side.value)
        arrays.extend(
            (
                (
                    "wing_motor_rate_hz/%s" % key,
                    np.asarray([sample.rate_hz for sample in channel.rate_samples]),
                    "Hz",
                    "exploratory_vnc_model_inference",
                ),
                (
                    "wing_motor_rate_measurement_time_s/%s" % key,
                    np.asarray(
                        [sample.measurement_time_s for sample in channel.rate_samples]
                    ),
                    "s",
                    "causal_vnc_rate_timing",
                ),
                (
                    "wing_motor_rate_availability_time_s/%s" % key,
                    np.asarray(
                        [sample.availability_time_s for sample in channel.rate_samples]
                    ),
                    "s",
                    "causal_vnc_rate_timing",
                ),
                (
                    "wing_motor_event_measurement_time_s/%s" % key,
                    np.asarray([event.event_time_s for event in channel.events]),
                    "s",
                    "seeded_synthetic_event_measurement_time",
                ),
                (
                    "wing_motor_event_availability_time_s/%s" % key,
                    np.asarray([event.availability_time_s for event in channel.events]),
                    "s",
                    "seeded_synthetic_event_actuation_time",
                ),
                (
                    "wing_motor_event_phase_rad/%s" % key,
                    np.asarray([event.wingbeat_phase_rad for event in channel.events]),
                    "rad",
                    "seeded_synthetic_phase_model",
                ),
            )
        )
    return tuple(arrays)


def _fly_fgs_replay_scientific_arrays(
    replay: Mapping[str, Any],
) -> Tuple[Tuple[str, np.ndarray, str, str], ...]:
    """Extract source-of-record fly-FGS matrices for chunked artifacts.

    The browser attachment remains a convenient, self-contained replay.  These
    arrays are its authoritative numeric counterpart: they make the complete
    circuit state and exact source stimulus independently hashable/readable
    without ever widening the four-channel motor-input cut.
    """

    def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("fly-FGS %s must be an object" % label)
        return value

    def numeric(value: Any, label: str, *, ndim: Optional[int] = None) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("fly-FGS %s must be numeric" % label) from exc
        if ndim is not None and array.ndim != ndim:
            raise ValueError("fly-FGS %s must have %d dimensions" % (label, ndim))
        if array.ndim == 0 or not np.all(np.isfinite(array)):
            raise ValueError("fly-FGS %s must be a finite array" % label)
        return array

    if not replay:
        raise ValueError("registered fly-FGS run is missing its circuit replay")
    scope = require_mapping(
        replay.get("full_cell_state_scope"), "full_cell_state_scope"
    )
    if bool(scope.get("eligible_motor_input")):
        raise ValueError("fly-FGS full-cell replay must never be motor eligible")
    # Lightweight manufactured interface fixtures can explicitly omit the rich
    # display state. Registered production intake always sets ``included`` and
    # is consequently checked and materialized in full below.
    if not bool(scope.get("included")):
        return ()
    fixed_step = require_mapping(replay.get("fixed_step"), "fixed_step")
    time_s = numeric(fixed_step.get("time_s"), "fixed_step.time_s", ndim=1)
    sample_count = int(fixed_step.get("sample_count", -1))
    if time_s.shape != (sample_count,):
        raise ValueError("fly-FGS fixed-step time axis does not match sample_count")

    result = [
        (
            "fly_fgs_source_time_s",
            time_s,
            "s",
            "registered_fly_fgs_fixed_step_source_clock",
        )
    ]
    full_state = require_mapping(replay.get("full_cell_state"), "full_cell_state")
    voltage = numeric(
        full_state.get("voltage_v"), "full_cell_state.voltage_v", ndim=2
    )
    activity = numeric(
        full_state.get("activity"), "full_cell_state.activity", ndim=2
    )
    if voltage.shape != activity.shape or voltage.shape[0] != sample_count:
        raise ValueError("fly-FGS full-cell matrices do not align with source time")
    cell_ids = full_state.get("cell_ids")
    if not isinstance(cell_ids, list) or len(cell_ids) != voltage.shape[1]:
        raise ValueError("fly-FGS full-cell ID axis does not align with matrices")
    result.extend(
        (
            (
                "fly_fgs_full_cell_voltage_v",
                voltage,
                "V",
                "registered_fly_fgs_full_cell_display_state_not_motor_input",
            ),
            (
                "fly_fgs_full_cell_activity",
                activity,
                "1",
                "registered_fly_fgs_full_cell_display_state_not_motor_input",
            ),
        )
    )

    pooled = require_mapping(
        replay.get("pooled_readout_traces"), "pooled_readout_traces"
    )
    for trace_name in sorted(pooled):
        channels = require_mapping(pooled[trace_name], "pooled trace %s" % trace_name)
        unit = "V" if str(trace_name).endswith("_voltage_v") else "1"
        for raw_side in ("raw_app_L", "raw_app_R"):
            values = numeric(
                channels.get(raw_side),
                "pooled trace %s.%s" % (trace_name, raw_side),
                ndim=1,
            )
            if values.shape != (sample_count,):
                raise ValueError("fly-FGS pooled trace does not align with source time")
            result.append(
                (
                    "fly_fgs_pooled/%s/%s" % (trace_name, raw_side),
                    values,
                    unit,
                    "registered_fly_fgs_app_side_label_anatomy_unknown",
                )
            )

    stimulus = require_mapping(replay.get("stimulus"), "stimulus")
    retinal_input = require_mapping(stimulus.get("retinal_input"), "retinal_input")
    retinal_luminance = numeric(
        retinal_input.get("luminance"), "retinal_input.luminance", ndim=2
    )
    retinal_azimuth_rad = np.deg2rad(
        numeric(retinal_input.get("azimuth_deg"), "retinal_input.azimuth_deg", ndim=1)
    )
    retinal_elevation_rad = np.deg2rad(
        numeric(
            retinal_input.get("elevation_deg"),
            "retinal_input.elevation_deg",
            ndim=1,
        )
    )
    retinal_indices = numeric(
        retinal_input.get("cell_indices"), "retinal_input.cell_indices", ndim=1
    )
    receptor_count = retinal_luminance.shape[1]
    if (
        retinal_luminance.shape[0] != sample_count
        or retinal_azimuth_rad.shape != (receptor_count,)
        or retinal_elevation_rad.shape != (receptor_count,)
        or retinal_indices.shape != (receptor_count,)
    ):
        raise ValueError("fly-FGS T4a input axes do not align with luminance")
    result.extend(
        (
            (
                "fly_fgs_t4a_input_luminance",
                retinal_luminance,
                "1",
                "registered_fly_fgs_exact_t4a_analytic_input",
            ),
            (
                "fly_fgs_t4a_input_azimuth_rad",
                retinal_azimuth_rad,
                "rad",
                "registered_fly_fgs_retinotopic_coordinate",
            ),
            (
                "fly_fgs_t4a_input_elevation_rad",
                retinal_elevation_rad,
                "rad",
                "registered_fly_fgs_retinotopic_coordinate",
            ),
            (
                "fly_fgs_t4a_input_cell_index",
                retinal_indices,
                "index",
                "registered_fly_fgs_bundle_cell_axis",
            ),
        )
    )

    display = require_mapping(stimulus.get("retinal_display"), "retinal_display")
    display_luminance = numeric(
        display.get("luminance"), "retinal_display.luminance", ndim=2
    )
    display_azimuth_rad = np.deg2rad(
        numeric(display.get("azimuth_deg"), "retinal_display.azimuth_deg", ndim=1)
    )
    if (
        display_luminance.shape[0] != sample_count
        or display_luminance.shape[1] != display_azimuth_rad.shape[0]
    ):
        raise ValueError("fly-FGS display strip axes do not align")
    result.extend(
        (
            (
                "fly_fgs_display_luminance",
                display_luminance,
                "1",
                "registered_fly_fgs_display_decimation",
            ),
            (
                "fly_fgs_display_azimuth_rad",
                display_azimuth_rad,
                "rad",
                "registered_fly_fgs_display_coordinate",
            ),
        )
    )

    schedule = require_mapping(stimulus.get("schedule"), "stimulus.schedule")
    for source_name in sorted(schedule):
        values = numeric(schedule[source_name], "stimulus.schedule.%s" % source_name, ndim=1)
        if values.shape != (sample_count,):
            raise ValueError("fly-FGS stimulus schedule does not align with source time")
        if source_name.endswith("_deg_s"):
            name = source_name[: -len("_deg_s")] + "_rad_s"
            values = np.deg2rad(values)
            unit = "rad/s"
        elif source_name.endswith("_deg"):
            name = source_name[: -len("_deg")] + "_rad"
            values = np.deg2rad(values)
            unit = "rad"
        else:
            name = source_name
            unit = "1"
        result.append(
            (
                "fly_fgs_stimulus/%s" % name,
                values,
                unit,
                "registered_fly_fgs_exact_stimulus_schedule",
            )
        )
    return tuple(result)


def _pipeline_manifest(run: NOD1FlightRun) -> Mapping[str, Any]:
    source_has_retinal = bool(run.retinal_frames)
    registered_fly_fgs_fixture = _is_registered_fly_fgs_source(
        run.source_metadata
    )
    frozen_browser_fixture = (
        run.source_metadata.get("input_mode")
        == "registered_frozen_browser_nod1_parity"
    )
    circuit_replay_contract: Optional[Mapping[str, Any]] = None
    if registered_fly_fgs_fixture:
        pipeline_id = "fly-fgs-retina-circuit-to-flight-v1"
        stages = [
            "registered_fly_fgs_visual_scene",
            "registered_fly_fgs_fixed_step_neural_circuit",
            "evidence_locked_nod1_circuit_cut",
            "contralateral_dnp26_encoder",
            "vnc_wing_motor_surrogate",
            "individual_steering_muscles",
            "virtual_six_axis_hinge",
            "selected_physics_backend",
        ]
        visual_notice = (
            "The registered fixed-step fly-FGS visual/circuit source is preserved "
            "for replay and drives the existing evidence-qualified DNp26/VNC path. "
            "The fly-FGS muscle, prescribed-wing, yaw, and body-motion equations are "
            "excluded; neuromuscular actuation and mechanics come from this project."
        )
        pipeline_mode = "open_loop_registered_fly_fgs_fixed_step_circuit_output"
        visual_status = (
            "registered_retinal_frames_and_circuit_replay"
            if source_has_retinal
            else "registered_visual_scene_and_circuit_replay"
        )
        circuit_replay_contract = {
            "attachment": "web_replay.circuit_replay",
            "scientific_array_prefix": "fly_fgs_",
            "source_time_array": "fly_fgs_source_time_s",
            "full_cell_voltage_array": "fly_fgs_full_cell_voltage_v",
            "full_cell_activity_array": "fly_fgs_full_cell_activity",
            "t4a_input_luminance_array": "fly_fgs_t4a_input_luminance",
            "pooled_trace_array_prefix": "fly_fgs_pooled/",
            "stimulus_schedule_array_prefix": "fly_fgs_stimulus/",
            "cell_axis": "web_replay.circuit_replay.circuit_topology.cell_axis",
            "motor_input_policy": "four CircuitOutputTrace NOD1 voltage channels only",
            "full_cell_state_eligible_motor_input": False,
            "raw_app_side_labels_are_anatomical": False,
        }
    elif source_has_retinal:
        pipeline_id = "causal-retinal-reduced-nod1-to-dnp26-to-flight-v1"
        stages = [
            "causal_finite_exposure_retinal_frames",
            "reduced_nod1_motion_surrogate",
            "contralateral_dnp26_encoder",
            "vnc_wing_motor_surrogate",
            "individual_steering_muscles",
            "virtual_six_axis_hinge",
            "selected_physics_backend",
        ]
        visual_notice = (
            "Source-of-record finite-exposure RetinalFrame samples and their "
            "measurement/availability timing are preserved in this artifact. "
            "The retinal sampler and reduced NOD1 circuit are uncalibrated "
            "software surrogates, not recorded neural data."
        )
        pipeline_mode = "open_loop_causal_retinal_frames"
        visual_status = "provided"
    elif frozen_browser_fixture:
        pipeline_id = "frozen-browser-nod1-parity-to-dnp26-to-flight-v1"
        stages = [
            "registered_frozen_chromium_browser_voltage_readouts",
            "contralateral_dnp26_encoder",
            "vnc_wing_motor_surrogate",
            "individual_steering_muscles",
            "virtual_six_axis_hinge",
            "selected_physics_backend",
        ]
        visual_notice = (
            "This historical registered fixture carries real-Chromium visual-circuit voltage "
            "output and exact figure-ground stimulus/configuration metadata. It "
            "contains no source retinal frames and does not establish a calibrated "
            "visual model, motor validity, or biological validity."
        )
        pipeline_mode = "open_loop_frozen_browser_visual_circuit_output"
        visual_status = "frozen_browser_output_no_retinal_frames"
    else:
        pipeline_id = "legacy-nod1-to-dnp26-to-flight-v1"
        stages = [
            "saved_legacy_nod1_voltage",
            "contralateral_dnp26_encoder",
            "vnc_wing_motor_surrogate",
            "individual_steering_muscles",
            "virtual_six_axis_hinge",
            "selected_physics_backend",
        ]
        visual_notice = (
            "The audited legacy result exports circuit voltage but not source-of-record "
            "retinal frames. Visual stimulus provenance must be carried in source metadata "
            "until the NOD1 exporter emits RetinalFrame artifacts."
        )
        pipeline_mode = "open_loop_saved_circuit_trace"
        visual_status = "unavailable_in_legacy_result"
    return {
        "pipeline_id": pipeline_id,
        "mode": pipeline_mode,
        "circuit_trace_sha256": run.circuit_trace_sha256,
        "circuit_output_contract": {
            "schema_version": run.circuit.schema_version,
            "dataset": run.circuit.dataset.to_dict(),
            "dataset_identity_space": run.circuit.dataset.identity_space,
            "exact_timebase": bool(run.circuit.exact_timebase),
            "sample_count": len(run.circuit.sample_times_s),
            "sample_time_array": "circuit_sample_time_s",
            "signals": [
                {
                    "neuron": signal.neuron.to_dict(),
                    "signal_kind": signal.signal_kind.value,
                    "unit": signal.unit,
                    "origin": signal.origin.value,
                    "confidence": signal.confidence.to_dict(),
                    "provenance": signal.provenance.to_dict(),
                    "value_array": (
                        "circuit_spike_times_s/%s" % signal.neuron.entity_id
                        if signal.signal_kind is CircuitSignalKind.SPIKE_EVENTS
                        else "circuit_%s/%s"
                        % (signal.signal_kind.value, signal.neuron.entity_id)
                    ),
                    "availability_time_array": (
                        "circuit_spike_availability_time_s/%s"
                        % signal.neuron.entity_id
                        if signal.signal_kind is CircuitSignalKind.SPIKE_EVENTS
                        else "circuit_availability_time_s/%s"
                        % signal.neuron.entity_id
                    ),
                }
                for signal in run.circuit.signals
            ],
        },
        "identity_boundary": {
            "circuit_source_identity_space": run.bridge.descending.source_dataset_identity,
            "downstream_identity_space": "simulation:exploratory-nod1-dnp26-bridge-v1",
            "join_policy": (
                "cell-type/side model only; FlyWire, BANC/FANC, and simulation "
                "identifiers are never directly joined"
            ),
        },
        "stages": stages,
        "visual_boundary": {
            "retinal_frames_present": source_has_retinal,
            "status": visual_status,
            "notice": visual_notice,
        },
        "circuit_replay_contract": circuit_replay_contract,
        "motor_semantics": "inferred_rate_with_seeded_synthetic_phase_events",
        "flight_state_drive": (
            "explicit_model_baseline_power_and_tension"
            if any(
                command.muscle_class is not MuscleClass.STEERING
                for command in run.flight_config.motor_commands
            )
            else "none"
        ),
        "source_metadata": dict(run.source_metadata),
        "scientific_status": "exploratory_uncalibrated",
    }


def write_nod1_flight_artifact(
    output_dir: Path,
    run: NOD1FlightRun,
    *,
    chunk_samples: int = 256,
    created_at_utc: Optional[str] = None,
    web_replay: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Write one immutable artifact containing every available pipeline stage."""

    return write_episode_artifact(
        output_dir,
        "nod1_visual_circuit_to_flight",
        run.flight_config,
        run.flight,
        chunk_samples=chunk_samples,
        created_at_utc=created_at_utc,
        supplemental_arrays=_pipeline_arrays(run),
        identity_context=nod1_run_identity_context(run),
        pipeline_manifest=_pipeline_manifest(run),
        web_replay=web_replay,
    )


def nod1_run_to_web_replay(
    run: NOD1FlightRun,
    *,
    target_sample_rate_hz: float = 200.0,
) -> Mapping[str, Any]:
    """Create a browser replay with attached raw NOD1/DN/MN/muscle traces.

    A saved legacy result is explicitly identified as an eight-cable-neuron
    export produced by the frozen 1,208-cell browser circuit.  A retinal run is
    explicitly identified as the four-readout reduced surrogate.  Those scopes
    are never treated as interchangeable in the interface.
    """

    registered_fly_fgs_fixture = _is_registered_fly_fgs_source(
        run.source_metadata
    )
    reduced = bool(run.retinal_frames) and not registered_fly_fgs_fixture
    frozen_browser_fixture = (
        run.source_metadata.get("input_mode")
        == "registered_frozen_browser_nod1_parity"
    )
    flybody_physics = run.flight.diagnostics.physics_backend == "flybody"
    if registered_fly_fgs_fixture:
        cell_count = int(run.source_metadata.get("circuit_cell_count", 1684))
        formatted_cell_count = format(cell_count, ",")
        scope = {
            "kind": "registered_fly_fgs_fixed_step_circuit",
            "label": "fly-FGS visual circuit · %s cells" % formatted_cell_count,
            "full_circuit_executed": True,
            "circuit_cell_count": cell_count,
            "exported_circuit_channel_count": len(run.circuit.signals),
            "notice": (
                "The registered fly-FGS visual and neural outputs are retained for "
                "inspection. Its muscle scaling, prescribed wing kinematics, yaw, "
                "and body animation are excluded; the displayed behavior is produced "
                "by the evidence-locked bridge and this project's flight mechanics."
            ),
        }
        episode_id = "fly_fgs_canonical"
        label = (
            "fly-FGS visual circuit → FlyBody"
            if flybody_physics
            else "fly-FGS visual circuit → reduced flight"
        )
        source_kind = (
            "exploratory_fly_fgs_circuit_flybody_pipeline"
            if flybody_physics
            else "exploratory_fly_fgs_circuit_reduced_pipeline"
        )
        stimulus_value = run.source_metadata.get(
            "stimulus",
            run.source_metadata.get("scene_schedule", "registered fly-FGS scene"),
        )
        stimulus = (
            json.dumps(stimulus_value, sort_keys=True, allow_nan=False)
            if isinstance(stimulus_value, (Mapping, list, tuple))
            else str(stimulus_value)
        )
    elif reduced:
        scope = {
            "kind": "reduced_nod1_surrogate",
            "label": "Reduced retinal-motion → four NOD1 readouts",
            "full_circuit_executed": False,
            "circuit_cell_count": None,
            "exported_circuit_channel_count": len(run.circuit.signals),
            "notice": (
                "This episode uses the causal image-derived reduced NOD1 surrogate, "
                "not the frozen 1,208-cell browser circuit."
            ),
        }
        episode_id = "retinal_reduced_nod1"
        label = (
            "Retina → reduced NOD1 surrogate → FlyBody"
            if flybody_physics
            else "Retina → reduced NOD1 surrogate → reduced flight"
        )
        source_kind = (
            "exploratory_retinal_nod1_flybody_pipeline"
            if flybody_physics
            else "exploratory_reduced_retinal_nod1_pipeline"
        )
        stimulus = "%d causal finite-exposure retinal frames" % len(run.retinal_frames)
    elif frozen_browser_fixture:
        cell_count = int(run.source_metadata["circuit_cell_count"])
        formatted_cell_count = format(cell_count, ",")
        scope = {
            "kind": "full_legacy_circuit_cable_export",
            "historical": True,
            "label": "Historical frozen real-Chromium %s-cell circuit · four NOD1 browser readouts"
            % formatted_cell_count,
            "full_circuit_executed": True,
            "circuit_cell_count": cell_count,
            "exported_circuit_channel_count": len(run.circuit.signals),
            "notice": (
                "This historical regression episode imports exactly four browser_voltage_v channels from "
                "the SHA-locked real-Chromium parity fixture. Python comparison "
                "arrays and deprecated motor proxies never enter the bridge."
            ),
        }
        episode_id = "frozen_browser_nod1"
        label = (
            "Frozen browser NOD1 circuit → FlyBody"
            if flybody_physics
            else "Frozen browser NOD1 circuit → reduced flight"
        )
        source_kind = (
            "exploratory_frozen_browser_nod1_flybody_pipeline"
            if flybody_physics
            else "exploratory_frozen_browser_nod1_pipeline"
        )
        stimulus_value = run.source_metadata["stimulus"]
        stimulus = json.dumps(stimulus_value, sort_keys=True)
    else:
        cell_count = int(run.source_metadata.get("circuit_cell_count", 1208))
        formatted_cell_count = format(cell_count, ",")
        scope = {
            "kind": "full_legacy_circuit_cable_export",
            "label": "Frozen %s-cell browser circuit · cable-neuron export" % formatted_cell_count,
            "full_circuit_executed": True,
            "circuit_cell_count": cell_count,
            "exported_circuit_channel_count": len(run.circuit.signals),
            "notice": (
                "The saved result was produced by the frozen %s-cell browser circuit; "
                "this replay carries only its exported vCH/DCH/NOD1 cable-neuron voltages."
                % formatted_cell_count
            ),
        }
        episode_id = "saved_nod1_circuit"
        label = (
            "Saved 1,208-cell NOD1 circuit → FlyBody"
            if flybody_physics
            else "Saved 1,208-cell NOD1 circuit → reduced flight"
        )
        source_kind = (
            "exploratory_saved_legacy_nod1_flybody_pipeline"
            if flybody_physics
            else "exploratory_saved_legacy_nod1_pipeline"
        )
        stimulus_value = run.source_metadata.get("stimulus", "not embedded in saved result")
        stimulus = (
            json.dumps(stimulus_value, sort_keys=True)
            if isinstance(stimulus_value, Mapping)
            else str(stimulus_value)
        )

    replay = dict(
        episode_to_web_replay(
            "baseline",
            run.flight_config,
            run.flight,
            target_sample_rate_hz=target_sample_rate_hz,
            circuit_trace=run.circuit,
            bridge_result=run.bridge,
            retinal_frames=run.retinal_frames,
            neural_model_scope=scope,
        )
    )
    replay.update(
        {
            "id": episode_id,
            "label": label,
            "source_kind": source_kind,
            "stimulus": stimulus,
            "perturbation": "encoded in the executed bridge trace; none inferred by replay",
            "source_run_id": episode_run_id(
                "nod1_visual_circuit_to_flight",
                run.flight_config,
                identity_context=nod1_run_identity_context(run),
            ),
            "source_circuit_trace_sha256": run.circuit_trace_sha256,
        }
    )
    if run.circuit_replay:
        replay["circuit_replay"] = _json_safe_mapping(
            run.circuit_replay,
            label="circuit_replay",
        )
    return replay


def write_nod1_web_replay(
    output_path: Path,
    run: NOD1FlightRun,
    *,
    target_sample_rate_hz: float = 200.0,
    overwrite: bool = False,
    source_artifact_manifest_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    """Write one pipeline-derived browser episode next to its run artifact."""

    replay = nod1_run_to_web_replay(
        run,
        target_sample_rate_hz=target_sample_rate_hz,
    )
    if source_artifact_manifest_path is not None:
        manifest_path = Path(source_artifact_manifest_path)
        artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if artifact_manifest.get("run_id") != replay["source_run_id"]:
            raise ValueError(
                "web replay source_run_id does not match its artifact manifest"
            )
        replay["source_artifact_manifest_sha256"] = sha256_file(manifest_path)
        replay["source_artifact_schema_version"] = artifact_manifest.get(
            "schema_version"
        )
    write_web_replay_episode(output_path, replay, overwrite=overwrite)
    return replay


def load_legacy_nod1_result(path: Path) -> Mapping[str, Any]:
    """Load a legacy JSON result without accepting a non-object top level."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("legacy NOD1 result must be a JSON object")
    return payload


__all__ = [
    "NOD1FlightPipelineConfig",
    "NOD1FlightRun",
    "load_legacy_nod1_result",
    "nod1_run_identity_context",
    "run_nod1_flight_pipeline",
    "run_registered_fly_fgs_flight_pipeline",
    "run_registered_nod1_browser_flight_pipeline",
    "run_retinal_flight_pipeline",
    "nod1_run_to_web_replay",
    "write_nod1_flight_artifact",
    "write_nod1_web_replay",
]
