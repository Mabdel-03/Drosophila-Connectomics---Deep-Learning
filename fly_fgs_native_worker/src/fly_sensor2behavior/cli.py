"""Command-line entry points for evidence audits, episodes, and replay export."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .artifacts import (
    SCENARIO_NAMES,
    audit_evidence,
    export_web_replay,
    run_scenario,
    write_episode_artifact,
)
from .evaluators import default_evaluators, sha256_file
from .validation import (
    GateStatus,
    SourceDigest,
    ValidationReport,
    ValidationRunner,
    default_validation_source_digests,
    default_benchmark_registry_path,
    load_benchmark_registry,
)


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _add_simulation_clock_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--duration-s", type=float, default=0.200)
    parser.add_argument("--physics-dt-s", type=float, default=0.0001)
    parser.add_argument("--logging-dt-s", type=float, default=0.001)
    parser.add_argument("--neural-dt-s", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)


def _add_pipeline_physics_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--physics-backend",
        choices=("reduced-order", "flybody"),
        default="reduced-order",
        help=(
            "select dependency-light exploratory mechanics or the pinned "
            "FlyBody worker; flybody requires the Python 3.12 worker image"
        ),
    )
    parser.add_argument(
        "--flybody-spawn-height-m",
        type=float,
        default=0.100,
        help="airborne FlyBody spawn height used only by --physics-backend flybody",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fly-s2b",
        description="Evidence-locked Drosophila neural-to-flight research simulator",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit-evidence", help="validate the evidence graph and report provisional coverage"
    )
    audit.add_argument("--evidence", type=Path, default=None)
    audit.set_defaults(handler=_handle_audit_evidence)

    simulate = subparsers.add_parser(
        "simulate", help="run one exploratory reduced-order episode and write hashed arrays"
    )
    simulate.add_argument("scenario", choices=SCENARIO_NAMES)
    simulate.add_argument("--output", type=Path, required=True)
    simulate.add_argument("--evidence", type=Path, default=None)
    simulate.add_argument("--chunk-samples", type=int, default=256)
    _add_simulation_clock_arguments(simulate)
    simulate.set_defaults(handler=_handle_simulate)

    simulate_nod1 = subparsers.add_parser(
        "simulate-nod1",
        help="run a saved legacy NOD1 voltage result through the causal flight pipeline",
    )
    simulate_nod1.add_argument("--input", type=Path, required=True)
    simulate_nod1.add_argument("--output", type=Path, required=True)
    simulate_nod1.add_argument("--physics-dt-s", type=float, default=0.0001)
    simulate_nod1.add_argument("--logging-dt-s", type=float, default=0.001)
    simulate_nod1.add_argument("--neural-dt-s", type=float, default=0.005)
    simulate_nod1.add_argument("--seed", type=int, default=0)
    simulate_nod1.add_argument("--chunk-samples", type=int, default=256)
    simulate_nod1.add_argument(
        "--web-replay-output",
        type=Path,
        default=None,
        help="also write a decimated episode JSON with attached raw neural channels",
    )
    simulate_nod1.add_argument("--web-replay-sample-rate-hz", type=float, default=200.0)
    _add_pipeline_physics_arguments(simulate_nod1)
    simulate_nod1.set_defaults(handler=_handle_simulate_nod1)

    simulate_nod1_fixture = subparsers.add_parser(
        "simulate-nod1-fixture",
        help=(
            "run the historical SHA-locked Chromium NOD1 parity fixture through the "
            "causal flight pipeline"
        ),
    )
    simulate_nod1_fixture.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="benchmark registry that authoritatively supplies the fixture URI and SHA-256",
    )
    simulate_nod1_fixture.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help=(
            "optional local copy of the registered compressed fixture; its bytes "
            "must match the registry SHA-256"
        ),
    )
    simulate_nod1_fixture.add_argument("--output", type=Path, required=True)
    simulate_nod1_fixture.add_argument("--physics-dt-s", type=float, default=0.0001)
    simulate_nod1_fixture.add_argument("--logging-dt-s", type=float, default=0.001)
    simulate_nod1_fixture.add_argument("--neural-dt-s", type=float, default=0.005)
    simulate_nod1_fixture.add_argument("--seed", type=int, default=0)
    simulate_nod1_fixture.add_argument("--chunk-samples", type=int, default=256)
    simulate_nod1_fixture.add_argument(
        "--web-replay-output",
        type=Path,
        default=None,
        help="also write a decimated replay with the four browser voltage channels",
    )
    simulate_nod1_fixture.add_argument(
        "--web-replay-sample-rate-hz", type=float, default=200.0
    )
    _add_pipeline_physics_arguments(simulate_nod1_fixture)
    simulate_nod1_fixture.set_defaults(handler=_handle_simulate_nod1_fixture)

    simulate_fly_fgs_fixture = subparsers.add_parser(
        "simulate-fly-fgs-fixture",
        help=(
            "run the registered fixed-step fly-FGS visual circuit through the "
            "project's DNp26/VNC/muscle/flight pipeline"
        ),
    )
    simulate_fly_fgs_fixture.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="optional local copy of the content-addressed fly-FGS source manifest",
    )
    simulate_fly_fgs_fixture.add_argument(
        "--capture",
        type=Path,
        default=None,
        help="optional local copy of the registered compressed fixed-step capture",
    )
    simulate_fly_fgs_fixture.add_argument("--output", type=Path, required=True)
    simulate_fly_fgs_fixture.add_argument(
        "--physics-dt-s", type=float, default=0.0001
    )
    simulate_fly_fgs_fixture.add_argument(
        "--logging-dt-s", type=float, default=0.001
    )
    simulate_fly_fgs_fixture.add_argument(
        "--neural-dt-s", type=float, default=0.005
    )
    simulate_fly_fgs_fixture.add_argument("--seed", type=int, default=0)
    simulate_fly_fgs_fixture.add_argument(
        "--chunk-samples", type=int, default=256
    )
    simulate_fly_fgs_fixture.add_argument(
        "--web-replay-output",
        type=Path,
        default=None,
        help=(
            "also write the canonical replay with synchronized fly-FGS circuit "
            "display data and project-generated mechanics"
        ),
    )
    simulate_fly_fgs_fixture.add_argument(
        "--web-replay-sample-rate-hz", type=float, default=200.0
    )
    _add_pipeline_physics_arguments(simulate_fly_fgs_fixture)
    simulate_fly_fgs_fixture.set_defaults(
        handler=_handle_simulate_fly_fgs_fixture
    )

    simulate_canonical_online = subparsers.add_parser(
        "simulate-canonical-online",
        help=(
            "run the live canonical fly-FGS circuit, causal streaming wing muscles, "
            "and pinned FlyBody physics, then publish an immutable multi-clock artifact"
        ),
    )
    simulate_canonical_online.add_argument("--output", type=Path, required=True)
    simulate_canonical_online.add_argument(
        "--scenario-id",
        default="canonical-online",
        help="stable lowercase ID used in the run identity and browser episode",
    )
    simulate_canonical_online.add_argument(
        "--label",
        default="Canonical online fly-FGS → FlyBody",
        help="human-readable browser episode label",
    )
    simulate_canonical_online.add_argument(
        "--web-replay-output",
        type=Path,
        default=None,
        help="optional derived browser JSON; it must be outside the immutable run directory",
    )
    simulate_canonical_online.add_argument("--duration-s", type=float, default=0.020)
    simulate_canonical_online.add_argument("--seed", type=int, default=73)
    simulate_canonical_online.add_argument(
        "--flybody-spawn-height-m", type=float, default=0.100
    )
    simulate_canonical_online.add_argument(
        "--effector-hypothesis",
        choices=("raw_l_to_physical_left", "raw_l_to_physical_right"),
        required=True,
        help=(
            "required unresolved physical-wing hypothesis; raw fly-FGS L/R are not "
            "anatomical sides and signed yaw/roll claims remain prohibited"
        ),
    )
    simulate_canonical_online.add_argument(
        "--figure-initial-world-azimuth-rad", type=float, default=0.0
    )
    simulate_canonical_online.add_argument(
        "--figure-velocity-rad-s", type=float, default=0.0
    )
    simulate_canonical_online.add_argument(
        "--ground-velocity-rad-s", type=float, default=0.0
    )
    simulate_canonical_online.add_argument(
        "--omit-full-cell-state",
        action="store_true",
        help="omit the 1,684-cell display/audit state; four NOD1 motor channels remain",
    )
    simulate_canonical_online.add_argument(
        "--muscle-interventions",
        type=Path,
        default=None,
        help=(
            "JSON array of force-stage iv2/i1/iv1/b3 SILENCE or SCALE contracts; "
            "times must lie on the 0.1 ms grid"
        ),
    )
    simulate_canonical_online.add_argument("--chunk-samples", type=int, default=256)
    simulate_canonical_online.set_defaults(handler=_handle_simulate_canonical_online)

    simulate_retinal = subparsers.add_parser(
        "simulate-retinal",
        help=(
            "run a causal analytic grating through retinal samples, the explicitly "
            "reduced NOD1 surrogate, muscles, and mechanics"
        ),
    )
    simulate_retinal.add_argument("--output", type=Path, required=True)
    simulate_retinal.add_argument("--duration-s", type=float, default=0.100)
    simulate_retinal.add_argument("--exposure-dt-s", type=float, default=0.001)
    simulate_retinal.add_argument("--sensor-latency-s", type=float, default=0.0005)
    simulate_retinal.add_argument("--receptors", type=int, default=64)
    simulate_retinal.add_argument("--angular-velocity-rad-s", type=float, default=1.0)
    simulate_retinal.add_argument("--phase-rad", type=float, default=0.0)
    simulate_retinal.add_argument("--physics-dt-s", type=float, default=0.0001)
    simulate_retinal.add_argument("--logging-dt-s", type=float, default=0.001)
    simulate_retinal.add_argument("--neural-dt-s", type=float, default=0.001)
    simulate_retinal.add_argument("--seed", type=int, default=0)
    simulate_retinal.add_argument("--chunk-samples", type=int, default=256)
    simulate_retinal.add_argument(
        "--web-replay-output",
        type=Path,
        default=None,
        help="also write a decimated episode JSON with attached raw neural channels",
    )
    simulate_retinal.add_argument("--web-replay-sample-rate-hz", type=float, default=200.0)
    _add_pipeline_physics_arguments(simulate_retinal)
    simulate_retinal.set_defaults(handler=_handle_simulate_retinal)

    export = subparsers.add_parser(
        "export-web", help="generate decimated static replay JSON for the TypeScript application"
    )
    export.add_argument("--output", type=Path, required=True)
    export.add_argument(
        "--scenarios", nargs="+", choices=SCENARIO_NAMES, default=list(SCENARIO_NAMES)
    )
    export.add_argument("--evidence", type=Path, default=None)
    export.add_argument(
        "--validation-report",
        type=Path,
        default=None,
        help="attach a validated fly-s2b validate JSON report to the static web bundle",
    )
    export.add_argument(
        "--validation-registry",
        type=Path,
        default=None,
        help=(
            "exact benchmark registry used to bind --validation-report; "
            "defaults to the packaged canonical registry"
        ),
    )
    export.add_argument("--sample-rate-hz", type=float, default=200.0)
    export.add_argument(
        "--nod1-result",
        type=Path,
        default=None,
        help="attach a saved frozen-browser NOD1 result as a trace-backed replay episode",
    )
    export.add_argument(
        "--include-reduced-retinal-pipeline",
        action="store_true",
        help="attach a causal analytic-retina/reduced-NOD1 trace-backed episode",
    )
    export.add_argument(
        "--include-registered-nod1-fixture",
        action="store_true",
        help=(
            "attach the SHA-locked real-Chromium 1,208-cell NOD1 fixture as a "
            "trace-backed flight episode"
        ),
    )
    export.add_argument(
        "--nod1-fixture",
        type=Path,
        default=None,
        help="optional local copy of the registered SHA-locked NOD1 fixture",
    )
    export.add_argument(
        "--include-registered-fly-fgs-fixture",
        action="store_true",
        help=(
            "attach the registered fixed-step fly-FGS visual/circuit fixture, "
            "using only its NOD1 voltages as motor-bound input"
        ),
    )
    export.add_argument(
        "--fly-fgs-manifest",
        type=Path,
        default=None,
        help="optional local copy of the registered fly-FGS source manifest",
    )
    export.add_argument(
        "--fly-fgs-capture",
        type=Path,
        default=None,
        help="optional local copy of the registered fly-FGS fixed-step capture",
    )
    export.add_argument("--retinal-receptors", type=int, default=64)
    export.add_argument(
        "--retinal-angular-velocity-rad-s", type=float, default=1.0
    )
    export.add_argument("--retinal-sensor-latency-s", type=float, default=None)
    export.add_argument("--overwrite", action="store_true")
    _add_simulation_clock_arguments(export)
    _add_pipeline_physics_arguments(export)
    export.set_defaults(handler=_handle_export_web)

    smoke = subparsers.add_parser(
        "flybody-smoke",
        help="run an actual pinned FlyGym/FlyBody compatibility smoke test",
    )
    smoke.add_argument(
        "--mode",
        choices=("analytic-wingbeat", "zero-torque"),
        default="analytic-wingbeat",
    )
    smoke.add_argument("--steps", type=int, default=2)
    smoke.add_argument("--duration-s", type=float, default=0.005)
    smoke.add_argument("--wingbeat-hz", type=float, default=218.0)
    smoke.add_argument("--timestep-s", type=float, default=None)
    smoke.add_argument("--output", type=Path, default=None)
    smoke.set_defaults(handler=_handle_flybody_smoke)

    validate = subparsers.add_parser(
        "validate",
        help="run evidence-aware scientific gates and write an immutable canonical report",
    )
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--registry", type=Path, default=None)
    validate.add_argument("--evaluation-id", default="local-validation")
    validate.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        default=None,
        help="select a case ID; repeat to select more than one",
    )
    validate.add_argument(
        "--dependency",
        dest="changed_dependencies",
        action="append",
        default=None,
        help="select cases impacted by a changed dependency key; repeat as needed",
    )
    validate.add_argument(
        "--no-dependents",
        action="store_false",
        dest="include_dependents",
        help="do not include downstream benchmark dependents",
    )
    validate.add_argument(
        "--no-prerequisites",
        action="store_false",
        dest="include_prerequisites",
        help="do not include prerequisite benchmark cases",
    )
    validate.set_defaults(
        handler=_handle_validate,
        include_dependents=True,
        include_prerequisites=True,
    )
    return parser


def _handle_audit_evidence(args: argparse.Namespace) -> int:
    _print_json(audit_evidence(args.evidence))
    return 0


def _handle_simulate(args: argparse.Namespace) -> int:
    config, result = run_scenario(
        args.scenario,
        duration_s=args.duration_s,
        physics_dt_s=args.physics_dt_s,
        logging_dt_s=args.logging_dt_s,
        neural_dt_s=args.neural_dt_s,
        seed=args.seed,
    )
    manifest = write_episode_artifact(
        args.output,
        args.scenario,
        config,
        result,
        evidence_path=args.evidence,
        chunk_samples=args.chunk_samples,
    )
    _print_json(
        {
            "run_id": manifest["run_id"],
            "manifest": str((args.output / "manifest.json").resolve()),
            "validation_status": manifest["validation_status"],
            "calibration": manifest["calibration"]["status"],
            "source_kind": manifest["source_kind"],
            "physics_steps": result.diagnostics.physics_steps,
        }
    )
    return 0


def _pipeline_summary(args: argparse.Namespace, manifest: Mapping[str, Any]) -> int:
    summary = {
        "run_id": manifest["run_id"],
        "manifest": str((args.output / "manifest.json").resolve()),
        "pipeline_mode": manifest["pipeline"]["mode"],
        "visual_boundary": manifest["pipeline"]["visual_boundary"]["status"],
        "validation_status": manifest["validation_status"],
        "scientific_status": manifest["pipeline"]["scientific_status"],
        "physics_backend": manifest["diagnostics"]["physics_backend"],
    }
    if args.web_replay_output is not None:
        summary["web_replay"] = str(args.web_replay_output.resolve())
    _print_json(summary)
    return 0


def _pipeline_physics_runner(args: argparse.Namespace):
    """Construct a worker-only FlyBody runner without importing it on the host."""

    if args.physics_backend == "reduced-order":
        return None
    if args.flybody_spawn_height_m <= 0.0:
        raise ValueError("--flybody-spawn-height-m must be positive")

    from .flight import FlightEpisodeRunner
    from .flybody_adapter import (
        FlyBodyPhysicsAdapter,
        FlyBodyWorkerConfig,
        WingAxisTorqueMap,
    )

    adapter = FlyBodyPhysicsAdapter(
        WingAxisTorqueMap(),
        FlyBodyWorkerConfig(
            timestep_s=args.physics_dt_s,
            spawn_height_m=args.flybody_spawn_height_m,
        ),
    )
    return FlightEpisodeRunner(physics_adapter=adapter)


def _handle_simulate_nod1(args: argparse.Namespace) -> int:
    from .pipeline import (
        NOD1FlightPipelineConfig,
        load_legacy_nod1_result,
        nod1_run_to_web_replay,
        run_nod1_flight_pipeline,
        write_nod1_flight_artifact,
        write_nod1_web_replay,
    )

    result = load_legacy_nod1_result(args.input)
    runner = _pipeline_physics_runner(args)
    run = run_nod1_flight_pipeline(
        result,
        config=NOD1FlightPipelineConfig(
            physics_dt_s=args.physics_dt_s,
            neural_dt_s=args.neural_dt_s,
            logging_dt_s=args.logging_dt_s,
            seed=args.seed,
        ),
        runner=runner,
    )
    replay = (
        None
        if args.web_replay_output is None
        else nod1_run_to_web_replay(
            run, target_sample_rate_hz=args.web_replay_sample_rate_hz
        )
    )
    manifest = write_nod1_flight_artifact(
        args.output,
        run,
        chunk_samples=args.chunk_samples,
        web_replay=replay,
    )
    if args.web_replay_output is not None:
        write_nod1_web_replay(
            args.web_replay_output,
            run,
            target_sample_rate_hz=args.web_replay_sample_rate_hz,
            source_artifact_manifest_path=args.output / "manifest.json",
        )
    return _pipeline_summary(args, manifest)


def _handle_simulate_nod1_fixture(args: argparse.Namespace) -> int:
    from .pipeline import (
        NOD1FlightPipelineConfig,
        nod1_run_to_web_replay,
        run_registered_nod1_browser_flight_pipeline,
        write_nod1_flight_artifact,
        write_nod1_web_replay,
    )

    runner = _pipeline_physics_runner(args)
    run = run_registered_nod1_browser_flight_pipeline(
        registry_path=args.registry,
        fixture_path=args.fixture,
        config=NOD1FlightPipelineConfig(
            physics_dt_s=args.physics_dt_s,
            neural_dt_s=args.neural_dt_s,
            logging_dt_s=args.logging_dt_s,
            seed=args.seed,
        ),
        runner=runner,
    )
    replay = (
        None
        if args.web_replay_output is None
        else nod1_run_to_web_replay(
            run, target_sample_rate_hz=args.web_replay_sample_rate_hz
        )
    )
    manifest = write_nod1_flight_artifact(
        args.output,
        run,
        chunk_samples=args.chunk_samples,
        web_replay=replay,
    )
    if args.web_replay_output is not None:
        write_nod1_web_replay(
            args.web_replay_output,
            run,
            target_sample_rate_hz=args.web_replay_sample_rate_hz,
            source_artifact_manifest_path=args.output / "manifest.json",
        )
    return _pipeline_summary(args, manifest)


def _handle_simulate_fly_fgs_fixture(args: argparse.Namespace) -> int:
    from .pipeline import (
        NOD1FlightPipelineConfig,
        nod1_run_to_web_replay,
        run_registered_fly_fgs_flight_pipeline,
        write_nod1_flight_artifact,
        write_nod1_web_replay,
    )

    runner = _pipeline_physics_runner(args)
    run = run_registered_fly_fgs_flight_pipeline(
        manifest_path=args.manifest,
        capture_path=args.capture,
        config=NOD1FlightPipelineConfig(
            physics_dt_s=args.physics_dt_s,
            neural_dt_s=args.neural_dt_s,
            logging_dt_s=args.logging_dt_s,
            seed=args.seed,
        ),
        runner=runner,
    )
    replay = (
        None
        if args.web_replay_output is None
        else nod1_run_to_web_replay(
            run, target_sample_rate_hz=args.web_replay_sample_rate_hz
        )
    )
    manifest = write_nod1_flight_artifact(
        args.output,
        run,
        chunk_samples=args.chunk_samples,
        web_replay=replay,
    )
    if args.web_replay_output is not None:
        write_nod1_web_replay(
            args.web_replay_output,
            run,
            target_sample_rate_hz=args.web_replay_sample_rate_hz,
            source_artifact_manifest_path=args.output / "manifest.json",
        )
    return _pipeline_summary(args, manifest)


def _load_streaming_muscle_interventions(path: Optional[Path]):
    if path is None:
        return ()
    from .flight.streaming_bridge import RawAppSide
    from .flight.streaming_mechanics import (
        MechanicsInterventionMode,
        StreamingMuscleIntervention,
    )

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--muscle-interventions must contain a JSON array")
    interventions = []
    allowed = {
        "intervention_id",
        "muscle",
        "start_s",
        "end_s",
        "mode",
        "raw_app_side",
        "output_scale",
    }
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping) or set(item).difference(allowed):
            raise ValueError("muscle intervention {} has unknown fields".format(index))
        required = {"intervention_id", "muscle", "start_s", "end_s", "mode"}
        if not required.issubset(item):
            raise ValueError("muscle intervention {} is missing required fields".format(index))
        mode = MechanicsInterventionMode(item["mode"])
        default_scale = 0.0 if mode is MechanicsInterventionMode.SILENCE else 1.0
        side_value = item.get("raw_app_side")
        interventions.append(
            StreamingMuscleIntervention(
                intervention_id=item["intervention_id"],
                muscle=item["muscle"],
                start_s=item["start_s"],
                end_s=item["end_s"],
                mode=mode,
                raw_app_side=(
                    None if side_value is None else RawAppSide(side_value)
                ),
                output_scale=item.get("output_scale", default_scale),
            )
        )
    if len({item.intervention_id for item in interventions}) != len(interventions):
        raise ValueError("muscle intervention IDs must be unique")
    return tuple(interventions)


def _handle_simulate_canonical_online(args: argparse.Namespace) -> int:
    """Run the live three-clock stack on the worker and publish atomically."""

    from .canonical_artifacts import (
        bind_canonical_web_replay_to_artifact,
        canonical_closed_loop_to_web_replay,
        canonical_run_id,
        write_canonical_closed_loop_artifact,
    )
    from .fly_fgs_runtime import NodeFlyFGSCircuitRuntime
    from .flybody_adapter import (
        FlyBodyPhysicsAdapter,
        FlyBodyWorkerConfig,
        WingAxisTorqueMap,
    )
    from .flight.canonical_closed_loop import (
        CanonicalClosedLoopConfig,
        CanonicalClosedLoopSimulator,
    )
    from .flight.effector_mapping import EffectorLateralityHypothesis
    from .flight.streaming_bridge import StreamingNOD1MotorBridge
    from .flight.streaming_mechanics import (
        StreamingMechanicsConfig,
        StreamingMuscleWingStepper,
    )

    if args.flybody_spawn_height_m <= 0.0:
        raise ValueError("--flybody-spawn-height-m must be positive")
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", args.scenario_id) is None:
        raise ValueError(
            "--scenario-id must contain lowercase letters, digits, dots, underscores, or hyphens"
        )
    if not isinstance(args.label, str) or not args.label.strip():
        raise ValueError("--label must be non-empty")
    if args.web_replay_output is not None:
        artifact_root = args.output.resolve()
        replay_path = args.web_replay_output.resolve()
        if replay_path == artifact_root or artifact_root in replay_path.parents:
            raise ValueError(
                "--web-replay-output must be outside the immutable --output directory"
            )
    muscle_interventions = _load_streaming_muscle_interventions(
        args.muscle_interventions
    )
    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=EffectorLateralityHypothesis(
            args.effector_hypothesis
        ),
        duration_s=args.duration_s,
        figure_initial_world_azimuth_rad=args.figure_initial_world_azimuth_rad,
        figure_velocity_rad_s=args.figure_velocity_rad_s,
        ground_velocity_rad_s=args.ground_velocity_rad_s,
        include_retinal_input=True,
        include_full_cell_state=not args.omit_full_cell_state,
    )
    adapter = FlyBodyPhysicsAdapter(
        WingAxisTorqueMap(),
        FlyBodyWorkerConfig(
            timestep_s=config.physics_dt_s,
            spawn_height_m=args.flybody_spawn_height_m,
        ),
    )
    simulator = CanonicalClosedLoopSimulator(
        config,
        circuit_runtime=NodeFlyFGSCircuitRuntime(expected_node_version="v22.22.1"),
        bridge=StreamingNOD1MotorBridge(seed=args.seed),
        mechanics=StreamingMuscleWingStepper(
            StreamingMechanicsConfig(muscle_interventions=muscle_interventions)
        ),
        physics_adapter=adapter,
    )
    try:
        result = simulator.run()
        checkpoint = simulator.checkpoint()
        run_id = canonical_run_id(
            result,
            scenario_id=args.scenario_id,
            checkpoint=checkpoint,
        )
        replay = canonical_closed_loop_to_web_replay(
            result,
            scenario_id=args.scenario_id,
            label=args.label,
            checkpoint=checkpoint,
            source_run_id=run_id,
        )
        runtime_receipts = {
            "fly_fgs_runtime": dict(simulator.circuit_runtime.ready_receipt),
            "flybody": dict(adapter.provenance_metadata()),
        }
        manifest = write_canonical_closed_loop_artifact(
            args.output,
            result,
            checkpoint=checkpoint,
            scenario_id=args.scenario_id,
            chunk_samples=args.chunk_samples,
            runtime_receipts=runtime_receipts,
            web_replay=replay,
        )
        manifest_path = args.output / "manifest.json"
        replay_digest = None
        if args.web_replay_output is not None:
            bound_replay = bind_canonical_web_replay_to_artifact(
                replay, manifest, sha256_file(manifest_path)
            )
            args.web_replay_output.parent.mkdir(parents=True, exist_ok=True)
            with args.web_replay_output.open("x", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        bound_replay,
                        allow_nan=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
            replay_digest = sha256_file(args.web_replay_output)
        transitions = tuple(
            transition
            for interval in result.intervals
            for transition in interval.physics
        )
        generated_events = tuple(
            event
            for interval in result.intervals
            for event in interval.bridge.generated_events
        )
        applied_event_ids = tuple(
            event_id
            for transition in transitions
            for event_id in transition.mechanics.applied_event_ids
        )
        suppressed_event_ids = tuple(
            event_id
            for transition in transitions
            for event_id in transition.mechanics.suppressed_event_ids
        )
        fluid_force = np.asarray(
            [
                transition.aerodynamic_wrench_after.force_body_n
                for transition in transitions
            ],
            dtype=float,
        )
        physical_axis_torque = np.asarray(
            [
                transition.physical_actuation_wing_kinematics.wing_axis_torque_n_m
                for transition in transitions
            ],
            dtype=float,
        )
        articulated = tuple(
            transition.articulated_telemetry_after
            for transition in transitions
            if transition.articulated_telemetry_after is not None
        )
        actuator_torque = np.asarray(
            [telemetry.actuator_torque_n_m for telemetry in articulated],
            dtype=float,
        )
        _print_json(
            {
                "run_id": manifest["run_id"],
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "web_replay": (
                    None
                    if args.web_replay_output is None
                    else str(args.web_replay_output.resolve())
                ),
                "web_replay_sha256": replay_digest,
                "online_circuit_replay_attached": (
                    "online_circuit_replay" in replay
                ),
                "validation_status": result.validation_status,
                "scientific_status": "exploratory; software-correctness gates only",
                "physics_backend": result.backend_name,
                "circuit_samples": result.circuit_sample_count,
                "physics_transitions": len(transitions),
                "generated_motor_events": len(generated_events),
                "applied_motor_events": len(applied_event_ids),
                "suppressed_motor_events": len(suppressed_event_ids),
                "muscle_intervention_ids": [
                    intervention.intervention_id
                    for intervention in muscle_interventions
                ],
                "checkpoint_payload_sha256": checkpoint.to_dict()[
                    "payload_sha256"
                ],
                "maximum_physical_wing_axis_command_n_m": float(
                    np.max(np.abs(physical_axis_torque), initial=0.0)
                ),
                "maximum_native_actuator_torque_n_m": float(
                    np.max(np.abs(actuator_torque), initial=0.0)
                ),
                "maximum_root_total_fluid_force_n": float(
                    np.max(np.linalg.norm(fluid_force, axis=1), initial=0.0)
                ),
                "ground_contact_transition_count": sum(
                    telemetry.ground_contact_count > 0 for telemetry in articulated
                ),
                "final_body_position_world_m": (
                    result.final_body_state.position_world_m.tolist()
                ),
                "final_body_velocity_world_m_s": (
                    result.final_body_state.velocity_world_m_s.tolist()
                ),
                "effector_mapping": dict(result.effector_mapping_receipt.to_dict()),
                "signed_behavior_claim_policy": config.effector_signed_behavior_claim_policy,
            }
        )
        return 0
    finally:
        simulator.close()


def _handle_simulate_retinal(args: argparse.Namespace) -> int:
    from .pipeline import (
        NOD1FlightPipelineConfig,
        nod1_run_to_web_replay,
        run_retinal_flight_pipeline,
        write_nod1_flight_artifact,
        write_nod1_web_replay,
    )
    from .vision import AnalyticGratingScene, PanoramicRetina

    if args.duration_s <= 0.0 or args.exposure_dt_s <= 0.0:
        raise ValueError("duration and exposure dt must be positive")
    exposure_steps = int(round(args.duration_s / args.exposure_dt_s))
    if (
        exposure_steps < 1
        or abs(exposure_steps * args.exposure_dt_s - args.duration_s) > 1.0e-12
    ):
        raise ValueError("duration_s must be an integer multiple of exposure_dt_s")
    retina = PanoramicRetina(
        receptor_count=args.receptors,
        sensor_latency_s=args.sensor_latency_s,
    )
    scene = AnalyticGratingScene(
        angular_velocity_rad_s=args.angular_velocity_rad_s,
        phase_rad=args.phase_rad,
    )
    frames = tuple(
        retina.sample(
            scene,
            exposure_start_s=index * args.exposure_dt_s,
            exposure_end_s=(index + 1) * args.exposure_dt_s,
        )
        for index in range(exposure_steps + 1)
    )
    runner = _pipeline_physics_runner(args)
    run = run_retinal_flight_pipeline(
        frames,
        config=NOD1FlightPipelineConfig(
            physics_dt_s=args.physics_dt_s,
            neural_dt_s=args.neural_dt_s,
            logging_dt_s=args.logging_dt_s,
            seed=args.seed,
        ),
        runner=runner,
    )
    replay = (
        None
        if args.web_replay_output is None
        else nod1_run_to_web_replay(
            run, target_sample_rate_hz=args.web_replay_sample_rate_hz
        )
    )
    manifest = write_nod1_flight_artifact(
        args.output,
        run,
        chunk_samples=args.chunk_samples,
        web_replay=replay,
    )
    if args.web_replay_output is not None:
        write_nod1_web_replay(
            args.web_replay_output,
            run,
            target_sample_rate_hz=args.web_replay_sample_rate_hz,
            source_artifact_manifest_path=args.output / "manifest.json",
        )
    return _pipeline_summary(args, manifest)


def _handle_export_web(args: argparse.Namespace) -> int:
    if args.validation_registry is not None and args.validation_report is None:
        raise ValueError("--validation-registry requires --validation-report")
    registry_path = None
    if args.validation_report is not None:
        registry_path = (
            default_benchmark_registry_path()
            if args.validation_registry is None
            else Path(args.validation_registry)
        )
        registry = load_benchmark_registry(registry_path)
        ValidationReport.from_json(
            Path(args.validation_report).read_text(encoding="utf-8"),
            registry=registry,
        )
    pipeline_requested = bool(
        args.nod1_result is not None
        or args.include_reduced_retinal_pipeline
        or args.include_registered_nod1_fixture
        or args.include_registered_fly_fgs_fixture
    )
    if args.nod1_fixture is not None and not args.include_registered_nod1_fixture:
        raise ValueError("--nod1-fixture requires --include-registered-nod1-fixture")
    if (
        args.fly_fgs_manifest is not None
        and not args.include_registered_fly_fgs_fixture
    ):
        raise ValueError(
            "--fly-fgs-manifest requires --include-registered-fly-fgs-fixture"
        )
    if (
        args.fly_fgs_capture is not None
        and not args.include_registered_fly_fgs_fixture
    ):
        raise ValueError(
            "--fly-fgs-capture requires --include-registered-fly-fgs-fixture"
        )
    if args.physics_backend == "flybody" and not pipeline_requested:
        raise ValueError(
            "--physics-backend flybody requires at least one attached neural pipeline"
        )
    pipeline_runner = _pipeline_physics_runner(args) if pipeline_requested else None
    manifest = export_web_replay(
        args.output,
        scenarios=args.scenarios,
        duration_s=args.duration_s,
        physics_dt_s=args.physics_dt_s,
        logging_dt_s=args.logging_dt_s,
        neural_dt_s=args.neural_dt_s,
        seed=args.seed,
        target_sample_rate_hz=args.sample_rate_hz,
        evidence_path=args.evidence,
        validation_report_path=args.validation_report,
        validation_registry_path=registry_path,
        legacy_nod1_result_path=args.nod1_result,
        include_reduced_retinal_pipeline=args.include_reduced_retinal_pipeline,
        include_registered_nod1_fixture=args.include_registered_nod1_fixture,
        registered_nod1_fixture_path=args.nod1_fixture,
        include_registered_fly_fgs_fixture=(
            args.include_registered_fly_fgs_fixture
        ),
        registered_fly_fgs_manifest_path=args.fly_fgs_manifest,
        registered_fly_fgs_capture_path=args.fly_fgs_capture,
        pipeline_runner=pipeline_runner,
        retinal_receptor_count=args.retinal_receptors,
        retinal_angular_velocity_rad_s=args.retinal_angular_velocity_rad_s,
        retinal_sensor_latency_s=args.retinal_sensor_latency_s,
        overwrite=args.overwrite,
    )
    _print_json(
        {
            "manifest": str((args.output / "manifest.json").resolve()),
            "episode_count": len(manifest["episodes"]),
            "validation_status": manifest["status"],
            "source_kind": manifest["source_kind"],
        }
    )
    return 0


def _handle_flybody_smoke(args: argparse.Namespace) -> int:
    if args.mode == "zero-torque" and args.steps < 1:
        raise ValueError("--steps must be positive")
    # Deliberately lazy: importing the evidence/CLI stack must not import
    # FlyGym, MuJoCo, or their native modules on the display host.
    from .flybody_adapter import (
        FlyBodyWorkerConfig,
        run_analytic_wingbeat_smoke,
        run_torque_trace,
    )

    if args.mode == "analytic-wingbeat":
        timestep_s = 5.0e-5 if args.timestep_s is None else args.timestep_s
        config = FlyBodyWorkerConfig(timestep_s=timestep_s)
        trace = run_analytic_wingbeat_smoke(
            duration_s=args.duration_s,
            wingbeat_frequency_hz=args.wingbeat_hz,
            config=config,
        )
    else:
        timestep_s = 1.0e-4 if args.timestep_s is None else args.timestep_s
        config = FlyBodyWorkerConfig(timestep_s=timestep_s)
        trace = run_torque_trace(np.zeros((args.steps, 6), dtype=float), config=config)
    integrated_root_fluid_impulse_n_s = (
        timestep_s * np.sum(trace.root_fluid_force_n[1:], axis=0)
    )
    summary = {
        "schema_version": "1.0.0",
        "kind": "flybody_worker_compatibility_smoke",
        "mode": args.mode,
        "status": "exploratory",
        "calibration": "none",
        "authority_notice": (
            "This compatibility smoke proves that the pinned FlyBody worker executes. "
            "The analytic wingbeat is upstream's fallback formula, not the released learned "
            "straight-flight/saccade policy and not a validated neuromuscular result."
        ),
        "steps": len(trace.time_s) - 1,
        "timestep_s": timestep_s,
        "worker_versions": trace.metadata["worker_versions"],
        "wing_dof_order": trace.metadata["wing_dof_order"],
        "fluid_geom_names": trace.metadata["fluid_geom_names"],
        "fluid_geoms_contact_disabled": trace.metadata[
            "fluid_geoms_contact_disabled"
        ],
        "tendon_count": trace.metadata["tendon_count"],
        "compiled_model_fingerprint": trace.metadata[
            "compiled_model_fingerprint"
        ],
        "dependency_record_sha256": trace.metadata[
            "dependency_record_sha256"
        ],
        "ground_contact_topology": trace.metadata["ground_contact_topology"],
        "released_policy_topology_equivalent": trace.metadata[
            "released_policy_topology_equivalent"
        ],
        "body_state_reference": trace.metadata["body_state_reference"],
        "root_fluid_force_sample_semantics": trace.metadata[
            "root_fluid_force_sample_semantics"
        ],
        "actuation_source": trace.metadata["actuation_source"],
        "control_timestep_s": trace.metadata.get("control_timestep_s"),
        "actuation_hold": trace.metadata.get("actuation_hold"),
        "control_phase_semantics": trace.metadata.get(
            "control_phase_semantics"
        ),
        "waveform_semantics": trace.metadata.get("waveform_semantics"),
        "air_unit_correction_applied": trace.metadata[
            "flygym_2_1_air_unit_correction_applied"
        ],
        "final_body_position_m": trace.body_position_m[-1].tolist(),
        "final_body_quaternion_wxyz": trace.body_quaternion_wxyz[-1].tolist(),
        "integrated_root_fluid_impulse_n_s": (
            integrated_root_fluid_impulse_n_s.tolist()
        ),
        "maximum_fluid_force_n": float(np.max(np.abs(trace.root_fluid_force_n))),
        "maximum_actuator_torque_n_m": float(
            np.max(np.abs(trace.actuator_torque_n_m))
        ),
        "maximum_measured_wing_excursion_rad": float(
            np.max(np.ptp(trace.wing_angles_rad, axis=0))
        ),
    }
    if trace.metadata.get("control_timestep_s") is not None:
        held_samples = int(round(trace.metadata["control_timestep_s"] / timestep_s))
        first_hold = trace.actuator_torque_n_m[1 : held_samples + 1]
        summary["first_control_hold_sample_count"] = held_samples
        summary["first_control_actuator_torque_n_m"] = first_hold[0].tolist()
        summary["first_control_hold_max_delta_n_m"] = float(
            np.max(np.abs(first_hold - first_hold[0]))
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "trace": trace.to_serializable()}
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        summary["output"] = str(args.output.resolve())
    _print_json(summary)
    return 0


def _handle_validate(args: argparse.Namespace) -> int:
    registry_path = (
        default_benchmark_registry_path()
        if args.registry is None
        else Path(args.registry)
    )
    registry = load_benchmark_registry(registry_path)
    source_digest_values = list(
        default_validation_source_digests(registry, registry_path)
    )
    if os.environ.get("FLY_S2B_WORKER_IMAGE_DIGEST") is not None:
        from .flybody_adapter import declared_worker_image_digest

        worker_image_digest = declared_worker_image_digest()
        assert worker_image_digest is not None
        source_digest_values.append(
            SourceDigest(
                kind="image",
                name="fly-s2b-worker-image",
                sha256=worker_image_digest.removeprefix("sha256:"),
            )
        )
    source_digests = tuple(source_digest_values)
    report = ValidationRunner(registry, default_evaluators(registry)).run(
        evaluation_id=args.evaluation_id,
        source_digests=source_digests,
        changed_dependencies=args.changed_dependencies,
        case_ids=args.case_ids,
        include_dependents=args.include_dependents,
        include_prerequisites=args.include_prerequisites,
    )
    # Generation and serialization are separate trust boundaries.  Recompute
    # every registered metric, evidence, prerequisite, coverage, and promotion
    # claim before any report bytes are allowed to leave the process.
    report.validate_against(registry)

    # Exclusive creation preserves prior scientific reports.  The file bytes
    # are exactly the canonical JSON whose digest is reported below.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(report.to_json())

    status_counts = {
        status.value: sum(result.status is status for result in report.results)
        for status in GateStatus
    }
    _print_json(
        {
            "evaluation_id": report.evaluation_id,
            "status": report.overall_status.value,
            "promotion_ceiling": (
                None
                if report.promotion_ceiling is None
                else report.promotion_ceiling.value
            ),
            "case_counts": status_counts,
            "report": str(output.resolve()),
            "sha256": sha256_file(output),
        }
    )
    # BLOCKED is an evidence-bearing scientific result, not a process error.
    return 1 if report.overall_status is GateStatus.FAIL else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileExistsError, FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, "fly-s2b: error: {}\n".format(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
