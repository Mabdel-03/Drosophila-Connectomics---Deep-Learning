"""Command line entry point for authoritative paper figure-ground runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .flight.paper_open_loop import PaperOpenLoopConfig, PaperOpenLoopSimulator
from .flybody_adapter import FlyBodyWorkerConfig
from .paper_fgs import (
    PAPER_FGS_PROTOCOL_IDS,
    analyze_phase_locked_trials,
    compare_convergence,
    compare_to_paper_figure3,
    load_protocol,
    write_paper_artifact,
)
from .paper_fgs_runtime import NodePaperFGSCircuitRuntime
from .paper_tether import PAPER_TORQUE_METER_MODES, make_paper_torque_meter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fly-s2b-paper-fgs",
        description=(
            "Run a prescribed Reichardt figure-ground protocol in the native "
            "rigidly tethered FlyBody/MuJoCo worker."
        ),
    )
    parser.add_argument(
        "--protocol",
        choices=PAPER_FGS_PROTOCOL_IDS,
        default="R83_Fig3a_0_to_90",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="phase-locked trial count (paper default: 100)",
    )
    parser.add_argument("--run-seed", type=int, default=73)
    parser.add_argument(
        "--circuit-rate-hz",
        type=int,
        choices=(400, 800),
        default=400,
        help="400 is canonical; 800 is the convergence sensitivity rate",
    )
    parser.add_argument(
        "--physics-rate-hz",
        type=int,
        choices=(10000, 20000),
        default=10000,
        help="10 kHz is canonical; 20 kHz is the convergence sensitivity rate",
    )
    parser.add_argument(
        "--torque-meter",
        choices=PAPER_TORQUE_METER_MODES,
        default="fixed-load-cell",
        help=(
            "fixed-load-cell is authoritative; equality-reaction is the weld "
            "validator; comparison advances both"
        ),
    )
    parser.add_argument(
        "--wingbeat-phase-mode",
        choices=("uniformly_stratified", "fixed"),
        default="uniformly_stratified",
        help="fixed is a sensitivity control only",
    )
    parser.add_argument(
        "--texture-mode",
        choices=("registered_copy", "independent_matched_statistics"),
        default="registered_copy",
    )
    parser.add_argument(
        "--texture-seed",
        type=int,
        default=123456,
        help="canonical texture seed is 123456; alternatives are sensitivity runs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--web-replay-output",
        type=Path,
        default=None,
        help=(
            "optional browser data/paper_replay.json destination; a checksum "
            "manifest is written alongside it"
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="fly_fgs_source directory (auto-discovered in a source checkout)",
    )
    parser.add_argument(
        "--expected-node-version",
        default=None,
        help="optional exact Node.js version gate, for example v22.17.0",
    )
    parser.add_argument(
        "--convergence-reference",
        type=Path,
        action="append",
        default=[],
        help="canonical artifact directory or analysis.json to compare against",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    protocol = load_protocol(arguments.protocol)
    repetitions = (
        protocol.repetitions
        if arguments.repetitions is None
        else arguments.repetitions
    )
    config = PaperOpenLoopConfig(
        protocol=protocol,
        repetitions=repetitions,
        run_seed=arguments.run_seed,
        texture_relationship_mode=arguments.texture_mode,
        texture_seed=arguments.texture_seed,
        circuit_rate_hz=arguments.circuit_rate_hz,
        physics_rate_hz=arguments.physics_rate_hz,
        torque_meter_mode=arguments.torque_meter,
        wingbeat_phase_mode=arguments.wingbeat_phase_mode,
    )
    worker = FlyBodyWorkerConfig(timestep_s=1.0 / arguments.physics_rate_hz)
    physics = make_paper_torque_meter(arguments.torque_meter, config=worker)

    def circuit_factory() -> NodePaperFGSCircuitRuntime:
        keywords = {
            "expected_node_version": arguments.expected_node_version,
            "circuit_dt_s": 1.0 / arguments.circuit_rate_hz,
        }
        if arguments.source_root is not None:
            keywords["source_root"] = arguments.source_root
        return NodePaperFGSCircuitRuntime(**keywords)

    simulator = PaperOpenLoopSimulator(
        config,
        physics_adapter=physics,
        circuit_factory=circuit_factory,
    )
    arrays, metadata = simulator.run()
    analysis = dict(analyze_phase_locked_trials(
        protocol,
        arrays["time_s"],
        arrays["reported_yaw_torque_Nm"],
    ))
    mechanical_sources = {
        "authoritative_total": "reported_yaw_torque_Nm",
        "left_wing": "wing_reported_yaw_torque_Nm",
        "right_wing": "wing_reported_yaw_torque_Nm",
        "wing_sum": "wing_sum_reported_yaw_torque_Nm",
        "nonwing_residual": "nonwing_reported_yaw_torque_residual_Nm",
        "left_wing_aerodynamic": "wing_aerodynamic_reported_yaw_torque_Nm",
        "right_wing_aerodynamic": "wing_aerodynamic_reported_yaw_torque_Nm",
    }
    mechanical_channels = {}
    channel_product_templates = {
        "authoritative_total": "torque_product_{}_Nm",
        "left_wing": "left_wing_torque_product_{}_Nm",
        "right_wing": "right_wing_torque_product_{}_Nm",
        "wing_sum": "wing_sum_torque_product_{}_Nm",
        "nonwing_residual": "nonwing_residual_torque_product_{}_Nm",
        "left_wing_aerodynamic": "left_wing_aerodynamic_torque_product_{}_Nm",
        "right_wing_aerodynamic": "right_wing_aerodynamic_torque_product_{}_Nm",
    }
    for channel, array_name in mechanical_sources.items():
        if array_name not in arrays:
            continue
        values = arrays[array_name]
        if channel in ("left_wing", "left_wing_aerodynamic"):
            values = values[:, :, 0]
        elif channel in ("right_wing", "right_wing_aerodynamic"):
            values = values[:, :, 1]
        if not np.all(np.isfinite(values)):
            continue
        channel_analysis = dict(
            analyze_phase_locked_trials(protocol, arrays["time_s"], values)
        )
        signal_product_analyses = {}
        template = channel_product_templates[channel]
        for product in (
            "paper_comparison",
            "lowpass_10hz",
            "lowpass_25hz",
            "lowpass_50hz",
            "wingbeat_averaged",
        ):
            product_array_name = template.format(product)
            if product_array_name not in arrays:
                continue
            product_values = arrays[product_array_name]
            if np.all(np.isfinite(product_values)):
                signal_product_analyses[product] = dict(
                    analyze_phase_locked_trials(
                        protocol, arrays["time_s"], product_values
                    )
                )
        channel_analysis["signal_products"] = signal_product_analyses
        mechanical_channels[channel] = channel_analysis
    analysis["mechanical_channels"] = mechanical_channels
    comparison_products = (
        "paper_comparison",
        "lowpass_10hz",
        "lowpass_25hz",
        "lowpass_50hz",
        "wingbeat_averaged",
    )
    analysis["paper_figure3_comparisons"] = {
        product: compare_to_paper_figure3(
            protocol,
            arrays["time_s"],
            arrays["torque_product_{}_Nm".format(product)],
            signal_product=product,
        )
        for product in comparison_products
    }
    convergence_checks = []
    for supplied_reference in arguments.convergence_reference:
        reference_path = supplied_reference
        if reference_path.is_dir():
            reference_path = reference_path / "analysis.json"
        reference = json.loads(reference_path.read_text("utf-8"))
        total_convergence = compare_convergence(reference, analysis)
        channel_convergence = {}
        for channel in ("left_wing", "right_wing", "wing_sum"):
            reference_channel = reference.get("mechanical_channels", {}).get(
                channel
            )
            candidate_channel = mechanical_channels.get(channel)
            if reference_channel is None or candidate_channel is None:
                channel_convergence[channel] = {
                    "available": False,
                    "passed": False,
                }
            else:
                channel_convergence[channel] = {
                    "available": True,
                    **compare_convergence(reference_channel, candidate_channel),
                }
        convergence = {
            "schema_version": "paper_multichannel_convergence.v3",
            "authoritative_total": total_convergence,
            "mechanical_channels": channel_convergence,
            "passed": bool(
                total_convergence["passed"] is True
                and all(
                    item["passed"] is True
                    for item in channel_convergence.values()
                )
            ),
        }
        convergence_checks.append(
            {
                "reference": str(reference_path.resolve()),
                "comparison": convergence,
            }
        )
        if convergence["passed"] is not True:
            raise RuntimeError("paper numerical convergence thresholds failed")
    if convergence_checks:
        analysis["convergence_against_reference"] = {
            "schema_version": "paper_convergence_collection.v1",
            "comparisons": convergence_checks,
            "passed": all(
                item["comparison"]["passed"] is True
                for item in convergence_checks
            ),
        }
    calibration_passed = (
        (metadata.get("torque_calibration") or {}).get("passed") is True
    )
    convergence_passed = (
        analysis.get("convergence_against_reference", {}).get("passed") is True
    )
    manifest = write_paper_artifact(
        arguments.output,
        protocol,
        arrays,
        analysis,
        run_metadata=metadata,
        authoritative=(
            arguments.torque_meter == "comparison"
            and calibration_passed
            and convergence_passed
        ),
        web_replay_output=arguments.web_replay_output,
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "run_id": manifest["run_id"],
                "scientific_status": manifest["scientific_status"],
                "protocol_id": protocol.protocol_id,
                "repetitions": repetitions,
                "post_transition_mean_torque_Nm": analysis[
                    "post_transition_mean_torque_n_m"
                ],
                "first_harmonic_amplitude_Nm": analysis[
                    "first_harmonic_amplitude_n_m"
                ],
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
