#!/usr/bin/env python3
"""Run the live fly-FGS stack against the pinned native FlyBody adapter.

This is an operational integration smoke, not a biological validation or a
released-policy flight baseline.  It is kept as a standalone script so the
same command can run inside a pinned worker image with the repository mounted
read-only, then later inside the newly sealed image without source mounts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fly_sensor2behavior.artifacts import sha256_file
from fly_sensor2behavior.canonical_artifacts import (
    bind_canonical_web_replay_to_artifact,
    canonical_closed_loop_to_web_replay,
    canonical_run_id,
    write_canonical_closed_loop_artifact,
)
from fly_sensor2behavior.fly_fgs_runtime import NodeFlyFGSCircuitRuntime
from fly_sensor2behavior.flybody_adapter import (
    FlyBodyPhysicsAdapter,
    FlyBodyWorkerConfig,
    WingAxisTorqueMap,
)
from fly_sensor2behavior.flight.canonical_closed_loop import (
    CanonicalClosedLoopConfig,
    CanonicalClosedLoopSimulator,
)
from fly_sensor2behavior.flight.effector_mapping import EffectorLateralityHypothesis
from fly_sensor2behavior.flight.streaming_bridge import StreamingNOD1MotorBridge
from fly_sensor2behavior.flight.streaming_mechanics import StreamingMuscleWingStepper


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=0.001)
    parser.add_argument("--spawn-height-m", type=float, default=0.100)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--figure-initial-world-azimuth-rad", type=float, default=0.0)
    parser.add_argument("--figure-velocity-rad-s", type=float, default=0.0)
    parser.add_argument("--ground-velocity-rad-s", type=float, default=0.0)
    parser.add_argument(
        "--effector-hypothesis",
        choices=[item.value for item in EffectorLateralityHypothesis],
        default=EffectorLateralityHypothesis.RAW_L_TO_PHYSICAL_LEFT.value,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-full-cell-state", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=EffectorLateralityHypothesis(
            args.effector_hypothesis
        ),
        duration_s=args.duration_s,
        figure_initial_world_azimuth_rad=args.figure_initial_world_azimuth_rad,
        figure_velocity_rad_s=args.figure_velocity_rad_s,
        ground_velocity_rad_s=args.ground_velocity_rad_s,
        include_retinal_input=True,
        include_full_cell_state=args.include_full_cell_state,
    )
    adapter = FlyBodyPhysicsAdapter(
        WingAxisTorqueMap(),
        FlyBodyWorkerConfig(
            timestep_s=config.physics_dt_s,
            spawn_height_m=args.spawn_height_m,
        ),
    )
    simulator = CanonicalClosedLoopSimulator(
        config,
        circuit_runtime=NodeFlyFGSCircuitRuntime(expected_node_version="v22.22.1"),
        bridge=StreamingNOD1MotorBridge(seed=args.seed),
        mechanics=StreamingMuscleWingStepper(),
        physics_adapter=adapter,
    )
    try:
        result = simulator.run()
        checkpoint = simulator.checkpoint()
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
        forces = np.asarray(
            [transition.aerodynamic_wrench_after.force_body_n for transition in transitions]
        )
        measured_wings = np.asarray(
            [
                transition.articulated_telemetry_after.measured_wing_position_rad
                for transition in transitions
                if transition.articulated_telemetry_after is not None
            ]
        )
        summary = {
            "schema_version": "1.0.0",
            "scientific_status": "software_integration_smoke_only",
            "completed": result.completed,
            "duration_s": config.duration_s,
            "circuit_sample_count": result.circuit_sample_count,
            "bridge_interval_count": len(result.intervals),
            "physics_transition_count": len(transitions),
            "generated_motor_event_count": len(generated_events),
            "applied_motor_event_count": len(applied_event_ids),
            "suppressed_motor_event_count": len(suppressed_event_ids),
            "maximum_root_total_fluid_force_n": float(
                np.max(np.linalg.norm(forces, axis=1), initial=0.0)
            ),
            "maximum_measured_wing_excursion_rad": float(
                np.max(np.abs(measured_wings), initial=0.0)
            ),
            "final_body_position_world_m": result.final_body_state.position_world_m.tolist(),
            "final_body_velocity_world_m_s": result.final_body_state.velocity_world_m_s.tolist(),
            "effector_mapping": dict(result.effector_mapping_receipt.to_dict()),
            "signed_behavior_claim_policy": config.effector_signed_behavior_claim_policy,
            "checkpoint_payload_sha256": checkpoint.to_dict()["payload_sha256"],
            "circuit_runtime_receipt": dict(
                simulator.circuit_runtime.ready_receipt
            ),
            "physics_provenance": dict(adapter.provenance_metadata()),
        }
        if args.output is not None:
            run_id = canonical_run_id(
                result,
                scenario_id="canonical-native-smoke",
                checkpoint=checkpoint,
            )
            replay = canonical_closed_loop_to_web_replay(
                result,
                scenario_id="canonical-native-smoke",
                label="Canonical online fly-FGS → FlyBody smoke",
                checkpoint=checkpoint,
                source_run_id=run_id,
            )
            manifest = write_canonical_closed_loop_artifact(
                args.output,
                result,
                checkpoint=checkpoint,
                scenario_id="canonical-native-smoke",
                runtime_receipts={
                    "fly_fgs_runtime": dict(
                        simulator.circuit_runtime.ready_receipt
                    ),
                    "flybody": dict(adapter.provenance_metadata()),
                },
                web_replay=replay,
            )
            manifest_digest = sha256_file(args.output / "manifest.json")
            bound_replay = bind_canonical_web_replay_to_artifact(
                replay, manifest, manifest_digest
            )
            # The run directory is immutable after atomic commit. Keep the
            # derived browser projection beside it, never inside it.
            replay_path = args.output.parent / (args.output.name + ".web-replay.json")
            with replay_path.open("x", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        bound_replay,
                        allow_nan=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
            summary["artifact"] = {
                "run_id": manifest["run_id"],
                "manifest_sha256": manifest_digest,
                "web_replay_sha256": sha256_file(replay_path),
                "output": str(args.output),
            }
        print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
        return 0
    finally:
        simulator.close()


if __name__ == "__main__":
    raise SystemExit(main())
