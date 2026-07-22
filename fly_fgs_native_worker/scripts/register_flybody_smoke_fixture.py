#!/usr/bin/env python3
"""Register an immutable FlyBody analytic-adapter compatibility fixture.

The input is the full JSON written by ``fly-s2b flybody-smoke --output`` in
the pinned worker. This script adds the preregistered comparison contract and
serializes deterministically. It does not promote the analytic fallback to a
released-policy or biological baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def comparison_contract() -> Mapping[str, Any]:
    return {
        "kind": "normalized_max_summary_error",
        "required_exact_summary_fields": [
            "actuation_hold",
            "actuation_source",
            "air_unit_correction_applied",
            "authority_notice",
            "body_state_reference",
            "calibration",
            "compiled_model_fingerprint",
            "control_phase_semantics",
            "control_timestep_s",
            "dependency_record_sha256",
            "first_control_hold_sample_count",
            "fluid_geom_names",
            "fluid_geoms_contact_disabled",
            "ground_contact_topology",
            "kind",
            "mode",
            "released_policy_topology_equivalent",
            "root_fluid_force_sample_semantics",
            "schema_version",
            "status",
            "steps",
            "tendon_count",
            "timestep_s",
            "waveform_semantics",
            "wing_dof_order",
            "worker_versions",
        ],
        "numeric_summary_tolerances": {
            "final_body_position_m": {"atol": 1.0e-8, "rtol": 1.0e-4},
            "final_body_quaternion_wxyz": {"atol": 1.0e-8, "rtol": 1.0e-4},
            "first_control_actuator_torque_n_m": {
                "atol": 1.0e-12,
                "rtol": 1.0e-4,
            },
            "first_control_hold_max_delta_n_m": {
                "atol": 1.0e-15,
                "rtol": 0.0,
            },
            "integrated_root_fluid_impulse_n_s": {
                "atol": 1.0e-11,
                "rtol": 1.0e-4,
            },
            "maximum_actuator_torque_n_m": {
                "atol": 1.0e-10,
                "rtol": 1.0e-4,
            },
            "maximum_fluid_force_n": {"atol": 1.0e-9, "rtol": 1.0e-4},
            "maximum_measured_wing_excursion_rad": {
                "atol": 1.0e-8,
                "rtol": 1.0e-4,
            },
        },
        "rationale": (
            "A compatibility regression for one exact compiled FlyBody/FlyGym/"
            "MuJoCo stack, beat-wrapped analytic fallback, advance-then-hold 0.2 ms "
            "controller clock, and collision-disabled fluid proxies. Numeric "
            "tolerances cover deterministic platform floating-point variation; "
            "this is not stable-flight, released-policy, muscle-calibration, or "
            "biological acceptance."
        ),
    }


def register(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(raw) != {"summary", "trace"}:
        raise ValueError("raw smoke output must contain exactly summary and trace")
    summary = raw["summary"]
    trace = raw["trace"]
    if not isinstance(summary, Mapping) or not isinstance(trace, Mapping):
        raise ValueError("summary and trace must be JSON objects")
    if summary.get("mode") != "analytic-wingbeat":
        raise ValueError("only the analytic-wingbeat compatibility trace is registrable")
    if summary.get("released_policy_topology_equivalent") is not False:
        raise ValueError("compatibility fixture must reject released-policy equivalence")
    summary = dict(summary)
    timestep_s = float(summary["timestep_s"])
    force_n = np.asarray(trace["root_fluid_force_n"], dtype=float)
    torque_n_m = np.asarray(trace["actuator_torque_n_m"], dtype=float)
    wing_rad = np.asarray(trace["wing_angles_rad"], dtype=float)
    quaternion = np.asarray(trace["body_quaternion_wxyz"], dtype=float)
    held_samples = int(round(float(summary["control_timestep_s"]) / timestep_s))
    first_hold = torque_n_m[1 : held_samples + 1]
    if first_hold.shape != (held_samples, 6):
        raise ValueError("trace does not contain the complete first control hold")
    summary.update(
        {
            "final_body_quaternion_wxyz": quaternion[-1].tolist(),
            "first_control_actuator_torque_n_m": first_hold[0].tolist(),
            "first_control_hold_max_delta_n_m": float(
                np.max(np.abs(first_hold - first_hold[0]))
            ),
            "first_control_hold_sample_count": held_samples,
            "integrated_root_fluid_impulse_n_s": (
                timestep_s * np.sum(force_n[1:], axis=0)
            ).tolist(),
            "maximum_measured_wing_excursion_rad": float(
                np.max(np.ptp(wing_rad, axis=0))
            ),
        }
    )
    contract = comparison_contract()
    required = set(contract["required_exact_summary_fields"])
    required.update(contract["numeric_summary_tolerances"])
    missing = sorted(required - set(summary))
    if missing:
        raise ValueError("raw smoke summary is missing: {}".format(", ".join(missing)))
    return {
        "fixture_schema_version": "3.0.0",
        "comparison": contract,
        "summary": dict(summary),
        "trace": dict(trace),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError("refusing to overwrite {}".format(args.output))
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    fixture = register(raw)
    args.output.write_text(
        json.dumps(fixture, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
