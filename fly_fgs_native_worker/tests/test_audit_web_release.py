import hashlib
import json
import os
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.audit_web_release import (
    AuditError,
    CANONICAL_ONLINE_VALIDATION_CASE_ID,
    FLY_FGS_CAUSAL_RETINAL_FRAME_COUNT,
    FLY_FGS_EPISODE_ID,
    FLY_FGS_FLYBODY_CASE_ID,
    FLY_FGS_CHECKPOINT_CASE_ID,
    FLY_FGS_INCREMENTAL_CASE_ID,
    FLY_FGS_INTEGRITY_CASE_ID,
    FLY_FGS_SOURCE_KIND,
    FROZEN_BROWSER_DURATION_S,
    FROZEN_BROWSER_EPISODE_ID,
    FROZEN_BROWSER_FLYBODY_CASE_ID,
    REPOSITORY_ROOT,
    SOURCE_ROOT,
    WINGBEAT_ANGLE_SAMPLE_SEMANTICS,
    WINGBEAT_TORQUE_SAMPLE_SEMANTICS,
    _audit_current_source_binding,
    _audit_canonical_online_episode,
    _audit_episode_content_binding,
    _audit_flybody_ground_contact_disclosure,
    _audit_flybody_wingbeat_inspection,
    _audit_fly_fgs_canonical_episode,
    _audit_fly_fgs_validation_inventory,
    _audit_frozen_browser_nod1_episode,
    _audit_no_fly_fgs_downstream_mechanics,
    _audit_logical_executable_provenance,
    _audit_publication_gate_status,
    _audit_required_validation_pass,
    _audit_run_manifest,
    _audit_web_replay_projection,
    _audit_worker_dependency_lock_binding,
    _audit_worker_image_binding,
    _canonical_fly_fgs_binding,
    _is_canonical_online_declaration,
    _declared_provenance_sha256,
    _load_json,
    _python_source_tree_sha256,
    _preferred_episode_index,
    _safe_local_path,
    _sha256_file,
)
from fly_sensor2behavior.canonical_artifacts import (
    CANONICAL_ARTIFACT_SCHEMA_VERSION,
    CANONICAL_WEB_PROJECTION_CANONICALIZATION,
    bind_canonical_web_replay_to_artifact,
    canonical_closed_loop_to_web_replay,
    canonical_run_id,
    write_canonical_closed_loop_artifact,
)
from fly_sensor2behavior.flight.canonical_closed_loop import (
    CanonicalClosedLoopConfig,
    CanonicalClosedLoopSimulator,
)
from fly_sensor2behavior.flight.streaming_mechanics import (
    MechanicsInterventionMode,
    StreamingMechanicsConfig,
    StreamingMuscleIntervention,
    StreamingMuscleWingStepper,
)
from fly_sensor2behavior.flight.streaming_bridge import (
    StreamingBridgeIntervention,
    StreamingBridgeStage,
    StreamingInterventionMode,
    StreamingNOD1MotorBridge,
)
from fly_sensor2behavior.artifacts import (
    GROUND_CONTACT_SAMPLE_SEMANTICS,
    GROUND_CONTACT_TELEMETRY_KIND,
    WEB_REPLAY_PROJECTION_CANONICALIZATION,
    WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS,
    WEB_REPLAY_PROJECTION_SCHEMA_VERSION,
    episode_run_id,
    run_scenario,
    web_replay_projection_sha256,
    write_episode_artifact,
    model_hashes,
)
from fly_sensor2behavior.evaluators import default_evaluators
from fly_sensor2behavior.fly_fgs import load_registered_fly_fgs_fixture
from fly_sensor2behavior.validation import (
    GateStatus,
    SourceDigest,
    ValidationRunner,
    load_benchmark_registry,
    verified_preregistered_protocol_files,
)
from fly_sensor2behavior.schema import REVIEWED_FLYBODY_WING_AXIS_ORDER
from fly_sensor2behavior.fly_fgs_runtime import (
    NodeFlyFGSCircuitRuntime,
    default_fly_fgs_runtime_script_path,
)
from test_canonical_closed_loop import (
    _DEFAULT_EFFECTOR_HYPOTHESIS,
    _high_gain_bridge,
    _initial_body,
    FakeFlightPhysics,
)


def test_logical_executable_provenance_binds_current_source_and_run_manifests():
    hashes = model_hashes()
    run_manifest = {"model_hashes": hashes}
    canonical_manifest = {
        "schema_version": "1.1.0",
        "scientific_content_sha256": "1" * 64,
    }
    assert _audit_logical_executable_provenance(
        "Reduced-order executable model",
        hashes["reduced_order_flight_source"],
        (canonical_manifest, run_manifest),
    )
    assert _audit_logical_executable_provenance(
        "FlyBody adapter executable model",
        hashes["flybody_adapter_source"],
        (run_manifest,),
    )
    assert not _audit_logical_executable_provenance(
        "unrelated provenance", None, (run_manifest,)
    )
    with pytest.raises(AuditError, match="current executable source"):
        _audit_logical_executable_provenance(
            "Reduced-order executable model", "0" * 64, (run_manifest,)
        )
    with pytest.raises(AuditError, match="not bound by run manifest"):
        _audit_logical_executable_provenance(
            "Reduced-order executable model",
            hashes["reduced_order_flight_source"],
            ({"model_hashes": {}},),
        )
    with pytest.raises(AuditError, match="no applicable legacy"):
        _audit_logical_executable_provenance(
            "Reduced-order executable model",
            hashes["reduced_order_flight_source"],
            (canonical_manifest,),
        )


def test_current_source_binding_requires_exact_dependency_lock_receipt():
    registry_path = REPOSITORY_ROOT / "data" / "benchmarks" / "registry.v1.json"
    evaluator_path = SOURCE_ROOT / "fly_sensor2behavior" / "evaluators.py"
    runtime_path = REPOSITORY_ROOT / "scripts" / "fly_fgs_runtime_rpc.mjs"
    package_path = SOURCE_ROOT / "fly_sensor2behavior"
    lock_path = REPOSITORY_ROOT / "requirements.lock"
    ordinary_flight_contract_path = (
        REPOSITORY_ROOT
        / "data"
        / "reference"
        / "flybody_ordinary_flight_release.expected.v1.json"
    )
    receipts = (
        SourceDigest("registry", registry_path.name, _sha256_file(registry_path)),
        SourceDigest("code", "built-in-evaluators.py", _sha256_file(evaluator_path)),
        SourceDigest(
            "code",
            "fly_sensor2behavior-python-tree",
            _python_source_tree_sha256(package_path),
        ),
        SourceDigest("code", "fly_fgs_runtime_rpc.mjs", _sha256_file(runtime_path)),
        SourceDigest(
            "dependency_lock",
            "requirements.lock",
            _sha256_file(lock_path),
        ),
        SourceDigest(
            "reference_contract",
            ordinary_flight_contract_path.name,
            _sha256_file(ordinary_flight_contract_path),
        ),
    ) + tuple(
        SourceDigest("protocol", source_uri, sha256)
        for source_uri, _path, sha256, _payload in verified_preregistered_protocol_files(
            load_benchmark_registry(registry_path), registry_path
        )
    )
    registry_bytes = registry_path.read_bytes()

    _audit_current_source_binding(
        SimpleNamespace(source_digests=receipts), registry_bytes
    )
    without_lock = tuple(
        item
        for item in receipts
        if not (item.kind == "dependency_lock" and item.name == "requirements.lock")
    )
    with pytest.raises(AuditError, match="dependency_lock:requirements.lock"):
        _audit_current_source_binding(
            SimpleNamespace(source_digests=without_lock), registry_bytes
        )


def test_publication_gate_allows_blocked_but_rejects_partial_or_failed():
    blocked = SimpleNamespace(
        suite_complete=True,
        results=(SimpleNamespace(case_id="blocked.case", status=GateStatus.BLOCKED),),
    )
    _audit_publication_gate_status(blocked)

    with pytest.raises(AuditError, match="complete validation suite"):
        _audit_publication_gate_status(
            SimpleNamespace(suite_complete=False, results=blocked.results)
        )
    failed = SimpleNamespace(
        suite_complete=True,
        results=(SimpleNamespace(case_id="failed.case", status=GateStatus.FAIL),),
    )
    with pytest.raises(AuditError, match="failed.case"):
        _audit_publication_gate_status(failed)


def test_web_replay_projection_binds_all_non_circular_fields():
    episode = {
        "schema_version": "1.0.0",
        "id": "fixture",
        "source_run_id": "fixture-run",
        "source_artifact_manifest_sha256": "1" * 64,
        "source_artifact_schema_version": "2.0.0",
        "frames": [{"time_s": 0.0}, {"time_s": 0.1}],
    }
    run_manifest = {
        "web_replay_projection": {
            "schema_version": WEB_REPLAY_PROJECTION_SCHEMA_VERSION,
            "sha256": web_replay_projection_sha256(episode),
            "canonicalization": WEB_REPLAY_PROJECTION_CANONICALIZATION,
            "excluded_top_level_fields": list(
                WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS
            ),
        }
    }
    _audit_web_replay_projection(episode, run_manifest)

    mutated = dict(episode)
    mutated["frames"] = [{"time_s": 0.0}, {"time_s": 0.2}]
    with pytest.raises(AuditError, match="does not match"):
        _audit_web_replay_projection(mutated, run_manifest)

    circular_only = dict(episode)
    circular_only["source_artifact_manifest_sha256"] = "2" * 64
    circular_only["source_artifact_schema_version"] = "9.0.0"
    _audit_web_replay_projection(circular_only, run_manifest)


def test_safe_local_path_accepts_only_existing_contained_files(tmp_path):
    target = tmp_path / "data" / "episode.json"
    target.parent.mkdir()
    target.write_text("{}", encoding="utf-8")

    assert _safe_local_path(tmp_path, "data/episode.json", "fixture") == target
    for unsafe in (
        "../episode.json",
        "data/../episode.json",
        "data//episode.json",
        "/data/episode.json",
        "https://example.org/episode.json",
        "//example.org/episode.json",
        "data/episode.json?mutable=1",
        "data/episode.json#fragment",
        "data\\episode.json",
        "data/%2e%2e/episode.json",
        "data/%252e%252e/episode.json",
        "data/%65pisode.json",
        "data%2fepisode.json",
        "data/epi\nsode.json",
    ):
        with pytest.raises(AuditError):
            _safe_local_path(tmp_path, unsafe, "fixture")


def test_safe_local_path_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside-release-audit.json"
    outside.write_text("{}", encoding="utf-8")
    link = tmp_path / "linked.json"
    try:
        link.symlink_to(outside)
        with pytest.raises(AuditError, match="escapes"):
            _safe_local_path(tmp_path, "linked.json", "fixture")
    finally:
        outside.unlink()


def test_provenance_digest_requires_a_full_sha256():
    digest = hashlib.sha256(b"release").hexdigest()
    assert _declared_provenance_sha256(
        {"value": "immutable bytes · sha256:{}".format(digest)}, "fixture"
    ) == digest
    assert _declared_provenance_sha256(
        {"value": "display prefix · sha256:{}".format(digest[:16])}, "fixture"
    ) is None
    with pytest.raises(AuditError, match="lowercase SHA-256"):
        _declared_provenance_sha256({"sha256": "not-a-digest"}, "fixture")
    with pytest.raises(AuditError, match="claims disagree"):
        _declared_provenance_sha256(
            {
                "sha256": "1" * 64,
                "value": "conflicting receipt · sha256:%s" % ("2" * 64),
            },
            "fixture",
        )


@pytest.mark.parametrize(
    "payload,error",
    [
        ('{"duplicate":1,"duplicate":2}', "duplicate JSON key"),
        ('{"invalid":NaN}', "non-finite JSON constant"),
    ],
)
def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(
    tmp_path: Path, payload: str, error: str
):
    path = tmp_path / "invalid.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(AuditError, match=error):
        _load_json(path, "fixture")


def _ground_contact_contract_fixture():
    time_s = np.array((0.0, 0.001, 0.002), dtype=float)
    contact = np.array((0, 0, 1), dtype=np.int64)
    physics_time_s = time_s.copy()
    transition_contact = contact.copy()
    run_manifest = {
        "configuration": {
            "duration_s": 0.002,
            "physics_timestep_s": 0.001,
        },
        "runtime": {
            "physics_backend": "flybody",
            "physics_provenance": {
                "ground_contact_telemetry": {
                    "ground_geom_name": "ground_plane",
                    "configured_pair_count": 48,
                }
            },
        },
        "diagnostics": {
            "physics_backend": "flybody",
            "physics_steps": 2,
            "metrics": {
                "ground_contact_telemetry_available": 1.0,
                "initial_ground_contact_count": 0.0,
                "ground_contact_transition_count": 1.0,
                "maximum_ground_contact_count": 1.0,
                "ground_contact_occurred": 1.0,
                "first_ground_contact_transition_start_s_or_duration_s": 0.001,
                "first_ground_contact_transition_end_s_or_duration_s": 0.002,
            },
        },
        "arrays": {
            "time_s": {
                "unit": "s",
                "provenance": "shared_episode_clock",
            },
            "ground_contact_count": {
                "unit": "1",
                "provenance": "decimated_projection_of_mujoco_transition_contact_points",
            },
            "physics_time_s": {
                "unit": "s",
                "provenance": "authoritative_external_physics_clock",
            },
            "ground_contact_transition_point_count": {
                "unit": "1",
                "provenance": GROUND_CONTACT_TELEMETRY_KIND,
                "sample_semantics": GROUND_CONTACT_SAMPLE_SEMANTICS,
            },
        },
    }
    episode = {
        "physics_backend": "flybody",
        "contact_telemetry_status": GROUND_CONTACT_TELEMETRY_KIND,
        "ground_contact_summary": {
            "telemetry": GROUND_CONTACT_TELEMETRY_KIND,
            "sample_semantics": GROUND_CONTACT_SAMPLE_SEMANTICS,
            "initial_count": 0,
            "occurred": True,
            "first_transition_start_s": 0.001,
            "first_transition_end_s": 0.002,
            "transition_count": 1,
            "maximum_count": 1,
        },
        "units": {"ground_contact_count": "1"},
        "frames": [
            {"t": float(time), "ground_contact_count": int(count)}
            for time, count in zip(time_s, contact)
        ],
    }
    return episode, run_manifest, {
        "time_s": time_s,
        "ground_contact_count": contact,
        "physics_time_s": physics_time_s,
        "ground_contact_transition_point_count": transition_contact,
    }


def test_flybody_contact_disclosure_binds_full_step_summary_and_replay_samples():
    episode, run_manifest, arrays = _ground_contact_contract_fixture()
    _audit_flybody_ground_contact_disclosure(episode, run_manifest, arrays)

    inconsistent_summary = deepcopy(episode)
    inconsistent_summary["ground_contact_summary"]["transition_count"] = 0
    with pytest.raises(AuditError, match="disagrees with full-step diagnostics"):
        _audit_flybody_ground_contact_disclosure(
            inconsistent_summary, run_manifest, arrays
        )

    inconsistent_frame = deepcopy(episode)
    inconsistent_frame["frames"][1]["ground_contact_count"] = 1
    with pytest.raises(AuditError, match="not a sample"):
        _audit_flybody_ground_contact_disclosure(
            inconsistent_frame, run_manifest, arrays
        )


def test_flybody_contact_disclosure_enforces_transition_interval_semantics():
    episode, run_manifest, arrays = _ground_contact_contract_fixture()

    wrong_interval = deepcopy(episode)
    wrong_interval["ground_contact_summary"]["first_transition_start_s"] = 0.0
    wrong_manifest = deepcopy(run_manifest)
    wrong_manifest["diagnostics"]["metrics"][
        "first_ground_contact_transition_start_s_or_duration_s"
    ] = 0.0
    with pytest.raises(AuditError, match="authoritative full-step telemetry"):
        _audit_flybody_ground_contact_disclosure(
            wrong_interval,
            wrong_manifest,
            arrays,
        )

    contact_at_reset = deepcopy(episode)
    contact_at_reset["ground_contact_summary"]["initial_count"] = 1
    contact_at_reset["ground_contact_summary"]["first_transition_start_s"] = 0.0
    contact_at_reset["ground_contact_summary"]["first_transition_end_s"] = 0.001
    contact_at_reset["frames"][0]["ground_contact_count"] = 1
    reset_manifest = deepcopy(run_manifest)
    reset_manifest["diagnostics"]["metrics"][
        "initial_ground_contact_count"
    ] = 1.0
    reset_manifest["diagnostics"]["metrics"][
        "first_ground_contact_transition_start_s_or_duration_s"
    ] = 0.0
    reset_manifest["diagnostics"]["metrics"][
        "first_ground_contact_transition_end_s_or_duration_s"
    ] = 0.001
    reset_arrays = dict(arrays)
    reset_arrays["ground_contact_count"] = np.array((1, 0, 1), dtype=np.int64)
    reset_arrays["ground_contact_transition_point_count"] = np.array(
        (1, 0, 1), dtype=np.int64
    )
    with pytest.raises(AuditError, match="first transition interval"):
        _audit_flybody_ground_contact_disclosure(
            contact_at_reset,
            reset_manifest,
            reset_arrays,
        )


def test_flybody_contact_disclosure_accepts_reset_and_no_contact_contracts():
    episode, run_manifest, arrays = _ground_contact_contract_fixture()

    reset_episode = deepcopy(episode)
    reset_episode["ground_contact_summary"].update(
        {
            "initial_count": 1,
            "first_transition_start_s": 0.0,
            "first_transition_end_s": 0.0,
        }
    )
    reset_episode["frames"][0]["ground_contact_count"] = 1
    reset_manifest = deepcopy(run_manifest)
    reset_manifest["diagnostics"]["metrics"].update(
        {
            "initial_ground_contact_count": 1.0,
            "first_ground_contact_transition_start_s_or_duration_s": 0.0,
            "first_ground_contact_transition_end_s_or_duration_s": 0.0,
        }
    )
    reset_arrays = dict(arrays)
    reset_arrays["ground_contact_count"] = np.array((1, 0, 1), dtype=np.int64)
    reset_arrays["ground_contact_transition_point_count"] = np.array(
        (1, 0, 1), dtype=np.int64
    )
    _audit_flybody_ground_contact_disclosure(
        reset_episode, reset_manifest, reset_arrays
    )

    no_contact_episode = deepcopy(episode)
    no_contact_episode["ground_contact_summary"].update(
        {
            "occurred": False,
            "first_transition_start_s": None,
            "first_transition_end_s": None,
            "transition_count": 0,
            "maximum_count": 0,
        }
    )
    for frame in no_contact_episode["frames"]:
        frame["ground_contact_count"] = 0
    no_contact_manifest = deepcopy(run_manifest)
    no_contact_manifest["diagnostics"]["metrics"].update(
        {
            "ground_contact_transition_count": 0.0,
            "maximum_ground_contact_count": 0.0,
            "ground_contact_occurred": 0.0,
            "first_ground_contact_transition_start_s_or_duration_s": 0.002,
            "first_ground_contact_transition_end_s_or_duration_s": 0.002,
        }
    )
    no_contact_arrays = dict(arrays)
    no_contact_arrays["ground_contact_count"] = np.zeros(3, dtype=np.int64)
    no_contact_arrays["ground_contact_transition_point_count"] = np.zeros(
        3, dtype=np.int64
    )
    _audit_flybody_ground_contact_disclosure(
        no_contact_episode, no_contact_manifest, no_contact_arrays
    )


def test_flybody_contact_disclosure_rejects_missing_sample_semantics():
    episode, run_manifest, arrays = _ground_contact_contract_fixture()
    episode["ground_contact_summary"].pop("sample_semantics")
    with pytest.raises(AuditError, match="sample semantics"):
        _audit_flybody_ground_contact_disclosure(episode, run_manifest, arrays)

    episode, run_manifest, arrays = _ground_contact_contract_fixture()
    run_manifest["arrays"]["ground_contact_transition_point_count"][
        "sample_semantics"
    ] = "ambiguous post-step collision query"
    with pytest.raises(AuditError, match="authoritative transition telemetry provenance"):
        _audit_flybody_ground_contact_disclosure(episode, run_manifest, arrays)


def test_flybody_contact_disclosure_rejects_tampered_full_step_trace():
    episode, run_manifest, arrays = _ground_contact_contract_fixture()
    arrays["ground_contact_transition_point_count"] = np.array(
        (0, 1, 0), dtype=np.int64
    )
    with pytest.raises(AuditError, match="authoritative full-step"):
        _audit_flybody_ground_contact_disclosure(episode, run_manifest, arrays)


def _wingbeat_inspection_contract_fixture(
    *,
    duration_s: float = 0.1,
    source_kind: str = "exploratory_frozen_browser_nod1_flybody_pipeline",
):
    physics_dt_s = 1.0e-4
    physics_time_s = np.arange(
        int(round(duration_s / physics_dt_s)) + 1, dtype=float
    ) * physics_dt_s
    phase = 2.0 * np.pi * 200.0 * physics_time_s
    physics_wing = np.column_stack(
        tuple((0.1 + 0.01 * axis) * np.sin(phase + 0.1 * axis) for axis in range(6))
    )
    physics_torque = np.column_stack(
        tuple(
            (axis + 1) * 1.0e-10 * np.cos(phase + 0.2 * axis)
            for axis in range(6)
        )
    )
    inspection_end_s = min(0.060, duration_s)
    inspection_count = int(round(inspection_end_s / physics_dt_s)) + 1
    wing_order = [
        "c_thorax-l_wing-yaw",
        "c_thorax-l_wing-roll",
        "c_thorax-l_wing-pitch",
        "c_thorax-r_wing-yaw",
        "c_thorax-r_wing-roll",
        "c_thorax-r_wing-pitch",
    ]
    episode = {
        "physics_backend": "flybody",
        "source_kind": source_kind,
        "duration_s": duration_s,
        "measured_wing_joint_order": wing_order,
        "wingbeat_inspection": {
            "sample_rate_hz": 10_000.0,
            "start_s": 0.0,
            "end_s": inspection_end_s,
            "time_s": physics_time_s[:inspection_count].tolist(),
            "wing_joint_order": wing_order.copy(),
            "measured_wing_joint_angle_rad": physics_wing[
                :inspection_count
            ].tolist(),
            "external_actuator_torque_n_m": physics_torque[
                :inspection_count
            ].tolist(),
        },
    }
    run_manifest = {
        "configuration": {
            "duration_s": duration_s,
            "physics_timestep_s": physics_dt_s,
        },
        "arrays": {
            "physics_time_s": {
                "unit": "s",
                "provenance": "authoritative_external_physics_clock",
            },
            "measured_wing_joint_angle_physics_rad": {
                "unit": "rad",
                "provenance": "physics_rate_external_measured_wing_output",
                "axis_labels": wing_order.copy(),
                "sample_semantics": WINGBEAT_ANGLE_SAMPLE_SEMANTICS,
            },
            "external_actuator_torque_physics_n_m": {
                "unit": "N m",
                "provenance": (
                    "physics_rate_external_mujoco_wing_actuator_torque"
                ),
                "axis_labels": wing_order.copy(),
                "sample_semantics": WINGBEAT_TORQUE_SAMPLE_SEMANTICS,
            },
        },
    }
    arrays = {
        "physics_time_s": physics_time_s,
        "measured_wing_joint_angle_physics_rad": physics_wing,
        "external_actuator_torque_physics_n_m": physics_torque,
    }
    return episode, run_manifest, arrays


@pytest.mark.parametrize(
    "source_kind,duration_s",
    (
        ("exploratory_frozen_browser_nod1_flybody_pipeline", 0.1),
        ("exploratory_retinal_nod1_flybody_pipeline", 0.02),
    ),
)
def test_flybody_wingbeat_inspection_binds_exact_physics_rate_excerpt(
    source_kind, duration_s
):
    episode, run_manifest, arrays = _wingbeat_inspection_contract_fixture(
        source_kind=source_kind,
        duration_s=duration_s,
    )
    _audit_flybody_wingbeat_inspection(episode, run_manifest, arrays)


def test_target_flybody_episode_requires_wingbeat_inspection():
    episode, run_manifest, arrays = _wingbeat_inspection_contract_fixture()
    episode.pop("wingbeat_inspection")
    with pytest.raises(AuditError, match="require wingbeat_inspection"):
        _audit_flybody_wingbeat_inspection(episode, run_manifest, arrays)

    episode["source_kind"] = "exploratory_flybody_mujoco_worker"
    _audit_flybody_wingbeat_inspection(episode, run_manifest, {})


@pytest.mark.parametrize(
    "mutation,expected",
    (
        ("spacing", "0.1 ms spacing"),
        ("sample_rate", "exactly 10 kHz"),
        ("wing_value", "authoritative physics-rate arrays"),
        ("nonfinite", "finite time-aligned"),
        ("axis", "six unique named"),
        ("provenance", "lack exact axis"),
    ),
)
def test_flybody_wingbeat_inspection_rejects_tampering(mutation, expected):
    episode, run_manifest, arrays = _wingbeat_inspection_contract_fixture()
    inspection = episode["wingbeat_inspection"]
    if mutation == "spacing":
        inspection["time_s"][10] += 1.0e-5
    elif mutation == "sample_rate":
        inspection["sample_rate_hz"] = 9_999.0
    elif mutation == "wing_value":
        inspection["measured_wing_joint_angle_rad"][10][0] += 1.0e-6
    elif mutation == "nonfinite":
        inspection["external_actuator_torque_n_m"][10][0] = float("nan")
    elif mutation == "axis":
        inspection["wing_joint_order"][1] = inspection["wing_joint_order"][0]
    else:
        run_manifest["arrays"]["measured_wing_joint_angle_physics_rad"][
            "provenance"
        ] = "decimated_proxy"
    with pytest.raises(AuditError, match=expected):
        _audit_flybody_wingbeat_inspection(episode, run_manifest, arrays)


def _frozen_browser_contract_fixture(registry):
    parity_case = registry.case("nod1.browser_python_parity")
    time_s = np.linspace(0.0, FROZEN_BROWSER_DURATION_S, 501)
    circuit_time_s = np.arange(100, dtype=float) * 0.005
    wing_phase = np.linspace(0.0, 8.0 * np.pi, len(time_s))
    measured_angle = np.zeros((len(time_s), 6), dtype=float)
    measured_angle[:, 0] = 0.1 * np.sin(wing_phase)
    measured_velocity = np.gradient(measured_angle, time_s, axis=0)
    whole_fly_com = np.zeros((len(time_s), 3), dtype=float)
    whole_fly_com[:, 2] = 0.1 - 0.01 * time_s
    actuator_torque = np.zeros((len(time_s), 6), dtype=float)
    actuator_torque[1:, 0] = 1.0e-9
    fluid_force = np.zeros((len(time_s), 3), dtype=float)
    fluid_force[1:, 2] = 1.0e-6
    circuit_ids = tuple("circuit_voltage/nod1-%d" % index for index in range(4))
    availability_ids = tuple(
        "circuit_availability_time_s/nod1-%d" % index for index in range(4)
    )
    arrays = {
        "time_s": time_s,
        "ground_contact_count": np.zeros(len(time_s), dtype=np.int64),
        "circuit_sample_time_s": circuit_time_s,
        "measured_wing_joint_angle_rad": measured_angle,
        "measured_wing_joint_velocity_rad_s": measured_velocity,
        "whole_fly_com_position_world_m": whole_fly_com,
        "external_actuator_torque_n_m": actuator_torque,
        "aerodynamic_force_body_n": fluid_force,
        "descending_rate_hz/DNp26/right": np.ones(1000),
        "wing_motor_rate_hz/MN-iv1/right": np.ones(1000),
        "wing_motor_event_availability_time_s/MN-iv1/right": np.array((0.03775,)),
        "muscle_activation/right:iv1": np.linspace(0.0, 0.5, len(time_s)),
    }
    descriptors = {
        array_id: {"unit": "1", "provenance": "manufactured"}
        for array_id in arrays
    }
    descriptors.update(
        {
            "time_s": {"unit": "s", "provenance": "shared_episode_clock"},
            "measured_wing_joint_angle_rad": {
                "unit": "rad",
                "provenance": "external_physics_measured_output",
            },
            "measured_wing_joint_velocity_rad_s": {
                "unit": "rad s^-1",
                "provenance": "external_physics_measured_output",
            },
            "whole_fly_com_position_world_m": {
                "unit": "m",
                "provenance": "flybody_mujoco_articulated_subtree_com",
            },
            "external_actuator_torque_n_m": {
                "unit": "N m",
                "provenance": (
                    "logged_projection_of_external_mujoco_wing_actuator_torque"
                ),
            },
            "aerodynamic_force_body_n": {
                "unit": "N",
                "provenance": "flybody_mujoco_root_total_output",
            },
        }
    )
    signals = []
    for index, (value_id, availability_id) in enumerate(
        zip(circuit_ids, availability_ids)
    ):
        arrays[value_id] = -0.055 + index * 1.0e-4 + np.linspace(0.0, 1.0e-4, 100)
        arrays[availability_id] = circuit_time_s.copy()
        descriptors[value_id] = {
            "unit": "V",
            "provenance": "registered_frozen_chromium_browser_voltage_v",
        }
        descriptors[availability_id] = {
            "unit": "s",
            "provenance": "registered_frozen_chromium_readout_timing",
        }
        signals.append(
            {
                "value_array": value_id,
                "availability_time_array": availability_id,
            }
        )
    trace_sha = hashlib.sha256(b"manufactured circuit trace").hexdigest()
    run_manifest = {
        "source_kind": "exploratory_flybody_mujoco_worker",
        "configuration": {
            "duration_s": FROZEN_BROWSER_DURATION_S,
            "physics_timestep_s": 1.0e-4,
        },
        "runtime": {"physics_backend": "flybody"},
        "diagnostics": {
            "metrics": {
                "maximum_external_actuator_torque_n_m": 1.0e-9,
                "maximum_measured_wing_excursion_rad": 0.2,
            }
        },
        "pipeline": {
            "pipeline_id": "frozen-browser-nod1-parity-to-dnp26-to-flight-v1",
            "mode": "open_loop_frozen_browser_visual_circuit_output",
            "circuit_trace_sha256": trace_sha,
            "visual_boundary": {
                "retinal_frames_present": False,
                "status": "frozen_browser_output_no_retinal_frames",
            },
            "source_metadata": {
                "input_mode": "registered_frozen_browser_nod1_parity",
                "visual_source": "frozen_browser_visual_circuit_output",
                "retinal_frames_present": False,
                "retinal_frame_count": 0,
                "fixture_uri": parity_case.fixture_uri,
                "fixture_sha256": parity_case.input_sha256,
                "benchmark_case_id": "nod1.browser_python_parity",
                "benchmark_case_version": parity_case.version,
                "source_duration_s": FROZEN_BROWSER_DURATION_S,
                "source_sample_dt_s": 0.005,
                "source_sample_count": 100,
                "source_sample_interval_end_s": FROZEN_BROWSER_DURATION_S,
                "circuit_model_scope": (
                    "frozen_real_chromium_1208_cell_circuit_four_nod1_readouts"
                ),
                "circuit_cell_count": 1208,
                "exported_circuit_channels": 4,
                "circuit_inventory": {"cells": 1208},
            },
            "circuit_output_contract": {
                "exact_timebase": True,
                "sample_count": 100,
                "signals": signals,
            },
        },
        "arrays": descriptors,
    }
    frame = {
        "root_position_m": [0.0, 0.0, 0.1],
        "position_m": [0.0, 0.0, 0.1],
        "measured_wing_joint_angle_rad": [0.0] * 6,
        "measured_wing_joint_velocity_rad_s": [0.0] * 6,
    }
    episode = {
        "id": FROZEN_BROWSER_EPISODE_ID,
        "physics_backend": "flybody",
        "source_kind": "exploratory_frozen_browser_nod1_flybody_pipeline",
        "wing_kinematics_source": "measured_flybody_yaw_axes",
        "body_state_reference": (
            "whole-fly articulated subtree COM for display; root/thorax retained separately"
        ),
        "duration_s": FROZEN_BROWSER_DURATION_S,
        "source_circuit_trace_sha256": trace_sha,
        "neural_model_scope": {
            "kind": "full_legacy_circuit_cable_export",
            "historical": True,
            "full_circuit_executed": True,
            "circuit_cell_count": 1208,
            "exported_circuit_channel_count": 4,
        },
        "measured_wing_joint_order": list(REVIEWED_FLYBODY_WING_AXIS_ORDER),
        "frames": [deepcopy(frame), deepcopy(frame)],
    }
    return episode, run_manifest, arrays


def test_frozen_browser_release_contract_binds_fixture_scope_and_causal_outputs():
    registry = load_benchmark_registry()
    episode, run_manifest, arrays = _frozen_browser_contract_fixture(registry)
    _audit_frozen_browser_nod1_episode(
        episode, run_manifest, registry, arrays
    )

    wrong_fixture = deepcopy(run_manifest)
    wrong_fixture["pipeline"]["source_metadata"]["fixture_sha256"] = "0" * 64
    with pytest.raises(AuditError, match="registered fixture SHA/scope/timebase"):
        _audit_frozen_browser_nod1_episode(
            episode, wrong_fixture, registry, arrays
        )

    wrong_backend = deepcopy(episode)
    wrong_backend["physics_backend"] = "reduced_order"
    with pytest.raises(AuditError, match="must execute the FlyBody worker"):
        _audit_frozen_browser_nod1_episode(
            wrong_backend, run_manifest, registry, arrays
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "duration",
        "scope",
        "missing_com",
        "dead_motor_events",
        "dead_actuator",
        "dead_measured_wing",
        "mechanics_provenance",
        "circuit_timebase",
    ),
)
def test_frozen_browser_release_contract_rejects_scope_or_output_substitution(mutation):
    registry = load_benchmark_registry()
    episode, run_manifest, arrays = _frozen_browser_contract_fixture(registry)
    arrays = dict(arrays)
    if mutation == "duration":
        episode["duration_s"] = 0.49
        expected = "exactly 0.5 seconds"
    elif mutation == "scope":
        episode["neural_model_scope"]["circuit_cell_count"] = 1207
        expected = "explicitly historical and retain its scope"
    elif mutation == "missing_com":
        arrays.pop("whole_fly_com_position_world_m")
        expected = "missing measured wing"
    elif mutation == "dead_motor_events":
        arrays["wing_motor_event_availability_time_s/MN-iv1/right"] = np.array(())
        expected = "causal output is dead"
    elif mutation == "dead_actuator":
        arrays["external_actuator_torque_n_m"] = np.zeros((501, 6))
        expected = "causal output is dead"
    elif mutation == "dead_measured_wing":
        arrays["measured_wing_joint_angle_rad"] = np.zeros((501, 6))
        expected = "causal output is dead"
    elif mutation == "mechanics_provenance":
        run_manifest["arrays"]["whole_fly_com_position_world_m"][
            "provenance"
        ] = "unreviewed_proxy"
        expected = "lack authoritative units or provenance"
    else:
        arrays["circuit_availability_time_s/nod1-0"] = (
            arrays["circuit_availability_time_s/nod1-0"] + 0.001
        )
        expected = "wrong shape or timing"
    with pytest.raises(AuditError, match=expected):
        _audit_frozen_browser_nod1_episode(
            episode, run_manifest, registry, arrays
        )


def test_frozen_browser_release_requires_matching_worker_validation_pass():
    registry = load_benchmark_registry()
    case = registry.case(FROZEN_BROWSER_FLYBODY_CASE_ID)
    passed = SimpleNamespace(
        results=(
            SimpleNamespace(
                case_id=case.case_id,
                case_version=case.version,
                status=GateStatus.PASS,
            ),
        )
    )
    _audit_required_validation_pass(
        passed, registry, FROZEN_BROWSER_FLYBODY_CASE_ID
    )

    blocked = SimpleNamespace(
        results=(
            SimpleNamespace(
                case_id=case.case_id,
                case_version=case.version,
                status=GateStatus.BLOCKED,
            ),
        )
    )
    with pytest.raises(AuditError, match="matching validation PASS"):
        _audit_required_validation_pass(
            blocked, registry, FROZEN_BROWSER_FLYBODY_CASE_ID
        )


@pytest.mark.parametrize("field", ("unit", "provenance"))
def test_run_manifest_requires_unit_and_provenance_per_array(tmp_path, field):
    config, result = run_scenario("baseline", duration_s=0.010)
    run_dir = tmp_path / episode_run_id("baseline", config)
    write_episode_artifact(run_dir, "baseline", config, result)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["arrays"]["time_s"][field]
    payload = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(payload)
    (run_dir / "manifest.sha256").write_text(
        "%s  manifest.json\n" % hashlib.sha256(payload).hexdigest(),
        encoding="ascii",
    )

    with pytest.raises(AuditError, match=field):
        _audit_run_manifest(manifest_path)


def test_complete_flybody_release_binds_report_and_run_to_one_worker_image():
    registry = load_benchmark_registry()
    report = ValidationRunner(registry, default_evaluators(registry)).run(
        evaluation_id="worker-image-audit-test"
    )
    assert report.suite_complete
    image_sha = hashlib.sha256(b"manufactured OCI worker image").hexdigest()
    lock_sha = hashlib.sha256(b"manufactured dependency lock").hexdigest()
    run_manifest = {
        "runtime": {
            "physics_provenance": {
                "worker_image_digest": "sha256:" + image_sha,
                "worker_dependency_lock": {
                    "name": "requirements.lock",
                    "sha256": "sha256:" + lock_sha,
                },
            }
        }
    }

    with pytest.raises(AuditError, match="requires one worker image receipt"):
        _audit_worker_image_binding(report, (run_manifest,))
    with pytest.raises(AuditError, match="requires one dependency-lock receipt"):
        _audit_worker_dependency_lock_binding(report, (run_manifest,))

    bound = replace(
        report,
        source_digests=report.source_digests
        + (
            SourceDigest(
                kind="image",
                name="fly-s2b-worker-image",
                sha256=image_sha,
            ),
            SourceDigest(
                kind="dependency_lock",
                name="requirements.lock",
                sha256=lock_sha,
            ),
        ),
    )
    assert _audit_worker_image_binding(bound, (run_manifest,)) == image_sha
    assert _audit_worker_dependency_lock_binding(bound, (run_manifest,)) == lock_sha
    canonical_run_manifest = {
        "schema_version": "1.1.0",
        "runtime": {
            "receipts": {
                "flybody": {
                    "worker_image_digest": "sha256:" + image_sha,
                    "worker_dependency_lock": {
                        "name": "requirements.lock",
                        "sha256": "sha256:" + lock_sha,
                    },
                }
            }
        },
    }
    assert (
        _audit_worker_image_binding(bound, (canonical_run_manifest,))
        == image_sha
    )
    assert (
        _audit_worker_dependency_lock_binding(bound, (canonical_run_manifest,))
        == lock_sha
    )

    wrong_run = {
        "runtime": {
            "physics_provenance": {
                "worker_image_digest": "sha256:" + "0" * 64,
            }
        }
    }
    with pytest.raises(AuditError, match="does not match"):
        _audit_worker_image_binding(bound, (wrong_run,))
    wrong_lock_run = {
        "runtime": {
            "physics_provenance": {
                "worker_dependency_lock": {
                    "name": "requirements.lock",
                    "sha256": "sha256:" + "0" * 64,
                }
            }
        }
    }
    with pytest.raises(AuditError, match="does not match"):
        _audit_worker_dependency_lock_binding(bound, (wrong_lock_run,))


@pytest.fixture(scope="module")
def canonical_fly_fgs_release_contract():
    fixture = load_registered_fly_fgs_fixture()
    replay = fixture.circuit_replay_attachment()
    source = dict(fixture.pipeline_source_metadata())
    source.update(
        {
            "fixture_kind": "fly_fgs_fixed_step_circuit_capture",
            "snapshot_id": fixture.snapshot_id,
            "circuit_inventory": dict(fixture.circuit_inventory),
            "exported_circuit_channels": 4,
            "motor_input_scope": "four_registered_nod1_voltage_channels_only",
            "circuit_replay_present": True,
            "full_cell_state_eligible_motor_input": False,
            "fly_fgs_downstream_mechanics_imported": False,
        }
    )
    source_time = np.arange(100, dtype=float) * 0.005
    full_state = replay["full_cell_state"]
    voltage = np.asarray(full_state["voltage_v"], dtype=float)
    activity = np.asarray(full_state["activity"], dtype=float)
    retinal = np.asarray(
        replay["stimulus"]["retinal_input"]["luminance"], dtype=float
    )
    logging_time = np.arange(501, dtype=float) * 0.001
    phase = 2.0 * np.pi * 200.0 * logging_time
    measured_angle = np.column_stack(
        tuple(0.1 * np.sin(phase + 0.1 * axis) for axis in range(6))
    )
    measured_velocity = np.gradient(measured_angle, 0.001, axis=0)
    actuator = np.column_stack(
        tuple(1.0e-10 * (axis + 1) * np.cos(phase) for axis in range(6))
    )
    fluid = np.column_stack(
        (1.0e-6 * np.cos(phase), 1.0e-6 * np.sin(phase), 2.0e-6 + 0.0 * phase)
    )
    whole_com = np.column_stack(
        (1.0e-3 * logging_time, 0.0 * logging_time, 0.01 + 0.0 * logging_time)
    )
    arrays = {
        "time_s": logging_time,
        "circuit_sample_time_s": source_time,
        "retinal_normalized_luminance": retinal[1:],
        "retinal_measurement_time_s": source_time[1:],
        "retinal_availability_time_s": source_time[1:],
        "retinal_exposure_interval_s": np.column_stack(
            (source_time[:-1], source_time[1:])
        ),
        "fly_fgs_source_time_s": source_time,
        "fly_fgs_full_cell_voltage_v": voltage,
        "fly_fgs_full_cell_activity": activity,
        "fly_fgs_t4a_input_luminance": retinal,
        "measured_wing_joint_angle_rad": measured_angle,
        "measured_wing_joint_velocity_rad_s": measured_velocity,
        "whole_fly_com_position_world_m": whole_com,
        "external_actuator_torque_n_m": actuator,
        "aerodynamic_force_body_n": fluid,
        "wing_motor_event_availability_time_s/MN-iv2/left": np.array((0.1,)),
        "muscle_activation/left:iv2": np.linspace(0.0, 0.5, len(logging_time)),
    }
    descriptors = {
        "time_s": {"unit": "s", "provenance": "shared_episode_clock"},
        "circuit_sample_time_s": {
            "unit": "s",
            "provenance": "registered_fly_fgs_exact_timebase",
        },
        "retinal_normalized_luminance": {
            "unit": "1",
            "provenance": "causal_retinal_frame",
        },
        "retinal_measurement_time_s": {
            "unit": "s",
            "provenance": "causal_retinal_frame_timing",
        },
        "retinal_availability_time_s": {
            "unit": "s",
            "provenance": "causal_retinal_frame_timing",
        },
        "retinal_exposure_interval_s": {
            "unit": "s",
            "provenance": "causal_retinal_frame_timing",
        },
        "fly_fgs_source_time_s": {
            "unit": "s",
            "provenance": "registered_fly_fgs_fixed_step_source_clock",
        },
        "fly_fgs_full_cell_voltage_v": {
            "unit": "V",
            "provenance": "registered_fly_fgs_full_cell_display_state_not_motor_input",
        },
        "fly_fgs_full_cell_activity": {
            "unit": "1",
            "provenance": "registered_fly_fgs_full_cell_display_state_not_motor_input",
        },
        "fly_fgs_t4a_input_luminance": {
            "unit": "1",
            "provenance": "registered_fly_fgs_exact_t4a_analytic_input",
        },
        "measured_wing_joint_angle_rad": {
            "unit": "rad",
            "provenance": "external_physics_measured_output",
        },
        "measured_wing_joint_velocity_rad_s": {
            "unit": "rad s^-1",
            "provenance": "external_physics_measured_output",
        },
        "whole_fly_com_position_world_m": {
            "unit": "m",
            "provenance": "flybody_mujoco_articulated_subtree_com",
        },
        "external_actuator_torque_n_m": {
            "unit": "N m",
            "provenance": "logged_projection_of_external_mujoco_wing_actuator_torque",
        },
        "aerodynamic_force_body_n": {
            "unit": "N",
            "provenance": "flybody_mujoco_root_total_output",
        },
        "wing_motor_event_availability_time_s/MN-iv2/left": {
            "unit": "s",
            "provenance": "seeded_synthetic_event_actuation_time",
        },
        "muscle_activation/left:iv2": {
            "unit": "1",
            "provenance": "virtual_hinge_muscle_state",
        },
    }
    signals = []
    for root_id in fixture.circuit_trace.signals:
        entity_id = root_id.neuron.entity_id
        raw_side = replay["side_semantics"]["raw_bundle_labels"][entity_id]
        value_id = "circuit_voltage/" + entity_id
        availability_id = "circuit_availability_time_s/" + entity_id
        column = full_state["cell_ids"].index("r" + entity_id)
        arrays[value_id] = voltage[:, column]
        arrays[availability_id] = source_time
        descriptors[value_id] = {
            "unit": "V",
            "provenance": "registered_fly_fgs_fixed_step_voltage_v",
        }
        descriptors[availability_id] = {
            "unit": "s",
            "provenance": "registered_fly_fgs_fixed_step_timing",
        }
        signals.append(
            {
                "neuron": {
                    "entity_id": entity_id,
                    "cell_type": "NOD1",
                    "anatomical_side": "unknown",
                    "side_context": {
                        "raw_dataset_side": raw_side,
                        "anatomical_side": "unknown",
                        "mapping_method": "simulation_convention",
                    },
                },
                "signal_kind": "voltage",
                "unit": "V",
                "value_array": value_id,
                "availability_time_array": availability_id,
            }
        )
    run_manifest = {
        "source_kind": "exploratory_flybody_mujoco_worker",
        "configuration": {
            "duration_s": 0.5,
            "physics_timestep_s": 1.0e-4,
        },
        "runtime": {"physics_backend": "flybody"},
        "diagnostics": {"metrics": {}},
        "pipeline": {
            "pipeline_id": "fly-fgs-retina-circuit-to-flight-v1",
            "mode": "open_loop_registered_fly_fgs_fixed_step_circuit_output",
            "circuit_trace_sha256": (
                "0db62b35fdcd5fd5ad615e8e52e8e390459c1d0d75ec90dfdfb3c8de5e5c1f75"
            ),
            "visual_boundary": {
                "retinal_frames_present": True,
                "status": "registered_retinal_frames_and_circuit_replay",
            },
            "circuit_replay_contract": {
                "attachment": "web_replay.circuit_replay",
                "scientific_array_prefix": "fly_fgs_",
                "source_time_array": "fly_fgs_source_time_s",
                "full_cell_voltage_array": "fly_fgs_full_cell_voltage_v",
                "full_cell_activity_array": "fly_fgs_full_cell_activity",
                "t4a_input_luminance_array": "fly_fgs_t4a_input_luminance",
                "motor_input_policy": "four CircuitOutputTrace NOD1 voltage channels only",
                "full_cell_state_eligible_motor_input": False,
                "raw_app_side_labels_are_anatomical": False,
            },
            "source_metadata": source,
            "circuit_output_contract": {
                "exact_timebase": True,
                "sample_count": 100,
                "signals": signals,
            },
        },
        "arrays": descriptors,
    }
    episode = {
        "id": FLY_FGS_EPISODE_ID,
        "physics_backend": "flybody",
        "source_kind": FLY_FGS_SOURCE_KIND,
        "duration_s": 0.5,
        "wing_kinematics_source": "measured_flybody_yaw_axes",
        "body_state_reference": (
            "whole-fly articulated subtree COM for display; root/thorax retained separately"
        ),
        "measured_wing_joint_order": list(REVIEWED_FLYBODY_WING_AXIS_ORDER),
        "source_circuit_trace_sha256": (
            "0db62b35fdcd5fd5ad615e8e52e8e390459c1d0d75ec90dfdfb3c8de5e5c1f75"
        ),
        "neural_model_scope": {
            "kind": "registered_fly_fgs_fixed_step_circuit",
            "full_circuit_executed": True,
            "circuit_cell_count": 1684,
            "exported_circuit_channel_count": 4,
        },
        "circuit_replay": replay,
    }
    return episode, run_manifest, arrays


def test_canonical_fly_fgs_release_binds_source_state_retina_and_motor_cut(
    canonical_fly_fgs_release_contract,
):
    episode, run_manifest, arrays = canonical_fly_fgs_release_contract
    _audit_fly_fgs_canonical_episode(
        episode,
        run_manifest,
        load_benchmark_registry(),
        arrays,
    )


def test_canonical_fly_fgs_release_rejects_motor_eligible_full_state(
    canonical_fly_fgs_release_contract,
):
    episode, run_manifest, arrays = canonical_fly_fgs_release_contract
    changed_episode = dict(episode)
    changed_replay = dict(episode["circuit_replay"])
    changed_scope = dict(changed_replay["full_cell_state_scope"])
    changed_scope["eligible_motor_input"] = True
    changed_replay["full_cell_state_scope"] = changed_scope
    changed_episode["circuit_replay"] = changed_replay
    with pytest.raises(AuditError, match="became motor eligible"):
        _audit_fly_fgs_canonical_episode(
            changed_episode,
            run_manifest,
            load_benchmark_registry(),
            arrays,
        )


def test_canonical_fly_fgs_release_rejects_retinal_shape_substitution(
    canonical_fly_fgs_release_contract,
):
    episode, run_manifest, arrays = canonical_fly_fgs_release_contract
    changed_arrays = dict(arrays)
    changed_arrays["retinal_normalized_luminance"] = changed_arrays[
        "retinal_normalized_luminance"
    ][:-1]
    with pytest.raises(AuditError, match="chunked state/retinal arrays"):
        _audit_fly_fgs_canonical_episode(
            episode,
            run_manifest,
            load_benchmark_registry(),
            changed_arrays,
        )


def test_fly_fgs_source_mechanics_are_only_allowed_in_rejected_disclosures():
    _audit_no_fly_fgs_downstream_mechanics(
        {
            "rejected_downstream_fields": ["SCALE_M", "state.muscleAct"],
            "notice": "SCALE_M and wingL are explicitly excluded",
            "muscle_activation/project_owned": [0.0, 0.1],
        },
        label="fixture",
    )
    with pytest.raises(AuditError, match="forbidden fly-FGS downstream mechanics"):
        _audit_no_fly_fgs_downstream_mechanics(
            {"state": {"muscleAct": [0.0, 1.0]}},
            label="fixture",
        )


def test_fly_fgs_validation_inventory_requires_all_v113_cases():
    registry = load_benchmark_registry()
    integrity = registry.case(FLY_FGS_INTEGRITY_CASE_ID)
    incremental = registry.case(FLY_FGS_INCREMENTAL_CASE_ID)
    checkpoint = registry.case(FLY_FGS_CHECKPOINT_CASE_ID)
    worker = registry.case(FLY_FGS_FLYBODY_CASE_ID)

    def report(worker_status):
        return SimpleNamespace(
            results=(
                SimpleNamespace(
                    case_id=integrity.case_id,
                    case_version=integrity.version,
                    status=GateStatus.PASS,
                ),
                SimpleNamespace(
                    case_id=incremental.case_id,
                    case_version=incremental.version,
                    status=GateStatus.PASS,
                ),
                SimpleNamespace(
                    case_id=checkpoint.case_id,
                    case_version=checkpoint.version,
                    status=GateStatus.PASS,
                ),
                SimpleNamespace(
                    case_id=worker.case_id,
                    case_version=worker.version,
                    status=worker_status,
                ),
            )
        )

    _audit_fly_fgs_validation_inventory(
        report(GateStatus.BLOCKED), registry, canonical_attached=False
    )
    _audit_fly_fgs_validation_inventory(
        report(GateStatus.PASS), registry, canonical_attached=True
    )
    with pytest.raises(AuditError, match="requires a PASS"):
        _audit_fly_fgs_validation_inventory(
            report(GateStatus.BLOCKED), registry, canonical_attached=True
        )


def test_preferred_browser_episode_mirrors_baseline_then_initial_selector_order():
    episodes = (
        {"id": "baseline", "physics_backend": "reduced_order"},
        {"id": "frozen_browser_nod1", "physics_backend": "flybody"},
        {"id": FLY_FGS_EPISODE_ID, "physics_backend": "flybody"},
    )
    assert _preferred_episode_index(episodes) == 1
    assert (
        _preferred_episode_index(
            (
                {"id": "baseline", "physics_backend": "reduced_order"},
                {"id": "worker", "physics_backend": "flybody"},
            )
        )
        == 1
    )
    assert _preferred_episode_index((episodes[0],)) == 0


def test_canonical_fly_fgs_declaration_is_optional_but_cannot_be_aliased():
    legacy_binding = (
        {"id": "baseline", "neural_model_scope": {"kind": "illustrative"}},
        Path("/tmp/baseline/manifest.json"),
        {"pipeline": {"pipeline_id": "illustrative"}},
    )
    assert _canonical_fly_fgs_binding(
        ({"id": "baseline"},), (legacy_binding,)
    ) is None

    canonical_binding = (
        {
            "id": FLY_FGS_EPISODE_ID,
            "neural_model_scope": {
                "kind": "registered_fly_fgs_fixed_step_circuit"
            },
        },
        Path("/tmp/fly-fgs/manifest.json"),
        {"pipeline": {"pipeline_id": "fly-fgs-retina-circuit-to-flight-v1"}},
    )
    assert _canonical_fly_fgs_binding(
        ({"id": FLY_FGS_EPISODE_ID},), (canonical_binding,)
    ) == canonical_binding

    aliased = (
        {
            "id": "misleading_alias",
            "neural_model_scope": {
                "kind": "registered_fly_fgs_fixed_step_circuit"
            },
        },
        Path("/tmp/alias/manifest.json"),
        {"pipeline": {"pipeline_id": "fly-fgs-retina-circuit-to-flight-v1"}},
    )
    with pytest.raises(AuditError, match="exactly one fly_fgs_canonical"):
        _canonical_fly_fgs_binding(
            ({"id": "misleading_alias"},), (aliased,)
        )


def _canonical_digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


class _AuditedFlyBodyPhysics(FakeFlightPhysics):
    """Fast deterministic double carrying the exact native identity boundary."""

    backend_name = "flybody"
    aerodynamic_owner = "flybody"
    wing_joint_order = REVIEWED_FLYBODY_WING_AXIS_ORDER

    def checkpoint(self):
        payload = dict(super().checkpoint())
        payload.pop("payload_sha256")
        payload.update(
            {
                "compiled_model_sha256": "sha256:" + "6" * 64,
                "worker_versions": {"flygym": "2.1.0", "mujoco": "3.9.0"},
                "physics_timestep_s": 0.0001,
                "wing_dof_order": list(REVIEWED_FLYBODY_WING_AXIS_ORDER),
            }
        )
        return {
            **payload,
            "payload_sha256": "sha256:" + _canonical_digest(payload),
        }


def _audited_flybody_receipt():
    return {
        "schema_version": "1.0.0",
        "engine": "FlyGym/FlyBody with native MuJoCo",
        "worker_versions": {"flygym": "2.1.0", "mujoco": "3.9.0"},
        "worker_image_digest": "sha256:" + "7" * 64,
        "worker_image_digest_status": "declared_oci_digest",
        "worker_dependency_lock": {
            "name": "requirements.lock",
            "sha256": "sha256:" + "8" * 64,
        },
        "dependency_record_sha256": {
            "flygym": "sha256:" + "9" * 64,
            "mujoco": "sha256:" + "a" * 64,
        },
        "compiled_model_fingerprint": {"sha256": "sha256:" + "6" * 64},
        "worker_config": {
            "timestep_s": 0.0001,
            "spawn_height_m": 0.1,
            "max_abs_wing_torque_n_m": 3.0e-6,
            "add_aerodynamic_geoms": True,
        },
        "wing_dof_order": list(REVIEWED_FLYBODY_WING_AXIS_ORDER),
        "fluid_geom_names": ["l_wing_fluid", "r_wing_fluid"],
        "fluid_geoms_contact_disabled": True,
        "ground_contact_topology": "legs_only",
        "ground_contact_telemetry": {
            "kind": "exact test contact count",
            "sample_semantics": "test double",
            "ground_geom_name": "ground_plane",
            "configured_pair_count": 6,
        },
        "released_policy_topology_equivalent": False,
        "tendon_count": 8,
        "source": {
            "flygym_tag": "v2.1.0",
            "flygym_commit": "ca65a510c2afe6ac61c51df4f274c8d190c2f95f",
            "flybody_aerodynamics_commit_reviewed": "d" * 40,
        },
        "licenses": {"flygym": "BSD-3-Clause", "mujoco": "Apache-2.0"},
        "body_state_reference": "deterministic test double",
        "root_fluid_wrench_scope": "root-total; no reviewed per-wing decomposition",
        "checkpoint_contract": {
            "schema_version": "1.0.0",
            "state_spec": "mjSTATE_INTEGRATION",
            "float_encoding": "float64_le_base64",
            "compatibility_bound_to_compiled_model": True,
            "fresh_adapter_exact_reentry_required": True,
        },
        "public_units": "SI",
    }


_NODE_CANONICAL_AVAILABLE = shutil.which("node") is not None and Path(
    default_fly_fgs_runtime_script_path()
).is_file()


@pytest.fixture(scope="module")
def audited_canonical_online_run(tmp_path_factory):
    if not _NODE_CANONICAL_AVAILABLE:
        pytest.skip("Node fly-FGS runtime is unavailable")
    config = CanonicalClosedLoopConfig(
        effector_laterality_hypothesis=_DEFAULT_EFFECTOR_HYPOTHESIS,
        duration_s=0.020,
        include_retinal_input=True,
        include_full_cell_state=True,
    )
    body = _initial_body(0.0, 2.0)
    base_bridge = _high_gain_bridge(seed=71)
    bridge = StreamingNOD1MotorBridge(
        base_bridge.config,
        seed=71,
        interventions=(
            StreamingBridgeIntervention(
                intervention_id="audit-dnp26-activate",
                target_stage=StreamingBridgeStage.DN,
                target_name="DNp26",
                raw_app_side=None,
                mode=StreamingInterventionMode.ACTIVATE,
                start_s=0.005,
                end_s=0.010,
                magnitude=50.0,
            ),
        ),
    )
    simulator = CanonicalClosedLoopSimulator(
        config,
        circuit_runtime=NodeFlyFGSCircuitRuntime(request_timeout_s=180.0),
        bridge=bridge,
        mechanics=StreamingMuscleWingStepper(
            StreamingMechanicsConfig(
                muscle_interventions=(
                    StreamingMuscleIntervention(
                        intervention_id="audit-iv2-silence",
                        muscle="iv2",
                        start_s=0.005,
                        end_s=0.015,
                        mode=MechanicsInterventionMode.SILENCE,
                        raw_app_side=None,
                    ),
                )
            )
        ),
        physics_adapter=_AuditedFlyBodyPhysics(body),
        initial_body_state=body,
    )
    try:
        result = simulator.run()
        checkpoint = simulator.checkpoint()
        run_id = canonical_run_id(
            result, scenario_id="audited-online", checkpoint=checkpoint
        )
        replay = canonical_closed_loop_to_web_replay(
            result,
            checkpoint=checkpoint,
            scenario_id="audited-online",
            source_run_id=run_id,
        )
        run_dir = tmp_path_factory.mktemp("canonical-online") / run_id
        runtime_receipts = {
            "fly_fgs_runtime": dict(simulator.circuit_runtime.ready_receipt),
            "flybody": _audited_flybody_receipt(),
        }
        manifest = write_canonical_closed_loop_artifact(
            run_dir,
            result,
            checkpoint=checkpoint,
            scenario_id="audited-online",
            created_at_utc="2026-07-18T00:00:00Z",
            chunk_samples=7,
            runtime_receipts=runtime_receipts,
            web_replay=replay,
        )
        bound = bind_canonical_web_replay_to_artifact(
            replay, manifest, _sha256_file(run_dir / "manifest.json")
        )
        return run_dir, manifest, bound
    finally:
        simulator.close()


def test_canonical_online_run_and_replay_are_independently_audited(
    audited_canonical_online_run,
):
    run_dir, manifest, replay = audited_canonical_online_run
    audited, array_count, chunk_count = _audit_run_manifest(
        run_dir / "manifest.json"
    )
    assert audited == manifest
    assert array_count >= 59
    assert chunk_count >= array_count
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["components"]["physics"]["payload_sha256"].startswith(
        "sha256:"
    )
    assert len(manifest["checkpoint"]["component_payload_sha256"]["physics"]) == 64
    _audit_web_replay_projection(replay, manifest)
    _audit_canonical_online_episode(replay, run_dir / "manifest.json", manifest)


def test_canonical_online_auditor_rejects_attachment_array_divergence(
    audited_canonical_online_run,
):
    run_dir, manifest, replay = audited_canonical_online_run
    changed = deepcopy(replay)
    changed["online_circuit_replay"]["full_cell_state"]["voltage_v"][0][0] += 1e-12
    with pytest.raises(AuditError, match="full-cell attachment"):
        _audit_canonical_online_episode(
            changed, run_dir / "manifest.json", manifest
        )


def test_canonical_online_auditor_rejects_stale_rate_and_wrong_boundary_receipts(
    audited_canonical_online_run,
):
    run_dir, manifest, replay = audited_canonical_online_run
    frames = replay["frames"]

    boundary_index = next(
        index
        for index in range(5, len(frames) - 1, 5)
        if {
            key: value
            for key, value in frames[index]["neural_signals"].items()
            if key.startswith(("dn:", "mn:"))
        }
        != {
            key: value
            for key, value in frames[index - 1]["neural_signals"].items()
            if key.startswith(("dn:", "mn:"))
        }
    )
    stale = deepcopy(replay)
    for prefix in ("dn:", "mn:"):
        for key in tuple(stale["frames"][boundary_index]["neural_signals"]):
            if key.startswith(prefix):
                del stale["frames"][boundary_index]["neural_signals"][key]
        stale["frames"][boundary_index]["neural_signals"].update(
            {
                key: value
                for key, value in frames[boundary_index - 1][
                    "neural_signals"
                ].items()
                if key.startswith(prefix)
            }
        )
    for pathway_index in (4, 5):
        stale["frames"][boundary_index]["pathway_values"][pathway_index] = frames[
            boundary_index - 1
        ]["pathway_values"][pathway_index]
        stale["frames"][boundary_index]["circuit"][pathway_index] = frames[
            boundary_index - 1
        ]["circuit"][pathway_index]
    with pytest.raises(AuditError, match="stale or non-authoritative held"):
        _audit_canonical_online_episode(
            stale, run_dir / "manifest.json", manifest
        )

    active_index = next(
        index
        for index, frame in enumerate(frames[:-1])
        if frame["active_intervention_ids"]
    )
    wrong_active = deepcopy(replay)
    wrong_active["frames"][active_index]["active_intervention_ids"] = []
    with pytest.raises(AuditError, match="outgoing transition"):
        _audit_canonical_online_episode(
            wrong_active, run_dir / "manifest.json", manifest
        )

    completion_index = next(
        index
        for index, frame in enumerate(frames)
        if frame["applied_motor_event_ids"]
        or frame["suppressed_motor_event_ids"]
    )
    wrong_completion = deepcopy(replay)
    wrong_completion["frames"][completion_index]["applied_motor_event_ids"] = []
    wrong_completion["frames"][completion_index][
        "suppressed_motor_event_ids"
    ] = []
    with pytest.raises(AuditError, match="completed event receipts"):
        _audit_canonical_online_episode(
            wrong_completion, run_dir / "manifest.json", manifest
        )


def test_canonical_pathway_mean_allows_only_binary_reduction_roundoff(
    audited_canonical_online_run,
):
    run_dir, manifest, replay = audited_canonical_online_run
    frame_index = next(
        index
        for index, frame in enumerate(replay["frames"])
        if frame["pathway_values"][5] not in (None, 0.0)
    )

    one_ulp = deepcopy(replay)
    original = one_ulp["frames"][frame_index]["pathway_values"][5]
    one_ulp["frames"][frame_index]["pathway_values"][5] = float(
        np.nextafter(original, np.inf)
    )
    _audit_canonical_online_episode(
        one_ulp, run_dir / "manifest.json", manifest
    )

    detached = deepcopy(replay)
    detached["frames"][frame_index]["pathway_values"][5] += 1.0e-6
    with pytest.raises(AuditError, match="pathway display is detached"):
        _audit_canonical_online_episode(
            detached, run_dir / "manifest.json", manifest
        )


def test_canonical_online_auditor_rejects_chunk_and_nested_receipt_tampering(
    audited_canonical_online_run, tmp_path
):
    run_dir, _manifest, _replay = audited_canonical_online_run
    copied = tmp_path / run_dir.name
    shutil.copytree(run_dir, copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_descriptor = manifest["arrays"][sorted(manifest["arrays"])[0]]
    chunk = copied / first_descriptor["chunks"][0]["path"]
    payload = bytearray(chunk.read_bytes())
    payload[-1] ^= 1
    chunk.write_bytes(bytes(payload))
    with pytest.raises(AuditError, match="chunk checksum"):
        _audit_run_manifest(manifest_path)

    shutil.rmtree(copied)
    shutil.copytree(run_dir, copied)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime"]["receipts"]["flybody"]["engine"] = "alias"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (copied / "manifest.sha256").write_text(
        _sha256_file(manifest_path) + "  manifest.json\n", encoding="ascii"
    )
    with pytest.raises(AuditError, match="nested FlyBody receipt"):
        _audit_run_manifest(manifest_path)

    shutil.rmtree(copied)
    shutil.copytree(run_dir, copied)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = copied / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["components"]["physics"]["payload_sha256"] = (
        "sha256:" + "0" * 64
    )
    checkpoint["payload_sha256"] = _canonical_digest(
        {
            key: value
            for key, value in checkpoint.items()
            if key != "payload_sha256"
        }
    )
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["checkpoint"]["file_sha256"] = _sha256_file(checkpoint_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (copied / "manifest.sha256").write_text(
        _sha256_file(manifest_path) + "  manifest.json\n", encoding="ascii"
    )
    with pytest.raises(AuditError, match="physics component checksum"):
        _audit_run_manifest(manifest_path)


def test_canonical_online_declaration_dispatch_rejects_aliases_and_allows_many():
    canonical_episode = {
        "source_kind": "canonical_online_fly_fgs_streaming_flybody",
        "online_closed_loop": {},
    }
    canonical_run = {
        "schema_version": "1.1.0",
        "source_kind": "canonical_online_fly_fgs_streaming_flight",
    }
    assert _is_canonical_online_declaration(canonical_episode, canonical_run)
    assert _is_canonical_online_declaration(
        {"source_kind": "canonical_online_alias"},
        {"schema_version": "2.0.0", "source_kind": "flybody_worker"},
    )
    bindings = [
        ({"id": "baseline", **canonical_episode}, Path("/a"), canonical_run),
        ({"id": "iv2-silence", **canonical_episode}, Path("/b"), canonical_run),
    ]
    assert len(bindings) == 2


def test_canonical_online_publication_gate_uses_one_class_level_pass():
    registry = load_benchmark_registry()
    case = registry.case(CANONICAL_ONLINE_VALIDATION_CASE_ID)
    report = SimpleNamespace(
        results=(
            SimpleNamespace(
                case_id=case.case_id,
                case_version=case.version,
                status=GateStatus.PASS,
            ),
        )
    )
    _audit_required_validation_pass(
        report, registry, CANONICAL_ONLINE_VALIDATION_CASE_ID
    )


def _content_binding_fixture(
    root: Path, *, episode_id: str, canonical: bool
):
    schema = CANONICAL_ARTIFACT_SCHEMA_VERSION if canonical else "2.0.0"
    run_id = "{}-run".format(episode_id)
    projection = {
        "schema_version": "1.0.0",
        "id": episode_id,
        "label": "{} replay".format(episode_id),
        "status": "exploratory",
        "source_run_id": run_id,
        "frames": [{"t": 0.0, "value": 1.25}],
    }
    projection_bytes = json.dumps(
        projection,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    manifest = {
        "schema_version": schema,
        "run_id": run_id,
        "web_replay_projection": {
            "schema_version": "1.0.0",
            "sha256": hashlib.sha256(projection_bytes).hexdigest(),
            "canonicalization": (
                CANONICAL_WEB_PROJECTION_CANONICALIZATION
                if canonical
                else WEB_REPLAY_PROJECTION_CANONICALIZATION
            ),
            "excluded_top_level_fields": list(
                WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS
            ),
        },
    }
    run_path = root / "data" / "runs" / run_id / "manifest.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_sha = _sha256_file(run_path)
    replay = {
        **projection,
        "source_artifact_manifest_sha256": artifact_sha,
        "source_artifact_schema_version": schema,
    }
    replay_path = root / "data" / "episodes" / (episode_id + ".json")
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    projection_path = (
        root
        / "data"
        / "episodes"
        / (episode_id + ".artifact-projection.json")
    )
    projection_path.write_bytes(projection_bytes)
    summary = {
        "id": episode_id,
        "label": replay["label"],
        "condition": "binding test",
        "description": "Exact public content binding fixture.",
        "color": "#123456",
        "status": replay["status"],
        "data_url": "data/episodes/{}.json".format(episode_id),
        "replay_sha256": _sha256_file(replay_path),
        "artifact_manifest_url": "data/runs/{}/manifest.json".format(run_id),
        "artifact_manifest_sha256": artifact_sha,
        "artifact_projection_url": (
            "data/episodes/{}.artifact-projection.json".format(episode_id)
        ),
    }
    return summary, replay_path, replay, run_path, manifest, projection_path


def test_episode_content_binding_accepts_mixed_legacy_and_canonical_release(
    tmp_path,
):
    fixtures = (
        _content_binding_fixture(
            tmp_path, episode_id="legacy-binding", canonical=False
        ),
        _content_binding_fixture(
            tmp_path, episode_id="canonical-binding", canonical=True
        ),
    )
    for fixture in fixtures:
        _audit_episode_content_binding(*fixture)


def test_episode_content_binding_rejects_replay_and_manifest_byte_mismatch(
    tmp_path,
):
    fixture = _content_binding_fixture(
        tmp_path, episode_id="byte-mismatch", canonical=True
    )
    summary, replay_path, replay, run_path, manifest, projection_path = fixture
    replay_path.write_bytes(replay_path.read_bytes() + b" ")
    with pytest.raises(AuditError, match="replay bytes"):
        _audit_episode_content_binding(*fixture)

    fixture = list(
        _content_binding_fixture(
            tmp_path, episode_id="replay-projection-tamper", canonical=True
        )
    )
    summary, replay_path, replay, run_path, manifest, projection_path = fixture
    replay = deepcopy(replay)
    replay["frames"][0]["value"] = 9.5
    replay_path.write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = dict(summary)
    summary["replay_sha256"] = _sha256_file(replay_path)
    fixture[0] = summary
    fixture[2] = replay
    with pytest.raises(AuditError, match="exact replay projection preimage"):
        _audit_episode_content_binding(*fixture)

    fixture = _content_binding_fixture(
        tmp_path, episode_id="manifest-mismatch", canonical=False
    )
    summary, replay_path, replay, run_path, manifest, projection_path = fixture
    run_path.write_bytes(run_path.read_bytes() + b" ")
    with pytest.raises(AuditError, match="manifest bytes"):
        _audit_episode_content_binding(*fixture)

    fixture = list(
        _content_binding_fixture(
            tmp_path, episode_id="manifest-cross-binding", canonical=True
        )
    )
    summary, replay_path, replay, run_path, manifest, projection_path = fixture
    run_path.write_bytes(run_path.read_bytes() + b" ")
    summary = dict(summary)
    summary["artifact_manifest_sha256"] = _sha256_file(run_path)
    fixture[0] = summary
    with pytest.raises(AuditError, match="does not bind the exact artifact"):
        _audit_episode_content_binding(*fixture)


def test_episode_content_binding_rejects_projection_tamper_alias_and_missing_sidecar(
    tmp_path,
):
    fixture = _content_binding_fixture(
        tmp_path, episode_id="projection-tamper", canonical=True
    )
    projection_path = fixture[-1]
    projection_path.write_bytes(projection_path.read_bytes() + b" ")
    with pytest.raises(AuditError, match="sidecar bytes"):
        _audit_episode_content_binding(*fixture)

    fixture = _content_binding_fixture(
        tmp_path, episode_id="projection-alias", canonical=False
    )
    summary, replay_path, replay, run_path, manifest, projection_path = fixture
    projection_path.unlink()
    os.link(replay_path, projection_path)
    with pytest.raises(AuditError, match="hard-link aliases"):
        _audit_episode_content_binding(*fixture)

    fixture = _content_binding_fixture(
        tmp_path, episode_id="projection-missing", canonical=True
    )
    fixture[-1].unlink()
    with pytest.raises(AuditError, match="regular non-symlink"):
        _audit_episode_content_binding(*fixture)


def test_episode_content_binding_rejects_projection_url_alias_even_with_valid_bytes(
    tmp_path,
):
    fixture = list(
        _content_binding_fixture(
            tmp_path, episode_id="projection-url-alias", canonical=True
        )
    )
    summary = dict(fixture[0])
    summary["artifact_projection_url"] = summary["data_url"]
    fixture[0] = summary
    with pytest.raises(AuditError, match="exact canonical path"):
        _audit_episode_content_binding(*fixture)
