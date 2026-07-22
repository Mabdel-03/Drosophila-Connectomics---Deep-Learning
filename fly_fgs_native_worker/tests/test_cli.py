import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from fly_sensor2behavior.artifacts import web_replay_projection_sha256
from fly_sensor2behavior.cli import (
    _load_streaming_muscle_interventions,
    build_parser,
    main,
)


def test_parser_has_required_subcommands():
    parser = build_parser()
    help_text = parser.format_help()
    for command in (
        "audit-evidence",
        "simulate",
        "simulate-nod1",
        "simulate-retinal",
        "simulate-canonical-online",
        "export-web",
        "flybody-smoke",
        "validate",
    ):
        assert command in help_text


def test_pipeline_commands_expose_explicit_worker_backend(capsys):
    for command in (
        "simulate-nod1",
        "simulate-nod1-fixture",
        "simulate-retinal",
        "export-web",
    ):
        with pytest.raises(SystemExit) as exc_info:
            main((command, "--help"))
        assert exc_info.value.code == 0
        help_text = capsys.readouterr().out
        assert "--physics-backend {reduced-order,flybody}" in help_text
        assert "--flybody-spawn-height-m" in help_text


def test_canonical_online_cli_requires_explicit_effector_and_external_replay(tmp_path, capsys):
    with pytest.raises(SystemExit) as missing:
        main(("simulate-canonical-online", "--output", str(tmp_path / "run")))
    assert missing.value.code == 2
    capsys.readouterr()

    with pytest.raises(SystemExit) as nested:
        main(
            (
                "simulate-canonical-online",
                "--output",
                str(tmp_path / "run"),
                "--web-replay-output",
                str(tmp_path / "run" / "replay.json"),
                "--effector-hypothesis",
                "raw_l_to_physical_left",
            )
        )
    assert nested.value.code == 2
    assert "outside the immutable" in capsys.readouterr().err

    with pytest.raises(SystemExit) as unsafe_id:
        main(
            (
                "simulate-canonical-online",
                "--output",
                str(tmp_path / "other-run"),
                "--scenario-id",
                "../unsafe",
                "--effector-hypothesis",
                "raw_l_to_physical_left",
            )
        )
    assert unsafe_id.value.code == 2
    assert "--scenario-id" in capsys.readouterr().err


def test_canonical_online_force_stage_intervention_input(tmp_path):
    path = tmp_path / "interventions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "intervention_id": "left-iv2-cut",
                    "muscle": "iv2",
                    "start_s": 0.001,
                    "end_s": 0.002,
                    "mode": "silence",
                    "raw_app_side": "L",
                },
                {
                    "intervention_id": "bilateral-b3-half",
                    "muscle": "b3",
                    "start_s": 0.003,
                    "end_s": 0.004,
                    "mode": "scale",
                    "output_scale": 0.5,
                },
            ]
        ),
        encoding="utf-8",
    )
    interventions = _load_streaming_muscle_interventions(path)
    assert [item.intervention_id for item in interventions] == [
        "left-iv2-cut",
        "bilateral-b3-half",
    ]
    assert interventions[0].raw_app_side.value == "L"
    assert interventions[0].output_scale == 0.0
    assert interventions[1].raw_app_side is None
    assert interventions[1].output_scale == 0.5


def test_registered_canonical_online_iv2_silence_scenario_is_exact():
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "benchmarks"
        / "scenarios"
        / "canonical-online-raw-l-iv2-silence.v1.json"
    )
    interventions = _load_streaming_muscle_interventions(path)
    assert len(interventions) == 1
    intervention = interventions[0]
    assert intervention.intervention_id == "raw-l-iv2-silence-60-100ms"
    assert intervention.muscle == "iv2"
    assert intervention.raw_app_side.value == "L"
    assert round(intervention.start_s / 0.0001) == 600
    assert round(intervention.end_s / 0.0001) == 1000
    assert intervention.mode.value == "silence"
    assert intervention.output_scale == 0.0


def test_audit_evidence_cli_prints_json(capsys):
    assert main(["audit-evidence"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["coverage_status"] == "provisional"
    assert len(payload["provisional_full_pathway_coverage"]) == 2


def test_simulate_cli_writes_scientific_artifact(tmp_path, capsys):
    output = tmp_path / "run"
    assert (
        main(
            [
                "simulate",
                "baseline",
                "--output",
                str(output),
                "--duration-s",
                "0.012",
                "--chunk-samples",
                "5",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    manifest = json.loads((output / "manifest.json").read_text())
    assert summary["validation_status"] == "exploratory"
    assert summary["calibration"] == "none"
    assert manifest["source_kind"] == "exploratory_reduced_order_numpy"


def test_export_web_cli_writes_contract(tmp_path, capsys):
    output = tmp_path / "web-data"
    assert (
        main(
            [
                "export-web",
                "--output",
                str(output),
                "--scenarios",
                "dng02_activation",
                "--duration-s",
                "0.012",
                "--sample-rate-hz",
                "100",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    # Baseline is injected because the UI's comparison contract requires it.
    assert summary["episode_count"] == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert [episode["id"] for episode in manifest["episodes"]] == [
        "baseline",
        "dng02_activation",
    ]


def test_cli_import_does_not_eagerly_import_flybody_native_modules():
    # Use a fresh interpreter: earlier native-worker tests may legitimately
    # have imported FlyGym, so process-global sys.modules is not a valid test
    # fixture for this import-boundary invariant.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "assert 'flygym' not in sys.modules; "
                "assert 'mujoco' not in sys.modules; "
                "import fly_sensor2behavior.cli; "
                "assert 'flygym' not in sys.modules; "
                "assert 'mujoco' not in sys.modules"
            ),
            str(Path(__file__).resolve().parents[1] / "src"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_simulate_retinal_cli_writes_full_vertical_slice(tmp_path, capsys):
    output = tmp_path / "retinal"
    replay_output = tmp_path / "retinal-replay.json"
    assert (
        main(
            [
                "simulate-retinal",
                "--output",
                str(output),
                "--duration-s",
                "0.020",
                "--exposure-dt-s",
                "0.001",
                "--receptors",
                "17",
                "--web-replay-output",
                str(replay_output),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    manifest = json.loads((output / "manifest.json").read_text())
    assert summary["visual_boundary"] == "provided"
    assert summary["scientific_status"] == "exploratory_uncalibrated"
    assert summary["web_replay"] == str(replay_output.resolve())
    assert "retinal_normalized_luminance" in manifest["arrays"]
    assert "descending_rate_hz/DNp26/left" in manifest["arrays"]
    replay = json.loads(replay_output.read_text())
    assert replay["neural_model_scope"]["kind"] == "reduced_nod1_surrogate"
    assert replay["neural_trace_channels"]
    assert replay["source_run_id"] == manifest["run_id"]
    assert replay["source_artifact_schema_version"] == manifest["schema_version"]
    assert replay["source_artifact_manifest_sha256"] == hashlib.sha256(
        (output / "manifest.json").read_bytes()
    ).hexdigest()
    assert manifest["web_replay_projection"]["sha256"] == (
        web_replay_projection_sha256(replay)
    )
