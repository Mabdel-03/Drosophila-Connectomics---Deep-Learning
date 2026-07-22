from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import fly_sensor2behavior.cli as cli_module
import fly_sensor2behavior.pipeline as pipeline_module
from fly_sensor2behavior.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    WEB_REPLAY_PROJECTION_CANONICALIZATION,
    WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS,
    WEB_REPLAY_PROJECTION_SCHEMA_VERSION,
    episode_run_id,
    export_web_replay,
    web_replay_projection_sha256,
)
from fly_sensor2behavior.cli import build_parser, main
from fly_sensor2behavior.flight import FlightSimulationConfig


def test_simulate_fly_fgs_fixture_cli_forwards_registered_inputs(
    tmp_path,
    monkeypatch,
    capsys,
):
    run = object()
    observed = {}

    def fake_pipeline(**kwargs):
        observed["pipeline"] = kwargs
        return run

    def fake_replay(observed_run, *, target_sample_rate_hz):
        assert observed_run is run
        observed["replay_rate_hz"] = target_sample_rate_hz
        return {"id": "fly_fgs_canonical", "finite": 1.0}

    def fake_artifact(output, observed_run, **kwargs):
        assert observed_run is run
        observed["artifact"] = {"output": output, **kwargs}
        return {
            "run_id": "fly-fgs-cli-test",
            "pipeline": {
                "mode": "open_loop_registered_fly_fgs_fixed_step_circuit_output",
                "visual_boundary": {
                    "status": "registered_retinal_frames_and_circuit_replay"
                },
                "scientific_status": "exploratory_uncalibrated",
            },
            "validation_status": "exploratory",
            "diagnostics": {"physics_backend": "reduced_order"},
        }

    def fake_write_replay(output, observed_run, **kwargs):
        assert observed_run is run
        observed["write_replay"] = {"output": output, **kwargs}
        return {"id": "fly_fgs_canonical"}

    monkeypatch.setattr(
        pipeline_module,
        "run_registered_fly_fgs_flight_pipeline",
        fake_pipeline,
    )
    monkeypatch.setattr(pipeline_module, "nod1_run_to_web_replay", fake_replay)
    monkeypatch.setattr(
        pipeline_module,
        "write_nod1_flight_artifact",
        fake_artifact,
    )
    monkeypatch.setattr(
        pipeline_module,
        "write_nod1_web_replay",
        fake_write_replay,
    )

    output = tmp_path / "run"
    replay = tmp_path / "fly-fgs.json"
    source_manifest = tmp_path / "source-manifest.json"
    capture = tmp_path / "capture.json.gz"
    assert main(
        (
            "simulate-fly-fgs-fixture",
            "--manifest",
            str(source_manifest),
            "--capture",
            str(capture),
            "--output",
            str(output),
            "--physics-dt-s",
            "0.005",
            "--logging-dt-s",
            "0.005",
            "--neural-dt-s",
            "0.005",
            "--seed",
            "29",
            "--chunk-samples",
            "17",
            "--web-replay-output",
            str(replay),
            "--web-replay-sample-rate-hz",
            "125",
        )
    ) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["run_id"] == "fly-fgs-cli-test"
    assert summary["visual_boundary"] == (
        "registered_retinal_frames_and_circuit_replay"
    )
    assert summary["web_replay"] == str(replay.resolve())
    assert observed["pipeline"]["manifest_path"] == source_manifest
    assert observed["pipeline"]["capture_path"] == capture
    assert observed["pipeline"]["runner"] is None
    config = observed["pipeline"]["config"]
    assert config.physics_dt_s == pytest.approx(0.005)
    assert config.logging_dt_s == pytest.approx(0.005)
    assert config.neural_dt_s == pytest.approx(0.005)
    assert config.seed == 29
    assert observed["artifact"]["chunk_samples"] == 17
    assert observed["artifact"]["web_replay"]["id"] == "fly_fgs_canonical"
    assert observed["replay_rate_hz"] == pytest.approx(125.0)
    assert observed["write_replay"]["output"] == replay
    assert observed["write_replay"]["source_artifact_manifest_path"] == (
        output / "manifest.json"
    )

    parsed = build_parser().parse_args(
        ("simulate-fly-fgs-fixture", "--output", str(tmp_path / "unused"))
    )
    assert parsed.command == "simulate-fly-fgs-fixture"
    assert not hasattr(parsed, "duration_s")


def test_export_rejects_fly_fgs_paths_without_include_flag(tmp_path):
    with pytest.raises(
        ValueError,
        match="registered_fly_fgs_manifest_path requires",
    ):
        export_web_replay(
            tmp_path / "manifest-only",
            scenarios=("baseline",),
            registered_fly_fgs_manifest_path=tmp_path / "source.json",
        )
    with pytest.raises(
        ValueError,
        match="registered_fly_fgs_capture_path requires",
    ):
        export_web_replay(
            tmp_path / "capture-only",
            scenarios=("baseline",),
            registered_fly_fgs_capture_path=tmp_path / "capture.json.gz",
        )


def test_export_cli_forwards_fly_fgs_fixture_paths(
    tmp_path,
    monkeypatch,
    capsys,
):
    observed = {}

    def fake_export(output, **kwargs):
        observed["output"] = output
        observed.update(kwargs)
        return {
            "episodes": [{"id": "fly_fgs_canonical"}],
            "status": "exploratory",
            "source_kind": "mixed_exploratory_replay_with_pipeline_traces",
        }

    monkeypatch.setattr(cli_module, "export_web_replay", fake_export)
    output = tmp_path / "web"
    source_manifest = tmp_path / "source.json"
    capture = tmp_path / "capture.json.gz"
    assert main(
        (
            "export-web",
            "--output",
            str(output),
            "--scenarios",
            "baseline",
            "--include-registered-fly-fgs-fixture",
            "--fly-fgs-manifest",
            str(source_manifest),
            "--fly-fgs-capture",
            str(capture),
        )
    ) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["episode_count"] == 1
    assert observed["output"] == output
    assert observed["include_registered_fly_fgs_fixture"] is True
    assert observed["registered_fly_fgs_manifest_path"] == source_manifest
    assert observed["registered_fly_fgs_capture_path"] == capture
    assert observed["pipeline_runner"] is None
    assert observed["include_registered_nod1_fixture"] is False


def test_export_attaches_canonical_before_historical_and_prefers_its_flybody(
    tmp_path,
    monkeypatch,
):
    calls = []

    def make_run(kind, config):
        return SimpleNamespace(
            kind=kind,
            flight_config=FlightSimulationConfig(
                duration_s=0.5,
                physics_dt_s=config.physics_dt_s,
                neural_dt_s=config.neural_dt_s,
                logging_dt_s=config.logging_dt_s,
                seed=config.seed,
            ),
        )

    monkeypatch.setattr(
        pipeline_module,
        "load_legacy_nod1_result",
        lambda _path: {"manufactured": True},
    )

    def fake_saved(_payload, *, config, runner):
        assert runner is pipeline_runner
        calls.append(("saved", None, None))
        return make_run("saved", config)

    def fake_fly_fgs(*, manifest_path, capture_path, config, runner, **_kwargs):
        assert runner is pipeline_runner
        calls.append(("fly_fgs", manifest_path, capture_path))
        return make_run("fly_fgs", config)

    def fake_historical(*, fixture_path, config, runner, **_kwargs):
        assert runner is pipeline_runner
        calls.append(("historical", None, fixture_path))
        return make_run("historical", config)

    monkeypatch.setattr(pipeline_module, "run_nod1_flight_pipeline", fake_saved)
    monkeypatch.setattr(
        pipeline_module,
        "run_registered_fly_fgs_flight_pipeline",
        fake_fly_fgs,
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_registered_nod1_browser_flight_pipeline",
        fake_historical,
    )
    monkeypatch.setattr(
        pipeline_module,
        "nod1_run_identity_context",
        lambda run: {"manufactured_kind": run.kind},
    )

    episode_contracts = {
        "saved": (
            "saved_nod1_circuit",
            "Saved circuit → FlyBody",
            "exploratory_saved_legacy_nod1_flybody_pipeline",
            "full_legacy_circuit_cable_export",
        ),
        "fly_fgs": (
            "fly_fgs_canonical",
            "fly-FGS visual circuit → FlyBody",
            "exploratory_fly_fgs_circuit_flybody_pipeline",
            "registered_fly_fgs_fixed_step_circuit",
        ),
        "historical": (
            "frozen_browser_nod1",
            "Historical frozen browser NOD1 → FlyBody",
            "exploratory_frozen_browser_nod1_flybody_pipeline",
            "full_legacy_circuit_cable_export",
        ),
    }

    def fake_replay(run, *, target_sample_rate_hz):
        assert target_sample_rate_hz == pytest.approx(100.0)
        episode_id, label, source_kind, scope = episode_contracts[run.kind]
        identity = {"manufactured_kind": run.kind}
        return {
            "schema_version": "1.0.0",
            "id": episode_id,
            "label": label,
            "status": "exploratory",
            "source_kind": source_kind,
            "physics_backend": "flybody",
            "neural_model_scope": {"kind": scope},
            "source_run_id": episode_run_id(
                "nod1_visual_circuit_to_flight",
                run.flight_config,
                identity_context=identity,
            ),
            "frames": [],
        }

    monkeypatch.setattr(
        pipeline_module,
        "nod1_run_to_web_replay",
        fake_replay,
    )

    def fake_write_artifact(output_dir, run, *, web_replay, **_kwargs):
        identity = {"manufactured_kind": run.kind}
        run_id = episode_run_id(
            "nod1_visual_circuit_to_flight",
            run.flight_config,
            identity_context=identity,
        )
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "web_replay_projection": {
                "schema_version": WEB_REPLAY_PROJECTION_SCHEMA_VERSION,
                "sha256": web_replay_projection_sha256(web_replay),
                "canonicalization": WEB_REPLAY_PROJECTION_CANONICALIZATION,
                "excluded_top_level_fields": list(
                    WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS
                ),
            },
            "runtime": {
                "physics_provenance": {
                    "worker_image_digest": None,
                    "worker_dependency_lock": None,
                }
            },
        }
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True),
            encoding="utf-8",
        )
        return manifest

    monkeypatch.setattr(
        pipeline_module,
        "write_nod1_flight_artifact",
        fake_write_artifact,
    )

    pipeline_runner = object()
    fly_fgs_manifest = tmp_path / "fly-fgs-manifest.json"
    fly_fgs_capture = tmp_path / "fly-fgs-capture.json.gz"
    historical_capture = tmp_path / "historical.json.gz"
    manifest = export_web_replay(
        tmp_path / "web",
        scenarios=("baseline",),
        duration_s=0.010,
        physics_dt_s=0.001,
        logging_dt_s=0.001,
        neural_dt_s=0.005,
        target_sample_rate_hz=100.0,
        legacy_nod1_result_path=tmp_path / "saved.json",
        include_registered_fly_fgs_fixture=True,
        registered_fly_fgs_manifest_path=fly_fgs_manifest,
        registered_fly_fgs_capture_path=fly_fgs_capture,
        include_registered_nod1_fixture=True,
        registered_nod1_fixture_path=historical_capture,
        pipeline_runner=pipeline_runner,
    )

    assert calls == [
        ("saved", None, None),
        ("fly_fgs", fly_fgs_manifest, fly_fgs_capture),
        ("historical", None, historical_capture),
    ]
    assert [item["id"] for item in manifest["episodes"]] == [
        "baseline",
        "fly_fgs_canonical",
        "saved_nod1_circuit",
        "frozen_browser_nod1",
    ]
    assert manifest["source_kind"] == (
        "mixed_exploratory_flybody_and_reduced_replay"
    )
    canonical_summary = manifest["episodes"][1]
    canonical = json.loads(
        (
            tmp_path
            / "web"
            / canonical_summary["data_url"].removeprefix("data/")
        ).read_text(encoding="utf-8")
    )
    assert canonical["id"] == "fly_fgs_canonical"
    assert canonical["source_kind"] == (
        "exploratory_fly_fgs_circuit_flybody_pipeline"
    )
    assert canonical["physics_backend"] == "flybody"
