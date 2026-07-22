import hashlib
import json
import os
from dataclasses import replace

import pytest

from fly_sensor2behavior.flybody_release_intake import (
    CANDIDATE_STATUS,
    POLICY_ROOT,
    REFERENCE_SENTINEL,
    REVIEWED_EXPECTED_STATUS,
    UPSTREAM_CODE_COMMIT,
    WPG_SENTINEL,
    BundleFile,
    BundleRole,
    CandidateBundleReceipt,
    InferenceMetadata,
    NormalizationMode,
    ReleaseIntakeError,
    ReviewAttestation,
    ReviewedExpectedReceipt,
    RuntimeMetadata,
    inventory_release_candidate,
    main,
    verify_release_readiness,
)


INFERENCE_PATH = "inference/manufactured_adapter.py"
RUNTIME_PATH = "runtime/manufactured.lock"
INFERENCE_PAYLOAD = b"# manufactured inference adapter\n"
RUNTIME_PAYLOAD = b"manufactured-runtime==1.0\n"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(root, relative_path: str, payload: bytes) -> None:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def _make_bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    model_root = POLICY_ROOT + "/manufactured-policy"
    normalization_path = model_root + "/normalization.json"
    files = {
        WPG_SENTINEL: b"manufactured wing pattern",
        REFERENCE_SENTINEL: b"manufactured reference trials",
        model_root + "/saved_model.pb": b"manufactured saved model",
        model_root + "/variables/variables.index": b"manufactured index",
        model_root + "/variables/variables.data-00000-of-00001": b"manufactured variables",
        normalization_path: b'{"mean":[0.0],"scale":[1.0]}',
        INFERENCE_PATH: INFERENCE_PAYLOAD,
        RUNTIME_PATH: RUNTIME_PAYLOAD,
        "LICENSE.txt": b"manufactured fixture; not upstream data\n",
    }
    for path, payload in files.items():
        _write(root, path, payload)
    overrides = {
        normalization_path: BundleRole.NORMALIZATION,
        INFERENCE_PATH: BundleRole.INFERENCE,
        RUNTIME_PATH: BundleRole.RUNTIME,
    }
    return root, model_root, normalization_path, overrides


def _candidate(tmp_path):
    root, model_root, normalization_path, overrides = _make_bundle(tmp_path)
    candidate = inventory_release_candidate(root, role_overrides=overrides)
    return root, model_root, normalization_path, overrides, candidate


def _expected(candidate, model_root: str, normalization_path: str) -> ReviewedExpectedReceipt:
    return ReviewedExpectedReceipt(
        expected_id="manufactured-reviewed-expectation",
        expected_version="1.0.0",
        status=REVIEWED_EXPECTED_STATUS,
        source=candidate.source,
        required_sentinels=candidate.required_sentinels,
        saved_model_roots=candidate.saved_model_roots,
        files=candidate.files,
        inference=InferenceMetadata(
            policy_format="tensorflow_saved_model",
            wpg_path=WPG_SENTINEL,
            reference_dataset_path=REFERENCE_SENTINEL,
            policy_saved_model_roots=(model_root,),
            normalization_mode=NormalizationMode.ARTIFACT_FILES,
            normalization_paths=(normalization_path,),
            normalization_description="Manufactured normalization for contract tests only.",
            inference_artifact_path=INFERENCE_PATH,
            inference_entrypoint_uri="urn:manufactured:inference-entrypoint:v1",
            inference_code_sha256=_sha(INFERENCE_PAYLOAD),
            observation_schema_uri="urn:manufactured:observation-schema:v1",
            observation_schema_sha256=_sha(b"manufactured observation schema"),
            action_schema_uri="urn:manufactured:action-schema:v1",
            action_schema_sha256=_sha(b"manufactured action schema"),
            controller_clock_s=0.0002,
        ),
        runtime=RuntimeMetadata(
            upstream_code_commit=UPSTREAM_CODE_COMMIT,
            python_version="manufactured-3.x",
            flygym_version="manufactured",
            mujoco_version="manufactured",
            tensorflow_version="manufactured",
            platform_tag="manufactured-test-platform",
            dependency_lock_path=RUNTIME_PATH,
            dependency_lock_uri="urn:manufactured:dependency-lock:v1",
            dependency_lock_sha256=_sha(RUNTIME_PAYLOAD),
            container_image_digest="sha256:" + _sha(b"manufactured image"),
            physics_timestep_s=0.0001,
            determinism_notes="Manufactured deterministic runtime metadata for tests.",
        ),
        review=ReviewAttestation(
            review_id="manufactured-review",
            reviewer="Manufactured unit-test reviewer",
            reviewed_at_utc="2026-07-18T00:00:00Z",
            decision_uri="urn:manufactured:review-decision:v1",
            decision_sha256=_sha(b"manufactured review decision"),
        ),
    )


def test_candidate_is_deterministic_unreviewed_and_canonical(tmp_path):
    root, _model_root, _normalization_path, overrides, first = _candidate(tmp_path)
    second = inventory_release_candidate(root, role_overrides=overrides)
    restored = CandidateBundleReceipt.from_json(first.to_json())

    assert first.status == CANDIDATE_STATUS
    assert first == second == restored
    assert first.content_sha256 == second.content_sha256
    assert len(first.files) == 9
    assert first.source.doi == "10.25378/janelia.25309105"
    assert first.source.version == 4
    assert first.source.flight_imitation_archive_file_id == "51196859"
    assert first.source.trained_policies_archive_file_id == "44815195"
    assert first.source.license == "GPL-3.0+"
    assert {item.role for item in first.files} >= {
        BundleRole.WPG,
        BundleRole.REFERENCE,
        BundleRole.POLICY,
        BundleRole.NORMALIZATION,
        BundleRole.INFERENCE,
        BundleRole.RUNTIME,
    }


def test_candidate_alone_is_never_ready_and_never_unblocks_gate(tmp_path):
    root, _model_root, _normalization_path, overrides, candidate = _candidate(tmp_path)
    report = verify_release_readiness(candidate, root, role_overrides=overrides)
    assert report.ready is False
    assert report.reason_codes == ("reviewed_expected_receipt_missing",)
    assert report.physics_timestep_convergence_unblocked is False


def test_exact_reviewed_expectation_can_only_prepare_offline_readiness(tmp_path):
    root, model_root, normalization_path, overrides, candidate = _candidate(tmp_path)
    expected = _expected(candidate, model_root, normalization_path)
    restored = ReviewedExpectedReceipt.from_json(expected.to_json())
    report = verify_release_readiness(
        candidate, root, expected=restored, role_overrides=overrides
    )

    assert report.ready is True
    assert report.status == "ready_for_offline_reproduction"
    assert report.reason_codes == ()
    assert report.physics_timestep_convergence_unblocked is False
    assert "does not modify, pass, or unblock" in report.note


def test_local_tampering_after_intake_fails_readiness(tmp_path):
    root, model_root, normalization_path, overrides, candidate = _candidate(tmp_path)
    expected = _expected(candidate, model_root, normalization_path)
    (root / WPG_SENTINEL).write_bytes(b"tampered wing pattern")
    report = verify_release_readiness(
        candidate, root, expected=expected, role_overrides=overrides
    )
    assert report.ready is False
    assert "local_candidate_mismatch" in report.reason_codes


@pytest.mark.parametrize("field", ("sha256", "bytes"))
def test_reviewed_expected_inventory_must_match_every_hash_and_byte_count(tmp_path, field):
    root, model_root, normalization_path, overrides, candidate = _candidate(tmp_path)
    expected = _expected(candidate, model_root, normalization_path)
    first = expected.files[0]
    changed = replace(
        first,
        **({"sha256": "0" * 64} if field == "sha256" else {"bytes": first.bytes + 1}),
    )
    expected = replace(expected, files=(changed,) + expected.files[1:])
    report = verify_release_readiness(
        candidate, root, expected=expected, role_overrides=overrides
    )
    assert report.ready is False
    assert "full_file_inventory_mismatch" in report.reason_codes


def test_reviewed_expected_inventory_must_include_every_file(tmp_path):
    root, model_root, normalization_path, overrides, candidate = _candidate(tmp_path)
    expected = _expected(candidate, model_root, normalization_path)
    expected = replace(
        expected,
        files=tuple(item for item in expected.files if item.path != "LICENSE.txt"),
    )
    report = verify_release_readiness(
        candidate, root, expected=expected, role_overrides=overrides
    )
    assert report.ready is False
    assert "full_file_inventory_mismatch" in report.reason_codes


def test_missing_or_empty_required_files_are_rejected(tmp_path):
    root, _model_root, _normalization_path, overrides = _make_bundle(tmp_path)
    (root / WPG_SENTINEL).unlink()
    with pytest.raises(ReleaseIntakeError, match="required sentinel is missing"):
        inventory_release_candidate(root, role_overrides=overrides)

    _write(root, WPG_SENTINEL, b"restored")
    (root / REFERENCE_SENTINEL).write_bytes(b"")
    with pytest.raises(ReleaseIntakeError, match="file is empty"):
        inventory_release_candidate(root, role_overrides=overrides)


@pytest.mark.parametrize(
    "relative_path",
    (
        POLICY_ROOT + "/manufactured-policy/variables/variables.index",
        POLICY_ROOT + "/manufactured-policy/variables/variables.data-00000-of-00001",
    ),
)
def test_incomplete_saved_model_inventory_is_rejected(tmp_path, relative_path):
    root, _model_root, _normalization_path, overrides = _make_bundle(tmp_path)
    (root / relative_path).unlink()
    with pytest.raises(ReleaseIntakeError, match="SavedModel variables"):
        inventory_release_candidate(root, role_overrides=overrides)


@pytest.mark.parametrize(
    ("replacement_name", "message"),
    (
        ("variables.data-not-a-shard", "invalid shard name"),
        ("variables.data-00000-of-00002", "shards are incomplete"),
    ),
)
def test_saved_model_data_shards_require_valid_complete_inventory(
    tmp_path, replacement_name, message
):
    root, model_root, _normalization_path, overrides = _make_bundle(tmp_path)
    original = root / model_root / "variables" / "variables.data-00000-of-00001"
    original.rename(original.with_name(replacement_name))

    with pytest.raises(ReleaseIntakeError, match=message):
        inventory_release_candidate(root, role_overrides=overrides)


def test_symlinks_and_hardlinked_duplicates_are_rejected(tmp_path):
    root, _model_root, _normalization_path, overrides = _make_bundle(tmp_path)
    (root / "forbidden-link").symlink_to(root / WPG_SENTINEL)
    with pytest.raises(ReleaseIntakeError, match="symlinks are forbidden"):
        inventory_release_candidate(root, role_overrides=overrides)

    (root / "forbidden-link").unlink()
    os.link(root / WPG_SENTINEL, root / "duplicate-hardlink")
    with pytest.raises(ReleaseIntakeError, match="duplicate hard-linked"):
        inventory_release_candidate(root, role_overrides=overrides)


def test_traversal_and_unknown_role_overrides_are_rejected(tmp_path):
    root, _model_root, _normalization_path, _overrides = _make_bundle(tmp_path)
    with pytest.raises(ReleaseIntakeError, match="normalized relative path"):
        inventory_release_candidate(root, role_overrides={"../escape": BundleRole.RUNTIME})
    with pytest.raises(ReleaseIntakeError, match="unknown files"):
        inventory_release_candidate(root, role_overrides={"absent.file": BundleRole.RUNTIME})


def test_normalization_and_runtime_metadata_fail_closed(tmp_path):
    _root, model_root, normalization_path, _overrides, candidate = _candidate(tmp_path)
    expected = _expected(candidate, model_root, normalization_path)
    normalization_file = next(
        item for item in expected.files if item.path == normalization_path
    )
    wrong_role = replace(normalization_file, role=BundleRole.POLICY)
    files = tuple(
        wrong_role if item.path == normalization_path else item for item in expected.files
    )
    with pytest.raises(ReleaseIntakeError, match="incompatible file role"):
        replace(expected, files=files)

    with pytest.raises(ReleaseIntakeError, match="upstream_code_commit"):
        replace(expected.runtime, upstream_code_commit="0" * 40)

    wrong_inference = replace(expected.inference, inference_code_sha256="0" * 64)
    with pytest.raises(ReleaseIntakeError, match="inference artifact hash"):
        replace(expected, inference=wrong_inference)

    wrong_runtime = replace(expected.runtime, dependency_lock_sha256="0" * 64)
    with pytest.raises(ReleaseIntakeError, match="dependency lock hash"):
        replace(expected, runtime=wrong_runtime)


@pytest.mark.parametrize("missing_field", ("inference", "runtime", "review"))
def test_reviewed_expectation_requires_inference_runtime_and_review_metadata(
    tmp_path, missing_field
):
    _root, model_root, normalization_path, _overrides, candidate = _candidate(tmp_path)
    payload = _expected(candidate, model_root, normalization_path).to_dict()
    del payload[missing_field]
    with pytest.raises(ReleaseIntakeError, match="missing fields"):
        ReviewedExpectedReceipt.from_dict(payload)


def test_duplicate_json_keys_are_rejected(tmp_path):
    _root, _model_root, _normalization_path, _overrides, candidate = _candidate(tmp_path)
    payload = candidate.to_json().replace(
        '"status":"candidate_unreviewed"',
        '"status":"candidate_unreviewed","status":"candidate_unreviewed"',
        1,
    )
    with pytest.raises(ReleaseIntakeError, match="duplicate JSON key"):
        CandidateBundleReceipt.from_json(payload)


def test_nonfinite_json_constants_are_rejected(tmp_path):
    _root, _model_root, _normalization_path, _overrides, candidate = _candidate(tmp_path)
    payload = candidate.to_json().replace(
        '"bytes":', '"unknown_numeric":NaN,"bytes":', 1
    )
    with pytest.raises(ReleaseIntakeError, match="non-finite JSON constant"):
        CandidateBundleReceipt.from_json(payload)


def test_candidate_json_cannot_claim_reviewed_status(tmp_path):
    _root, _model_root, _normalization_path, _overrides, candidate = _candidate(tmp_path)
    with pytest.raises(ReleaseIntakeError, match="candidate_unreviewed"):
        replace(candidate, status=REVIEWED_EXPECTED_STATUS)


def test_cli_writes_exclusive_candidate_outside_bundle(tmp_path, capsys):
    root, _model_root, normalization_path, overrides = _make_bundle(tmp_path)
    roles_path = tmp_path / "roles.json"
    roles_path.write_text(
        "{" + ",".join(
            '"%s":"%s"' % (path, role.value)
            for path, role in sorted(overrides.items())
        ) + "}",
        encoding="utf-8",
    )
    output = tmp_path / "candidate.json"
    assert main(("intake", "--root", str(root), "--output", str(output), "--roles", str(roles_path))) == 0
    candidate = CandidateBundleReceipt.from_json(output.read_text(encoding="utf-8"))
    assert any(item.path == normalization_path for item in candidate.files)
    assert main(("intake", "--root", str(root), "--output", str(output), "--roles", str(roles_path))) == 2
    assert "output already exists" in capsys.readouterr().err


def test_cli_verifier_reports_readiness_but_never_gate_unblocking(tmp_path, capsys):
    root, model_root, normalization_path, overrides, candidate = _candidate(tmp_path)
    expected = _expected(candidate, model_root, normalization_path)
    candidate_path = tmp_path / "candidate.json"
    expected_path = tmp_path / "expected.json"
    roles_path = tmp_path / "roles.json"
    candidate_path.write_text(candidate.to_json(), encoding="utf-8")
    expected_path.write_text(expected.to_json(), encoding="utf-8")
    roles_path.write_text(
        json.dumps({path: role.value for path, role in overrides.items()}, sort_keys=True),
        encoding="utf-8",
    )
    exit_code = main(
        (
            "verify",
            "--root",
            str(root),
            "--candidate",
            str(candidate_path),
            "--expected",
            str(expected_path),
            "--roles",
            str(roles_path),
        )
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ready"] is True
    assert payload["physics_timestep_convergence_unblocked"] is False
