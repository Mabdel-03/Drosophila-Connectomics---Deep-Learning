import copy
import hashlib
from pathlib import Path

import pytest

from fly_sensor2behavior.flybody_ordinary_flight_release import (
    CONTRACT_ID,
    CONTROLLER_REUSE_FILE_ID,
    CONTROL_TIMESTEP_S,
    EXPECTED_INFERENCE_STAGE_IDS,
    EXPECTED_WRAPPER_ORDER,
    FLIGHT_DATA_FILE_ID,
    ORDINARY_POLICY_FILE_ID,
    OrdinaryFlightReleaseError,
    PHYSICS_TIMESTEP_S,
    POLICY_ROOT,
    REFERENCE_PATH,
    SAVED_MODEL_PATH,
    UPSTREAM_COMMIT,
    VARIABLES_INDEX_PATH,
    WPG_BASE_FREQUENCY_HZ,
    WPG_DISCRETE_FREQUENCY_COUNT,
    WPG_PATH,
    WPG_RELATIVE_FREQUENCY_RANGE,
    canonical_inventory_sha256,
    default_expected_manifest_path,
    load_expected_manifest,
    load_runtime_audit,
    scan_staged_tree,
    validate_expected_manifest,
    validate_staged_release,
)


MANIFEST_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "reference"
    / "flybody_ordinary_flight_release.expected.v1.json"
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(root: Path, relative_path: str, payload: bytes) -> None:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def _make_staged_tree(tmp_path: Path) -> Path:
    root = tmp_path / "staged"
    files = {
        REFERENCE_PATH: b"manufactured reference trajectory",
        WPG_PATH: b"manufactured measured wing pattern",
        SAVED_MODEL_PATH: b"manufactured saved model descriptor",
        VARIABLES_INDEX_PATH: b"manufactured variables index",
        POLICY_ROOT
        + "/variables/variables.data-00000-of-00001": b"manufactured variables",
        "LICENSE.manufactured.txt": b"unit-test fixture only",
    }
    for path, payload in files.items():
        _write(root, path, payload)
    return root


def _resolved_contract_and_audit(tmp_path: Path):
    root = _make_staged_tree(tmp_path)
    contract = copy.deepcopy(load_expected_manifest(MANIFEST_PATH))
    observed = scan_staged_tree(root)
    members = [entry.to_dict() for entry in observed]

    archive_receipts = []
    for receipt in contract["expected_receipts"]["archives"]:
        file_id = receipt["file_id"]
        payload = ("manufactured archive " + file_id).encode("ascii")
        receipt.update(
            {
                "status": "resolved_reviewed",
                "bytes": len(payload),
                "sha256": _sha(payload),
                "reason": "Manufactured reviewed receipt for contract tests only.",
            }
        )
        archive_receipts.append(
            {
                "file_id": file_id,
                "source_uri": (
                    "https://janelia.figshare.com/ndownloader/files/" + file_id
                ),
                "bytes": len(payload),
                "sha256": _sha(payload),
                "verification_method": "manufactured-fixture-byte-hash",
            }
        )

    inventory = contract["expected_receipts"]["extracted_inventory"]
    inventory.update(
        {
            "status": "resolved_reviewed",
            "member_count": len(members),
            "members": members,
            "inventory_sha256": canonical_inventory_sha256(members),
            "reason": "Manufactured reviewed inventory for contract tests only.",
        }
    )

    fixture_hash = _sha(b"manufactured audit fixture")
    audit = {
        "schema_version": "1.0.0",
        "contract_id": CONTRACT_ID,
        "archive_receipts": archive_receipts,
        "inventory": members,
        "checkpoint_audit": {
            "policy_root": POLICY_ROOT,
            "format": "tensorflow_saved_model",
            "saved_model_load_succeeded": True,
            "callable_succeeded": True,
        },
        "saved_model_signature_audit": {
            "policy_root": POLICY_ROOT,
            "signature_inventory_complete": True,
            "signature_names": [],
            "callable_python_type": "manufactured.DistributionPolicy",
            "callable_input_signature": {
                "walker/joints_pos": {"shape": [25], "dtype": "float32"}
            },
            "callable_output_signature": {
                "distribution": "manufactured.Normal"
            },
            "policy_return_type": "manufactured.Normal",
            "policy_output_semantics": "distribution",
            "mean_action_shape": [10],
            "mean_action_dtype": "float32",
            "external_observation_normalization": "none_documented",
            "layer_norm_variables_in_saved_model": True,
            "observation_spec_sha256": fixture_hash,
            "action_spec_sha256": fixture_hash,
        },
        "runtime_audit": {
            "upstream_commit": UPSTREAM_COMMIT,
            "python_version": "manufactured-3.x",
            "numpy_version": "manufactured",
            "tensorflow_version": "manufactured",
            "tensorflow_probability_version": "manufactured",
            "acme_version": "manufactured",
            "sonnet_version": "manufactured",
            "dm_control_version": "manufactured",
            "mujoco_version": "manufactured",
            "platform_tag": "manufactured-test-platform",
            "dependency_lock_sha256": fixture_hash,
            "container_image_digest": "sha256:" + fixture_hash,
            "physics_timestep_s": PHYSICS_TIMESTEP_S,
            "control_timestep_s": CONTROL_TIMESTEP_S,
        },
        "action_wrapper_audit": {
            "environment_factory": "flybody.fly_envs.flight_imitation",
            "wrapper_order": list(EXPECTED_WRAPPER_ORDER),
            "canonical_clip": True,
            "policy_action_selection": "distribution_mean",
            "canonical_action_bounds": [-1.0, 1.0],
            "unwrapped_action_spec_sha256": fixture_hash,
            "wrapped_action_spec_sha256": fixture_hash,
            "action_vector_shape": [10],
            "wing_action_indices": [3, 4, 5, 6, 7, 8],
            "user_frequency_action_index": 9,
            "wpg_frequency_mapping_verified": True,
        },
        "inference_sequence_audit": {
            "executed_stage_ids": list(EXPECTED_INFERENCE_STAGE_IDS),
            "observation_precedes_action": True,
            "one_policy_call_per_control_step": True,
            "environment_step_receives_wrapped_action": True,
        },
    }
    validate_expected_manifest(contract)
    return root, contract, audit


def _subjects(report, code: str):
    return {blocker.subject for blocker in report.blockers if blocker.code == code}


def test_checked_in_contract_pins_only_the_official_ordinary_flight_path():
    manifest = load_expected_manifest(MANIFEST_PATH)

    assert manifest["source"]["upstream_commit"] == UPSTREAM_COMMIT
    assert {
        item["file_id"] for item in manifest["archive_policy"]["required"]
    } == {FLIGHT_DATA_FILE_ID, ORDINARY_POLICY_FILE_ID}
    assert manifest["archive_policy"]["excluded"] == [
        {
            "file_id": CONTROLLER_REUSE_FILE_ID,
            "name": "controller-reuse-checkpoints",
            "reason": (
                "The controller-reuse checkpoint is a separate transfer-learning "
                "artifact with a different runtime path and is not interchangeable "
                "with the ordinary flight SavedModel."
            ),
            "status": "excluded_non_interchangeable",
        }
    ]
    assert manifest["runtime_facts"] == {
        "control_timestep_s": CONTROL_TIMESTEP_S,
        "physics_timestep_s": PHYSICS_TIMESTEP_S,
        "wing_pattern_generator": {
            "base_frequency_hz": WPG_BASE_FREQUENCY_HZ,
            "control_action_mapping": (
                "base_frequency_hz * (1 + relative_frequency_range * "
                "canonical_user_action)"
            ),
            "discrete_frequency_count": WPG_DISCRETE_FREQUENCY_COUNT,
            "relative_frequency_range": WPG_RELATIVE_FREQUENCY_RANGE,
        },
    }
    assert tuple(
        stage["stage_id"] for stage in manifest["original_inference_sequence"]
    ) == EXPECTED_INFERENCE_STAGE_IDS


def test_default_expected_manifest_and_strict_audit_loader(tmp_path):
    assert default_expected_manifest_path().resolve() == MANIFEST_PATH.resolve()
    audit_path = tmp_path / "audit.json"
    audit_path.write_text('{"schema_version":"1.0.0"}', encoding="utf-8")
    assert load_runtime_audit(audit_path) == {"schema_version": "1.0.0"}

    audit_path.write_text('{"x":1,"x":2}', encoding="utf-8")
    with pytest.raises(OrdinaryFlightReleaseError, match="duplicate JSON key"):
        load_runtime_audit(audit_path)


def test_checked_in_contract_explicitly_leaves_unknown_byte_receipts_unresolved():
    manifest = load_expected_manifest(MANIFEST_PATH)

    for receipt in manifest["expected_receipts"]["archives"]:
        assert receipt["status"] == "unresolved_not_downloaded"
        assert receipt["bytes"] is None
        assert receipt["sha256"] is None
        assert "not" in receipt["reason"].lower()
    inventory = manifest["expected_receipts"]["extracted_inventory"]
    assert inventory["status"] == "unresolved_not_downloaded"
    assert inventory["members"] is None
    assert inventory["member_count"] is None
    assert inventory["inventory_sha256"] is None


def test_missing_stage_and_audit_report_distinct_fail_closed_blockers(tmp_path):
    manifest = load_expected_manifest(MANIFEST_PATH)
    report = validate_staged_release(manifest, tmp_path / "absent")

    assert report.ready is False
    assert "staged_inventory_invalid" in report.blocker_codes
    assert "checkpoint_descriptor_missing" in report.blocker_codes
    assert "checkpoint_variables_index_missing" in report.blocker_codes
    assert "checkpoint_variable_shards_missing" in report.blocker_codes
    assert _subjects(report, "archive_receipt_missing") == {
        FLIGHT_DATA_FILE_ID,
        ORDINARY_POLICY_FILE_ID,
    }
    assert "inventory_audit_missing" in report.blocker_codes
    assert "checkpoint_audit_missing" in report.blocker_codes
    assert "saved_model_signature_audit_missing" in report.blocker_codes
    assert "runtime_audit_missing" in report.blocker_codes
    assert "action_wrapper_audit_missing" in report.blocker_codes
    assert "inference_sequence_audit_missing" in report.blocker_codes
    assert report.physics_timestep_convergence_unblocked is False


def test_complete_manufactured_contract_and_stage_can_reach_intake_readiness(tmp_path):
    root, contract, audit = _resolved_contract_and_audit(tmp_path)
    report = validate_staged_release(contract, root, audit=audit)

    assert report.ready is True
    assert report.blockers == ()
    assert len(report.observed_inventory) == 6
    assert report.physics_timestep_convergence_unblocked is False


def test_missing_checkpoint_descriptor_is_not_conflated_with_signature(tmp_path):
    root, contract, audit = _resolved_contract_and_audit(tmp_path)
    (root / SAVED_MODEL_PATH).unlink()
    audit.pop("saved_model_signature_audit")

    report = validate_staged_release(contract, root, audit=audit)

    assert "checkpoint_descriptor_missing" in report.blocker_codes
    assert "saved_model_signature_audit_missing" in report.blocker_codes
    assert "inventory_audit_mismatch" in report.blocker_codes
    assert "expected_inventory_mismatch" in report.blocker_codes


def test_missing_archive_receipt_and_excluded_reuse_receipt_are_distinct(tmp_path):
    root, contract, audit = _resolved_contract_and_audit(tmp_path)
    audit["archive_receipts"] = [
        item
        for item in audit["archive_receipts"]
        if item["file_id"] != ORDINARY_POLICY_FILE_ID
    ]
    audit["archive_receipts"].append(
        {
            "file_id": CONTROLLER_REUSE_FILE_ID,
            "source_uri": (
                "https://janelia.figshare.com/ndownloader/files/"
                + CONTROLLER_REUSE_FILE_ID
            ),
            "bytes": 1,
            "sha256": _sha(b"controller reuse"),
            "verification_method": "manufactured-fixture-byte-hash",
        }
    )

    report = validate_staged_release(contract, root, audit=audit)

    assert ORDINARY_POLICY_FILE_ID in _subjects(report, "archive_receipt_missing")
    assert CONTROLLER_REUSE_FILE_ID in _subjects(report, "excluded_archive_present")


def test_inventory_tampering_is_detected_against_scan_and_expectation(tmp_path):
    root, contract, audit = _resolved_contract_and_audit(tmp_path)
    (root / WPG_PATH).write_bytes(b"tampered wing pattern")

    report = validate_staged_release(contract, root, audit=audit)

    assert "inventory_audit_mismatch" in report.blocker_codes
    assert "expected_inventory_mismatch" in report.blocker_codes


@pytest.mark.parametrize(
    "bad_name",
    (
        "variables.data-not-a-shard",
        "variables.data-00000-of-00002",
    ),
)
def test_checkpoint_variable_shards_must_be_a_complete_tensorflow_set(
    tmp_path, bad_name
):
    root, contract, audit = _resolved_contract_and_audit(tmp_path)
    original = root / POLICY_ROOT / "variables" / "variables.data-00000-of-00001"
    original.rename(original.with_name(bad_name))

    report = validate_staged_release(contract, root, audit=audit)

    assert "checkpoint_variable_shards_invalid" in report.blocker_codes


def test_required_runtime_and_action_audits_fail_closed_independently(tmp_path):
    root, contract, audit = _resolved_contract_and_audit(tmp_path)
    audit["runtime_audit"]["physics_timestep_s"] = 0.0001
    audit["action_wrapper_audit"]["policy_action_selection"] = "sample"

    report = validate_staged_release(contract, root, audit=audit)

    assert "runtime_audit_invalid" in report.blocker_codes
    assert "action_wrapper_audit_invalid" in report.blocker_codes


def test_malformed_audit_sequences_are_reported_instead_of_raising(tmp_path):
    root, contract, audit = _resolved_contract_and_audit(tmp_path)
    audit["action_wrapper_audit"]["wrapper_order"] = 7
    audit["inference_sequence_audit"]["executed_stage_ids"] = None

    report = validate_staged_release(contract, root, audit=audit)

    assert "action_wrapper_audit_invalid" in report.blocker_codes
    assert "inference_sequence_audit_invalid" in report.blocker_codes


def test_contract_rejects_wrong_clock_and_fabricated_partial_receipts():
    manifest = copy.deepcopy(load_expected_manifest(MANIFEST_PATH))
    manifest["runtime_facts"]["physics_timestep_s"] = 0.0001
    with pytest.raises(OrdinaryFlightReleaseError, match="clocks are wrong"):
        validate_expected_manifest(manifest)

    manifest = copy.deepcopy(load_expected_manifest(MANIFEST_PATH))
    receipt = manifest["expected_receipts"]["archives"][0]
    receipt["sha256"] = "0" * 64
    with pytest.raises(OrdinaryFlightReleaseError, match="null bytes and sha256"):
        validate_expected_manifest(manifest)


def test_staged_scanner_rejects_links_instead_of_following_them(tmp_path):
    root = _make_staged_tree(tmp_path)
    (root / "forbidden-link").symlink_to(root / WPG_PATH)

    with pytest.raises(OrdinaryFlightReleaseError, match="symlink"):
        scan_staged_tree(root)
