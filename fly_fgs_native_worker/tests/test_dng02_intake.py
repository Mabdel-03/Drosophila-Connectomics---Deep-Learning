"""Leakage-safe tests for the metadata-only DNg02 v3 intake."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from fly_sensor2behavior import dng02_intake as dng02


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "benchmarks"
    / "protocols"
    / "dng02-driver-line-heldout.v3.json"
)
RECEIPT_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "benchmarks"
    / "receipts"
    / "dng02-mendeley-public-artifacts.v1.json"
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_locked_intake_reads_json_only_and_never_unblocks_promotion(monkeypatch):
    observed_paths = []
    original = Path.read_bytes

    def guarded_read_bytes(path):
        observed_paths.append(Path(path))
        assert Path(path).suffix == ".json"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    intake = dng02.load_locked_dng02_intake(PROTOCOL_PATH)

    assert observed_paths == [PROTOCOL_PATH, RECEIPT_PATH]
    assert intake.hdf5_opened is False
    assert intake.promotion_unblocked is False
    assert intake.artifact_count == 6
    assert intake.sealed_outcome_artifact_count == 2
    assert intake.receipt_file_sha256 == dng02.EXPECTED_RECEIPT_FILE_SHA256
    assert intake.protocol_file_sha256 == hashlib.sha256(
        PROTOCOL_PATH.read_bytes()
    ).hexdigest()


def test_public_receipt_freezes_exact_artifact_identities_and_totals():
    receipt = _json(RECEIPT_PATH)
    dng02.validate_public_artifact_receipt(receipt)
    by_uuid = {item["file_uuid"]: item for item in receipt["artifacts"]}

    assert receipt["source"] == {
        "dataset_id": "7g984jm2zc",
        "version": 1,
        "doi": "10.17632/7g984jm2zc.1",
        "record_uri": "https://data.mendeley.com/datasets/7g984jm2zc/1",
        "publisher": "Mendeley Data",
        "license": "CC-BY-4.0",
        "license_uri": "https://creativecommons.org/licenses/by/4.0/legalcode",
        "complete_record_bytes": 170_906_272,
    }
    assert receipt["inventory_totals"] == {
        "canonical_relevant_artifact_count": 6,
        "canonical_relevant_bytes": 160_245_629,
        "sealed_outcome_artifact_count": 2,
        "sealed_outcome_bytes": 160_188_760,
        "non_outcome_artifact_count": 4,
        "non_outcome_bytes": 56_869,
    }
    assert by_uuid["71d7fc6f-c1d6-453b-a508-e5c18da58537"]["bytes"] == 84_162_848
    assert by_uuid["71d7fc6f-c1d6-453b-a508-e5c18da58537"]["sha256"] == (
        "e48b0dd500769e3a987bf8e4c16a57f94d66ae0ac88aa51ac165b359b121f148"
    )
    assert by_uuid["7ace3eba-e1f4-4862-8c9a-b970b9fcce28"]["bytes"] == 76_025_912
    assert by_uuid["7ace3eba-e1f4-4862-8c9a-b970b9fcce28"]["sha256"] == (
        "b1ef68ef01ac8cd4b48beba1274ab677cc395c55bb78f1c685a4d3216d263777"
    )
    for uuid in (
        "71d7fc6f-c1d6-453b-a508-e5c18da58537",
        "7ace3eba-e1f4-4862-8c9a-b970b9fcce28",
    ):
        assert by_uuid[uuid]["contains_outcome_values"] is True
        assert by_uuid[uuid]["access_before_seal"] == "forbidden"

    readme = by_uuid["f3848fc9-7cf2-401d-a4bd-1409b8af7887"]
    assert readme["logical_name"] == "README"
    assert readme["source_file_name_asserted"] is False
    assert readme["bytes"] == 17_929
    assert readme["sha256"] == (
        "721148ac0d20e4482f3d03d0fb34bfc3b553f613cd3c3c69179817d94b423476"
    )


def test_v3_materializes_exact_global_hash_rank_split():
    protocol = _json(PROTOCOL_PATH)
    lines = dng02.validate_protocol_v3(protocol)
    derived = dng02.derive_global_split()

    assert lines == derived
    assert len(lines) == 15
    assert [line.source_order for line in lines] == list(range(15))
    assert len({line.driver_line_id for line in lines}) == 15
    assert {
        partition: sum(line.partition == partition for line in lines)
        for partition in dng02.EXPECTED_PARTITION_COUNTS
    } == dng02.EXPECTED_PARTITION_COUNTS
    assert [
        (line.driver_line_label, line.targeted_pair_count)
        for line in lines
        if line.partition == "sealed_test"
    ] == [
        ("SS03500", 0),
        ("SS01578", 5),
        ("SS01073", 5),
        ("SS02544", 9),
        ("SS01562", 12),
    ]
    assert len(
        {
            line.targeted_pair_count
            for line in lines
            if line.partition == "sealed_test"
        }
    ) == 4
    assert protocol["permanent_split"]["missing_protocol_rule"].startswith(
        "A line need not occur in both protocols"
    )
    assert "BLOCK" in protocol["permanent_split"]["coverage_failure"]


def test_split_algorithm_is_outcome_independent_and_exact():
    first = dng02.derive_global_split()
    second = dng02.derive_global_split()
    assert first == second

    line = next(item for item in first if item.driver_line_label == "SS02634")
    payload = (
        dng02.SPLIT_SEED + "\0" + "mendeley-7g984jm2zc-v1:SS02634"
    ).encode("utf-8")
    assert line.split_rank_sha256 == hashlib.sha256(payload).hexdigest()
    assert line.split_rank == 2

    with pytest.raises(dng02.DNg02IntakeError, match="metadata drifted"):
        dng02.derive_global_split(dng02.EXPECTED_LINE_METADATA[:-1])


def test_ids_are_stable_dataset_local_and_do_not_merge_protocols():
    line_id = dng02.stable_driver_line_id("SS01073")
    open_fly = dng02.stable_fly_id(
        "71d7fc6f-c1d6-453b-a508-e5c18da58537", line_id, 2
    )
    closed_fly = dng02.stable_fly_id(
        "7ace3eba-e1f4-4862-8c9a-b970b9fcce28", line_id, 2
    )
    assert open_fly != closed_fly
    assert open_fly == dng02.stable_fly_id(
        "71d7fc6f-c1d6-453b-a508-e5c18da58537", line_id, 2
    )
    assert dng02.stable_trial_id(open_fly, 0) != dng02.stable_trial_id(open_fly, 1)
    with pytest.raises(dng02.DNg02IntakeError, match="non-negative"):
        dng02.stable_trial_id(open_fly, -1)


def test_primary_endpoint_is_methods_mean_not_fig3_maximum():
    protocol = _json(PROTOCOL_PATH)
    endpoint = protocol["primary_methods_endpoint"]

    assert endpoint["status"] == "promotion_authoritative"
    assert endpoint["baseline"]["interval_s"] == [-0.5, 0.0]
    assert endpoint["response"]["interval_s"] == [0.0, 0.5]
    assert endpoint["activation_trials_per_fly"] == 30
    assert endpoint["unit_conversion"]["factor_expression"] == "pi/180"
    assert "_mx_WBA" in endpoint["source_signal"]
    assert endpoint["time_sampling_rule"].startswith("Use source timestamps")

    diagnostic = protocol["source_code_regression_diagnostic"]
    assert diagnostic["status"] == (
        "blocked_pending_static_code_constant_extraction_receipt"
    )
    assert diagnostic["promotion_eligible"] is False
    assert diagnostic["model_selection_eligible"] is False
    assert diagnostic["blocks_primary_methods_endpoint"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "hdf_access",
        "artifact_digest",
        "split_assignment",
        "methods_window",
        "diagnostic_promotion",
        "bootstrap_seed",
    ),
)
def test_validator_rejects_leakage_or_contract_drift(mutation):
    receipt = _json(RECEIPT_PATH)
    protocol = _json(PROTOCOL_PATH)

    if mutation == "hdf_access":
        receipt["artifacts"][-1]["access_before_seal"] = "allowed_read_only"
        with pytest.raises(dng02.DNg02IntakeError, match="access_before_seal"):
            dng02.validate_public_artifact_receipt(receipt)
    elif mutation == "artifact_digest":
        receipt["artifacts"][0]["sha256"] = "0" * 64
        with pytest.raises(dng02.DNg02IntakeError, match="SHA-256 drifted"):
            dng02.validate_public_artifact_receipt(receipt)
    elif mutation == "split_assignment":
        protocol["non_outcome_driver_line_metadata"]["driver_lines"][0][
            "partition"
        ] = "calibration"
        with pytest.raises(dng02.DNg02IntakeError, match="assignment drifted"):
            dng02.validate_protocol_v3(protocol)
    elif mutation == "methods_window":
        protocol["primary_methods_endpoint"]["response"]["interval_s"] = [0.0, 3.0]
        with pytest.raises(dng02.DNg02IntakeError, match="windows drifted"):
            dng02.validate_protocol_v3(protocol)
    elif mutation == "diagnostic_promotion":
        protocol["source_code_regression_diagnostic"]["promotion_eligible"] = True
        with pytest.raises(dng02.DNg02IntakeError, match="leak into promotion"):
            dng02.validate_protocol_v3(protocol)
    else:
        protocol["null_and_inference"]["bootstrap"]["seed"] = "picked-after-test"
        with pytest.raises(dng02.DNg02IntakeError, match="bootstrap contract"):
            dng02.validate_protocol_v3(protocol)


def test_file_loader_rejects_receipt_tampering_before_semantic_parse(tmp_path):
    receipt = RECEIPT_PATH.read_text(encoding="utf-8")
    tampered = tmp_path / "receipt.json"
    tampered.write_text(receipt.replace("160245629", "160245628"), encoding="utf-8")

    with pytest.raises(dng02.DNg02IntakeError, match="file SHA-256 mismatch"):
        dng02.load_locked_dng02_intake(PROTOCOL_PATH, receipt_path=tampered)


def test_loader_resolves_wheel_style_sibling_receipt(tmp_path):
    protocols = tmp_path / "share" / "fly-sensor2behavior" / "benchmarks" / "protocols"
    receipts = protocols.parent / "receipts"
    protocols.mkdir(parents=True)
    receipts.mkdir()
    installed_protocol = protocols / PROTOCOL_PATH.name
    installed_receipt = receipts / RECEIPT_PATH.name
    installed_protocol.write_bytes(PROTOCOL_PATH.read_bytes())
    installed_receipt.write_bytes(RECEIPT_PATH.read_bytes())

    intake = dng02.load_locked_dng02_intake(installed_protocol)
    assert Path(intake.receipt_path) == installed_receipt
    assert intake.receipt_file_sha256 == dng02.EXPECTED_RECEIPT_FILE_SHA256


def test_duplicate_json_keys_and_non_json_inputs_fail_closed(tmp_path):
    duplicate = tmp_path / "protocol.json"
    duplicate.write_text(
        '{"protocol_id":"dng02-driver-line-heldout",'
        '"protocol_id":"dng02-driver-line-heldout"}',
        encoding="utf-8",
    )
    with pytest.raises(dng02.DNg02IntakeError, match="duplicate JSON key"):
        dng02.load_locked_dng02_intake(duplicate, receipt_path=RECEIPT_PATH)

    non_json = tmp_path / "outcomes.hdf5"
    non_json.write_bytes(b"not opened as HDF5")
    with pytest.raises(dng02.DNg02IntakeError, match="JSON control artifacts only"):
        dng02.load_locked_dng02_intake(non_json, receipt_path=RECEIPT_PATH)


def test_missingness_null_and_sealed_rules_are_explicit():
    protocol = _json(PROTOCOL_PATH)
    missingness = protocol["missingness_and_exclusions"]
    inference = protocol["null_and_inference"]
    sealing = protocol["sealed_outcome_access"]

    assert missingness["imputation"] == "forbidden"
    assert missingness["winsorization"] == "forbidden"
    assert "effect size" in missingness["forbidden_exclusions"][0]
    assert inference["independent_cluster"] == "driver_line_id"
    assert inference["bootstrap"]["resamples"] == 10_000
    assert inference["bootstrap"]["seed"] == dng02.BOOTSTRAP_SEED
    assert "calibration-partition" in inference["calibration_only_null"]
    assert sealing["outcome_files"] == [
        "fig1_3A_dataset.hdf5",
        "fig3B-E_dataset.hdf5",
    ]
    assert "sealed-test values" in sealing["trusted_splitter"]
