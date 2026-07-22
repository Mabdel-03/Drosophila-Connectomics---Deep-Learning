import hashlib
import json
import os
from dataclasses import FrozenInstanceError, replace

import pytest

from fly_sensor2behavior import CalibrationEvidenceContract
from fly_sensor2behavior.calibration import (
    ArtifactFile,
    CalibrationContractError,
    CalibrationSplit,
    FitDiagnostic,
    FitProvenance,
    LicenseRecord,
    LicensedSource,
    MetricProtocol,
    PermanentSplitPlan,
    ProtocolReference,
    RedistributionStatus,
    TrialAssignment,
    TrialRecord,
    verify_artifact_files,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact(tmp_path, artifact_id: str, relative_path: str, payload: bytes) -> ArtifactFile:
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return ArtifactFile(
        artifact_id=artifact_id,
        uri="urn:manufactured-artifact:%s" % artifact_id,
        relative_path=relative_path,
        sha256=_sha256(payload),
        bytes=len(payload),
        media_type="application/octet-stream",
    )


def _contract(tmp_path) -> CalibrationEvidenceContract:
    raw = _artifact(tmp_path, "raw-trials", "inputs/trials.bin", b"manufactured trials")
    posterior = _artifact(
        tmp_path, "parameter-posterior", "fit/posterior.bin", b"manufactured posterior"
    )
    diagnostics = _artifact(
        tmp_path, "fit-diagnostics", "fit/diagnostics.json", b'{"loss":0.125}'
    )
    source = LicensedSource(
        source_id="manufactured-source",
        source_uri="urn:manufactured-source:calibration-contract-test",
        citation="Manufactured fixture generated inside the unit test.",
        license=LicenseRecord(
            license_name="Test-only manufactured data",
            license_uri="urn:license:test-only-manufactured",
            redistribution=RedistributionStatus.REDISTRIBUTABLE,
        ),
        artifact_ids=(raw.artifact_id,),
    )
    trials = tuple(
        TrialRecord(
            trial_id="trial-%s-%d" % (prefix, index),
            individual_id="individual-%s-%d" % (prefix, index),
            session_id="session-%s-%d" % (prefix, index),
            source_id=source.source_id,
            artifact_ids=(raw.artifact_id,),
        )
        for prefix in ("cal", "val", "held")
        for index in (1, 2)
    )
    split_for_prefix = {
        "cal": CalibrationSplit.CALIBRATION,
        "val": CalibrationSplit.VALIDATION,
        "held": CalibrationSplit.HELD_OUT,
    }
    assignments = tuple(
        TrialAssignment(
            trial_id=trial.trial_id,
            split=split_for_prefix[trial.trial_id.split("-")[1]],
        )
        for trial in trials
    )
    metric = MetricProtocol(
        metric_id="manufactured-error",
        protocol_uri="urn:protocol:manufactured-error:v1",
        protocol_sha256=_sha256(b"metric protocol v1"),
        formula="absolute(predicted - observed)",
        aggregation="median per individual, then arithmetic mean across individuals",
        uncertainty="individual-level percentile bootstrap with 1000 fixed-seed resamples",
        unit="rad",
    )
    fit = FitProvenance(
        fit_id="manufactured-fit",
        code_uri="urn:code:manufactured-fit:v1",
        code_sha256=_sha256(b"fit code v1"),
        objective="minimize the declared manufactured-error metric",
        likelihood="Gaussian residual model with fixed manufactured variance",
        metric_ids=(metric.metric_id,),
        calibration_trial_ids=tuple(
            assignment.trial_id
            for assignment in assignments
            if assignment.split is CalibrationSplit.CALIBRATION
        ),
        parameter_posterior_artifact_id=posterior.artifact_id,
        diagnostics_artifact_id=diagnostics.artifact_id,
        diagnostics=(
            FitDiagnostic(
                diagnostic_id="finite-objective",
                value=0.125,
                unit="rad",
                interpretation="Manufactured finite objective used only for contract testing.",
            ),
        ),
    )
    return CalibrationEvidenceContract(
        contract_id="manufactured-calibration-contract",
        contract_version="1.0.0",
        sources=(source,),
        artifacts=(raw, posterior, diagnostics),
        trials=trials,
        split_plan=PermanentSplitPlan(
            split_id="manufactured-permanent-split",
            split_version="1.0.0",
            assignments=assignments,
        ),
        preprocessing_protocols=(
            ProtocolReference(
                protocol_id="manufactured-preprocessing",
                uri="urn:protocol:manufactured-preprocessing:v1",
                sha256=_sha256(b"preprocessing protocol v1"),
            ),
        ),
        metric_protocols=(metric,),
        fit=fit,
    )


def test_contract_is_immutable_canonical_and_losslessly_round_trips(tmp_path):
    contract = _contract(tmp_path)
    restored = CalibrationEvidenceContract.from_json(contract.to_json())

    assert restored == contract
    assert restored.to_json() == contract.to_json()
    assert restored.content_sha256 == contract.content_sha256
    assert len(contract.content_sha256) == 64
    assert isinstance(contract.trials, tuple)
    with pytest.raises(FrozenInstanceError):
        contract.contract_id = "changed"


def test_json_loader_rejects_duplicate_and_unknown_fields(tmp_path):
    contract = _contract(tmp_path)
    raw = contract.to_json()
    duplicate = raw.replace(
        '"contract_id":"manufactured-calibration-contract",',
        '"contract_id":"first","contract_id":"manufactured-calibration-contract",',
        1,
    )
    with pytest.raises(CalibrationContractError, match="duplicate JSON key"):
        CalibrationEvidenceContract.from_json(duplicate)

    decoded = json.loads(raw)
    decoded["unregistered_claim"] = True
    with pytest.raises(CalibrationContractError, match="unknown fields"):
        CalibrationEvidenceContract.from_dict(decoded)


def test_verification_accepts_exact_manufactured_files(tmp_path):
    contract = _contract(tmp_path)
    assert contract.verify_files(tmp_path) == contract.artifacts
    assert verify_artifact_files(contract.artifacts, tmp_path) == contract.artifacts


def test_verification_rejects_missing_file(tmp_path):
    contract = _contract(tmp_path)
    (tmp_path / contract.artifacts[0].relative_path).unlink()
    with pytest.raises(CalibrationContractError, match="is missing"):
        contract.verify_files(tmp_path)


def test_verification_rejects_same_size_tampering(tmp_path):
    contract = _contract(tmp_path)
    artifact = contract.artifacts[0]
    (tmp_path / artifact.relative_path).write_bytes(b"x" * artifact.bytes)
    with pytest.raises(CalibrationContractError, match="SHA-256 mismatch"):
        contract.verify_files(tmp_path)


def test_verification_rejects_size_mismatch_before_hashing(tmp_path):
    contract = _contract(tmp_path)
    artifact = contract.artifacts[0]
    (tmp_path / artifact.relative_path).write_bytes(b"changed size")
    with pytest.raises(CalibrationContractError, match="byte count mismatch"):
        contract.verify_files(tmp_path)


def test_contract_rejects_duplicate_local_artifact_paths(tmp_path):
    contract = _contract(tmp_path)
    duplicate_path = replace(
        contract.artifacts[1],
        relative_path=contract.artifacts[0].relative_path,
    )

    with pytest.raises(CalibrationContractError, match="duplicate relative_path"):
        replace(
            contract,
            artifacts=(contract.artifacts[0], duplicate_path, contract.artifacts[2]),
        )


def test_verification_rejects_symlink_and_hardlink_aliases(tmp_path):
    contract = _contract(tmp_path)
    raw = contract.artifacts[0]
    raw_path = tmp_path / raw.relative_path
    replacement = tmp_path / "inputs" / "replacement.bin"
    replacement.write_bytes(raw_path.read_bytes())
    raw_path.unlink()
    raw_path.symlink_to(replacement)
    with pytest.raises(CalibrationContractError, match="symlink path component"):
        contract.verify_files(tmp_path)

    raw_path.unlink()
    raw_path.write_bytes(b"manufactured trials")
    posterior = contract.artifacts[1]
    diagnostics = replace(
        contract.artifacts[2],
        sha256=posterior.sha256,
        bytes=posterior.bytes,
    )
    diagnostics_path = tmp_path / diagnostics.relative_path
    diagnostics_path.unlink()
    os.link(tmp_path / posterior.relative_path, diagnostics_path)
    aliased = replace(
        contract,
        artifacts=(contract.artifacts[0], posterior, diagnostics),
    )
    with pytest.raises(CalibrationContractError, match="aliases the same file"):
        aliased.verify_files(tmp_path)


def test_artifact_paths_cannot_escape_verification_root():
    with pytest.raises(CalibrationContractError, match="without traversal"):
        ArtifactFile(
            artifact_id="escape",
            uri="urn:artifact:escape",
            relative_path="../outside.bin",
            sha256="0" * 64,
            bytes=0,
            media_type="application/octet-stream",
        )


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("individual_id", "individual .* leaks"),
        ("session_id", "session .* leaks"),
    ),
)
def test_group_leakage_across_splits_is_rejected(tmp_path, field, message):
    contract = _contract(tmp_path)
    calibration = contract.trials[0]
    validation = contract.trials[2]
    changed = replace(validation, **{field: getattr(calibration, field)})
    trials = tuple(changed if trial.trial_id == changed.trial_id else trial for trial in contract.trials)

    with pytest.raises(CalibrationContractError, match=message):
        replace(contract, trials=trials)


def test_duplicate_and_unknown_split_trials_are_rejected(tmp_path):
    contract = _contract(tmp_path)
    assignments = contract.split_plan.assignments
    with pytest.raises(CalibrationContractError, match="duplicate trials"):
        replace(contract.split_plan, assignments=assignments + (assignments[0],))

    unknown = replace(assignments[0], trial_id="unknown-trial")
    plan = replace(contract.split_plan, assignments=(unknown,) + assignments[1:])
    with pytest.raises(CalibrationContractError, match="unknown trials"):
        replace(contract, split_plan=plan)


def test_duplicate_and_unassigned_inventory_trials_are_rejected(tmp_path):
    contract = _contract(tmp_path)
    with pytest.raises(CalibrationContractError, match="duplicate trial_id"):
        replace(contract, trials=contract.trials + (contract.trials[0],))

    reduced_assignments = tuple(
        assignment
        for assignment in contract.split_plan.assignments
        if assignment.trial_id != "trial-held-2"
    )
    plan = replace(contract.split_plan, assignments=reduced_assignments)
    with pytest.raises(CalibrationContractError, match="omits trials"):
        replace(contract, split_plan=plan)


def test_fit_inputs_must_equal_only_the_calibration_partition(tmp_path):
    contract = _contract(tmp_path)
    leaked = replace(
        contract.fit,
        calibration_trial_ids=("trial-cal-1", "trial-val-1"),
    )
    with pytest.raises(CalibrationContractError, match="must exactly equal"):
        replace(contract, fit=leaked)


def test_unknown_artifact_and_metric_references_are_rejected(tmp_path):
    contract = _contract(tmp_path)
    bad_trial = replace(contract.trials[0], artifact_ids=("unknown-artifact",))
    trials = (bad_trial,) + contract.trials[1:]
    with pytest.raises(CalibrationContractError, match="unknown IDs"):
        replace(contract, trials=trials)

    bad_fit = replace(contract.fit, metric_ids=("unknown-metric",))
    with pytest.raises(CalibrationContractError, match="unknown IDs"):
        replace(contract, fit=bad_fit)


def test_restricted_or_unknown_license_requires_access_notes():
    with pytest.raises(CalibrationContractError, match="require access_notes"):
        LicenseRecord(
            license_name="Unresolved terms",
            license_uri="urn:license:unknown",
            redistribution=RedistributionStatus.UNKNOWN,
        )
