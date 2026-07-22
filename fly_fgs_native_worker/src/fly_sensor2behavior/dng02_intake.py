"""Metadata-only intake for the preregistered DNg02 held-out evaluation.

This module validates two small JSON control-plane artifacts.  It has no HDF5
dependency, performs no network requests, and never opens a source outcome
file.  The expected HDF5 byte counts and digests are evidence locks, not an
authorization to inspect their contents.

The validator deliberately fails closed on split drift.  Driver-line labels
and targeted-pair proxies come from hash-locked, non-outcome source code; one
SHA-256 rank assigns each line globally across open- and closed-loop data.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


DOI = "10.17632/7g984jm2zc.1"
LICENSE = "CC-BY-4.0"
PROTOCOL_ID = "dng02-driver-line-heldout"
PROTOCOL_VERSION = "3.0.0"
PROTOCOL_STATUS = "frozen_before_outcome_access"
RECEIPT_ID = "dng02-mendeley-public-artifacts-v1"
RECEIPT_STATUS = "audited_expected_public_artifacts_outcomes_sealed"
EXPECTED_RECEIPT_FILE_SHA256 = (
    "edb81f26d6c0b1c75ec04d020366aa45b401f70defc2e1e8514b5fd7bb74c019"
)
EXPECTED_RECEIPT_RELATIVE_PATH = (
    "data/benchmarks/receipts/dng02-mendeley-public-artifacts.v1.json"
)
SPLIT_SEED = "dng02-driver-line-heldout-v3-line-split"
BOOTSTRAP_SEED = "dng02-driver-line-heldout-v3-bootstrap"
LINE_ID_PREFIX = "mendeley-7g984jm2zc-v1:"
EXPECTED_PARTITION_COUNTS = {
    "calibration": 7,
    "validation": 3,
    "sealed_test": 5,
}
EXPECTED_LINE_METADATA = (
    ("SS03500", 0),
    ("SS02634", 1),
    ("SS02627", 2),
    ("SS01577", 3),
    ("SS02535", 3),
    ("SS01578", 5),
    ("SS01073", 5),
    ("SS02550", 6),
    ("SS01563", 8),
    ("SS02624", 8),
    ("SS02625", 8),
    ("SS02544", 9),
    ("SS02630", 10),
    ("SS01562", 12),
    ("R42B02", 15),
)
EXPECTED_LINE_PAYLOAD = (
    '[["SS03500",0],["SS02634",1],["SS02627",2],["SS01577",3],'
    '["SS02535",3],["SS01578",5],["SS01073",5],["SS02550",6],'
    '["SS01563",8],["SS02624",8],["SS02625",8],["SS02544",9],'
    '["SS02630",10],["SS01562",12],["R42B02",15]]'
)
EXPECTED_LINE_PAYLOAD_SHA256 = (
    "717e041167de3c295837dd2cc67de4f4385e316be8b51866efd242988fe1f6e5"
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_CONTENT_URL_PREFIX = (
    "https://data.mendeley.com/public-files/datasets/7g984jm2zc/files/"
)

# UUID is the stable key because the README's exact repository filename was
# not independently asserted during the audit.  Its logical name is locked.
EXPECTED_ARTIFACTS = {
    "e5c85743-1d8f-433e-8e05-d320990df39d": {
        "name_field": "file_name",
        "name": "DNg02Fig1_CB.py",
        "bytes": 7_024,
        "sha256": "80540f3aeef6bdc31d91e760e87558aeaf08727d2e183bd182f47289d3e6b72e",
        "role": "source_analysis_code",
        "contains_outcome_values": False,
        "access_before_seal": "allowed_read_only",
    },
    "e38287b2-a759-415c-be0a-51990f25e2b4": {
        "name_field": "file_name",
        "name": "DNg02Fig3AB_CB.py",
        "bytes": 11_535,
        "sha256": "3e1be876cb465b81ec1f77a9088ad7bc6cc10deaa7507a86d1f224cc9f5d8186",
        "role": "source_analysis_code",
        "contains_outcome_values": False,
        "access_before_seal": "allowed_read_only",
    },
    "99292709-45ed-4b82-977d-a51298ac8f05": {
        "name_field": "file_name",
        "name": "DNg02Fig3CDE_CB.py",
        "bytes": 20_381,
        "sha256": "7e5cc0158711be27a7e3f4a4a124dd88566583da90929de5a1b38ff6f96872aa",
        "role": "source_analysis_code",
        "contains_outcome_values": False,
        "access_before_seal": "allowed_read_only",
    },
    "f3848fc9-7cf2-401d-a4bd-1409b8af7887": {
        "name_field": "logical_name",
        "name": "README",
        "bytes": 17_929,
        "sha256": "721148ac0d20e4482f3d03d0fb34bfc3b553f613cd3c3c69179817d94b423476",
        "role": "readme_axis_and_unit_metadata",
        "contains_outcome_values": False,
        "access_before_seal": "allowed_read_only",
    },
    "71d7fc6f-c1d6-453b-a508-e5c18da58537": {
        "name_field": "file_name",
        "name": "fig1_3A_dataset.hdf5",
        "bytes": 84_162_848,
        "sha256": "e48b0dd500769e3a987bf8e4c16a57f94d66ae0ac88aa51ac165b359b121f148",
        "role": "sealed_open_loop_outcomes",
        "contains_outcome_values": True,
        "access_before_seal": "forbidden",
    },
    "7ace3eba-e1f4-4862-8c9a-b970b9fcce28": {
        "name_field": "file_name",
        "name": "fig3B-E_dataset.hdf5",
        "bytes": 76_025_912,
        "sha256": "b1ef68ef01ac8cd4b48beba1274ab677cc395c55bb78f1c685a4d3216d263777",
        "role": "sealed_closed_loop_outcomes",
        "contains_outcome_values": True,
        "access_before_seal": "forbidden",
    },
}


class DNg02IntakeError(ValueError):
    """Raised when a metadata receipt or preregistration drifts."""


@dataclass(frozen=True)
class SplitLine:
    """One frozen, outcome-independent driver-line assignment."""

    source_order: int
    driver_line_label: str
    driver_line_id: str
    targeted_pair_count: int
    split_rank: int
    split_rank_sha256: str
    partition: str


@dataclass(frozen=True)
class ValidatedDNg02Intake:
    """Proof that only the two locked JSON control artifacts were validated."""

    protocol_path: str
    protocol_file_sha256: str
    receipt_path: str
    receipt_file_sha256: str
    driver_lines: Tuple[SplitLine, ...]
    artifact_count: int
    sealed_outcome_artifact_count: int
    hdf5_opened: bool = False
    promotion_unblocked: bool = False


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DNg02IntakeError("duplicate JSON key %r" % key)
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise DNg02IntakeError("non-finite JSON constant %r is forbidden" % value)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DNg02IntakeError("%s must be a JSON object" % label)
    return value


def _sequence(value: Any, label: str) -> Tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise DNg02IntakeError("%s must be a JSON array" % label)
    try:
        return tuple(value)
    except TypeError as exc:
        raise DNg02IntakeError("%s must be a JSON array" % label) from exc


def _exact_keys(
    value: Mapping[str, Any], expected: Iterable[str], label: str
) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        raise DNg02IntakeError(
            "%s fields drifted (missing=%s, extra=%s)"
            % (label, missing, extra)
        )


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DNg02IntakeError("%s must be a positive integer" % label)
    return value


def _zero_based_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DNg02IntakeError("%s must be a non-negative integer" % label)
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise DNg02IntakeError("%s must be a lowercase SHA-256" % label)
    return value


def _regular_json_bytes(path: os.PathLike[str] | str) -> Tuple[Path, bytes]:
    candidate = Path(path).absolute()
    try:
        info = candidate.lstat()
    except OSError as exc:
        raise DNg02IntakeError("cannot stat JSON control artifact %s" % candidate) from exc
    if stat.S_ISLNK(info.st_mode):
        raise DNg02IntakeError("JSON control artifact must not be a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise DNg02IntakeError("JSON control artifact must be a regular file")
    if candidate.suffix != ".json":
        raise DNg02IntakeError("metadata intake opens JSON control artifacts only")
    try:
        return candidate, candidate.read_bytes()
    except OSError as exc:
        raise DNg02IntakeError("cannot read JSON control artifact %s" % candidate) from exc


def _parse_json(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DNg02IntakeError("invalid %s JSON: %s" % (label, exc)) from exc
    return _mapping(value, label)


def _file_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def stable_driver_line_id(label: str) -> str:
    """Return the locked readable dataset-local ID without normalization."""

    if not isinstance(label, str) or not label or label != label.strip():
        raise DNg02IntakeError("driver-line label must be a non-empty exact string")
    if label not in {item[0] for item in EXPECTED_LINE_METADATA}:
        raise DNg02IntakeError("driver-line label is outside frozen metadata")
    return LINE_ID_PREFIX + label


def split_rank_sha256(driver_line_id: str, *, seed: str = SPLIT_SEED) -> str:
    """Compute the outcome-independent rank digest from exact UTF-8 bytes."""

    if not isinstance(seed, str) or not seed:
        raise DNg02IntakeError("split seed must be non-empty")
    if not isinstance(driver_line_id, str) or not driver_line_id.startswith(
        LINE_ID_PREFIX
    ):
        raise DNg02IntakeError("driver_line_id is outside the frozen namespace")
    return hashlib.sha256((seed + "\0" + driver_line_id).encode("utf-8")).hexdigest()


def derive_global_split(
    line_metadata: Sequence[Tuple[str, int]] = EXPECTED_LINE_METADATA,
    *,
    seed: str = SPLIT_SEED,
) -> Tuple[SplitLine, ...]:
    """Materialize the exact 7/3/5 line split without outcome information."""

    metadata = tuple(line_metadata)
    if metadata != EXPECTED_LINE_METADATA:
        raise DNg02IntakeError("eligible driver-line metadata drifted")
    ranked = sorted(
        (
            split_rank_sha256(stable_driver_line_id(label), seed=seed),
            stable_driver_line_id(label),
            label,
            pair_count,
            source_order,
        )
        for source_order, (label, pair_count) in enumerate(metadata)
    )
    digests = tuple(row[0] for row in ranked)
    if len(digests) != len(set(digests)):
        raise DNg02IntakeError("driver-line split digest collision")
    by_source: Dict[int, SplitLine] = {}
    for zero_rank, (digest, line_id, label, pair_count, source_order) in enumerate(
        ranked
    ):
        rank = zero_rank + 1
        partition = (
            "calibration" if rank <= 7 else "validation" if rank <= 10 else "sealed_test"
        )
        by_source[source_order] = SplitLine(
            source_order=source_order,
            driver_line_label=label,
            driver_line_id=line_id,
            targeted_pair_count=pair_count,
            split_rank=rank,
            split_rank_sha256=digest,
            partition=partition,
        )
    return tuple(by_source[index] for index in range(len(metadata)))


def stable_fly_id(source_file_uuid: str, driver_line_id: str, fly_axis_index: int) -> str:
    """Derive a protocol-local fly ID; it intentionally cannot merge files."""

    if not isinstance(source_file_uuid, str) or not _UUID_PATTERN.fullmatch(
        source_file_uuid
    ):
        raise DNg02IntakeError("source file UUID is malformed")
    if source_file_uuid not in EXPECTED_ARTIFACTS:
        raise DNg02IntakeError("source file UUID is not registered")
    if not EXPECTED_ARTIFACTS[source_file_uuid]["contains_outcome_values"]:
        raise DNg02IntakeError("fly IDs require a registered outcome file UUID")
    if driver_line_id not in {
        stable_driver_line_id(label) for label, _count in EXPECTED_LINE_METADATA
    }:
        raise DNg02IntakeError("driver_line_id is not registered")
    index = _zero_based_integer(fly_axis_index, "fly axis index")
    payload = "%s\0%s\0%d" % (source_file_uuid, driver_line_id, index)
    return "dng02-fly-v1:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_trial_id(fly_id: str, activation_trial_index: int) -> str:
    """Derive a stable source-coordinate trial identity."""

    if not isinstance(fly_id, str) or not re.fullmatch(
        r"dng02-fly-v1:[0-9a-f]{64}", fly_id
    ):
        raise DNg02IntakeError("fly_id is malformed")
    index = _zero_based_integer(activation_trial_index, "activation trial index")
    payload = "%s\0%d" % (fly_id, index)
    return "dng02-trial-v1:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_public_artifact_receipt(data: Mapping[str, Any]) -> None:
    """Validate exact metadata receipts without touching the referenced URLs."""

    data = _mapping(data, "artifact receipt")
    _exact_keys(
        data,
        (
            "schema_version",
            "receipt_id",
            "receipt_status",
            "created_utc",
            "source",
            "construction_access_boundary",
            "canonical_readme_deduplication",
            "artifacts",
            "inventory_totals",
            "scientific_boundary",
        ),
        "artifact receipt",
    )
    if data["schema_version"] != "1.0.0":
        raise DNg02IntakeError("artifact receipt schema version drifted")
    if data["receipt_id"] != RECEIPT_ID or data["receipt_status"] != RECEIPT_STATUS:
        raise DNg02IntakeError("artifact receipt identity or status drifted")

    source = _mapping(data["source"], "artifact source")
    expected_source = {
        "dataset_id": "7g984jm2zc",
        "version": 1,
        "doi": DOI,
        "record_uri": "https://data.mendeley.com/datasets/7g984jm2zc/1",
        "publisher": "Mendeley Data",
        "license": LICENSE,
        "license_uri": "https://creativecommons.org/licenses/by/4.0/legalcode",
        "complete_record_bytes": 170_906_272,
    }
    if dict(source) != expected_source:
        raise DNg02IntakeError("artifact source metadata drifted")

    access = _mapping(data["construction_access_boundary"], "access boundary")
    if (
        access.get("mode") != "metadata_and_non_outcome_source_only"
        or access.get("hdf5_outcome_files_downloaded_during_preregistration") is not False
        or access.get("hdf5_outcome_arrays_opened_during_preregistration") is not False
        or access.get("expected_digests_recomputed_from_hdf5_during_preregistration")
        is not False
        or access.get("receipt_authorizes_outcome_access") is not False
    ):
        raise DNg02IntakeError("outcome access boundary is not fail-closed")

    readme = _mapping(
        data["canonical_readme_deduplication"], "README deduplication"
    )
    if (
        readme.get("logical_name") != "README"
        or readme.get("source_file_name_asserted") is not False
        or readme.get("canonical_file_uuid")
        != "f3848fc9-7cf2-401d-a4bd-1409b8af7887"
        or readme.get("excluded_duplicate_uuids_registered") is not False
    ):
        raise DNg02IntakeError("README canonicalization drifted")

    artifacts = _sequence(data["artifacts"], "artifacts")
    if len(artifacts) != len(EXPECTED_ARTIFACTS):
        raise DNg02IntakeError("artifact inventory length drifted")
    by_uuid: Dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(artifacts):
        artifact = _mapping(raw, "artifact[%d]" % index)
        uuid = artifact.get("file_uuid")
        if not isinstance(uuid, str) or not _UUID_PATTERN.fullmatch(uuid):
            raise DNg02IntakeError("artifact UUID is malformed")
        if uuid in by_uuid:
            raise DNg02IntakeError("artifact UUID is duplicated")
        by_uuid[uuid] = artifact
    if set(by_uuid) != set(EXPECTED_ARTIFACTS):
        raise DNg02IntakeError("artifact UUID set drifted")

    for uuid, expected in EXPECTED_ARTIFACTS.items():
        artifact = by_uuid[uuid]
        if artifact.get(expected["name_field"]) != expected["name"]:
            raise DNg02IntakeError("artifact name drifted for %s" % uuid)
        if expected["name_field"] == "logical_name":
            if artifact.get("source_file_name_asserted") is not False:
                raise DNg02IntakeError("README source filename must remain unasserted")
        if artifact.get("bytes") != expected["bytes"]:
            raise DNg02IntakeError("artifact byte count drifted for %s" % uuid)
        if artifact.get("sha256") != expected["sha256"]:
            raise DNg02IntakeError("artifact SHA-256 drifted for %s" % uuid)
        _sha256(artifact.get("sha256"), "artifact SHA-256")
        expected_url = _CONTENT_URL_PREFIX + uuid + "/file_downloaded"
        if artifact.get("content_url") != expected_url:
            raise DNg02IntakeError("artifact content URL drifted for %s" % uuid)
        for field in ("role", "contains_outcome_values", "access_before_seal"):
            if artifact.get(field) != expected[field]:
                raise DNg02IntakeError("artifact %s drifted for %s" % (field, uuid))
        if expected["contains_outcome_values"]:
            if not str(expected["name"]).endswith(".hdf5"):
                raise DNg02IntakeError("outcome artifact must be HDF5")
            if expected["access_before_seal"] != "forbidden":
                raise DNg02IntakeError("outcome artifact is not sealed")

    totals = _mapping(data["inventory_totals"], "inventory totals")
    expected_total_bytes = sum(int(item["bytes"]) for item in EXPECTED_ARTIFACTS.values())
    outcome_items = [
        item for item in EXPECTED_ARTIFACTS.values() if item["contains_outcome_values"]
    ]
    expected_totals = {
        "canonical_relevant_artifact_count": len(EXPECTED_ARTIFACTS),
        "canonical_relevant_bytes": expected_total_bytes,
        "sealed_outcome_artifact_count": len(outcome_items),
        "sealed_outcome_bytes": sum(int(item["bytes"]) for item in outcome_items),
        "non_outcome_artifact_count": len(EXPECTED_ARTIFACTS) - len(outcome_items),
        "non_outcome_bytes": expected_total_bytes
        - sum(int(item["bytes"]) for item in outcome_items),
    }
    if dict(totals) != expected_totals:
        raise DNg02IntakeError("artifact inventory totals drifted")


def _validate_protocol_sources(data: Mapping[str, Any]) -> None:
    public = _mapping(data["public_source"], "public source")
    if public.get("doi") != DOI or public.get("license") != LICENSE:
        raise DNg02IntakeError("protocol public source drifted")
    if public.get("artifact_receipt_path") != EXPECTED_RECEIPT_RELATIVE_PATH:
        raise DNg02IntakeError("protocol receipt path drifted")
    if public.get("artifact_receipt_file_sha256") != EXPECTED_RECEIPT_FILE_SHA256:
        raise DNg02IntakeError("protocol receipt digest drifted")

    protocols = _mapping(data["protocols"], "protocols")
    if set(protocols) != {"open_loop", "closed_loop"}:
        raise DNg02IntakeError("open/closed protocol set drifted")
    expected = {
        "open_loop": (
            "striped_drum_open_loop",
            "fig1_3A_dataset.hdf5",
            "71d7fc6f-c1d6-453b-a508-e5c18da58537",
        ),
        "closed_loop": (
            "closed_loop_stripe",
            "fig3B-E_dataset.hdf5",
            "7ace3eba-e1f4-4862-8c9a-b970b9fcce28",
        ),
    }
    for name, (stimulus, file_name, uuid) in expected.items():
        protocol = _mapping(protocols[name], "%s protocol" % name)
        if (
            protocol.get("stimulus") != stimulus
            or protocol.get("source_file_name") != file_name
            or protocol.get("source_file_uuid") != uuid
            or protocol.get("time_key_template") != "{driver_line_label}_time"
            or protocol.get("trial_key_template") != "{driver_line_label}_WBAs"
            or tuple(protocol.get("trial_axes", ()))
            != ("time", "activation_trial", "dataset_local_fly")
            or protocol.get("fit_predict_resample_report_separately") is not True
        ):
            raise DNg02IntakeError("%s source schema drifted" % name)


def _validate_protocol_split(data: Mapping[str, Any]) -> Tuple[SplitLine, ...]:
    metadata = _mapping(
        data["non_outcome_driver_line_metadata"], "driver-line metadata"
    )
    if (
        metadata.get("authority_file") != "DNg02Fig3AB_CB.py"
        or metadata.get("authority_file_uuid")
        != "e38287b2-a759-415c-be0a-51990f25e2b4"
        or metadata.get("authority_file_sha256")
        != EXPECTED_ARTIFACTS["e38287b2-a759-415c-be0a-51990f25e2b4"]["sha256"]
        or metadata.get("canonical_payload") != EXPECTED_LINE_PAYLOAD
        or metadata.get("canonical_payload_sha256") != EXPECTED_LINE_PAYLOAD_SHA256
        or _file_sha256(EXPECTED_LINE_PAYLOAD.encode("utf-8"))
        != EXPECTED_LINE_PAYLOAD_SHA256
    ):
        raise DNg02IntakeError("non-outcome driver-line authority drifted")

    expected_lines = derive_global_split()
    raw_lines = _sequence(metadata.get("driver_lines"), "driver lines")
    if len(raw_lines) != len(expected_lines):
        raise DNg02IntakeError("driver-line count drifted")
    observed_lines = []
    for index, (raw, expected) in enumerate(zip(raw_lines, expected_lines)):
        line = _mapping(raw, "driver line %d" % index)
        expected_dict = {
            "source_order": expected.source_order,
            "driver_line_label": expected.driver_line_label,
            "driver_line_id": expected.driver_line_id,
            "targeted_pair_count": expected.targeted_pair_count,
            "split_rank": expected.split_rank,
            "split_rank_sha256": expected.split_rank_sha256,
            "partition": expected.partition,
        }
        if dict(line) != expected_dict:
            raise DNg02IntakeError("driver-line assignment drifted at source order %d" % index)
        observed_lines.append(expected)

    split = _mapping(data["permanent_split"], "permanent split")
    if (
        split.get("group") != "driver_line_id"
        or split.get("eligible_unique_driver_line_count") != 15
        or split.get("seed") != SPLIT_SEED
        or dict(_mapping(split.get("fixed_counts"), "fixed split counts"))
        != EXPECTED_PARTITION_COUNTS
    ):
        raise DNg02IntakeError("permanent split contract drifted")
    counts = {
        name: sum(line.partition == name for line in observed_lines)
        for name in EXPECTED_PARTITION_COUNTS
    }
    if counts != EXPECTED_PARTITION_COUNTS:
        raise DNg02IntakeError("materialized split counts drifted")
    coverage = _mapping(
        split.get("minimum_sealed_test_coverage_per_endpoint"),
        "sealed-test coverage",
    )
    if dict(coverage) != {
        "evaluable_driver_lines": 5,
        "distinct_targeted_pair_count_levels": 2,
        "minimum_evaluable_flies_per_line": 3,
    }:
        raise DNg02IntakeError("sealed-test coverage contract drifted")
    return tuple(observed_lines)


def _validate_primary_endpoint(data: Mapping[str, Any]) -> None:
    endpoint = _mapping(data["primary_methods_endpoint"], "Methods endpoint")
    if (
        endpoint.get("endpoint_id") != "methods_half_second_mean_delta_wba"
        or endpoint.get("status") != "promotion_authoritative"
        or tuple(endpoint.get("execute_separately_for", ()))
        != ("open_loop", "closed_loop")
        or endpoint.get("activation_trials_per_fly") != 30
    ):
        raise DNg02IntakeError("primary Methods endpoint identity drifted")
    baseline = _mapping(endpoint.get("baseline"), "baseline endpoint")
    response = _mapping(endpoint.get("response"), "response endpoint")
    if (
        tuple(baseline.get("interval_s", ())) != (-0.5, 0.0)
        or tuple(response.get("interval_s", ())) != (0.0, 0.5)
        or baseline.get("interval_semantics") != "left_closed_right_open"
        or response.get("interval_semantics") != "left_closed_right_open"
    ):
        raise DNg02IntakeError("Methods windows drifted")
    conversion = _mapping(endpoint.get("unit_conversion"), "unit conversion")
    if (
        conversion.get("source_unit") != "degree"
        or conversion.get("internal_unit") != "radian"
        or conversion.get("factor_expression") != "pi/180"
    ):
        raise DNg02IntakeError("Methods unit conversion drifted")
    if "Raw {driver_line_label}_WBAs" not in str(endpoint.get("source_signal", "")):
        raise DNg02IntakeError("Methods endpoint is not locked to raw trial data")

    missingness = _mapping(data["missingness_and_exclusions"], "missingness rules")
    if missingness.get("imputation") != "forbidden" or missingness.get(
        "winsorization"
    ) != "forbidden":
        raise DNg02IntakeError("missingness rules permit leakage")

    inference = _mapping(data["null_and_inference"], "inference")
    bootstrap = _mapping(inference.get("bootstrap"), "bootstrap")
    confidence = bootstrap.get("confidence")
    if (
        bootstrap.get("resamples") != 10_000
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isclose(float(confidence), 0.95, rel_tol=0.0, abs_tol=0.0)
        or bootstrap.get("seed") != BOOTSTRAP_SEED
    ):
        raise DNg02IntakeError("bootstrap contract drifted")
    if "driver_line_id" not in str(inference.get("independent_cluster", "")):
        raise DNg02IntakeError("inference is not driver-line clustered")


def _validate_sealing_and_diagnostic(data: Mapping[str, Any]) -> None:
    diagnostic = _mapping(
        data["source_code_regression_diagnostic"], "source diagnostic"
    )
    if (
        diagnostic.get("status")
        != "blocked_pending_static_code_constant_extraction_receipt"
        or diagnostic.get("promotion_eligible") is not False
        or diagnostic.get("model_selection_eligible") is not False
        or diagnostic.get("blocks_primary_methods_endpoint") is not False
    ):
        raise DNg02IntakeError("source diagnostic could leak into promotion")
    expected_scripts = {
        str(item["name"]): str(item["sha256"])
        for item in EXPECTED_ARTIFACTS.values()
        if item["role"] == "source_analysis_code"
    }
    observed_scripts = {
        str(item.get("file_name")): str(item.get("sha256"))
        for item in (
            _mapping(raw, "diagnostic script")
            for raw in _sequence(
                diagnostic.get("authoritative_scripts"), "diagnostic scripts"
            )
        )
    }
    if observed_scripts != expected_scripts:
        raise DNg02IntakeError("diagnostic script receipts drifted")

    sealed = _mapping(data["sealed_outcome_access"], "sealed outcome access")
    if tuple(sealed.get("outcome_files", ())) != (
        "fig1_3A_dataset.hdf5",
        "fig3B-E_dataset.hdf5",
    ):
        raise DNg02IntakeError("sealed outcome file set drifted")
    if sealed.get("base_protocol") != (
        "data/benchmarks/protocols/sealed-heldout-execution.v1.json"
    ):
        raise DNg02IntakeError("sealed execution protocol drifted")
    blockers = _sequence(data.get("remaining_pre_execution_blockers"), "blockers")
    if not any("activation-onset" in str(item) for item in blockers):
        raise DNg02IntakeError("activation-onset receipt blocker was removed")
    if not any("prerequisite" in str(item) for item in blockers):
        raise DNg02IntakeError("upstream prerequisite blocker was removed")


def validate_protocol_v3(data: Mapping[str, Any]) -> Tuple[SplitLine, ...]:
    """Validate the decision-complete v3 preregistration contract."""

    data = _mapping(data, "DNg02 protocol")
    required_root_fields = (
        "protocol_id",
        "version",
        "supersedes_without_modifying",
        "registered_utc",
        "status",
        "claim_scope",
        "promotion_scope_limits",
        "public_source",
        "protocols",
        "non_outcome_driver_line_metadata",
        "permanent_split",
        "stable_dataset_local_identities",
        "primary_methods_endpoint",
        "missingness_and_exclusions",
        "candidate_prediction_contract",
        "null_and_inference",
        "promotion_metrics",
        "source_code_regression_diagnostic",
        "sealed_outcome_access",
        "remaining_pre_execution_blockers",
    )
    _exact_keys(data, required_root_fields, "DNg02 protocol")
    if (
        data["protocol_id"] != PROTOCOL_ID
        or data["version"] != PROTOCOL_VERSION
        or data["status"] != PROTOCOL_STATUS
        or data["supersedes_without_modifying"]
        != "data/benchmarks/protocols/dng02-driver-line-heldout.v2.json"
    ):
        raise DNg02IntakeError("DNg02 protocol identity or status drifted")
    _validate_protocol_sources(data)
    lines = _validate_protocol_split(data)
    _validate_primary_endpoint(data)
    _validate_sealing_and_diagnostic(data)
    return lines


def load_locked_dng02_intake(
    protocol_path: os.PathLike[str] | str,
    *,
    receipt_path: Optional[os.PathLike[str] | str] = None,
) -> ValidatedDNg02Intake:
    """Read and validate only JSON metadata; referenced HDF5 is never opened."""

    protocol_file, protocol_raw = _regular_json_bytes(protocol_path)
    protocol_data = _parse_json(protocol_raw, "DNg02 protocol")
    lines = validate_protocol_v3(protocol_data)

    if receipt_path is None:
        try:
            repository_root = protocol_file.parents[3]
        except IndexError as exc:
            raise DNg02IntakeError("cannot resolve repository-relative receipt") from exc
        repository_candidate = repository_root / EXPECTED_RECEIPT_RELATIVE_PATH
        installed_candidate = (
            protocol_file.parent.parent
            / "receipts"
            / "dng02-mendeley-public-artifacts.v1.json"
        )
        if repository_candidate.is_file():
            receipt_path = repository_candidate
        elif installed_candidate.is_file():
            receipt_path = installed_candidate
        else:
            raise DNg02IntakeError(
                "cannot resolve the content-addressed artifact receipt"
            )
    receipt_file, receipt_raw = _regular_json_bytes(receipt_path)
    receipt_digest = _file_sha256(receipt_raw)
    declared_digest = _mapping(
        protocol_data["public_source"], "public source"
    ).get("artifact_receipt_file_sha256")
    if receipt_digest != declared_digest or receipt_digest != EXPECTED_RECEIPT_FILE_SHA256:
        raise DNg02IntakeError("artifact receipt file SHA-256 mismatch")
    receipt_data = _parse_json(receipt_raw, "artifact receipt")
    validate_public_artifact_receipt(receipt_data)

    return ValidatedDNg02Intake(
        protocol_path=str(protocol_file),
        protocol_file_sha256=_file_sha256(protocol_raw),
        receipt_path=str(receipt_file),
        receipt_file_sha256=receipt_digest,
        driver_lines=lines,
        artifact_count=len(EXPECTED_ARTIFACTS),
        sealed_outcome_artifact_count=sum(
            bool(item["contains_outcome_values"])
            for item in EXPECTED_ARTIFACTS.values()
        ),
    )
