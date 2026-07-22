"""Strict loader for the frozen BANC/FANC wing-pathway evidence fixture.

This module validates a registered structural observation artifact; it does
not infer synaptic physiology or cross-atlas neuron identity.  The helper is
kept independent of the benchmark runner so a benchmark evaluator can consume
its scalar metrics without duplicating scientific integrity checks.
"""

from __future__ import annotations

import hashlib
import json
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


REGISTERED_ARTIFACT_SHA256 = (
    "91763c068c74520d9c9f1aadd0ef80a295bae01cefc95c9091014eb0a0859c21"
)
REGISTERED_ARTIFACT_ID = "banc-fanc-wing-pathway-evidence-v1"
REGISTERED_SCHEMA_VERSION = "1.0.0"

EXPECTED_SOURCE_RECEIPTS: Mapping[str, Tuple[int, str]] = {
    "banc_v888_edgelist_paper_v2": (
        305_250_378,
        "363fdef3813b72a5e45a42f17034cd5a544b654c838929cecf6e1ce5f60625cb",
    ),
    "banc_v888_meta": (
        57_550_610,
        "819bbcff476e52702d6f8d8604ce1f12d1d7b11942281df2f49df2a73a6f15a5",
    ),
    "fanc_dn_crosswalk": (
        156_324,
        "3acb2bb2159e2cc6d5e005f66d0526c13dc50e933780ca4dbfe4460da72aee33",
    ),
    "fanc_v840_premotor_to_wing_mn_matrix": (
        457_291,
        "ce48e91c462ae7ffbea122429e0e0b064064d20431e82dcb8735892b6292e46c",
    ),
    "fanc_v840_wing_mn_properties": (
        2_046,
        "265a9a8f93f6d297a79503e6d0c32e66f4e63801b61307ed3e0f234e48de9740",
    ),
}

EXPECTED_NOD1_ROOT_IDS = frozenset(
    {
        "720575941416849692",
        "720575941461200710",
        "720575941467991830",
        "720575941652138225",
    }
)
EXPECTED_DNP26_ROOT_IDS = frozenset(
    {"720575941392812548", "720575941566309174"}
)


class BancFancEvidenceError(ValueError):
    """Raised when the registered structural fixture is missing or inconsistent."""


@dataclass(frozen=True)
class BancFancEvidenceMetrics:
    """Scalar observations suitable for a frozen structural benchmark gate."""

    source_receipt_mismatch_count: int
    direct_cross_atlas_join_count: int
    premotor_candidate_promotion_violation_count: int
    banc_nod1_to_dnp26_edge_count: int
    banc_nod1_to_dnp26_structural_synapse_count: int
    banc_wing_motor_neuron_count: int
    banc_unresolved_wing_motor_neuron_count: int
    banc_dnp26_to_wing_mn_edge_count: int
    banc_dnp26_to_wing_mn_structural_synapse_count: int
    fanc_premotor_row_count: int
    fanc_nonzero_pair_count: int
    fanc_structural_synapse_count: int
    fanc_row_total_mismatch_count: int
    fanc_dnp26_to_wing_mn_structural_synapse_count: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "source_receipt_mismatch_count": self.source_receipt_mismatch_count,
            "direct_cross_atlas_join_count": self.direct_cross_atlas_join_count,
            "premotor_candidate_promotion_violation_count": (
                self.premotor_candidate_promotion_violation_count
            ),
            "banc_nod1_to_dnp26_edge_count": self.banc_nod1_to_dnp26_edge_count,
            "banc_nod1_to_dnp26_structural_synapse_count": (
                self.banc_nod1_to_dnp26_structural_synapse_count
            ),
            "banc_wing_motor_neuron_count": self.banc_wing_motor_neuron_count,
            "banc_unresolved_wing_motor_neuron_count": (
                self.banc_unresolved_wing_motor_neuron_count
            ),
            "banc_dnp26_to_wing_mn_edge_count": self.banc_dnp26_to_wing_mn_edge_count,
            "banc_dnp26_to_wing_mn_structural_synapse_count": (
                self.banc_dnp26_to_wing_mn_structural_synapse_count
            ),
            "fanc_premotor_row_count": self.fanc_premotor_row_count,
            "fanc_nonzero_pair_count": self.fanc_nonzero_pair_count,
            "fanc_structural_synapse_count": self.fanc_structural_synapse_count,
            "fanc_row_total_mismatch_count": self.fanc_row_total_mismatch_count,
            "fanc_dnp26_to_wing_mn_structural_synapse_count": (
                self.fanc_dnp26_to_wing_mn_structural_synapse_count
            ),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_banc_fanc_evidence_path() -> Path:
    source_tree = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "reference"
        / "banc_fanc_wing_pathway_evidence.v1.json"
    )
    if source_tree.is_file():
        return source_tree
    return (
        Path(sysconfig.get_path("data"))
        / "share"
        / "fly-sensor2behavior"
        / "reference"
        / "banc_fanc_wing_pathway_evidence.v1.json"
    )


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BancFancEvidenceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _expect_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise BancFancEvidenceError(
            f"{label} mismatch: expected {expected!r}, observed {actual!r}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BancFancEvidenceError(f"{label} must be an object")
    return value


def _rows(value: Any, label: str) -> Sequence[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise BancFancEvidenceError(f"{label} must be an array of objects")
    return value


def _positive_structural_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BancFancEvidenceError(f"{label} must be a positive integer")
    return value


def _decimal_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.isdecimal():
        raise BancFancEvidenceError(f"{label} must be a decimal identifier string")
    return value


def _unique_decimal_ids(rows: Iterable[Mapping[str, Any]], key: str, label: str) -> set:
    values = [_decimal_id(row.get(key), f"{label}.{key}") for row in rows]
    if len(set(values)) != len(values):
        raise BancFancEvidenceError(f"{label}.{key} values must be unique")
    return set(values)


def _sum_edges(rows: Iterable[Mapping[str, Any]], label: str) -> int:
    return sum(
        _positive_structural_count(row.get("structural_synapse_count"), label)
        for row in rows
    )


def load_banc_fanc_evidence(
    path: Path = None, *, verify_registered_digest: bool = True
) -> Mapping[str, Any]:
    fixture = default_banc_fanc_evidence_path() if path is None else Path(path)
    if not fixture.is_file():
        raise BancFancEvidenceError(f"BANC/FANC evidence fixture is missing: {fixture}")
    if verify_registered_digest:
        _expect_equal(
            sha256_file(fixture),
            REGISTERED_ARTIFACT_SHA256,
            "registered artifact SHA-256",
        )
    try:
        data = json.loads(
            fixture.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except json.JSONDecodeError as exc:
        raise BancFancEvidenceError(f"invalid BANC/FANC evidence JSON: {exc}") from exc
    return _mapping(data, "fixture")


def validate_banc_fanc_evidence(
    path: Path = None, *, verify_registered_digest: bool = True
) -> BancFancEvidenceMetrics:
    """Validate provenance, identifier separation, and registered topology counts."""

    data = load_banc_fanc_evidence(
        path, verify_registered_digest=verify_registered_digest
    )
    _expect_equal(data.get("artifact_id"), REGISTERED_ARTIFACT_ID, "artifact_id")
    _expect_equal(data.get("schema_version"), REGISTERED_SCHEMA_VERSION, "schema_version")

    scope = _mapping(data.get("scientific_scope"), "scientific_scope")
    _expect_equal(
        scope.get("claim_level"),
        "structural_connectome_evidence",
        "scientific claim level",
    )
    _expect_equal(
        scope.get("physiological_weight_interpretation"),
        "forbidden",
        "physiological weight interpretation",
    )
    _expect_equal(scope.get("functional_sign"), "unknown", "functional sign")
    _expect_equal(
        scope.get("autapse_policy"),
        "excluded from selected pathway edges",
        "autapse policy",
    )

    receipts = _rows(data.get("source_artifacts"), "source_artifacts")
    observed_receipts = {row.get("source_id"): row for row in receipts}
    mismatch_count = 0
    if set(observed_receipts) != set(EXPECTED_SOURCE_RECEIPTS):
        mismatch_count += len(set(observed_receipts) ^ set(EXPECTED_SOURCE_RECEIPTS))
    for source_id, (expected_bytes, expected_sha) in EXPECTED_SOURCE_RECEIPTS.items():
        row = observed_receipts.get(source_id, {})
        if row.get("bytes") != expected_bytes:
            mismatch_count += 1
        if row.get("sha256") != expected_sha:
            mismatch_count += 1
        if not isinstance(row.get("source_uri"), str) or not row.get("source_uri"):
            mismatch_count += 1
    _expect_equal(mismatch_count, 0, "source receipt mismatch count")

    identity_spaces = _rows(data.get("identity_spaces"), "identity_spaces")
    _expect_equal(
        {row.get("identity_space") for row in identity_spaces},
        {"banc:banc_888", "fanc:fanc_production_mar2021@840"},
        "identity spaces",
    )

    banc = _mapping(data.get("banc_v888"), "banc_v888")
    selected = _rows(banc.get("selected_pathway_neurons"), "selected_pathway_neurons")
    selected_ids = _unique_decimal_ids(selected, "root_id", "selected_pathway_neurons")
    _expect_equal(
        selected_ids,
        set(EXPECTED_NOD1_ROOT_IDS | EXPECTED_DNP26_ROOT_IDS),
        "selected BANC pathway roots",
    )
    selected_types = {row["root_id"]: row.get("cell_type") for row in selected}
    _expect_equal(
        {root for root, kind in selected_types.items() if kind == "Nod1"},
        set(EXPECTED_NOD1_ROOT_IDS),
        "BANC Nod1 roots",
    )
    _expect_equal(
        {root for root, kind in selected_types.items() if kind == "DNp26"},
        set(EXPECTED_DNP26_ROOT_IDS),
        "BANC DNp26 roots",
    )

    nod_edges = _rows(banc.get("nod1_to_dnp26_edges"), "nod1_to_dnp26_edges")
    for row in nod_edges:
        if _decimal_id(row.get("pre_root_id"), "Nod1 edge pre") not in EXPECTED_NOD1_ROOT_IDS:
            raise BancFancEvidenceError("Nod1 edge has an unexpected presynaptic root")
        if _decimal_id(row.get("post_root_id"), "Nod1 edge post") not in EXPECTED_DNP26_ROOT_IDS:
            raise BancFancEvidenceError("Nod1 edge has an unexpected postsynaptic root")
        if row.get("pre_root_id") == row.get("post_root_id"):
            raise BancFancEvidenceError("selected BANC pathway contains an autapse")
    nod_total = _sum_edges(nod_edges, "Nod1-to-DNp26 count")
    nod_summary = _mapping(banc.get("nod1_to_dnp26_summary"), "nod1 summary")
    _expect_equal(nod_summary.get("edge_count"), len(nod_edges), "Nod1 edge summary")
    _expect_equal(
        nod_summary.get("total_structural_synapse_count"), nod_total, "Nod1 synapse summary"
    )
    _expect_equal(len(nod_edges), 4, "registered Nod1-to-DNp26 edge count")
    _expect_equal(nod_total, 487, "registered Nod1-to-DNp26 structural count")

    wing_mns = _rows(banc.get("wing_motor_neurons"), "BANC wing_motor_neurons")
    wing_ids = _unique_decimal_ids(wing_mns, "root_id", "BANC wing_motor_neurons")
    _expect_equal(len(wing_mns), 62, "BANC wing-MN row count")
    if not all(row.get("proofread") is True for row in wing_mns):
        raise BancFancEvidenceError("all registered BANC wing MNs must be proofread")
    unresolved = [
        row for row in wing_mns if row.get("project_named_atlas_status") == "unresolved"
    ]
    _expect_equal(
        {row.get("cell_type") for row in unresolved},
        {"MNxm01", "PSn_u"},
        "unresolved BANC wing-MN types",
    )
    _expect_equal(len(unresolved), 4, "unresolved BANC wing-MN row count")

    dn_wing_edges = _rows(
        banc.get("dnp26_to_wing_motor_neuron_edges"), "BANC DNp26-to-wing-MN edges"
    )
    for row in dn_wing_edges:
        if _decimal_id(row.get("pre_root_id"), "DNp26 edge pre") not in EXPECTED_DNP26_ROOT_IDS:
            raise BancFancEvidenceError("DNp26 edge has an unexpected presynaptic root")
        if _decimal_id(row.get("post_root_id"), "DNp26 edge post") not in wing_ids:
            raise BancFancEvidenceError("DNp26 edge target is not a registered BANC wing MN")
        if row.get("pre_root_id") == row.get("post_root_id"):
            raise BancFancEvidenceError("selected BANC pathway contains an autapse")
    dn_wing_total = _sum_edges(dn_wing_edges, "DNp26-to-wing-MN count")
    dn_wing_summary = _mapping(
        banc.get("dnp26_to_wing_motor_neuron_summary"), "BANC DNp26 wing summary"
    )
    _expect_equal(
        dn_wing_summary.get("edge_count"), len(dn_wing_edges), "BANC DNp26 edge summary"
    )
    _expect_equal(
        dn_wing_summary.get("total_structural_synapse_count"),
        dn_wing_total,
        "BANC DNp26 synapse summary",
    )
    _expect_equal(len(dn_wing_edges), 18, "registered BANC DNp26 wing edge count")
    _expect_equal(dn_wing_total, 129, "registered BANC DNp26 wing structural count")

    fanc = _mapping(data.get("fanc_v840"), "fanc_v840")
    _expect_equal(fanc.get("materialization"), 840, "FANC materialization")
    fanc_mns = _rows(fanc.get("wing_motor_neurons"), "FANC wing_motor_neurons")
    _unique_decimal_ids(fanc_mns, "root_id", "FANC wing_motor_neurons")
    _expect_equal(len(fanc_mns), 29, "FANC wing-MN row count")
    matrix = _mapping(
        fanc.get("published_wing_premotor_matrix_summary"), "FANC matrix summary"
    )
    expected_matrix_values = {
        "premotor_rows": 1784,
        "motor_columns": 29,
        "nonzero_pairs": 7289,
        "thresholded_structural_synapse_count": 144668,
        "minimum_positive_pair_count": 3,
        "maximum_pair_count": 416,
        "row_total_mismatch_count": 0,
        "matrix_only_motor_labels": [],
        "properties_only_motor_labels": [],
    }
    for key, expected in expected_matrix_values.items():
        _expect_equal(matrix.get(key), expected, f"FANC matrix {key}")

    fanc_dnp26 = _rows(fanc.get("dnp26_rows"), "FANC DNp26 rows")
    _unique_decimal_ids(fanc_dnp26, "root_id", "FANC DNp26 rows")
    _expect_equal(len(fanc_dnp26), 2, "FANC DNp26 row count")
    fanc_dnp26_total = 0
    for row in fanc_dnp26:
        if row.get("cell_type") != "DNp26" or row.get("matrix_membership") is not True:
            raise BancFancEvidenceError("FANC DNp26 row lacks its explicit matrix membership")
        edges = _rows(row.get("wing_motor_neuron_edges"), "FANC DNp26 wing-MN edges")
        edge_total = _sum_edges(edges, "FANC DNp26-to-wing-MN count")
        _expect_equal(
            row.get("total_structural_synapse_count"), edge_total, "FANC DNp26 row total"
        )
        fanc_dnp26_total += edge_total
    _expect_equal(fanc_dnp26_total, 153, "registered FANC DNp26 wing structural count")

    cross = _mapping(data.get("cross_atlas_identity"), "cross_atlas_identity")
    _expect_equal(cross.get("premotor_matching_performed"), False, "premotor matching flag")
    _expect_equal(cross.get("premotor_identity_status"), "incomplete", "premotor identity status")
    direct_joins = cross.get("direct_individual_neuron_id_joins")
    promoted = cross.get("candidate_premotor_pairs_promoted_to_matches")
    matched = cross.get("matched_premotor_pairs")
    _expect_equal(direct_joins, [], "direct individual-neuron joins")
    _expect_equal(promoted, [], "promoted premotor candidates")
    _expect_equal(matched, [], "matched premotor pairs")

    return BancFancEvidenceMetrics(
        source_receipt_mismatch_count=mismatch_count,
        direct_cross_atlas_join_count=len(direct_joins),
        premotor_candidate_promotion_violation_count=len(promoted),
        banc_nod1_to_dnp26_edge_count=len(nod_edges),
        banc_nod1_to_dnp26_structural_synapse_count=nod_total,
        banc_wing_motor_neuron_count=len(wing_mns),
        banc_unresolved_wing_motor_neuron_count=len(unresolved),
        banc_dnp26_to_wing_mn_edge_count=len(dn_wing_edges),
        banc_dnp26_to_wing_mn_structural_synapse_count=dn_wing_total,
        fanc_premotor_row_count=int(matrix["premotor_rows"]),
        fanc_nonzero_pair_count=int(matrix["nonzero_pairs"]),
        fanc_structural_synapse_count=int(matrix["thresholded_structural_synapse_count"]),
        fanc_row_total_mismatch_count=int(matrix["row_total_mismatch_count"]),
        fanc_dnp26_to_wing_mn_structural_synapse_count=fanc_dnp26_total,
    )


__all__ = [
    "BancFancEvidenceError",
    "BancFancEvidenceMetrics",
    "EXPECTED_SOURCE_RECEIPTS",
    "REGISTERED_ARTIFACT_ID",
    "REGISTERED_ARTIFACT_SHA256",
    "default_banc_fanc_evidence_path",
    "load_banc_fanc_evidence",
    "sha256_file",
    "validate_banc_fanc_evidence",
]
