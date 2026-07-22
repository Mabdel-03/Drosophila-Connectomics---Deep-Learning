import copy
import json

import pytest

from scripts.extract_banc_fanc_wing_evidence import (
    SOURCE_SPECS,
    _source_receipts,
    render_artifact,
)
from fly_sensor2behavior.banc_fanc_evidence import (
    BancFancEvidenceError,
    EXPECTED_SOURCE_RECEIPTS,
    REGISTERED_ARTIFACT_SHA256,
    default_banc_fanc_evidence_path,
    load_banc_fanc_evidence,
    sha256_file,
    validate_banc_fanc_evidence,
)
from fly_sensor2behavior.evaluators import evaluate_banc_fanc_wing_pathway
from fly_sensor2behavior.validation import load_benchmark_registry


def _write_fixture(tmp_path, data):
    path = tmp_path / "banc-fanc-evidence.json"
    path.write_text(render_artifact(data), encoding="utf-8")
    return path


def test_registered_banc_fanc_fixture_is_content_addressed_and_structurally_valid():
    path = default_banc_fanc_evidence_path()
    assert sha256_file(path) == REGISTERED_ARTIFACT_SHA256

    metrics = validate_banc_fanc_evidence(path)
    assert metrics.as_dict() == {
        "source_receipt_mismatch_count": 0,
        "direct_cross_atlas_join_count": 0,
        "premotor_candidate_promotion_violation_count": 0,
        "banc_nod1_to_dnp26_edge_count": 4,
        "banc_nod1_to_dnp26_structural_synapse_count": 487,
        "banc_wing_motor_neuron_count": 62,
        "banc_unresolved_wing_motor_neuron_count": 4,
        "banc_dnp26_to_wing_mn_edge_count": 18,
        "banc_dnp26_to_wing_mn_structural_synapse_count": 129,
        "fanc_premotor_row_count": 1784,
        "fanc_nonzero_pair_count": 7289,
        "fanc_structural_synapse_count": 144668,
        "fanc_row_total_mismatch_count": 0,
        "fanc_dnp26_to_wing_mn_structural_synapse_count": 153,
    }


def test_runtime_steering_structural_masks_match_registered_banc_aggregates():
    case = load_benchmark_registry().case("evidence.banc_fanc_wing_pathway")
    result = evaluate_banc_fanc_wing_pathway(case)
    observed = {value.metric_id: value.value for value in result.values}
    assert observed["runtime_structural_mask_mismatch_count"] == 0.0


def test_fixture_is_canonical_and_source_receipts_match_strict_extractor_constants():
    path = default_banc_fanc_evidence_path()
    data = load_banc_fanc_evidence(path)
    assert path.read_text(encoding="utf-8") == render_artifact(data)

    fixture_receipts = {
        row["source_id"]: (row["bytes"], row["sha256"])
        for row in data["source_artifacts"]
    }
    extractor_receipts = {
        source_id: (spec["bytes"], spec["sha256"])
        for source_id, spec in SOURCE_SPECS.items()
    }
    assert fixture_receipts == EXPECTED_SOURCE_RECEIPTS == extractor_receipts


def test_every_connectome_identifier_is_a_string_and_atlas_spaces_are_not_joined():
    data = load_banc_fanc_evidence()

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"root_id", "banc_v888_id", "nucleus_id", "cell_id"}:
                    assert item is None or (isinstance(item, str) and item.isdecimal())
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(data)
    cross = data["cross_atlas_identity"]
    assert cross["premotor_matching_performed"] is False
    assert cross["premotor_identity_status"] == "incomplete"
    assert cross["matched_premotor_pairs"] == []
    assert cross["candidate_premotor_pairs_promoted_to_matches"] == []
    assert cross["direct_individual_neuron_id_joins"] == []
    assert "remain candidates" in cross["policy"]


def test_registered_digest_and_internal_cross_atlas_guards_fail_closed(tmp_path):
    original = load_banc_fanc_evidence()
    tampered = copy.deepcopy(original)
    tampered["cross_atlas_identity"]["candidate_premotor_pairs_promoted_to_matches"] = [
        {
            "banc_root_id": "720575941000000001",
            "fanc_root_id": "648518346000000001",
        }
    ]
    path = _write_fixture(tmp_path, tampered)

    with pytest.raises(BancFancEvidenceError, match="registered artifact SHA-256"):
        validate_banc_fanc_evidence(path)
    with pytest.raises(BancFancEvidenceError, match="promoted premotor candidates"):
        validate_banc_fanc_evidence(path, verify_registered_digest=False)


def test_source_receipt_and_structural_summary_tampering_are_rejected(tmp_path):
    original = load_banc_fanc_evidence()
    bad_receipt = copy.deepcopy(original)
    bad_receipt["source_artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(BancFancEvidenceError, match="source receipt mismatch count"):
        validate_banc_fanc_evidence(
            _write_fixture(tmp_path, bad_receipt), verify_registered_digest=False
        )

    bad_summary = copy.deepcopy(original)
    bad_summary["banc_v888"]["dnp26_to_wing_motor_neuron_summary"][
        "total_structural_synapse_count"
    ] = 131
    with pytest.raises(BancFancEvidenceError, match="BANC DNp26 synapse summary"):
        validate_banc_fanc_evidence(
            _write_fixture(tmp_path, bad_summary), verify_registered_digest=False
        )


def test_extractor_refuses_source_drift_without_an_override(tmp_path):
    drifted = tmp_path / "banc_888_edgelist_simple_v2.feather"
    drifted.write_bytes(b"not the registered source")
    paths = {source_id: drifted for source_id in SOURCE_SPECS}
    with pytest.raises(ValueError, match="source drift"):
        _source_receipts(paths)


def test_fixture_has_no_physiological_weight_field():
    data = load_banc_fanc_evidence()
    serialized = json.dumps(data, sort_keys=True)
    assert '"physiological_weight"' not in serialized
    assert data["scientific_scope"]["physiological_weight_interpretation"] == "forbidden"
