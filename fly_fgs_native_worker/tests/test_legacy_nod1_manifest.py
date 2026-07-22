import json
from pathlib import Path


def _manifest():
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "manifests"
        / "legacy_nod1_v0.5.0.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_legacy_nod1_snapshot_is_frozen_and_atlas_qualified():
    manifest = _manifest()
    assert manifest["snapshot_id"] == "legacy-nod1-v0.5.0"
    assert manifest["status"] == "frozen_reference"
    assert manifest["dataset"]["family"] == "flywire_fafb"
    assert manifest["dataset"]["materialization"] == 783
    assert manifest["dataset"]["identifier_space"] == "root_id"
    assert manifest["dataset"]["cleft_score_threshold"] is None
    assert "higher soma x is fly-left" in manifest["dataset"]["anatomical_side_rule"]


def test_legacy_contact_inventory_exposes_synthetic_fallbacks():
    inventory = _manifest()["circuit_inventory"]
    assert inventory["prepared_real_coordinate_events"] + inventory[
        "prepared_synthetic_fallback_events"
    ] == inventory["prepared_events"]
    assert inventory["synthetic_fraction"] > 0.8
    assert len(_manifest()["cable_neurons"]) == 8


def test_legacy_manifest_uses_string_ids_and_sha256_digests():
    manifest = _manifest()
    assert all(isinstance(item["root_id"], str) for item in manifest["cable_neurons"])
    digests = tuple(manifest["files"].values()) + tuple(
        manifest["morphology_sha256"].values()
    )
    assert all(value.startswith("sha256:") and len(value) == 71 for value in digests)
    forbidden = manifest["compatibility_contract"]["motor_forbidden_fields"]
    assert "steering" in forbidden
