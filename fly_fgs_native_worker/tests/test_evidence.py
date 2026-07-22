from dataclasses import fields

import pytest

from fly_sensor2behavior.evidence import (
    EdgeRelation,
    EvidenceEdge,
    EvidenceGraph,
    FunctionalSign,
    load_seed_graph,
)
from fly_sensor2behavior.schema import (
    AnatomicalSide,
    Confidence,
    ConfidenceLevel,
    CrossAtlasJoinError,
    DatasetRef,
    EntityKind,
    EntityRef,
    EvidenceTier,
    Provenance,
    SchemaValidationError,
)


def datasets():
    fafb = DatasetRef("flywire_fafb", "FAFB", materialization=783, coordinate_units="m")
    manc = DatasetRef("manc", "male-cns:v1.0", coordinate_units="m")
    return fafb, manc


def provenance():
    return Provenance(source_uri="urn:test:evidence", method="fixture")


def confidence(tier=EvidenceTier.CURATED_CONNECTOME):
    return Confidence(tier=tier, level=ConfidenceLevel.HIGH, score=0.9, basis="fixture")


def test_seed_graph_loads_with_locked_datasets_and_expected_totals():
    graph = load_seed_graph()
    assert graph.graph_id == "fly-fgs-flight-evidence-seed-v1"
    assert {dataset.identity_space for dataset in graph.datasets} == {
        "flywire_fafb:FAFB@783",
        "manc:male-cns:v1.0",
    }
    assert len(graph.edges) == 48
    assert graph.structural_synapse_total() == 7736

    structural = [edge for edge in graph.edges if edge.structural_synapse_count is not None]
    assert sum(edge.structural_synapse_count for edge in structural) == 7736
    assert all("weight" not in field.name for field in fields(EvidenceEdge))


def test_seed_graph_never_directly_joins_cross_atlas_ids():
    graph = load_seed_graph()
    cross_atlas = [edge for edge in graph.edges if edge.is_cross_atlas]
    assert len(cross_atlas) == 8
    assert all(edge.relation is EdgeRelation.TYPE_CROSSWALK for edge in cross_atlas)
    assert all(edge.source.kind is EntityKind.CELL_TYPE for edge in cross_atlas)
    assert all(edge.target.kind is EntityKind.CELL_TYPE for edge in cross_atlas)
    assert all(edge.structural_synapse_count is None for edge in cross_atlas)


def test_seed_dng02_and_priority_dn_muscle_summaries_match_source_artifact():
    graph = load_seed_graph()

    def muscle_total(cell_type):
        return sum(
            edge.structural_synapse_count or 0
            for edge in graph.edges_for_cell_type(cell_type)
            if edge.relation is EdgeRelation.AGGREGATED_VIA_MOTOR_NEURONS
        )

    assert muscle_total("DNg02") == 4304
    assert muscle_total("DNa04") == 819
    assert muscle_total("DNp26") == 494
    assert muscle_total("DNg32") == 288

    dng02_tp2 = next(edge for edge in graph.edges if edge.edge_id == "manc-DNg02-tp2")
    assert dng02_tp2.structural_synapse_count == 1375
    assert dng02_tp2.pathway_via == (EntityKind.MOTOR_NEURON,)
    assert dng02_tp2.confidence.tier is EvidenceTier.CURATED_CONNECTOME


def test_seed_coverage_exposes_scope_and_provisional_uncertainty():
    graph = load_seed_graph()
    coverage = {record.branch: record for record in graph.coverage}
    assert coverage["LLPC1_display_selected_targets"].resolved_fraction == 1.0
    assert "not the complete" in coverage["LLPC1_display_selected_targets"].scope
    assert coverage["LLPC1_full_pathway_provisional"].resolved_fraction == pytest.approx(618 / 2174)
    assert coverage["NOD1_full_pathway_provisional"].resolved_fraction == pytest.approx(545 / 712)
    assert coverage["LLPC1_full_pathway_provisional"].confidence.level is ConfidenceLevel.LOW
    assert coverage["LLPC1_full_pathway_provisional"].confidence.tier is EvidenceTier.MODEL_INFERENCE


def test_direct_cross_atlas_neuron_edge_is_rejected_even_with_type_relation():
    fafb, manc = datasets()
    with pytest.raises(CrossAtlasJoinError, match="cell types only"):
        EvidenceEdge(
            edge_id="bad-root-body-join",
            source=EntityRef(fafb, "720575940627502338", EntityKind.NEURON),
            target=EntityRef(manc, "10360", EntityKind.NEURON),
            relation=EdgeRelation.TYPE_CROSSWALK,
            provenance=provenance(),
            confidence=confidence(EvidenceTier.TYPE_CROSSWALK),
        )


def test_cross_atlas_connectivity_edge_is_rejected_and_type_crosswalk_is_allowed():
    fafb, manc = datasets()
    source_type = EntityRef(fafb, "DNp26", EntityKind.CELL_TYPE, cell_type="DNp26")
    target_type = EntityRef(manc, "DNp26", EntityKind.CELL_TYPE, cell_type="DNp26")
    with pytest.raises(CrossAtlasJoinError, match="explicit type_crosswalk"):
        EvidenceEdge(
            edge_id="bad-connectivity",
            source=source_type,
            target=target_type,
            relation=EdgeRelation.SYNAPSES_ONTO,
            structural_synapse_count=10,
            provenance=provenance(),
            confidence=confidence(),
        )

    edge = EvidenceEdge(
        edge_id="good-crosswalk",
        source=source_type,
        target=target_type,
        relation=EdgeRelation.TYPE_CROSSWALK,
        provenance=provenance(),
        confidence=confidence(EvidenceTier.TYPE_CROSSWALK),
    )
    assert edge.is_cross_atlas


def test_muscles_cannot_be_connectome_synaptic_nodes():
    _, manc = datasets()
    with pytest.raises(SchemaValidationError, match="not connectome synaptic nodes"):
        EvidenceEdge(
            edge_id="bad-muscle-synapse",
            source=EntityRef(manc, "DNa04", EntityKind.CELL_TYPE, cell_type="DNa04"),
            target=EntityRef(manc, "b3", EntityKind.MUSCLE),
            relation=EdgeRelation.SYNAPSES_ONTO,
            structural_synapse_count=167,
            provenance=provenance(),
            confidence=confidence(),
        )


def test_aggregated_dn_to_muscle_edge_must_disclose_motor_neuron_intermediate():
    _, manc = datasets()
    source = EntityRef(manc, "DNa04", EntityKind.CELL_TYPE, cell_type="DNa04")
    target = EntityRef(manc, "b3", EntityKind.MUSCLE)
    with pytest.raises(SchemaValidationError, match="pathway_via motor_neuron"):
        EvidenceEdge(
            edge_id="hidden-intermediate",
            source=source,
            target=target,
            relation=EdgeRelation.AGGREGATED_VIA_MOTOR_NEURONS,
            structural_synapse_count=167,
            provenance=provenance(),
            confidence=confidence(),
        )


def test_evidence_graph_rejects_duplicate_edge_ids_and_undeclared_datasets():
    fafb, manc = datasets()
    edge = EvidenceEdge(
        edge_id="crosswalk",
        source=EntityRef(fafb, "DNp26", EntityKind.CELL_TYPE),
        target=EntityRef(manc, "DNp26", EntityKind.CELL_TYPE),
        relation="type_crosswalk",
        provenance=provenance(),
        confidence=confidence(EvidenceTier.TYPE_CROSSWALK),
    )
    with pytest.raises(SchemaValidationError, match="unique"):
        EvidenceGraph(
            graph_id="duplicate",
            generated_at_utc="2026-07-17T00:00:00Z",
            datasets=(fafb, manc),
            edges=(edge, edge),
            coverage=(),
            provenance=provenance(),
        )
    with pytest.raises(SchemaValidationError, match="declared"):
        EvidenceGraph(
            graph_id="missing-dataset",
            generated_at_utc="2026-07-17T00:00:00Z",
            datasets=(fafb,),
            edges=(edge,),
            coverage=(),
            provenance=provenance(),
        )


def test_seed_provenance_records_hashes_and_structural_not_physiological_semantics():
    graph = load_seed_graph()
    flywire_edge = next(edge for edge in graph.edges if edge.edge_id == "fafb-nod1-DNp26")
    assert flywire_edge.structural_synapse_count == 289
    assert flywire_edge.provenance.filters["source_sha256"] == (
        "f19434ee872984035fafbfb492538609cf458111ddb64ad9f998e6c371187932"
    )
    assert "not physiological weights" in flywire_edge.provenance.notes
    assert flywire_edge.functional_sign is FunctionalSign.UNKNOWN
