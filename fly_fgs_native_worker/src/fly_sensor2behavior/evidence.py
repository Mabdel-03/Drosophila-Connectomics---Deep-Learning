"""Evidence-locked connectome graph and seed-data loader.

The graph separates structural observations from physiological parameters.
``structural_synapse_count`` is intentionally named and never exposed as a
model gain or activation weight.  Cross-atlas routes must pass through an
explicit cell-type crosswalk; root/body identifiers can never be directly
joined across FlyWire, BANC/FANC, and MANC.
"""

from __future__ import annotations

import json
import sysconfig
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .schema import (
    SCHEMA_VERSION,
    Confidence,
    CoverageSummary,
    CrossAtlasJoinError,
    DatasetRef,
    EntityKind,
    EntityRef,
    EvidenceTier,
    JsonSchemaMixin,
    Provenance,
    SchemaValidationError,
    _check_schema_version,
    _enum,
    _finite,
    _nonempty,
    require_same_identifier_space,
)


class EdgeRelation(str, Enum):
    SYNAPSES_ONTO = "synapses_onto"
    TYPE_ASSIGNMENT = "type_assignment"
    TYPE_CROSSWALK = "type_crosswalk"
    INNERVATES = "innervates"
    AGGREGATED_VIA_MOTOR_NEURONS = "aggregated_via_motor_neurons"
    FUNCTIONAL_ASSOCIATION = "functional_association"


class FunctionalSign(str, Enum):
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EvidenceEdge(JsonSchemaMixin):
    edge_id: str
    source: EntityRef
    target: EntityRef
    relation: EdgeRelation
    provenance: Provenance
    confidence: Confidence
    structural_synapse_count: Optional[int] = None
    cleft_score_min: Optional[float] = None
    autapses_included: Optional[bool] = None
    functional_sign: FunctionalSign = FunctionalSign.UNKNOWN
    pathway_via: Tuple[EntityKind, ...] = ()
    notes: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.edge_id, "edge_id")
        relation = _enum(EdgeRelation, self.relation, "relation")
        sign = _enum(FunctionalSign, self.functional_sign, "functional_sign")
        via = tuple(_enum(EntityKind, item, "pathway_via") for item in self.pathway_via)
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "functional_sign", sign)
        object.__setattr__(self, "pathway_via", via)
        _check_schema_version(self.schema_version)

        count = self.structural_synapse_count
        if count is not None:
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise SchemaValidationError("structural_synapse_count must be a positive integer")
        if self.cleft_score_min is not None:
            object.__setattr__(
                self, "cleft_score_min", _finite(self.cleft_score_min, "cleft_score_min", minimum=0.0)
            )
        if self.autapses_included is not None and not isinstance(self.autapses_included, bool):
            raise SchemaValidationError("autapses_included must be boolean or null")

        same_space = self.source.dataset.identity_space == self.target.dataset.identity_space
        if not same_space:
            if relation is not EdgeRelation.TYPE_CROSSWALK:
                raise CrossAtlasJoinError(
                    "cross-atlas edges must be explicit type_crosswalk edges, not %s" % relation.value
                )
            if self.source.kind is not EntityKind.CELL_TYPE or self.target.kind is not EntityKind.CELL_TYPE:
                raise CrossAtlasJoinError("cross-atlas edges may join cell types only, never neuron identifiers")
            if count is not None:
                raise SchemaValidationError("type crosswalks cannot carry structural synapse counts")
            if self.confidence.tier not in (EvidenceTier.TYPE_CROSSWALK, EvidenceTier.MODEL_INFERENCE):
                raise SchemaValidationError("cross-atlas matches require type-crosswalk or inference confidence")
        else:
            if relation is EdgeRelation.TYPE_CROSSWALK:
                raise SchemaValidationError("type_crosswalk is reserved for different identifier spaces")

        if relation is EdgeRelation.SYNAPSES_ONTO:
            if count is None:
                raise SchemaValidationError("synapses_onto edges require a structural synapse count")
            if self.target.kind is EntityKind.MUSCLE:
                raise SchemaValidationError("muscles are not connectome synaptic nodes")
        elif relation is EdgeRelation.TYPE_ASSIGNMENT:
            require_same_identifier_space(self.source, self.target)
            if self.target.kind is not EntityKind.CELL_TYPE or count is not None:
                raise SchemaValidationError("type assignments end at a cell type and carry no synapse count")
        elif relation is EdgeRelation.INNERVATES:
            require_same_identifier_space(self.source, self.target)
            if self.source.kind is not EntityKind.MOTOR_NEURON or self.target.kind is not EntityKind.MUSCLE:
                raise SchemaValidationError("innervates edges must be motor_neuron -> muscle")
        elif relation is EdgeRelation.AGGREGATED_VIA_MOTOR_NEURONS:
            require_same_identifier_space(self.source, self.target)
            if self.source.kind is not EntityKind.CELL_TYPE or self.target.kind is not EntityKind.MUSCLE:
                raise SchemaValidationError("aggregated motor routes must be cell_type -> muscle")
            if EntityKind.MOTOR_NEURON not in via:
                raise SchemaValidationError("aggregated motor routes must state pathway_via motor_neuron")
            if count is None:
                raise SchemaValidationError("aggregated motor routes require a structural synapse count")

    @property
    def is_cross_atlas(self) -> bool:
        return self.source.dataset.identity_space != self.target.dataset.identity_space

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        datasets: Optional[Mapping[str, DatasetRef]] = None,
        provenance_records: Optional[Mapping[str, Provenance]] = None,
        confidence_records: Optional[Mapping[str, Confidence]] = None,
    ) -> "EvidenceEdge":
        def entity(payload: Mapping[str, Any]) -> EntityRef:
            if "dataset_key" in payload:
                if datasets is None or payload["dataset_key"] not in datasets:
                    raise SchemaValidationError("unknown dataset_key %r" % payload.get("dataset_key"))
                dataset = datasets[payload["dataset_key"]]
            elif "dataset" in payload:
                dataset = DatasetRef.from_dict(payload["dataset"])
            else:
                raise SchemaValidationError("entity requires dataset_key or dataset")
            return EntityRef(
                dataset=dataset,
                entity_id=payload["entity_id"],
                kind=payload["kind"],
                cell_type=payload.get("cell_type"),
                anatomical_side=payload.get("anatomical_side", "unknown"),
                schema_version=payload.get("schema_version", SCHEMA_VERSION),
            )

        if "provenance_key" in data:
            if provenance_records is None or data["provenance_key"] not in provenance_records:
                raise SchemaValidationError("unknown provenance_key %r" % data.get("provenance_key"))
            provenance = provenance_records[data["provenance_key"]]
        else:
            provenance = Provenance.from_dict(data["provenance"])
        if "confidence_key" in data:
            if confidence_records is None or data["confidence_key"] not in confidence_records:
                raise SchemaValidationError("unknown confidence_key %r" % data.get("confidence_key"))
            confidence = confidence_records[data["confidence_key"]]
        else:
            confidence = Confidence.from_dict(data["confidence"])

        return cls(
            edge_id=data["edge_id"],
            source=entity(data["source"]),
            target=entity(data["target"]),
            relation=data["relation"],
            provenance=provenance,
            confidence=confidence,
            structural_synapse_count=data.get("structural_synapse_count"),
            cleft_score_min=data.get("cleft_score_min"),
            autapses_included=data.get("autapses_included"),
            functional_sign=data.get("functional_sign", FunctionalSign.UNKNOWN.value),
            pathway_via=tuple(data.get("pathway_via", ())),
            notes=data.get("notes"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class EvidenceGraph(JsonSchemaMixin):
    graph_id: str
    generated_at_utc: str
    datasets: Tuple[DatasetRef, ...]
    edges: Tuple[EvidenceEdge, ...]
    coverage: Tuple[CoverageSummary, ...]
    provenance: Provenance
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.graph_id, "graph_id")
        _nonempty(self.generated_at_utc, "generated_at_utc")
        datasets = tuple(self.datasets)
        edges = tuple(self.edges)
        coverage = tuple(self.coverage)
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "coverage", coverage)
        _check_schema_version(self.schema_version)
        identities = [dataset.identity_space for dataset in datasets]
        if len(set(identities)) != len(identities):
            raise SchemaValidationError("dataset identity spaces must be unique")
        edge_ids = [edge.edge_id for edge in edges]
        if len(set(edge_ids)) != len(edge_ids):
            raise SchemaValidationError("evidence edge IDs must be unique")
        declared = set(identities)
        for edge in edges:
            if edge.source.dataset.identity_space not in declared or edge.target.dataset.identity_space not in declared:
                raise SchemaValidationError("all edge datasets must be declared by the evidence graph")

    def outgoing(self, source: EntityRef, relation: Optional[EdgeRelation] = None) -> Tuple[EvidenceEdge, ...]:
        relation_value = None if relation is None else _enum(EdgeRelation, relation, "relation")
        return tuple(
            edge
            for edge in self.edges
            if edge.source.scoped_id == source.scoped_id
            and (relation_value is None or edge.relation is relation_value)
        )

    def edges_for_cell_type(self, cell_type: str) -> Tuple[EvidenceEdge, ...]:
        _nonempty(cell_type, "cell_type")
        return tuple(
            edge
            for edge in self.edges
            if edge.source.cell_type == cell_type
            or edge.target.cell_type == cell_type
            or (edge.source.kind is EntityKind.CELL_TYPE and edge.source.entity_id == cell_type)
            or (edge.target.kind is EntityKind.CELL_TYPE and edge.target.entity_id == cell_type)
        )

    def structural_synapse_total(self, edges: Optional[Iterable[EvidenceEdge]] = None) -> int:
        """Sum anatomical counts only; this value is never a physiological gain."""

        selected = self.edges if edges is None else tuple(edges)
        return sum(edge.structural_synapse_count or 0 for edge in selected)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceGraph":
        dataset_map: Dict[str, DatasetRef] = {}
        dataset_list = []
        for index, payload in enumerate(data.get("datasets", ())):
            dataset = DatasetRef.from_dict(payload)
            key = payload.get("key", "dataset_%d" % index)
            _nonempty(key, "dataset key")
            if key in dataset_map:
                raise SchemaValidationError("duplicate dataset key %r" % key)
            dataset_map[key] = dataset
            dataset_list.append(dataset)

        provenance_records = {
            key: Provenance.from_dict(payload)
            for key, payload in data.get("provenance_records", {}).items()
        }
        confidence_records = {
            key: Confidence.from_dict(payload)
            for key, payload in data.get("confidence_records", {}).items()
        }
        edges = tuple(
            EvidenceEdge.from_dict(payload, dataset_map, provenance_records, confidence_records)
            for payload in data.get("edges", ())
        )
        coverage = tuple(
            _coverage_from_dict(payload, provenance_records, confidence_records)
            for payload in data.get("coverage", ())
        )
        return cls(
            graph_id=data["graph_id"],
            generated_at_utc=data["generated_at_utc"],
            datasets=tuple(dataset_list),
            edges=edges,
            coverage=coverage,
            provenance=Provenance.from_dict(data["provenance"]),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


def _coverage_from_dict(
    data: Mapping[str, Any],
    provenance_records: Optional[Mapping[str, Provenance]] = None,
    confidence_records: Optional[Mapping[str, Confidence]] = None,
) -> CoverageSummary:
    if "provenance_key" in data:
        if provenance_records is None or data["provenance_key"] not in provenance_records:
            raise SchemaValidationError("unknown coverage provenance_key")
        provenance = provenance_records[data["provenance_key"]]
    else:
        provenance = Provenance.from_dict(data["provenance"])
    if "confidence_key" in data:
        if confidence_records is None or data["confidence_key"] not in confidence_records:
            raise SchemaValidationError("unknown coverage confidence_key")
        confidence = confidence_records[data["confidence_key"]]
    else:
        confidence = Confidence.from_dict(data["confidence"])
    return CoverageSummary(
        branch=data["branch"],
        total_structural_synapses=data["total_structural_synapses"],
        resolved_structural_synapses=data["resolved_structural_synapses"],
        unresolved_structural_synapses=data["unresolved_structural_synapses"],
        provenance=provenance,
        confidence=confidence,
        scope=data["scope"],
        schema_version=data.get("schema_version", SCHEMA_VERSION),
    )


def default_seed_graph_path() -> Path:
    repository_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "evidence"
        / "seed_graph.v1.json"
    )
    if repository_path.is_file():
        return repository_path
    installed_path = (
        Path(sysconfig.get_path("data"))
        / "share"
        / "fly-sensor2behavior"
        / "evidence"
        / "seed_graph.v1.json"
    )
    if installed_path.is_file():
        return installed_path
    raise FileNotFoundError(
        "seed_graph.v1.json was not found in the source tree or installed data files"
    )


def load_evidence_graph(path: Path) -> EvidenceGraph:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise SchemaValidationError("evidence graph JSON must contain an object")
    return EvidenceGraph.from_dict(payload)


def load_seed_graph(path: Optional[Path] = None) -> EvidenceGraph:
    return load_evidence_graph(default_seed_graph_path() if path is None else path)


__all__ = [
    "EdgeRelation",
    "FunctionalSign",
    "EvidenceEdge",
    "EvidenceGraph",
    "default_seed_graph_path",
    "load_evidence_graph",
    "load_seed_graph",
]
