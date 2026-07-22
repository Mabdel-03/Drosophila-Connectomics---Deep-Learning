#!/usr/bin/env python3
"""Build the frozen BANC v888 / FANC v840 wing-pathway evidence fixture.

This is an offline extraction tool.  It reads the public, content-addressed
source snapshots listed in ``SOURCE_SPECS`` and refuses to run if any source
digest differs.  The emitted JSON is byte-deterministic and keeps every
connectome identifier as a decimal string.

The fixture supports structural-integrity evaluation only.  Synapse counts
are anatomical observations, not physiological weights.  In particular, the
script does not match BANC and FANC premotor neuron identities: the two data
sets remain independent identity spaces and the output records that cross-
atlas premotor matching is incomplete.

Offline dependencies (not simulator runtime dependencies): pandas, numpy,
pyarrow, and a pandas version capable of reading the two published pickle
files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


ARTIFACT_ID = "banc-fanc-wing-pathway-evidence-v1"
SCHEMA_VERSION = "1.0.0"
GENERATOR_VERSION = "1.0.0"
SOURCE_SNAPSHOT_DATE = "2026-07-18"

NOD1_ROOT_IDS = frozenset(
    {
        "720575941461200710",
        "720575941416849692",
        "720575941652138225",
        "720575941467991830",
    }
)
DNP26_ROOT_IDS = frozenset(
    {
        "720575941566309174",
        "720575941392812548",
    }
)
UNRESOLVED_WING_MN_TYPES = frozenset({"PSn_u", "MNxm01"})


# These are the exact bytes used for the registered v1 artifact.  Updating a
# source requires a new artifact version rather than a drift override.
SOURCE_SPECS: Mapping[str, Mapping[str, Any]] = {
    "banc_v888_meta": {
        "dataset_identity_space": "banc:banc_888",
        "release": "BANC v888",
        "logical_name": "banc_888_meta.feather",
        "source_uri": (
            "gs://lee-lab_brain-and-nerve-cord-fly-connectome/"
            "compiled_data/banc_888/banc_888_meta.feather"
        ),
        "bytes": 57_550_610,
        "sha256": "819bbcff476e52702d6f8d8604ce1f12d1d7b11942281df2f49df2a73a6f15a5",
    },
    "banc_v888_edgelist_paper_v2": {
        "dataset_identity_space": "banc:banc_888",
        "release": "BANC v888 synapses v2 (paper version)",
        "logical_name": "banc_888_edgelist_simple_v2.feather",
        "source_uri": (
            "gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/"
            "banc_888/banc_888_edgelist_simple_v2.feather"
        ),
        "bytes": 305_250_378,
        "sha256": "363fdef3813b72a5e45a42f17034cd5a544b654c838929cecf6e1ce5f60625cb",
    },
    "fanc_v840_premotor_to_wing_mn_matrix": {
        "dataset_identity_space": "fanc:fanc_production_mar2021@840",
        "release": "FANC materialization 840",
        "logical_name": "pkls/preMN_to_MN_wing_v840.pkl",
        "source_uri": (
            "https://github.com/tuthill-lab/Lesser_Azevedo_2023/blob/"
            "93cafa55b8bbdb1493e8d73c941035969349b223/"
            "pkls/preMN_to_MN_wing_v840.pkl"
        ),
        "repository_commit": "93cafa55b8bbdb1493e8d73c941035969349b223",
        "bytes": 457_291,
        "sha256": "ce48e91c462ae7ffbea122429e0e0b064064d20431e82dcb8735892b6292e46c",
    },
    "fanc_v840_wing_mn_properties": {
        "dataset_identity_space": "fanc:fanc_production_mar2021@840",
        "release": "FANC materialization 840",
        "logical_name": "pkls/mn_properties_wing_v840.pkl",
        "source_uri": (
            "https://github.com/tuthill-lab/Lesser_Azevedo_2023/blob/"
            "93cafa55b8bbdb1493e8d73c941035969349b223/"
            "pkls/mn_properties_wing_v840.pkl"
        ),
        "repository_commit": "93cafa55b8bbdb1493e8d73c941035969349b223",
        "bytes": 2_046,
        "sha256": "265a9a8f93f6d297a79503e6d0c32e66f4e63801b61307ed3e0f234e48de9740",
    },
    "fanc_dn_crosswalk": {
        "dataset_identity_space": "fanc:fanc_production_mar2021",
        "release": "2023neckconnective supplemental annotations",
        "logical_name": "Supplemental_files/Supplemental_file6_FANC_DNs.tsv",
        "source_uri": (
            "https://github.com/flyconnectome/2023neckconnective/blob/"
            "94637d7d82920234ef0bfdbf383c000e74a8b45a/"
            "Supplemental_files/Supplemental_file6_FANC_DNs.tsv"
        ),
        "repository_commit": "94637d7d82920234ef0bfdbf383c000e74a8b45a",
        "bytes": 156_324,
        "sha256": "3acb2bb2159e2cc6d5e005f66d0526c13dc50e933780ca4dbfe4460da72aee33",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_missing(value: Any) -> bool:
    # Imported lazily so importing constants from this script does not make
    # pandas/NumPy runtime dependencies of the simulator or its tests.
    import pandas as pd

    result = pd.isna(value)
    return bool(result) if not hasattr(result, "__len__") else False


def _required_id(value: Any, label: str) -> str:
    if _is_missing(value):
        raise ValueError(f"{label} is missing")
    text = str(value)
    if not text.isdecimal():
        raise ValueError(f"{label} must be a decimal identifier string, got {text!r}")
    return text


def _optional_id(value: Any) -> Any:
    return None if _is_missing(value) else _required_id(value, "optional identifier")


def _optional_text(value: Any) -> Any:
    return None if _is_missing(value) else str(value)


def _status_flags(value: Any) -> Sequence[str]:
    if _is_missing(value):
        return []
    return sorted({part.strip() for part in str(value).split(",") if part.strip()})


def _source_receipts(paths: Mapping[str, Path]) -> Sequence[Dict[str, Any]]:
    receipts = []
    for source_id in sorted(SOURCE_SPECS):
        spec = SOURCE_SPECS[source_id]
        path = paths[source_id]
        if not path.is_file():
            raise FileNotFoundError(f"missing {source_id} source: {path}")
        byte_count = path.stat().st_size
        digest = sha256_file(path)
        if byte_count != spec["bytes"] or digest != spec["sha256"]:
            raise ValueError(
                f"{source_id} source drift: expected {spec['bytes']} bytes / "
                f"{spec['sha256']}, observed {byte_count} bytes / {digest}"
            )
        receipt = {"source_id": source_id}
        receipt.update(spec)
        receipts.append(receipt)
    return receipts


def _counts_by(rows: Iterable[Mapping[str, Any]], key: str) -> Sequence[Dict[str, Any]]:
    totals: Dict[str, int] = {}
    for row in rows:
        label = str(row[key])
        totals[label] = totals.get(label, 0) + int(row["structural_synapse_count"])
    return [
        {key: label, "structural_synapse_count": count}
        for label, count in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    ]


def build_artifact(paths: Mapping[str, Path]) -> Dict[str, Any]:
    """Extract and normalize the registered source snapshots."""

    import pandas as pd

    receipts = _source_receipts(paths)

    meta_columns = [
        "banc_888_id",
        "root_id",
        "nucleus_id",
        "proofread",
        "side",
        "cell_class",
        "cell_sub_class",
        "cell_type",
        "body_part_effector",
        "peripheral_target_type",
        "status",
    ]
    meta = pd.read_feather(paths["banc_v888_meta"], columns=meta_columns)
    edges = pd.read_feather(
        paths["banc_v888_edgelist_paper_v2"],
        columns=["pre", "post", "count", "norm", "post_count", "pre_count"],
    )
    for column in ("banc_888_id", "root_id", "nucleus_id"):
        meta[column] = meta[column].astype("string")
    for column in ("pre", "post"):
        edges[column] = edges[column].astype("string")

    selected = meta[meta["root_id"].isin(NOD1_ROOT_IDS | DNP26_ROOT_IDS)].copy()
    selected_rows = []
    for row in selected.to_dict(orient="records"):
        selected_rows.append(
            {
                "root_id": _required_id(row["root_id"], "BANC root_id"),
                "banc_v888_id": _required_id(row["banc_888_id"], "BANC v888 ID"),
                "nucleus_id": _optional_id(row["nucleus_id"]),
                "cell_type": str(row["cell_type"]),
                "dataset_side": str(row["side"]),
                "proofread": str(row["proofread"]).upper() == "TRUE",
                "status_flags": _status_flags(row["status"]),
            }
        )
    selected_rows.sort(
        key=lambda row: (row["cell_type"].lower(), row["dataset_side"], row["root_id"])
    )
    selected_lookup = {row["root_id"]: row for row in selected_rows}

    nod1_dnp26 = edges[
        edges["pre"].isin(NOD1_ROOT_IDS)
        & edges["post"].isin(DNP26_ROOT_IDS)
        & edges["pre"].ne(edges["post"])
    ].copy()
    nod1_rows = []
    for row in nod1_dnp26.to_dict(orient="records"):
        pre = _required_id(row["pre"], "BANC pre root")
        post = _required_id(row["post"], "BANC post root")
        nod1_rows.append(
            {
                "pre_root_id": pre,
                "pre_cell_type": selected_lookup[pre]["cell_type"],
                "pre_dataset_side": selected_lookup[pre]["dataset_side"],
                "post_root_id": post,
                "post_cell_type": selected_lookup[post]["cell_type"],
                "post_dataset_side": selected_lookup[post]["dataset_side"],
                "structural_synapse_count": int(row["count"]),
            }
        )
    nod1_rows.sort(key=lambda row: (row["pre_root_id"], row["post_root_id"]))

    wing = meta[meta["cell_class"].eq("wing_motor_neuron")].copy()
    wing_rows = []
    for row in wing.to_dict(orient="records"):
        cell_type = str(row["cell_type"])
        wing_rows.append(
            {
                "root_id": _required_id(row["root_id"], "BANC wing-MN root_id"),
                "banc_v888_id": _required_id(
                    row["banc_888_id"], "BANC wing-MN v888 ID"
                ),
                "nucleus_id": _optional_id(row["nucleus_id"]),
                "dataset_side": str(row["side"]),
                "cell_type": cell_type,
                "cell_sub_class": _optional_text(row["cell_sub_class"]),
                "peripheral_target_type": str(row["peripheral_target_type"]),
                "proofread": str(row["proofread"]).upper() == "TRUE",
                "project_named_atlas_status": (
                    "unresolved" if cell_type in UNRESOLVED_WING_MN_TYPES else "accepted"
                ),
                "status_flags": _status_flags(row["status"]),
            }
        )
    wing_rows.sort(
        key=lambda row: (row["dataset_side"], row["cell_type"], row["root_id"])
    )
    wing_lookup = {row["root_id"]: row for row in wing_rows}

    dnp26_wing = edges[
        edges["pre"].isin(DNP26_ROOT_IDS)
        & edges["post"].isin(set(wing_lookup))
        & edges["pre"].ne(edges["post"])
    ].copy()
    dnp26_wing_rows = []
    for row in dnp26_wing.to_dict(orient="records"):
        pre = _required_id(row["pre"], "BANC DNp26 root")
        post = _required_id(row["post"], "BANC wing-MN root")
        target = wing_lookup[post]
        dnp26_wing_rows.append(
            {
                "pre_root_id": pre,
                "pre_cell_type": "DNp26",
                "pre_dataset_side": selected_lookup[pre]["dataset_side"],
                "post_root_id": post,
                "post_cell_type": target["cell_type"],
                "post_dataset_side": target["dataset_side"],
                "post_cell_sub_class": target["cell_sub_class"],
                "post_peripheral_target_type": target["peripheral_target_type"],
                "structural_synapse_count": int(row["count"]),
            }
        )
    dnp26_wing_rows.sort(
        key=lambda row: (
            row["pre_root_id"],
            -row["structural_synapse_count"],
            row["post_root_id"],
        )
    )

    fanc = pd.read_pickle(paths["fanc_v840_premotor_to_wing_mn_matrix"])
    fanc_mn = pd.read_pickle(paths["fanc_v840_wing_mn_properties"]).copy()
    fanc_index = fanc.index.to_frame(index=False)
    fanc_matrix = fanc.drop(columns=["MN_syn_total"])
    fanc_index["pre_pt_root_id"] = fanc_index["pre_pt_root_id"].map(str)

    fanc_mn_rows = []
    for row in fanc_mn.to_dict(orient="records"):
        fanc_mn_rows.append(
            {
                "root_id": _required_id(row["pt_root_id"], "FANC wing-MN root_id"),
                "cell_type": str(row["cell_type"]),
                "total_structural_inputs": int(row["total_inputs"]),
            }
        )
    fanc_mn_rows.sort(key=lambda row: (row["cell_type"], row["root_id"]))

    fanc_dns = pd.read_csv(paths["fanc_dn_crosswalk"], sep="\t", dtype=str)
    fanc_dnp26 = fanc_dns[fanc_dns["type"].eq("DNp26")].copy()
    fanc_dnp26_rows = []
    fanc_root_ids = set(fanc_index["pre_pt_root_id"])
    for annotation in fanc_dnp26.to_dict(orient="records"):
        root_id = _required_id(annotation["root_id"], "FANC DNp26 root_id")
        row_record: Dict[str, Any] = {
            "root_id": root_id,
            "cell_id": _required_id(annotation["cell_id"], "FANC DNp26 cell_id"),
            "dataset_side": str(annotation["side"]),
            "cell_type": str(annotation["type"]),
            "systematic_type": str(annotation["systematic_type"]),
            "matrix_membership": root_id in fanc_root_ids,
            "wing_motor_neuron_edges": [],
            "total_structural_synapse_count": 0,
        }
        if root_id in fanc_root_ids:
            values = fanc.xs(int(root_id), level="pre_pt_root_id")
            if isinstance(values, pd.DataFrame):
                if len(values) != 1:
                    raise ValueError(f"expected one FANC matrix row for DNp26 {root_id}")
                values = values.iloc[0]
            edge_rows = [
                {
                    "post_cell_type": str(label),
                    "structural_synapse_count": int(value),
                }
                for label, value in values.items()
                if label != "MN_syn_total" and int(value) > 0
            ]
            edge_rows.sort(key=lambda row: row["post_cell_type"])
            row_record["wing_motor_neuron_edges"] = edge_rows
            row_record["total_structural_synapse_count"] = int(values["MN_syn_total"])
        fanc_dnp26_rows.append(row_record)
    fanc_dnp26_rows.sort(key=lambda row: (row["dataset_side"], row["root_id"]))

    matrix_values = fanc_matrix.to_numpy()
    positive_values = matrix_values[matrix_values > 0]
    classification_counts = {
        str(key): int(value)
        for key, value in fanc_index["classification_system"].value_counts().items()
    }
    matrix_only = sorted(set(map(str, fanc_matrix.columns)) - {r["cell_type"] for r in fanc_mn_rows})
    properties_only = sorted({r["cell_type"] for r in fanc_mn_rows} - set(map(str, fanc_matrix.columns)))

    banc_source_totals = _counts_by(dnp26_wing_rows, "pre_root_id")
    accepted_count = sum(
        row["project_named_atlas_status"] == "accepted" for row in wing_rows
    )
    artifact: Dict[str, Any] = {
        "artifact_id": ARTIFACT_ID,
        "schema_version": SCHEMA_VERSION,
        "generator": {
            "script": "scripts/extract_banc_fanc_wing_evidence.py",
            "version": GENERATOR_VERSION,
            "source_snapshot_date": SOURCE_SNAPSHOT_DATE,
            "serialization": "UTF-8 JSON; keys sorted; two-space indent; one trailing newline",
        },
        "source_artifacts": receipts,
        "identity_spaces": [
            {
                "identity_space": "banc:banc_888",
                "sex": "female",
                "scope": "brain and ventral nerve cord",
            },
            {
                "identity_space": "fanc:fanc_production_mar2021@840",
                "sex": "female",
                "scope": "ventral nerve cord",
            },
        ],
        "scientific_scope": {
            "claim_level": "structural_connectome_evidence",
            "count_semantics": "raw aggregate structural synapse counts",
            "functional_sign": "unknown",
            "physiological_weight_interpretation": "forbidden",
            "autapse_policy": "excluded from selected pathway edges",
            "additional_user_synapse_threshold": "none",
            "banc_edge_source": "positive paper-v2 aggregate edges",
            "fanc_pair_threshold": "published matrix retains pairs with at least 3 synapses",
            "laterality": "dataset side labels are preserved without cross-dataset remapping",
        },
        "banc_v888": {
            "source_table_counts": {
                "metadata_rows": int(len(meta)),
                "aggregate_edge_rows": int(len(edges)),
            },
            "selected_pathway_neurons": selected_rows,
            "nod1_to_dnp26_edges": nod1_rows,
            "nod1_to_dnp26_summary": {
                "edge_count": len(nod1_rows),
                "total_structural_synapse_count": sum(
                    row["structural_synapse_count"] for row in nod1_rows
                ),
                "all_dataset_side_labels_opposed": all(
                    row["pre_dataset_side"] != row["post_dataset_side"]
                    for row in nod1_rows
                ),
            },
            "wing_motor_neurons": wing_rows,
            "wing_motor_neuron_summary": {
                "row_count": len(wing_rows),
                "dataset_side_counts": {
                    side: sum(row["dataset_side"] == side for row in wing_rows)
                    for side in sorted({row["dataset_side"] for row in wing_rows})
                },
                "proofread_count": sum(row["proofread"] for row in wing_rows),
                "project_named_atlas_accepted_count": accepted_count,
                "project_named_atlas_unresolved_count": len(wing_rows) - accepted_count,
                "unresolved_cell_types": sorted(UNRESOLVED_WING_MN_TYPES),
            },
            "dnp26_to_wing_motor_neuron_edges": dnp26_wing_rows,
            "dnp26_to_wing_motor_neuron_summary": {
                "edge_count": len(dnp26_wing_rows),
                "total_structural_synapse_count": sum(
                    row["structural_synapse_count"] for row in dnp26_wing_rows
                ),
                "by_source_dnp26": banc_source_totals,
                "by_target_cell_type": _counts_by(dnp26_wing_rows, "post_cell_type"),
            },
        },
        "fanc_v840": {
            "dataset_table": "fanc_production_mar2021",
            "materialization": 840,
            "published_wing_premotor_matrix_summary": {
                "premotor_rows": int(fanc.shape[0]),
                "motor_columns": int(fanc_matrix.shape[1]),
                "nonzero_pairs": int((matrix_values > 0).sum()),
                "thresholded_structural_synapse_count": int(matrix_values.sum()),
                "minimum_positive_pair_count": int(positive_values.min()),
                "maximum_pair_count": int(positive_values.max()),
                "row_total_mismatch_count": int(
                    (
                        fanc_matrix.sum(axis=1).to_numpy()
                        != fanc["MN_syn_total"].to_numpy()
                    ).sum()
                ),
                "matrix_only_motor_labels": matrix_only,
                "properties_only_motor_labels": properties_only,
                "premotor_classification_counts": dict(sorted(classification_counts.items())),
                "scope": "published one-sided wing premotor-to-MN matrix",
            },
            "wing_motor_neurons": fanc_mn_rows,
            "dnp26_rows": fanc_dnp26_rows,
            "dnp26_to_wing_motor_neuron_summary": {
                "source_neuron_count": len(fanc_dnp26_rows),
                "edge_count": sum(
                    len(row["wing_motor_neuron_edges"]) for row in fanc_dnp26_rows
                ),
                "total_structural_synapse_count": sum(
                    row["total_structural_synapse_count"] for row in fanc_dnp26_rows
                ),
            },
        },
        "cross_atlas_identity": {
            "premotor_matching_performed": False,
            "premotor_identity_status": "incomplete",
            "matched_premotor_pairs": [],
            "candidate_premotor_pairs_promoted_to_matches": [],
            "direct_individual_neuron_id_joins": [],
            "policy": (
                "BANC and FANC connectivity are independent observations. "
                "No BANC/FANC/FlyWire/MANC neuron IDs may be joined directly; "
                "external topological candidates remain candidates until an "
                "explicit, evidence-scored crosswalk is registered."
            ),
        },
    }
    # Guard JSON compliance here rather than allowing NaN to surface only at
    # final serialization.
    json.dumps(artifact, allow_nan=False)
    return artifact


def render_artifact(artifact: Mapping[str, Any]) -> str:
    return json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banc-meta", type=Path, default=Path("/tmp/banc_888_meta.feather"))
    parser.add_argument(
        "--banc-edges",
        type=Path,
        default=Path("/tmp/banc_888_edgelist_simple_v2.feather"),
    )
    parser.add_argument(
        "--fanc-matrix",
        type=Path,
        default=Path("/tmp/Lesser_Azevedo_2023/pkls/preMN_to_MN_wing_v840.pkl"),
    )
    parser.add_argument(
        "--fanc-mn-properties",
        type=Path,
        default=Path("/tmp/Lesser_Azevedo_2023/pkls/mn_properties_wing_v840.pkl"),
    )
    parser.add_argument(
        "--fanc-dn-crosswalk",
        type=Path,
        default=Path(
            "/tmp/2023neckconnective/Supplemental_files/"
            "Supplemental_file6_FANC_DNs.tsv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path; parent directory must already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = {
        "banc_v888_meta": args.banc_meta,
        "banc_v888_edgelist_paper_v2": args.banc_edges,
        "fanc_v840_premotor_to_wing_mn_matrix": args.fanc_matrix,
        "fanc_v840_wing_mn_properties": args.fanc_mn_properties,
        "fanc_dn_crosswalk": args.fanc_dn_crosswalk,
    }
    if not args.output.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {args.output.parent}")
    args.output.write_text(render_artifact(build_artifact(paths)), encoding="utf-8")


if __name__ == "__main__":
    main()
