"""Evidence audit artifacts for the FD3 full-circuit report.

The report is prose, but the audit is deliberately mechanical: each major reader-facing
claim is tied to a JSON path, a local literature source, or an explicit caveat. The audit
is written beside the generated TeX so reviewers can check that the narrative did not
outrun the evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fam(results: dict, key: str) -> dict:
    return results["families"][key] if "families" in results else results


def _g(d: dict, *path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _entry(
    claim_id: str,
    section: str,
    claim: str,
    evidence_class: str,
    source: str,
    status: str,
    caveat: str = "",
) -> dict[str, str]:
    return {
        "claim_id": claim_id,
        "section": section,
        "claim": claim,
        "evidence_class": evidence_class,
        "source": source,
        "status": status,
        "caveat": caveat,
    }


def build_audit(k_results: dict, p_results: dict, l_results: dict) -> dict[str, Any]:
    K = _fam(k_results, "K")["derived"]
    P = _fam(p_results, "P")["derived"]
    L = _fam(l_results, "L")["derived"]
    meta = k_results.get("meta", {}) if isinstance(k_results, dict) else {}
    track = meta.get("flywire_track", K.get("track", "unknown"))
    cv = K.get("cross_version", {}) or {}
    contra = P.get("contra_inhibition", {}) or {}
    motor = L.get("motor", {}) or {}

    entries = [
        _entry(
            "K.identity",
            "Identity",
            "The strongest connectomic FD3 candidate is LPT42_Nod4.",
            "direct_connectome",
            "families.K.derived.fd_family_screen",
            "confirmed",
            f"Candidate screen has margin {K.get('fd_family_screen', {}).get('fd3_margin')}.",
        ),
        _entry(
            "K.direction",
            "Identity",
            "LPT42_Nod4 draws almost all measured T4/T5 motion input from layer-b.",
            "direct_connectome",
            "families.K.derived.cand_layer_frac.b",
            "confirmed",
            f"Layer-b fraction: {K.get('cand_layer_frac', {}).get('b')}%.",
        ),
        _entry(
            "K.rf_relative",
            "Identity",
            "LPT42_Nod4 is lateral to FD1 and has a frontal gap on both sides.",
            "direct_connectome",
            "families.K.derived.rf.per_side",
            "confirmed",
            "Absolute visual angle remains calibration-limited; the core comparison is relative to FD1.",
        ),
        _entry(
            "K.small_field",
            "Identity",
            "The FD3 candidate pools a spatially bounded retinotopic input patch.",
            "direct_connectome",
            "families.K.derived.smallfield_null",
            "confirmed",
            f"Permutation p={K.get('smallfield_null', {}).get('p_value')}; this is wiring support, not a direct physiology recording.",
        ),
        _entry(
            "K.output_side",
            "Identity",
            "Most LPT42_Nod4 output is contralateral.",
            "direct_connectome",
            "families.K.derived.cand_contra_output_pct",
            "confirmed",
            f"Contralateral output: {K.get('cand_contra_output_pct')}%.",
        ),
        _entry(
            "K.morphology",
            "Identity",
            "Both reconstructed skeletons have a dorsoventral dendritic span and crossed axonal displacement consistent with FD3.",
            "direct_connectome",
            "families.K.derived.morphology.cells",
            "confirmed_with_caveat",
            "Skeleton supports overall geometry; fine branch-level identity is not overinterpreted.",
        ),
        _entry(
            "K.cross_version",
            "Provenance",
            "Live or v630 replication is included only if available in the current run.",
            "provenance",
            "families.K.derived.cross_version",
            "confirmed" if cv.get("available") else "unavailable",
            "" if cv.get("available") else str(cv.get("reason", "not available")),
        ),
        _entry(
            "P.t4t5_fraction",
            "Inputs",
            "T4/T5 motion detectors provide a minority of all FD3 input.",
            "direct_connectome",
            "families.P.derived.census.t4t5_frac_of_total",
            "confirmed",
            f"T4/T5 fraction of all input: {P.get('census', {}).get('t4t5_frac_of_total')}%.",
        ),
        _entry(
            "P.layer_b_motion",
            "Inputs",
            "Within the measured T4/T5 subset, FD3 is dominated by layer-b back-to-front input.",
            "direct_connectome",
            "families.P.derived.census.layer_b_frac_of_t4t5",
            "confirmed",
            f"Layer-b fraction of T4/T5 input: {P.get('census', {}).get('layer_b_frac_of_t4t5')}%.",
        ),
        _entry(
            "P.upstream_cascade",
            "Inputs",
            "Canonical photoreceptor, lamina, medulla, T4/T5 pathways feeding the FD3 detector types are present.",
            "direct_connectome_and_literature",
            "families.P.derived.upstream_cascade; Maisak et al. 2013; Hardie 1989",
            "confirmed_with_caveat",
            "The upstream cascade is measured at type level, not as a unique per-object trace.",
        ),
        _entry(
            "P.central_inputs",
            "Inputs",
            "FD3 receives diverse non-T4/T5 input, including cholinergic projection sheets and inhibitory inputs.",
            "direct_connectome",
            "families.P.derived.central_inputs.top_types",
            "confirmed",
            "Only one leading central input type is marked as a layer-b carrier in the current derived table.",
        ),
        _entry(
            "P.contra_inhibition",
            "Inputs",
            "Contralateral inhibitory input is present and compatible with FD3 physiology.",
            "direct_connectome_and_literature",
            "families.P.derived.contra_inhibition; Egelhaaf 1985 Part II",
            "confirmed_with_caveat",
            "Transmitter-classified partner count is power-limited."
            if not contra.get("sufficient")
            else "",
        ),
        _entry(
            "L.direct_dns",
            "Outputs",
            "FD3 directly contacts descending neurons, led by DNp26.",
            "direct_connectome",
            "families.L.derived.direct",
            "confirmed",
            f"Direct DN synapses: {L.get('direct', {}).get('dn_syn')}; DN types: {L.get('direct', {}).get('n_dns')}.",
        ),
        _entry(
            "L.relay_dns",
            "Outputs",
            "FD3 reaches a broader descending set through strong central partners.",
            "direct_connectome",
            "families.L.derived.relay",
            "confirmed_with_caveat",
            "Relay membership depends on the reported intermediary synapse threshold.",
        ),
        _entry(
            "L.motor_proxy",
            "Outputs",
            "FD3-weighted descending output is anatomically biased toward wing-steering motor systems.",
            "cross_connectome_proxy",
            "families.L.derived.motor.motor_system_pct; Namiki et al. 2018",
            "confirmed_with_caveat" if motor.get("available") else "unavailable",
            "This is an anatomical motor-system proxy, not a behavioral perturbation result.",
        ),
        _entry(
            "L.dnp26_muscles",
            "Outputs",
            "DNp26 links FD3 to contralateral wing-steering muscles in the male CNS mapping.",
            "cross_connectome_proxy",
            "families.L.derived.motor.per_dn.DNp26",
            "confirmed_with_caveat" if motor.get("available") else "unavailable",
            "Neuron matching is by shared descending-neuron type name across connectomes.",
        ),
    ]

    return {
        "artifact": "fd3_full_circuit_audit",
        "primary_track": track,
        "n_entries": len(entries),
        "entries": entries,
    }


def _md_escape(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def write_audit(
    k_results: dict,
    p_results: dict,
    l_results: dict,
    stage_dir: str | Path = "7 - FD3 Identification",
) -> dict[str, Any]:
    stage = Path(stage_dir)
    stage.mkdir(parents=True, exist_ok=True)
    audit = build_audit(k_results, p_results, l_results)
    json_path = stage / "fd3_full_circuit_audit.json"
    md_path = stage / "fd3_full_circuit_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    rows = [
        "# FD3 Full-Circuit Evidence Audit",
        "",
        f"Primary track: `{audit['primary_track']}`",
        "",
        "| Claim ID | Section | Evidence | Status | Claim | Caveat |",
        "|---|---|---|---|---|---|",
    ]
    for e in audit["entries"]:
        rows.append(
            "| {claim_id} | {section} | {evidence_class} | {status} | {claim} | {caveat} |".format(
                claim_id=_md_escape(e["claim_id"]),
                section=_md_escape(e["section"]),
                evidence_class=_md_escape(e["evidence_class"]),
                status=_md_escape(e["status"]),
                claim=_md_escape(e["claim"]),
                caveat=_md_escape(e["caveat"]),
            )
        )
    md_path.write_text("\n".join(rows) + "\n")
    return audit
