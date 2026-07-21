"""Stage 9 orchestrator: how the left and right figure-ground circuits connect and communicate.

Runs five channels on the live CAVE v783 connectome (cache-backed):
  N1  readout crossing / command convergence (Nod1 -> contralateral DNp26)
  N2  heterolateral input bridges + the Egelhaaf regressive-inhibition test
  N3  centrifugal gating (the soma-vs-arbor laterality distinction; VCH/DCH not axonal bridges)
  N4  shared downstream convergence (the bilateral integration locus)
  N5  systematic discovery of all inter-hemispheric edges (completeness guard + artifact contrast)

Plus a bridge manifest naming every inter-hemispheric bridge cell. Emits to
"9 - Interhemispheric Coupling/": interhemispheric_results.json, bridge_manifest.json,
VERIFICATION.md, REPORT.md, figures/.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from flyconn.io import write_json
from flyconn.motif import compare as K

from . import fw_access as FW
from .derive import interhemi_common as IH

REPO = Path("/orcd/data/tpoggio/001/mabdel03/Connectomics")
STAGE = REPO / "9 - Interhemispheric Coupling"

CHANNELS = [
    ("N1", "n1_readout_crossing", "Readout crossing: Nod1 -> contralateral DNp26 command"),
    ("N2", "n2_heterolateral_bridges", "Heterolateral input bridges (Egelhaaf regressive inhibition)"),
    ("N3", "n3_centrifugal_gating", "Centrifugal gating: soma-vs-arbor laterality"),
    ("N4", "n4_shared_targets", "Shared downstream convergence (bilateral integration)"),
    ("N5", "n5_discovery", "Systematic discovery of inter-hemispheric edges"),
]


def run_all(prefer: str = "auto", only: list[str] | None = None) -> dict:
    src = FW.make_source(prefer)
    meta = FW.NeuronMeta.load()
    print(f"[stage9] source track = {src.track}", flush=True)

    midline = IH.synapse_space_midline(src, meta)
    print(f"[stage9] synapse-space midline = {midline.get('midline_x_um')} um "
          f"(self_test_ok={midline.get('self_test_ok')})", flush=True)

    out: dict = {"src_track": src.track, "midline": midline, "channels": {}}

    for cid, mod_name, title in CHANNELS:
        if only and cid not in only:
            continue
        dmod = importlib.import_module(f".derive.{mod_name}", __package__)
        omod = importlib.import_module(f".oracle.{mod_name}", __package__)
        try:
            d = dmod.run(src, meta, midline)
            claims = omod.build_claims(d)
        except Exception as e:
            import traceback
            traceback.print_exc()
            d = {"error": str(e)}
            claims = [K.ClaimResult(id=f"{cid}.error", description=f"Channel {cid} error",
                                    report_value=None, computed_primary=None, verdict=K.UNVERIFIABLE,
                                    notes=f"derivation raised: {e}")]
        out["channels"][cid] = {
            "title": title,
            "verdicts": K.verdict_counts(claims),
            "claims": [c.to_dict() for c in claims],
            "derived": {k: v for k, v in d.items() if not k.startswith("_")},
        }
        print(f"[stage9] channel {cid}: {K.verdict_counts(claims)}", flush=True)

    # bridge manifest (use the discovery's genuine bridge types as the extra set)
    if not only or "MANIFEST" in only:
        BM = importlib.import_module(".derive.bridge_manifest", __package__)
        extra = (out["channels"].get("N5", {}).get("derived", {}) or {}).get("genuine_bridge_types", [])
        out["bridge_manifest"] = BM.build(src, meta, midline, extra_types=extra)
        print("[stage9] bridge manifest built", flush=True)

    return out


def _all_claims(run: dict) -> list[dict]:
    return [c for ch in run["channels"].values() for c in ch["claims"]]


def _verdict_counts(claims: list[dict]) -> dict:
    out: dict = {}
    for c in claims:
        out[c["verdict"]] = out.get(c["verdict"], 0) + 1
    return out


def emit(run: dict, stage: Path = STAGE) -> None:
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "figures").mkdir(parents=True, exist_ok=True)
    all_claims = _all_claims(run)
    meta = {
        "investigation": "Stage 9 — how the left and right figure-ground circuits connect and communicate",
        "papers": ["Figure_Ground_Circuit.pdf (Ziyin ... Poggio)", "Egelhaaf 1985 (FD cells)"],
        "flywire_track": run["src_track"],
        "flywire_datastack": "flywire_fafb_public v783 (synapses_nt_v1, no cleft threshold)",
        "synapse_space_midline_um": run["midline"].get("midline_x_um"),
        "midline_self_test_ok": run["midline"].get("self_test_ok"),
        "verdict_counts": _verdict_counts(all_claims),
        "n_claims": len(all_claims),
    }
    payload = {"meta": meta, "midline": run["midline"], "channels": run["channels"]}
    write_json(stage / "interhemispheric_results.json", payload)
    if "bridge_manifest" in run:
        write_json(stage / "bridge_manifest.json", run["bridge_manifest"])
    _write_markdown(run, meta, stage)
    _write_report(run, meta, stage)
    try:
        from . import figures_inter as FI
        FI.render_all(run, stage / "figures")
        print("[stage9] figures written", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[stage9] figures skipped: {e}", flush=True)
    print(f"[stage9] emitted to {stage}", flush=True)


def _icon(v: str) -> str:
    return {"CONFIRMED": "PASS", "CONFIRMED_WITH_CAVEAT": "PASS*",
            "REFUTED": "FAIL", "UNVERIFIABLE": "N/A"}.get(v, "?")


def _claims_table(claims: list[dict]) -> str:
    lines = ["| Finding | Computed | Verdict |", "|---|---|---|"]
    for c in claims:
        lines.append(f"| {c['description']} | {c.get('computed_primary')} | {_icon(c['verdict'])} {c['verdict']} |")
    return "\n".join(lines)


def _write_markdown(run: dict, meta: dict, stage: Path) -> None:
    L = []
    A = L.append
    A("# Stage 9 — Inter-Hemispheric Coupling of the Figure-Ground Circuits: VERIFICATION\n")
    A(f"**FlyWire track:** {meta['flywire_track']} · v783 · no cleft threshold  ")
    A(f"**Synapse-space midline:** {meta['synapse_space_midline_um']} um "
      f"(self-test {meta['midline_self_test_ok']})  ")
    A(f"**Verdict counts:** {meta['verdict_counts']} (n={meta['n_claims']})\n")
    for cid, _mod, title in CHANNELS:
        ch = run["channels"].get(cid)
        if not ch:
            continue
        A(f"## Channel {cid} — {title}\n")
        A(f"Verdicts: {ch['verdicts']}\n")
        A(_claims_table(ch["claims"]))
        A("")
    (stage / "VERIFICATION.md").write_text("\n".join(L))


def _write_report(run: dict, meta: dict, stage: Path) -> None:
    ch = run["channels"]
    n1 = ch.get("N1", {}).get("derived", {})
    n2 = ch.get("N2", {}).get("derived", {})
    n3 = ch.get("N3", {}).get("derived", {})
    n4 = ch.get("N4", {}).get("derived", {})
    n5 = ch.get("N5", {}).get("derived", {})
    L = []
    A = L.append
    A("# Stage 9 — How the left and right figure-ground circuits connect and communicate\n")
    A("## Headline\n")
    A("The two hemispheric figure circuits are coupled by three structural routes, identified and "
      "quantified at synapse resolution: a readout crossing in which each side's Nod1 drives the "
      "opposite hemisphere's steering command DNp26; heterolateral input bridges (H1, H2) that "
      "carry regressive-tuned contralateral inhibition onto the figure cells, matching Egelhaaf "
      "(1985); and a shared downstream convergence where the two readouts meet on common premotor, "
      "central, and neuromodulatory cells. The centrifugal gaters VCH and DCH, though their somata "
      "are registered to the opposite hemisphere, are shown NOT to be axonal bridges.\n")

    A("## Channel N1 — readout crossing / command convergence\n")
    A(f"- Left Nod1 -> right DNp26: {n1.get('left_nod1_to_right_dnp26')} synapses  ")
    A(f"- Right Nod1 -> left DNp26: {n1.get('right_nod1_to_left_dnp26')} synapses  ")
    A(f"- Nod1 output crossing (position-based): {n1.get('nod1_output_crossing_pct')}%  ")
    comp = n1.get("dnp26_input_composition", {})
    for sd, c in comp.items():
        A(f"- DNp26_{sd}: {c.get('contra_frac_of_nod1_input')}% of its Nod1 input is contralateral  ")
    A(f"- Asymmetry ratio: {n1.get('asymmetry_ratio')}\n")
    A("Each steering command is driven by the opposite hemisphere's figure readout, which is the "
      "structural basis of the established wing-flip.\n")

    A("## Channel N2 — heterolateral input bridges (Egelhaaf regressive inhibition)\n")
    if n2.get("bridges"):
        for b in n2["bridges"][:8]:
            lay = b.get("input_layer", {})
            A(f"- {b['bridge_type']} ({b['nt']}, {b['super_class']}): "
              f"{b['cross_syn_onto_machinery']} crossing synapses onto the figure machinery, "
              f"input layer {lay.get('dominant_layer')} ({lay.get('dominant_direction')})  ")
        A(f"\nInhibitory bridges onto the FD1/Nod1 pathway: {n2.get('inhibitory_fd1_bridges')}; "
          f"regressive-tuned: {n2.get('regressive_inhibitory_fd1_bridges')}. Egelhaaf "
          f"regressive-inhibition pattern holds: {n2.get('egelhaaf_regressive_inhibition_holds')}.\n")

    A("## Channel N3 — centrifugal gating (soma vs arbor)\n")
    for t, g in (n3.get("gater_laterality", {}) or {}).items():
        A(f"- {t}: soma {g.get('soma_contra_pct')}% contralateral, but position-based crossing "
          f"{g.get('position_cross_pct')}% (soma and arbor swapped: {g.get('soma_arbor_swapped')})  ")
    A("\nThe gaters are soma-displaced cells local to the lobe they gate, not axonal bridges, and "
      "they do not couple to each other directly.\n")

    A("## Channel N4 — shared downstream convergence\n")
    A(f"- {n4.get('n_shared')} cells receive from both the left and the right figure readout  ")
    A(f"- Roles: {n4.get('shared_by_role')}  ")
    A(f"- Readout cross-talk: {[c['cell_type'] for c in n4.get('nod1_crosstalk_cells', [])]}  ")
    A(f"- Bridge feedback: {[c['cell_type'] for c in n4.get('bridge_feedback_cells', [])]}  ")
    A(f"- Neuromodulatory convergence: {[c['cell_type'] for c in n4.get('neuromodulatory_convergence', [])]}\n")

    A("## Channel N5 — systematic discovery\n")
    A(f"- Of {n5.get('total_input_syn')} input synapses onto the circuit, "
      f"{n5.get('position_cross_syn')} ({n5.get('position_cross_frac')}%) are genuine "
      f"position-based inter-hemispheric crossings  ")
    A(f"- Genuine bridge types: {n5.get('genuine_bridge_types')}  ")
    A(f"- Soma-side artifact types removed by the position test: {n5.get('artifact_types_soma_only')}\n")

    A("## Caveats\n")
    A("- A true crossing is defined by synapse position relative to the synapse-space midline "
      f"({meta['synapse_space_midline_um']} um), not by the soma-side annotation, because the soma "
      "side conflates arbor geometry with axonal crossing (the VCH/DCH case).  \n")
    A("- Predicted neurotransmitter is used as an anatomical sign annotation, not a measured "
      "transmitter. Synapse counts are anatomical evidence, not efficacy.  \n")
    A("- The left-right asymmetries are reported with the per-hemisphere proofreading-completeness "
      "caveat established in Stage 8.\n")
    (stage / "REPORT.md").write_text("\n".join(L))
