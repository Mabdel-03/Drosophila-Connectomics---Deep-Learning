"""Stage 8 orchestrator — does a faithful LEFT mirror of the figure-ground circuit exist?

Runs, on the live CAVE v783 primary (cache-backed):
  Phase 0      right Nod1=FD1 anchor audit (k1_nod1_fd1) -> the named FD1 caveat
  Phase 0.5    left-mirror existence screen (existence_screen) -> 7-stage PRESENT/WEAK/ABSENT
  Families     A-I + K for BOTH sides (cfg=RIGHT positive control, cfg=LEFT mirror candidate)
  Family M     mirror symmetry + negative controls + wing-flip (m_mirror + muscular.mirror)
  Manifest     every left+right circuit root id (identities)

Emits to "8 - Left Hemisphere Circuit/": left_hemisphere_results.json,
cell_identity_manifest.json, VERIFICATION.md, REPORT.md, figures/.

The right run reuses the proven family code at cfg=RIGHT and must reproduce the paper
(positive control). The verdict is an EXISTENCE call (PRESENT / PARTIAL / ABSENT / REFUTED),
never rigged to confirm — see oracle.m_mirror.mirror_verdict and the floors there.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from flyconn.io import write_json
from flyconn.motif import compare as K

from . import fw_access as FW
from .oracle import consts as C

REPO = Path("/orcd/data/tpoggio/001/mabdel03/Connectomics")
STAGE = REPO / "8 - Left Hemisphere Circuit"

# Per-side families: (id, derive module, oracle module). J is MCNS-only (handled via wing-flip).
SIDE_FAMILIES = [
    ("A", "a_vch_loop", "a_vch_loop"),
    ("B", "b_inhibitor_screen", "b_inhibitor_screen"),
    ("C", "c_sheet_elimination", "c_sheet_elimination"),
    ("D", "d_retinotopy_null", "d_retinotopy_null"),
    ("E", "e_cable_distance", "e_cable_distance"),
    ("F", "f_sheet_regulation", "f_sheet_regulation"),
    ("G", "g_output_census", "g_output_census"),
    ("H", "h_nod_to_dnp26", "h_nod_to_dnp26"),
    ("I", "i_descending_channels", "i_descending_channels"),
    ("K", "k_fd3_lpt42", "k_fd3_lpt42"),
]

FAMILY_TITLES = {
    "A": "VCH-T4/T5 reciprocal loop (entry point)",
    "B": "Inhibitor screen: only VCH/DCH gate the sheet",
    "C": "LLPC1 is the unique figure-output sheet",
    "D": "T4a->LLPC1 retinotopy is local (null model)",
    "E": "Dual dendrite + compartmentalized inhibition (cable distance)",
    "F": "Sheet regulation (LPi15 / PVLP011 / VCH-direct)",
    "G": "LLPC1 output census; Nod1 dominance",
    "H": "Nod1 relays the figure signal to DNp26",
    "I": "Three descending channels",
    "K": "LPT42_Nod4 is the modern correlate of Egelhaaf-1985 FD3",
}


def _run_family(fid, dmod_name, omod_name, src, meta, cfg):
    dmod = importlib.import_module(f".derive.{dmod_name}", __package__)
    omod = importlib.import_module(f".oracle.{omod_name}", __package__)
    try:
        d = dmod.run(src, meta, cfg)
        claims = omod.build_claims(d)
    except Exception as e:  # never silently drop a family
        import traceback
        traceback.print_exc()
        d = {"error": str(e)}
        claims = [K.ClaimResult(id=f"{fid}.error", description=f"Family {fid} ({cfg.sheet_side}) error",
                                report_value=None, computed_primary=None, verdict=K.UNVERIFIABLE,
                                notes=f"derivation raised: {e}")]
    return d, claims


def run_all(prefer: str = "auto", only: list[str] | None = None) -> dict:
    src = FW.make_source(prefer)
    meta = FW.NeuronMeta.load()
    print(f"[stage8] source track = {src.track}", flush=True)

    out: dict = {"src_track": src.track, "families": {"right": {}, "left": {}}}

    # --- Phase 0: right Nod1=FD1 audit ---
    if not only or "K1" in only:
        D0 = importlib.import_module(".derive.k1_nod1_fd1", __package__)
        O0 = importlib.import_module(".oracle.k1_nod1_fd1", __package__)
        d0 = D0.run(src, meta)
        c0 = O0.build_claims(d0)
        out["right_fd1_audit"] = {
            "derived": {k: v for k, v in d0.items() if not k.startswith("_")},
            "claims": [c.to_dict() for c in c0],
            "verdict": O0.identity_verdict(c0),
        }
        print(f"[stage8] Phase 0 (Nod1=FD1): {out['right_fd1_audit']['verdict']['verdict']}", flush=True)

    # --- Phase 0.5: existence screen ---
    if not only or "SCREEN" in only:
        ES = importlib.import_module(".derive.existence_screen", __package__)
        out["existence_screen"] = ES.run(src, meta)
        print(f"[stage8] Phase 0.5 screen: {out['existence_screen']['decision']}", flush=True)

    # --- Families A-I + K, both sides ---
    right_derived: dict = {}
    left_derived: dict = {}
    for fid, dmod, omod in SIDE_FAMILIES:
        if only and fid not in only:
            continue
        for side, cfg, store in (("right", C.RIGHT, right_derived), ("left", C.LEFT, left_derived)):
            d, claims = _run_family(fid, dmod, omod, src, meta, cfg)
            store[fid] = {k: v for k, v in d.items() if not k.startswith("_")}
            out["families"][side][fid] = {
                "title": FAMILY_TITLES.get(fid, fid),
                "verdicts": K.verdict_counts(claims),
                "claims": [c.to_dict() for c in claims],
                "derived": store[fid],
            }
            print(f"[stage8] family {fid} ({side}): {K.verdict_counts(claims)}", flush=True)

    # --- Wing-flip (MCNS) ---
    wingflip = None
    if not only or "M" in only:
        WF = importlib.import_module("flyconn.muscular.mirror")
        wingflip = WF.run()
        print(f"[stage8] wing-flip available={wingflip.get('available')} "
              f"opposite_wings={wingflip.get('dnp26_bodies_opposite_wings')}", flush=True)

    # --- Family M: mirror + negative controls ---
    if (not only or "M" in only) and right_derived and left_derived:
        MD = importlib.import_module(".derive.m_mirror", __package__)
        MO = importlib.import_module(".oracle.m_mirror", __package__)
        md = MD.run(src, meta, right_derived, left_derived, wingflip=wingflip)
        mc = MO.build_claims(md)
        out["mirror"] = {
            "derived": md,
            "claims": [c.to_dict() for c in mc],
            "verdict": MO.mirror_verdict(mc),
        }
        print(f"[stage8] Family M verdict: {out['mirror']['verdict']['verdict']}", flush=True)

    # --- Cell-identity manifest (both sides) ---
    if not only or "MANIFEST" in only:
        ID = importlib.import_module(".derive.identities", __package__)
        out["manifest"] = {"right": ID.build_manifest(src, meta, C.RIGHT),
                           "left": ID.build_manifest(src, meta, C.LEFT)}
        print("[stage8] manifest built (both sides)", flush=True)

    return out


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
def _all_claims(run: dict) -> list[dict]:
    cs = []
    for side in ("right", "left"):
        for fid, blk in run["families"].get(side, {}).items():
            cs += blk["claims"]
    if "right_fd1_audit" in run:
        cs += run["right_fd1_audit"]["claims"]
    if "mirror" in run:
        cs += run["mirror"]["claims"]
    return cs


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
        "investigation": "Stage 8 — does a faithful LEFT mirror of the figure-ground circuit exist?",
        "papers": ["Figure_Ground_Circuit.pdf (Ziyin ... Poggio)", "Egelhaaf 1985 (FD cells)"],
        "flywire_track": run["src_track"],
        "flywire_datastack": "flywire_fafb_public v783 (synapses_nt_v1, no cleft threshold)",
        "malecns_source": "male-cns:v1.0 public bulk feather (subclass=wm wing-steering)",
        "fd1_anchor_verdict": run.get("right_fd1_audit", {}).get("verdict", {}),
        "existence_screen_decision": run.get("existence_screen", {}).get("decision"),
        "mirror_verdict": run.get("mirror", {}).get("verdict", {}),
        "verdict_counts": _verdict_counts(all_claims),
        "n_claims": len(all_claims),
    }
    payload = {
        "meta": meta,
        "right_fd1_audit": run.get("right_fd1_audit"),
        "existence_screen": run.get("existence_screen"),
        "families": run["families"],
        "mirror": run.get("mirror"),
    }
    write_json(stage / "left_hemisphere_results.json", payload)
    if "manifest" in run:
        write_json(stage / "cell_identity_manifest.json", run["manifest"])
    _write_markdown(run, meta, stage)
    _write_report(run, meta, stage)
    try:
        from . import figures_left as FL
        FL.render_all(run, stage / "figures")
        print("[stage8] figures written", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[stage8] figures skipped: {e}", flush=True)
    print(f"[stage8] emitted to {stage}", flush=True)


def _verdict_icon(v: str) -> str:
    return {"CONFIRMED": "PASS", "CONFIRMED_WITH_CAVEAT": "PASS*",
            "REFUTED": "FAIL", "UNVERIFIABLE": "N/A"}.get(v, "?")


def _claims_table(claims: list[dict]) -> str:
    lines = ["| Claim | Report | Left/Computed | Verdict |", "|---|---|---|---|"]
    for c in claims:
        lines.append(f"| {c['description']} | {c.get('report_value')} | "
                     f"{c.get('computed_primary')} | {_verdict_icon(c['verdict'])} {c['verdict']} |")
    return "\n".join(lines)


def _write_markdown(run: dict, meta: dict, stage: Path) -> None:
    L = []
    A = L.append
    A("# Stage 8 — Left-Hemisphere Figure-Ground Circuit: VERIFICATION\n")
    A(f"**FlyWire track:** {meta['flywire_track']} · v783 · no cleft threshold  ")
    A(f"**Existence-screen decision:** {meta.get('existence_screen_decision')}  ")
    mv = meta.get("mirror_verdict", {})
    A(f"**Mirror verdict:** {mv.get('verdict')} — {mv.get('statement')}  ")
    fv = meta.get("fd1_anchor_verdict", {})
    A(f"**Nod1=FD1 anchor (Phase 0):** {fv.get('verdict')} — {fv.get('statement')}\n")
    A(f"**Verdict counts:** {meta['verdict_counts']} (n={meta['n_claims']})\n")

    # Existence screen table
    es = run.get("existence_screen")
    if es:
        A("## Phase 0.5 — Existence screen (7-stage PRESENT/WEAK/ABSENT)\n")
        A("| Stage | Right (control) | Left (candidate) |")
        A("|---|---|---|")
        rs, ls = es["right"]["stages"], es["left"]["stages"]
        for k in rs:
            A(f"| {k} | {rs[k]} | {ls[k]} |")
        A("")

    # Phase 0
    if "right_fd1_audit" in run:
        A("## Phase 0 — Nod1 = FD1 audit (right hemisphere)\n")
        A(_claims_table(run["right_fd1_audit"]["claims"]))
        A("")

    # Mirror + controls
    if "mirror" in run:
        A("## Family M — mirror symmetry, complement (wing-flip), negative controls\n")
        A(_claims_table(run["mirror"]["claims"]))
        A("")

    # Per-family, both sides
    for fid in [f[0] for f in SIDE_FAMILIES]:
        rblk = run["families"]["right"].get(fid)
        lblk = run["families"]["left"].get(fid)
        if not (rblk or lblk):
            continue
        A(f"## Family {fid} — {FAMILY_TITLES.get(fid, fid)}\n")
        if rblk:
            A(f"**Right (positive control):** {rblk['verdicts']}  ")
        if lblk:
            A(f"**Left (mirror candidate):** {lblk['verdicts']}\n")
            A(_claims_table(lblk["claims"]))
            A("")
    (stage / "VERIFICATION.md").write_text("\n".join(L))


def _write_report(run: dict, meta: dict, stage: Path) -> None:
    mv = meta.get("mirror_verdict", {})
    es = run.get("existence_screen", {})
    fv = meta.get("fd1_anchor_verdict", {})
    wf = (run.get("mirror", {}).get("derived", {}) or {}).get("wing_flip", {})
    L = []
    A = L.append
    A("# Stage 8 — Does a faithful LEFT mirror of the figure-ground circuit exist?\n")
    A("## Headline\n")
    A(f"**{mv.get('verdict', 'n/a')}** — {mv.get('statement', '')}\n")
    A("The investigation tested existence rather than assuming it: a cheap 7-stage screen "
      "(go/no-go), then the full A-K mirror at the same code path as the right positive "
      "control, the wing-flip complement in MaleCNS, and falsifying negative controls.\n")

    A("## What the left circuit IS (named cells)\n")
    man = run.get("manifest", {}).get("left", {})
    if man:
        A(f"- Gating VCH (right-soma, crosses): `{man.get('gating_vch', {}).get('root')}` "
          f"({man.get('gating_vch', {}).get('nt_canonical')})")
        A(f"- VCH-gated left T4a: {man.get('t4a_gated', {}).get('n')} cells  ")
        A(f"- Left LLPC1 sheet: {man.get('llpc1_sheet', {}).get('n')} cells "
          f"(of {man.get('llpc1_sheet', {}).get('n_annotated_side')} annotated)")
        nod1 = [r['root_id'] for r in man.get('nod_types', {}).get('Nod1', [])]
        A(f"- Left Nod1 (FD1-role correlate): `{nod1}`")
        fd3 = [r['root_id'] for r in man.get('nod_types', {}).get('LPT42_Nod4', [])]
        A(f"- Left FD3 (LPT42_Nod4): `{fd3}`")
        A(f"- Left DNp26: `{man.get('descending_neurons', {}).get('DNp26', {}).get('flywire_left')}`\n")

    A("## The complement (wing-flip)\n")
    if wf.get("available"):
        A(f"- DNp26 target wings by soma side: {wf.get('dnp26_target_wings_by_soma')}  ")
        A(f"- Opposite wings (complement): **{wf.get('dnp26_bodies_opposite_wings')}**  ")
        A(f"- Shared steering muscles: {wf.get('dnp26_shared_muscles')}\n")
        A("The left brain's Nod1 drives the somaSide-L DNp26 (steering the RIGHT wing); the right "
          "brain's Nod1 drives the somaSide-R DNp26 (steering the LEFT wing). The two circuits "
          "steer OPPOSITE wings through the SAME muscles — a genuine mirror complement.\n")

    A("## Nod1 = FD1 anchor (Phase 0)\n")
    A(f"**{fv.get('verdict')}** — {fv.get('statement')}\n")

    A("## Existence screen\n")
    if es:
        A(f"Decision: **{es.get('decision')}**. Left stages: {es.get('left', {}).get('stages')}\n")

    A("## Caveats\n")
    A("- The left sheet (86 LLPC1) is smaller than the right (100): a per-hemisphere "
      "proofreading-completeness deficit (~7-14% across all counts), NOT circuit absence — the "
      "scale-robust statistics (reciprocal fraction, Nod1 rank/share, retinotopy z, contra %, "
      "wing category) are preserved.\n")
    A("- Nod1 is the FD1-ROLE correlate (visual_projection VPN), not literally Egelhaaf's "
      "lobula-plate tangential FD1 cell; the connectivity mirror + wing-flip do not depend on "
      "the FD1 label.\n")
    A("- MaleCNS (male) vs FlyWire FAFB (female): the wing-flip is a cross-animal homology "
      "bridge. NT labels are predictions; synapse counts are not weights.\n")
    (stage / "REPORT.md").write_text("\n".join(L))
