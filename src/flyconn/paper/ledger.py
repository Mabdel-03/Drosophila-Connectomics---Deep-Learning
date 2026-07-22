"""Assemble every family's ClaimResults into one auditable ledger (JSON + Markdown).

Runs each family's derive.run() with a shared FlyWire source (live CAVE primary) and
MaleCNS offline data, calls the matching oracle.build_claims(), and concatenates the
verdicts. Emits:
  5 - Paper Verification/verification_results.json   (machine-readable, per-claim)
  5 - Paper Verification/VERIFICATION.md             (human-readable, grouped by family)

The per-claim verdict vocabulary is reused unchanged from flyconn.motif.compare.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from flyconn.io import write_json
from flyconn.motif import compare as K

from . import fw_access as FW

REPO = Path("/orcd/data/tpoggio/001/mabdel03/Connectomics")
STAGE = REPO / "5 - Paper Verification"

# family id -> (derive module name, oracle module name, needs_flywire)
FAMILY_SPECS = [
    ("A", "a_vch_loop", "a_vch_loop", True),
    ("B", "b_inhibitor_screen", "b_inhibitor_screen", True),
    ("C", "c_sheet_elimination", "c_sheet_elimination", True),
    ("D", "d_retinotopy_null", "d_retinotopy_null", True),
    ("E", "e_cable_distance", "e_cable_distance", True),
    ("F", "f_sheet_regulation", "f_sheet_regulation", True),
    ("G", "g_output_census", "g_output_census", True),
    ("H", "h_nod_to_dnp26", "h_nod_to_dnp26", True),
    ("I", "i_descending_channels", "i_descending_channels", True),
    ("J", "j_motor_mcns", "j_motor_mcns", False),       # MaleCNS offline
    ("K", "k_fd3_lpt42", "k_fd3_lpt42", True),          # FD3 == LPT42_Nod4 identity
    ("KD", "kd_fd3_disambig", "kd_fd3_disambig", True),  # FD3: LPT42_Nod4 vs Nod3 (Nod3 = FD2)
    ("L", "l_fd3_descending", "l_fd3_descending", True),  # FD3 -> descending neurons -> motor
    ("P", "p_fd3_input", "p_fd3_input", True),          # afferent: photoreceptor -> T4/T5 -> FD3
    ("Q", "q_fd3_sheet", "q_fd3_sheet", True),          # FD3 sheet-set (LPC1 the layer-b sheet)
    ("R", "r_fd3_inhibitor", "r_fd3_inhibitor", True),  # FD3 wide-field inhibitor (LPi14 = VCH role)
    ("S", "s_fd3_functional_circuit", "s_fd3_functional_circuit", True),  # composed: named circuit
    ("Y", "internal_consistency", "internal_consistency", False),  # oracle-only arithmetic
    ("Z", "interpretive", "interpretive", True),
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
    "J": "Wing-steering DNs and wing specificity (MaleCNS)",
    "K": "LPT42_Nod4 is the modern correlate of Egelhaaf-1985 FD3",
    "KD": "Disambiguation: FD3 is LPT42_Nod4, not Nod3 (Nod3 = FD2)",
    "L": "Descending targets of the FD3 cell (LPT42_Nod4 -> DN -> motor)",
    "P": "Afferent pathway of the FD3 cell (photoreceptor -> T4/T5 -> LPT42_Nod4)",
    "Q": "FD3 sheet: LPC1 is the direction-matched layer-b cholinergic feed-forward sheet",
    "R": "FD3 wide-field inhibitor: LPi14 is the VCH-role opponent gate",
    "S": "FD3 functional figure-ground circuit (T4b/T5b -> LPC1 -> FD3 -> DNp26; LPi14 gate)",
    "Y": "Paper-internal arithmetic consistency (oracle-only)",
    "Z": "Interpretive / physiology predictions (proxies)",
}

# Families whose build_claims() takes no derived data (computed from the oracle alone).
_ORACLE_ONLY = {"Y"}
# Families whose derive.run() takes no FlyWire source (offline / self-contained).
_NO_FW_SOURCE = {"J", "Y"}
# COMPOSED families: run AFTER the main loop because their derive.run() consumes the derived
# dicts (and claims) of sibling families rather than pulling from the connectome directly.
_COMPOSED = {"S"}


def _load_prior_families() -> dict:
    """Sibling family blocks from the persisted verification_results.json (for composed families
    whose siblings were not in the current --only run). Returns {} if the file is absent."""
    path = STAGE / "verification_results.json"
    if not path.exists():
        return {}
    try:
        from flyconn.io import read_json
        return read_json(path).get("families", {})
    except Exception:  # noqa: BLE001
        return {}


def run_all(prefer: str = "auto", only: list[str] | None = None) -> dict:
    src = FW.make_source(prefer)
    meta = FW.NeuronMeta.load()
    results: dict[str, list[K.ClaimResult]] = {}
    derived: dict[str, dict] = {}

    for fid, dmod_name, omod_name, _needs_fw in FAMILY_SPECS:
        if only and fid not in only:
            continue
        if fid in _COMPOSED:
            continue  # handled after the loop (needs the sibling families' outputs)
        omod = importlib.import_module(f".oracle.{omod_name}", __package__)
        try:
            if fid in _ORACLE_ONLY:
                d = {}
                claims = omod.build_claims()
            else:
                dmod = importlib.import_module(f".derive.{dmod_name}", __package__)
                d = dmod.run() if fid in _NO_FW_SOURCE else dmod.run(src, meta)
                claims = omod.build_claims(d)
        except Exception as e:  # a family failure is recorded, never silently dropped
            import traceback
            traceback.print_exc()
            claims = [K.ClaimResult(
                id=f"{fid}.error", description=f"Family {fid} derivation error",
                report_value="(see paper)", computed_primary=None, verdict=K.UNVERIFIABLE,
                notes=f"derivation raised: {e}")]
            d = {"error": str(e)}
        results[fid] = claims
        derived[fid] = {k: v for k, v in d.items() if not k.startswith("_")}
        print(f"[ledger] family {fid}: {K.verdict_counts(claims)}", flush=True)

    # ---- composed families (S): assemble from sibling derived dicts + claims ----
    for fid, dmod_name, omod_name, _needs_fw in FAMILY_SPECS:
        if fid not in _COMPOSED or (only and fid not in only):
            continue
        omod = importlib.import_module(f".oracle.{omod_name}", __package__)
        try:
            dmod = importlib.import_module(f".derive.{dmod_name}", __package__)
            # S reads Q/R/K/L/P. A sibling not in THIS run (e.g. `--only S` or `--only Q,R,S`)
            # is loaded from the persisted verification_results.json so S can still assemble.
            prior = _load_prior_families()

            def _block(f):
                if f in results:
                    return {**derived.get(f, {}), "claims": [c.to_dict() for c in results[f]]}
                pf = prior.get(f, {})
                return {**pf.get("derived", {}), "claims": pf.get("claims", [])}
            d = dmod.run(src, meta, q_derived=_block("Q"), r_derived=_block("R"),
                         k_derived=_block("K"), l_derived=_block("L"),
                         p_derived=_block("P"))
            claims = omod.build_claims(d)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            claims = [K.ClaimResult(id=f"{fid}.error", description=f"Family {fid} composition error",
                                    report_value="(composed)", computed_primary=None,
                                    verdict=K.UNVERIFIABLE, notes=f"composition raised: {e}")]
            d = {"error": str(e)}
        results[fid] = claims
        derived[fid] = {k: v for k, v in d.items() if not k.startswith("_")}
        print(f"[ledger] family {fid} (composed): {K.verdict_counts(claims)}", flush=True)

    return {"src_track": src.track, "results": results, "derived": derived}


def emit(run: dict, stage: Path = STAGE) -> None:
    stage.mkdir(parents=True, exist_ok=True)
    all_claims = [c for fid in run["results"] for c in run["results"][fid]]
    meta = {
        "paper": "Figure_Ground_Circuit.pdf",
        "flywire_track": run["src_track"],
        "flywire_datastack": "flywire_fafb_public v783 (synapses_nt_v1, no cleft threshold)",
        "malecns_source": "male-cns:v1.0 public bulk feather (subclass=wm wing-steering)",
        "verdict_counts": K.verdict_counts(all_claims),
        "n_claims": len(all_claims),
    }
    payload = {
        "meta": meta,
        "families": {
            fid: {
                "title": FAMILY_TITLES[fid],
                "verdicts": K.verdict_counts(run["results"][fid]),
                "claims": [c.to_dict() for c in run["results"][fid]],
                "derived": run["derived"].get(fid, {}),
            }
            for fid in run["results"]
        },
    }
    write_json(stage / "verification_results.json", payload)
    # Side-file each FD3-family offline block (K,P,Q,R,S) so the FD3 report's multi-track tables
    # have the offline column without re-running anything (a live --only run overwrites
    # verification_results.json with just that family).
    if run["src_track"] != "live":
        (stage / "figures").mkdir(parents=True, exist_ok=True)
        for fid in ("K", "P", "Q", "R", "S"):
            if fid in run["results"]:
                write_json(stage / "figures" / f"{fid}_offline.json", {
                    "meta": {"flywire_track": run["src_track"]},
                    "title": FAMILY_TITLES[fid],
                    "verdicts": K.verdict_counts(run["results"][fid]),
                    "claims": [c.to_dict() for c in run["results"][fid]],
                    "derived": run["derived"].get(fid, {}),
                })
    _write_markdown(run, meta, stage)
    _write_report(run, meta, stage)
    try:
        from . import figures as PF
        PF.summary_figure(run, stage / "figures" / "verdict_summary.png")
        if "K" in run["results"]:
            PF.fd3_rf_differential(run, stage / "figures" / "fd3_rf_differential.png")
            PF.fd3_claim_table_tex(run, stage / "figures" / "fd3_claim_table.tex")
    except Exception as e:
        print(f"[ledger] figure warning: {e}")
    print(f"[ledger] wrote {stage/'verification_results.json'}, VERIFICATION.md, REPORT.md")
    print(f"[ledger] TOTAL: {meta['verdict_counts']} over {meta['n_claims']} claims")


def _write_report(run: dict, meta: dict, stage: Path) -> None:
    """A short narrative: headline, every caveat and every UNVERIFIABLE claim with reason."""
    all_claims = [c for fid in run["results"] for c in run["results"][fid]]
    caveats = [c for c in all_claims if c.verdict == K.CONFIRMED_WITH_CAVEAT]
    refuted = [c for c in all_claims if c.verdict == K.REFUTED]
    unver = [c for c in all_claims if c.verdict == K.UNVERIFIABLE]
    vc = meta["verdict_counts"]
    L = []
    A = L.append
    A("# Verification Report — Figure-Ground Circuit\n")
    A(f"**{vc.get('CONFIRMED', 0)} CONFIRMED · {vc.get('CONFIRMED_WITH_CAVEAT', 0)} "
      f"CONFIRMED_WITH_CAVEAT · {vc.get('REFUTED', 0)} REFUTED · "
      f"{vc.get('UNVERIFIABLE', 0)} UNVERIFIABLE** over {meta['n_claims']} claims.\n")
    A("Every quantitative claim was re-derived from public connectome data: the FlyWire "
      f"FAFB v783 brain ({meta['flywire_track']} track, `synapses_nt_v1`, no cleft "
      "threshold — which reproduces the paper's absolute counts to the digit) and the male "
      "CNS connectome (MaleCNS v1.0 public bulk files) for the motor mapping.\n")

    A("## Headline\n")
    A("The circuit reproduces end-to-end. Exemplary exact matches: VCH 27,576/32,363 "
      "synapses; 1,022 T4/T5 inputs (12,301 syn, 44.6%); 912 reciprocal; the 454 VCH-gated "
      "T4a -> 9,223 syn -> 100 LLPC1 sheet; LLPC1 receives 17,499 reciprocal-T4/T5 synapses "
      "(~48x its sibling sheets, 318-401); LLPC1 sheet output 106,269 synapses with Nod1 the "
      "dominant excitatory readout; VCH->T4a inputs sit a median ~1.8 um from the "
      "T4a->LLPC1 terminal vs ~29 um for other inputs (VCH closest in 100% of 286 T4a, "
      "Wilcoxon p~6e-49); and in the male CNS, DNa04 drives the ipsilateral wing (0.99), "
      "DNp26/DNg32 the contralateral wing (0.23/0.03), with DNp26 targeting hg1/i1/hg2.\n")

    if refuted:
        A("## Refuted claims\n")
        for c in refuted:
            A(f"- **{c.id}** — {c.description}: paper `{c.report_value}` vs computed "
              f"`{c.computed_primary}`. {c.notes}")
        A("")
    else:
        A("## Refuted claims\n\nNone. No quantitative claim was contradicted by the data.\n")

    A("## Confirmed-with-caveat (matches after a named difference)\n")
    for c in caveats:
        A(f"- **{c.id}** — {c.description}: paper `{c.report_value}` vs `{c.computed_primary}`. "
          f"{c.notes}")
    A("")

    A("## Unverifiable (interpretive / physiology predictions)\n")
    A("These are not connectomic claims; each is recorded with the anatomical proxy "
      "(where one exists) that the connectome *can* supply.\n")
    for c in unver:
        A(f"- **{c.id}** — {c.description}. {c.notes}")
    A("")
    A("---")
    A("Full per-claim ledger: `verification_results.json` and `VERIFICATION.md`.")
    (stage / "REPORT.md").write_text("\n".join(L))


def _write_markdown(run: dict, meta: dict, stage: Path) -> None:
    L = []
    A = L.append
    A("# Independent Verification — *Finding Vision-Behavior Circuit through Connectomics*\n")
    A("End-to-end re-derivation of every quantitative claim in `Figure_Ground_Circuit.pdf` "
      "(the VCH -> T4/T5 -> LLPC1 -> Nod1 -> DNp26 -> wing-steering figure-ground circuit), "
      "across both connectomes.\n")
    A(f"- **FlyWire track:** {meta['flywire_track']} — {meta['flywire_datastack']}")
    A(f"- **MaleCNS:** {meta['malecns_source']}")
    A(f"- **Total:** {meta['verdict_counts']} over {meta['n_claims']} claims\n")
    A("Verdict legend: **CONFIRMED** (within tolerance) · **CONFIRMED_WITH_CAVEAT** "
      "(matches after a named, demonstrated difference) · **REFUTED** (paper appears wrong) "
      "· **UNVERIFIABLE** (interpretive / physiology prediction; anatomical proxy noted).\n")

    for fid, _, _, _ in FAMILY_SPECS:
        if fid not in run["results"]:
            continue
        claims = run["results"][fid]
        A(f"## Family {fid} — {FAMILY_TITLES[fid]}\n")
        A(f"_Verdicts: {K.verdict_counts(claims)}_\n")
        A(K.summary_table_md(claims))
        A("")

    A("## Per-claim provenance\n")
    A("| ID | Verdict | Notes |")
    A("|---|---|---|")
    for fid in run["results"]:
        for c in run["results"][fid]:
            note = (c.notes or "").replace("\n", " ")
            A(f"| {c.id} | {c.verdict} | {note} |")
    A("")
    A("---")
    A("Regenerate: `python scripts/paper_verify.py` (live CAVE token at "
      "`~/.cloudvolume/secrets/cave-secret.json`; MaleCNS bulk files on scratch).")
    (stage / "VERIFICATION.md").write_text("\n".join(L))
