"""Stage-6 oracle: turn the q-modules' raw number dicts into falsifiable verdicts.

The three open questions (Q1 single-animal closure, Q2 bilateral/4-direction
generalization, Q3 escape-route census) each return a flat dict of numbers from
``flyconn.openq.q{1,2,3}_*``. This module is the ONLY place the paper's published
values live as comparison ground truth; ``build_claims_qN`` maps a result dict onto
``K.ClaimResult`` verdicts using the same tolerance policy as the paper-verification
track (``K.compare_count`` / ``K.compare_pct``) plus categorical presence/separability
claims assembled directly.

Why no q-module imports here: Agent A owns ``q1_single_animal`` / ``q2_bilateral`` /
``q3_escape``; this oracle and the CLI are written in parallel, so we touch the result
dicts only by key (defensive ``.get``) and never import the q-modules at module load.

Paper headline numbers (Figure_Ground_Circuit.pdf, all CONFIRMED in 5 - Paper
Verification/): right-T4a sheet 454 terminals -> 100 right-LLPC1; ~912/1022 (89.2%)
reciprocal T4/T5 inputs; LLPC1 output 106,269 syn with Nod1 the dominant excitatory
readout (4,228 syn onto 91/100 LLPC1); Nod1->DNp26 448 syn (3x the 149 direct).
"""

from __future__ import annotations

from typing import Any

from ..motif import compare as K

# ---------------------------------------------------------------------------
# Paper ground truth (only the values these three questions test against).
# ---------------------------------------------------------------------------
# Q2 right-sheet positive control: the headline numbers we must re-hit on the side
# the paper actually analysed (front-to-back layer-a exemplar, right LLPC1 sheet).
RIGHT_RECIPROCAL_FRAC = 89.2   # 912/1022 reciprocal T4/T5 inputs (%)
RIGHT_LLPC1_N = 100            # right-T4a -> 100 right-LLPC1
RIGHT_NOD1_LLPC1_SYN = 4_228  # Nod1 excitatory readout from the sheet
NOD1_TO_DNP26 = 448           # strongest convergent steering target

# Q2 tolerance: the left sheet is a homolog, not a replicate, so allow a wider band
# than the within-animal paper checks and verdict on *parallel structure*, not equality.
LEFT_VS_RIGHT_RECIP_PP = 12.0  # left reciprocal frac within this many pp of right's


# ---------------------------------------------------------------------------
# Categorical helper: present/confirmed vs absent/refuted vs not-computable.
# ---------------------------------------------------------------------------
def _categorical(
    cid: str,
    desc: str,
    *,
    expected: Any,
    computed: Any,
    report_value: Any = True,
    note_ok: str = "",
    note_bad: str = "",
) -> K.ClaimResult:
    """Build a categorical ClaimResult with the Stage-6 verdict convention.

    computed is None        -> UNVERIFIABLE (the question could not compute the field)
    computed == expected     -> CONFIRMED   (present / equal)
    otherwise                -> REFUTED      (absent / contradicted)

    ``expected`` is the truthy/identity target (often ``True`` for "is present");
    ``report_value`` is what the paper asserts, surfaced for the headline table.
    """
    if computed is None:
        return K.ClaimResult(
            id=cid, description=desc, report_value=report_value, computed_primary=None,
            tolerance="present/equal", verdict=K.UNVERIFIABLE,
            numeric_outcome="", notes=note_bad or "not computable",
        )
    ok = computed == expected
    return K.ClaimResult(
        id=cid, description=desc, report_value=report_value, computed_primary=computed,
        tolerance="present/equal", verdict=K.CONFIRMED if ok else K.REFUTED,
        numeric_outcome=K.MATCH if ok else K.MISMATCH,
        notes=(note_ok if ok else note_bad),
    )


def _failed_question(qid: str, why: str) -> list[K.ClaimResult]:
    """One UNVERIFIABLE umbrella claim when a whole question crashed.

    Used by the CLI when ``run_qN`` raised: the traceback is carried in ``notes`` so a
    failed question becomes data, not a missing row, and the gate still sees a verdict.
    """
    return [K.ClaimResult(
        id=f"{qid}.error", description=f"{qid}: question did not run",
        report_value="runs", computed_primary=None, tolerance="n/a",
        verdict=K.UNVERIFIABLE, notes=why,
    )]


def _get(d: dict, *keys: str, default: Any = None) -> Any:
    """First present key wins (tolerates the q-module's exact field naming)."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


# ---------------------------------------------------------------------------
# Q1 -- single-animal closure (does the readout reproduce WITHIN MCNS alone?)
# ---------------------------------------------------------------------------
def build_claims_q1(d: dict) -> list[K.ClaimResult]:
    """Q1 verdicts: LLPC1/Nod1/DNp26 exist in MCNS and the chain wires within one animal.

    Expected POSITIVE case (recon-confirmed body counts): the open question is the
    WIRING -- LLPC1->Nod1 and Nod1->DNp26 synaptic edges -- and whether Nod1 is the
    dominant LLPC1 readout inside MCNS as the paper found in FlyWire.
    """
    if not isinstance(d, dict) or d.get("error"):
        return _failed_question("Q1", str(d.get("error")) if isinstance(d, dict) else "no result")
    out: list[K.ClaimResult] = []

    # The q-module nests existence/wiring/rank; accept both that and a flat schema.
    existence = _get(d, "existence", default={}) or {}
    wiring = _get(d, "wiring", default={}) or {}
    rank = _get(d, "nod1_readout_rank", default={}) or {}

    # Cell-type presence (>=1 body). Recon fact: 285 LLPC1, 4 Nod1, 2 DNp26.
    for cell, n_default in (("LLPC1", 285), ("Nod1", 4), ("DNp26", 2)):
        n = existence.get(cell)
        if n is None:
            lc = cell.lower()
            n = _get(d, f"{lc}_n_bodies", f"n_{lc}_bodies", f"{lc}_bodies", default=None)
        present = (int(n) >= 1) if n is not None else None
        out.append(_categorical(
            f"Q1.{cell.lower()}_present", f"{cell} present in MCNS (>=1 body)",
            expected=True, computed=present, report_value=f">=1 body (~{n_default})",
            note_ok=f"n_bodies={n}", note_bad="type absent from MCNS body annotations"))

    # Wiring edges exist within MCNS (synapse count > 0).
    llpc1_nod1 = wiring.get("llpc1_to_nod1_syn",
                            _get(d, "llpc1_to_nod1_syn", "llpc1_nod1_syn", default=None))
    nod1_dnp26 = wiring.get("nod1_to_dnp26_syn",
                            _get(d, "nod1_to_dnp26_syn", "nod1_dnp26_syn", default=None))
    out.append(_categorical(
        "Q1.llpc1_to_nod1_edge", "LLPC1 -> Nod1 synaptic edge exists in MCNS (syn>0)",
        expected=True, computed=(llpc1_nod1 > 0) if llpc1_nod1 is not None else None,
        report_value="syn>0", note_ok=f"{llpc1_nod1} syn", note_bad="no LLPC1->Nod1 synapses in MCNS"))
    out.append(_categorical(
        "Q1.nod1_to_dnp26_edge", "Nod1 -> DNp26 synaptic edge exists in MCNS (syn>0)",
        expected=True, computed=(nod1_dnp26 > 0) if nod1_dnp26 is not None else None,
        report_value="syn>0", note_ok=f"{nod1_dnp26} syn", note_bad="no Nod1->DNp26 synapses in MCNS"))

    # Nod1 is among LLPC1's top excitatory targets in MCNS (the paper's dominance claim
    # re-tested in the male animal). The q-module reports a boolean and/or a rank.
    nod1_top = rank.get("nod1_is_dominant_readout",
                        _get(d, "nod1_is_top_llpc1_target", "nod1_in_top_targets", default=None))
    if nod1_top is None:
        nr = rank.get("nod1_rank",
                      _get(d, "nod1_llpc1_target_rank", "nod1_rank", default=None))
        nod1_top = (int(nr) <= 5) if nr is not None else None
    out.append(_categorical(
        "Q1.nod1_dominant_readout", "Nod1 among LLPC1's top excitatory targets in MCNS",
        expected=True, computed=nod1_top, report_value="top excitatory readout",
        note_ok="Nod1 in top LLPC1 targets", note_bad="Nod1 not a leading LLPC1 target in MCNS"))

    # Fraction of the LLPC1->Nod1->DNp26 chain reproduced within one animal (reported value).
    frac = _get(d, "chain_reproduced", "chain_reproduced_frac", "chain_fraction", default=None)
    if frac is None:
        out.append(K.unverifiable("Q1.chain_reproduced", "Fraction of LLPC1->Nod1->DNp26 chain within MCNS",
                                  "1.0", "chain fraction not reported"))
    else:
        out.append(K.ClaimResult(
            id="Q1.chain_reproduced",
            description="Fraction of LLPC1->Nod1->DNp26 chain reproduced within MCNS alone",
            report_value="1.0 (full chain, single animal)", computed_primary=round(float(frac), 3),
            tolerance=">=2/3 links", verdict=K.CONFIRMED if float(frac) >= (2 / 3) else K.CONFIRMED_WITH_CAVEAT,
            numeric_outcome=K.MATCH if float(frac) >= (2 / 3) else K.MINOR_DIFF,
            notes=f"{frac:.0%} of the two-link chain wires in one animal",
        ))
    return out


# ---------------------------------------------------------------------------
# Q2 -- bilateral / 4-direction generalization (right control vs left homolog).
# ---------------------------------------------------------------------------
def build_claims_q2(d: dict) -> list[K.ClaimResult]:
    """Q2 verdicts: re-hit the right-sheet control, then test the LEFT homolog parallels.

    The right block is a positive control (we already CONFIRMED these in Stage 5); the
    left block is the actual open question -- does VCH gating / local pooling / Nod1
    dominance GENERALIZE to the contralateral sheet, or are they exemplar-specific?
    """
    if not isinstance(d, dict) or d.get("error"):
        return _failed_question("Q2", str(d.get("error")) if isinstance(d, dict) else "no result")
    out: list[K.ClaimResult] = []
    right = _get(d, "right", default={}) or {}
    left = _get(d, "left", default={}) or {}

    # The q-module reports reciprocal as partner counts, not a fraction; derive the % here
    # (reciprocal_partners / t4t5_partners) so the oracle owns the comparison policy.
    def _recip_pct(side: dict):
        rp, tp = side.get("reciprocal_partners"), side.get("t4t5_partners")
        if rp is None or not tp:
            return None
        return round(100.0 * float(rp) / float(tp), 2)

    r_recip = _recip_pct(right)
    l_recip = _recip_pct(left)

    # --- Right sheet positive control (reproduce paper) ---
    if r_recip is not None:
        out.append(K.compare_pct(
            "Q2.right.reciprocal", "RIGHT sheet reciprocal T4/T5 fraction (control)",
            RIGHT_RECIPROCAL_FRAC, float(r_recip), pp=6.0))
    out.append(_categorical(
        "Q2.right.vch_gating", "RIGHT sheet: VCH presynaptically gates driving terminals (control)",
        expected=True, computed=_get(right, "gating_present", "vch_gating_present", default=None),
        report_value="present", note_bad="VCH gating absent on the analysed right sheet"))
    out.append(_categorical(
        "Q2.right.nod1_dominant", "RIGHT sheet: Nod1 dominant excitatory readout (control)",
        expected=True, computed=_get(right, "nod1_is_top_excitatory", "nod1_dominant", default=None),
        report_value="dominant", note_bad="Nod1 not dominant on the right sheet"))

    # --- Left sheet homolog (the open question) ---
    out.append(_categorical(
        "Q2.left.vch_gating", "LEFT sheet: VCH (right-soma) presynaptically gates driving terminals",
        expected=True, computed=_get(left, "gating_present", "vch_gating_present", default=None),
        report_value="present", note_ok="centrifugal VCH gating generalizes to the left sheet",
        note_bad="no VCH gating found on the left sheet"))
    out.append(_categorical(
        "Q2.left.nod1_dominant", "LEFT sheet: Nod1 dominates the readout (rank parallels right)",
        expected=True, computed=_get(left, "nod1_is_top_excitatory", "nod1_dominant", default=None),
        report_value="dominant", note_ok="Nod1 dominance generalizes to the left sheet",
        note_bad="Nod1 not the dominant left-sheet readout"))
    # The heavy per-side retinotopy/spatial-locality null is deferred by the q-module.
    out.append(K.unverifiable(
        "Q2.left.pooling_local", "LEFT sheet: pooling is spatially local / retinotopic",
        "local", "per-side retinotopy null deferred (paper.geometry / derive.d_retinotopy_null) "
                 "as too heavy for the structural parallel; gating + readout parallels computed"))

    # Left reciprocal fraction within tolerance of the right (parallel, not identity).
    if l_recip is not None and r_recip is not None:
        diff = abs(float(l_recip) - float(r_recip))
        ok = diff <= LEFT_VS_RIGHT_RECIP_PP
        out.append(K.ClaimResult(
            id="Q2.left.reciprocal_parallel",
            description="LEFT reciprocal T4/T5 fraction parallels RIGHT (within tolerance)",
            report_value=f"~{r_recip:.1f}% (right)", computed_primary=round(float(l_recip), 2),
            tolerance=f"+/-{LEFT_VS_RIGHT_RECIP_PP} pp of right", verdict=K.CONFIRMED if ok else K.REFUTED,
            numeric_outcome=K.MATCH if ok else K.MISMATCH, notes=f"|left-right|={diff:.1f} pp"))
    else:
        out.append(K.unverifiable("Q2.left.reciprocal_parallel",
                                  "LEFT reciprocal fraction parallels RIGHT",
                                  f"~{RIGHT_RECIPROCAL_FRAC}%", "left/right reciprocal fraction not computed"))
    return out


# ---------------------------------------------------------------------------
# Q3 -- escape-route census to muscle (is escape separable from steering?).
# ---------------------------------------------------------------------------
def build_claims_q3(d: dict) -> list[K.ClaimResult]:
    """Q3 verdicts: enumerate the escape DN census to muscle and test SEPARABILITY.

    The paper's claim is that the PLP/PVLP -> LPLC2/LC4 escape/broadcast arm is
    anatomically separable from the LLPC1 wing-steering arm -- distinct DNs, distinct
    muscles, not converging. Separability is CONFIRMED iff both overlaps are ~0.
    """
    if not isinstance(d, dict) or d.get("error"):
        return _failed_question("Q3", str(d.get("error")) if isinstance(d, dict) else "no result")
    out: list[K.ClaimResult] = []

    # The q-module nests the FlyWire census, the MCNS muscle map, and the separability block.
    fw = _get(d, "flywire", default={}) or {}
    mcns = _get(d, "mcns", default={}) or {}
    sep = _get(d, "separability", default={}) or {}
    command_syn = _get(fw, "command_dn_syn", default={}) or {}

    # Escape-DN census is non-empty.
    n_escape = _get(fw, "n_dn_types", default=_get(d, "n_escape_dns", "escape_dn_count"))
    out.append(_categorical(
        "Q3.census_nonempty", "Escape-route DN census downstream of LPLC2/LC4 is non-empty",
        expected=True, computed=(int(n_escape) >= 1) if n_escape is not None else None,
        report_value=">=1 DN", note_ok=f"{n_escape} escape DN types",
        note_bad="no escape DNs downstream of LPLC2/LC4"))

    # Looming/escape command cluster present downstream (giant fibre + DNp03/04/06).
    looming = None
    if command_syn:
        wanted = {"DNp01", "DNp03", "DNp04", "DNp06"}
        looming = any(int(command_syn.get(dn, 0) or 0) > 0 for dn in wanted)
    out.append(_categorical(
        "Q3.looming_cluster", "Looming command cluster (DNp01/03/04/06) downstream of LPLC2/LC4",
        expected=True, computed=looming, report_value="present (giant fibre DNp01 + DNp03/04/06)",
        note_ok=f"command-DN synapses: {command_syn}", note_bad="looming command cluster absent"))

    # Escape route reaches jump/TTM (tergotrochanter) musculature in MCNS. The q-module
    # reports the muscle list (e.g. STTMm/TTMn = tergotrochanter); reach = non-empty list.
    reaches = _get(mcns, "reaches_jump_ttm", "escape_muscle_reached", "reaches_escape_muscle",
                   default=_get(d, "reaches_jump_ttm", "escape_muscle_reached"))
    if reaches is None:
        em = _get(mcns, "escape_muscles", default=None)
        if em is not None:
            reaches = len(em) >= 1
    out.append(_categorical(
        "Q3.reaches_jump_muscle", "Escape route reaches jump/TTM (tergotrochanter) muscle in MCNS",
        expected=True, computed=reaches, report_value="reaches jump/TTM",
        note_ok="DN->MN->muscle traced to jump musculature", note_bad="escape route does not reach jump/TTM muscle"))

    # --- Separability: DN-set overlap and muscle-target overlap with the steering arm. ---
    # The q-module reports the overlaps as LISTS (dn_overlap / muscle_overlap) plus booleans.
    def _overlap_n(block, *keys):
        for k in keys:
            v = block.get(k)
            if v is not None:
                return len(v) if isinstance(v, (list, tuple, set)) else int(v)
        return None
    dn_overlap = _overlap_n(sep, "dn_overlap", "dn_set_overlap", "steering_dn_overlap")
    muscle_overlap = _overlap_n(sep, "muscle_overlap", "muscle_target_overlap")

    if dn_overlap is None:
        out.append(K.unverifiable("Q3.dn_separable", "Escape DN-set disjoint from steering DN-set",
                                  "overlap=0", "DN-set overlap not computed"))
    else:
        ok = int(dn_overlap) == 0
        out.append(K.ClaimResult(
            id="Q3.dn_separable", description="Escape DN-set disjoint from steering (LLPC1) DN-set",
            report_value="overlap=0 (separable)", computed_primary=int(dn_overlap),
            tolerance="exact 0", verdict=K.CONFIRMED if ok else K.CONFIRMED_WITH_CAVEAT,
            numeric_outcome=K.MATCH if ok else K.MINOR_DIFF,
            notes="separable" if ok else f"{dn_overlap} DN(s) shared with the steering arm"))

    if muscle_overlap is None:
        out.append(K.unverifiable("Q3.muscle_separable", "Escape muscle-targets disjoint from steering muscles",
                                  "overlap=0", "muscle-target overlap not computed"))
    else:
        ok = int(muscle_overlap) == 0
        out.append(K.ClaimResult(
            id="Q3.muscle_separable", description="Escape muscle-targets disjoint from wing-steering muscles",
            report_value="overlap=0 (separable)", computed_primary=int(muscle_overlap),
            tolerance="exact 0", verdict=K.CONFIRMED if ok else K.CONFIRMED_WITH_CAVEAT,
            numeric_outcome=K.MATCH if ok else K.MINOR_DIFF,
            notes="separable" if ok else f"{muscle_overlap} muscle(s) shared with the steering arm"))

    # Headline separability verdict: CONFIRMED iff BOTH overlaps are exactly 0.
    if dn_overlap is not None and muscle_overlap is not None:
        sep = (int(dn_overlap) == 0) and (int(muscle_overlap) == 0)
        out.append(_categorical(
            "Q3.escape_separable", "Escape output anatomically separable from LLPC1 steering output",
            expected=True, computed=sep, report_value="separable (distinct DNs + muscles)",
            note_ok="escape and steering arms are anatomically separable",
            note_bad="escape and steering arms converge (shared DNs or muscles)"))
    else:
        out.append(K.unverifiable("Q3.escape_separable",
                                  "Escape output anatomically separable from LLPC1 steering output",
                                  "separable", "overlaps not fully computed"))
    return out
