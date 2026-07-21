"""Family Z — interpretive / mechanistic / physiology-prediction claims.

The user asked that we attempt an anatomical PROXY for interpretive claims where one
exists, noting it is indirect, and mark UNVERIFIABLE only where no proxy is possible.
These claims are about *function* (what the circuit computes) or *predictions* (what a
silencing experiment would do); the connectome can supply at most an anatomical
plausibility proxy, never the physiological result.

Source: Discussion (p.8), S12 (p.26), S13 (p.26).
"""

from __future__ import annotations

from flyconn.motif import compare as K


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []

    # 1. Divisive normalization / proportional gain control. Proxy: a graded, reciprocal
    #    same-direction loop (excitation ~2.4x feedback, 912 reciprocal partners) is the
    #    wiring signature of proportional (not all-or-none) gain control. Anatomy is
    #    *consistent* with it but cannot distinguish divisive vs subtractive.
    px = d.get("gain_loop_proxy", {})
    out.append(K.ClaimResult(
        id="Z.divisive_norm", description="Circuit implements divisive normalization / gain control",
        report_value="divisive (untested)",
        computed_primary=f"reciprocal loop present (n={px.get('reciprocal_n')}, exc/inhib={px.get('ratio')})",
        tolerance="anatomical proxy only", verdict=K.UNVERIFIABLE,
        notes="PROXY: graded reciprocal same-direction VCH-T4/T5 loop is consistent with "
              "proportional gain control, but divisive-vs-subtractive is a dynamical "
              "property not determinable from static synapse counts (paper concurs, S9)."))

    # 2. "LLPC1 reads out the gated local signal" — proxy: LLPC1 carries BOTH local T4/T5
    #    motion AND lobula form on separate fields (family E) and projects centrally
    #    (family A/S7). Anatomy supports the readout role; the response property needs
    #    physiology.
    out.append(K.ClaimResult(
        id="Z.llpc1_readout", description="LLPC1 reads out the VCH-gated local motion residual",
        report_value="readout (untested)",
        computed_primary=f"dual-field motion+form integrator, central-projecting "
                         f"(motion {d.get('motion_pct')}% / form {d.get('form_pct')}%)",
        tolerance="anatomical proxy only", verdict=K.UNVERIFIABLE,
        notes="PROXY: the motion+form dual-dendrite, retinotopic, central-projecting "
              "architecture is the substrate for a readout; whether the RESPONSE encodes "
              "the residual requires LLPC1 imaging (paper S13)."))

    # 3. VCH-silencing prediction (S13). No connectomic test possible.
    out.append(K.unverifiable(
        "Z.vch_silencing", "Silencing VCH makes LLPC1 more raw-motion-like / less figure-selective",
        "physiology prediction (S13)",
        why="A perturbation-response prediction; not testable from the static connectome. "
            "Anatomical precondition (VCH presynaptically gates the T4a terminals driving "
            "LLPC1) IS confirmed (family E), which is the strongest available support."))

    # 4. LPi15 / PVLP011 silencing predictions (S13). No connectomic test.
    out.append(K.unverifiable(
        "Z.lpi15_pvlp_silencing", "Silencing LPi15/PVLP011 changes operating point/gain, not separation",
        "physiology prediction (S13)",
        why="Perturbation-response prediction. Anatomical precondition (LPi15 feed-forward "
            "opponent; PVLP011 recurrent on a distal compartment) is confirmed (families F/E)."))

    # 5. LPLC2/LC escape separability (S13). Proxy: the figure (direct/Nod) routes avoid
    #    jump/TTM, which lives on the broadcast route (family I). Anatomically separable.
    out.append(K.ClaimResult(
        id="Z.escape_separable", description="LPLC2/LC4 escape is separable from the LLPC1 course-control route",
        report_value="separable (prediction)",
        computed_primary="direct/Nod routes avoid jump/TTM; broadcast is the only jump bridge",
        tolerance="anatomical proxy only", verdict=K.UNVERIFIABLE,
        notes="PROXY: the channel separation IS in the anatomy (family I/J), but the "
              "behavioural double-dissociation needs perturbation experiments (S13)."))
    return out
