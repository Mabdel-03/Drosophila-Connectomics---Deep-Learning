"""Family L — the descending neurons of the FD3 cell (LPT42_Nod4 -> DN -> motor output).

Family K identified LPT42_Nod4 as Egelhaaf's FD3. This family characterises FD3's DOWNSTREAM
descending output: the descending neurons (DNs) it contacts directly and reaches through one
relay hop, and (in the male CNS) the motor neurons / muscles / motor systems those DNs drive.

Unlike the earlier families, FD3's DN set is a DISCOVERY rather than a number stated in a paper,
so the claims are of three kinds:

  EXISTENCE / SELF-CONSISTENCY  re-derivable connectome facts that must hold for the result to
                                be real (FD3 makes descending output; it reaches a known
                                figure-steering DN; both the direct and relay routes are
                                non-empty; the headline ranking reproduces across data sources).
  LITERATURE CROSS-CHECK        any FD3-targeted DN that is also one of the seven figure-steering
                                DNs the Figure-Ground paper resolved in the male CNS must
                                reproduce that DN's published wing-laterality / muscle targets
                                (Tables S14-S15; reused from family J's oracle).
  FUNCTIONAL PREDICTION         the behavioural reading ("FD3 drive is wing-steering-dominant")
                                is recorded UNVERIFIABLE with the measured motor-system % as the
                                connectome proxy, the repo's convention for a physiology claim.

Sources: Egelhaaf 1985 Part III (behavioural significance of the FD cells); the Figure-Ground
circuit paper Fig 5b / Tables S10-S15 (the FD1=Nod1 -> DNp26 steering arm this extends); Namiki
et al. 2018 (the descending-neuron atlas / nomenclature).
"""

from __future__ import annotations

from flyconn.motif import compare as K

# Published wing-laterality of the seven figure-steering DNs (Fig 6 / Table S14) and DNp26's top
# steering muscles (Table S15) — reused verbatim from family J as the cross-check oracle.
WING_SPECIFICITY = {
    "DNa04": "ipsilateral", "DNbe001": "bilateral", "DNge107": "bilateral",
    "DNbe005": "bilateral", "DNp26": "contralateral", "DNg32": "contralateral",
    "DNge094": "contralateral",
}
DNP26_TOP_MUSCLES = {"hg1", "i1", "hg2"}
PAGE = "Egelhaaf 1985 III; Figure-Ground paper Fig 5b / Tables S10-S15"


def _category(ipsi_frac):
    if ipsi_frac is None:
        return None
    if ipsi_frac >= 0.60:
        return "ipsilateral"
    if ipsi_frac <= 0.40:
        return "contralateral"
    return "bilateral"


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []
    direct = d.get("direct", {})
    relay = d.get("relay", {})
    motor = d.get("motor", {})

    # --- EXISTENCE / SELF-CONSISTENCY -----------------------------------------------------
    # FD3's direct descending output is a DISCOVERY, and its absolute size differs by data
    # source in a known, bidirectional way: the live track (synapses_nt_v1, no cleft threshold)
    # reports MORE synapses/partners than the frozen proofread offline dump (live ~216 syn / 34
    # DNs vs offline ~138 / 24 — the same drift documented for the identity family). So the size
    # is checked as an order-of-magnitude band that both tracks fall in, not an exact count; the
    # biology below (which DNs, with what laterality) is what carries the result and is
    # track-invariant.
    dsyn = direct.get("dn_syn", 0)
    ndn = direct.get("n_dns", 0)
    out.append(K.compare_categorical(
        "L.direct_dn_syn_scale", "FD3 (LPT42_Nod4) direct descending output is substantial "
        "(~100-300 synapses across data sources)",
        True, 80 <= dsyn <= 350, refuted_note=f"computed {dsyn} synapses"))
    out.append(K.compare_categorical(
        "L.direct_n_dns_scale", "FD3 contacts ~20-40 descending neurons directly (across sources)",
        True, 15 <= ndn <= 45, refuted_note=f"computed {ndn} DNs"))
    out.append(K.compare_categorical(
        "L.direct_nonempty", "FD3 makes a direct descending projection",
        True, dsyn > 0))
    out.append(K.compare_categorical(
        "L.relay_nonempty", "FD3 also reaches descending neurons through a relay hop",
        True, relay.get("dn_syn", 0) > 0))

    # FD3's strongest direct descending target is a known figure-steering DN (the convergence
    # with the FD1=Nod1 arm). We assert the top direct DN is one of the seven steering DNs.
    top_dn = direct.get("top_dn")
    out.append(K.compare_categorical(
        "L.top_direct_is_steering", "FD3's strongest direct descending target is a figure-steering DN",
        True, top_dn in WING_SPECIFICITY,
        refuted_note=f"top direct DN = {top_dn}"))
    # Specifically, FD3 and the FD1=Nod1 arm converge on DNp26 (the principal steering command).
    out.append(K.compare_categorical(
        "L.converges_on_dnp26", "FD3's top direct steering target is DNp26 (shared with the FD1 arm)",
        "DNp26", top_dn, refuted_note=f"top direct DN = {top_dn}"))

    # A meaningful share of FD3's direct descending output goes to known steering DNs.
    out.append(K.compare_pct(
        "L.direct_steering_frac", "FD3 direct descending output reaching known steering DNs (%)",
        58.0, direct.get("steering_frac", float("nan")), pp=20.0))

    # Direct and relay routes overlap on a set of figure-steering DNs (the result is coherent,
    # not two disjoint accidents).
    ov = d.get("overlap", {})
    out.append(K.compare_categorical(
        "L.routes_overlap", "Direct and relay routes share descending targets",
        True, ov.get("n_overlap", 0) >= 3,
        refuted_note=f"overlap = {ov.get('overlap')}"))

    # --- LITERATURE CROSS-CHECK (MaleCNS) -------------------------------------------------
    if not motor.get("available", False):
        out.append(K.unverifiable(
            "L.motor_unavailable", "DN -> motor-neuron mapping (male CNS)", "male-cns:v1.0",
            why=motor.get("reason", "MaleCNS data not available")))
    else:
        per_dn = motor.get("per_dn", {})
        # For every figure-steering DN that FD3 reaches and that has a MaleCNS body, its measured
        # wing-laterality must reproduce the paper's published category (Table S14).
        for dn, expected in WING_SPECIFICITY.items():
            row = per_dn.get(dn)
            if not row or not row.get("n_bodies") or row.get("ipsi_frac") is None:
                continue
            out.append(K.compare_categorical(
                f"L.{dn}.wing", f"{dn} wing-steering laterality reproduces the paper",
                expected, _category(row.get("ipsi_frac")),
                refuted_note=f"ipsi_frac={row.get('ipsi_frac')}"))
        # DNp26's top steering muscles include hg1/i1/hg2 (Table S15).
        dnp26 = per_dn.get("DNp26", {})
        top = set(dnp26.get("top_muscles", {}).keys())
        if dnp26.get("n_bodies"):
            out.append(K.compare_categorical(
                "L.dnp26_muscles", "DNp26's strongest steering muscles include hg1/i1/hg2",
                True, len(DNP26_TOP_MUSCLES & top) >= 2,
                refuted_note=f"computed top DNp26 muscles = {sorted(top)}"))

        # --- FUNCTIONAL PREDICTION (proxy) ------------------------------------------------
        dom = motor.get("dominant_motor_system")
        msp = motor.get("motor_system_pct", {})
        out.append(K.unverifiable(
            "L.steering_dominant",
            "FD3's descending output predominantly drives wing-steering (figure-tracking)",
            "behavioural prediction (Egelhaaf 1985 III)",
            why=("anatomical proxy: of FD3's descending drive that resolves to a male-CNS motor "
                 f"system, the largest share is '{dom}' "
                 f"(motor-system %, FD3-drive-weighted: {msp}).")))

    return out


# ---------------------------------------------------------------------------
# Aggregate summary (analogous to family K's identity_verdict): a one-glance read of the
# headline result for the report and the ledger.
# ---------------------------------------------------------------------------
def descending_summary(d: dict) -> dict:
    direct = d.get("direct", {})
    relay = d.get("relay", {})
    motor = d.get("motor", {})
    return {
        "top_direct_dn": direct.get("top_dn"),
        "n_direct_dns": direct.get("n_dns"),
        "direct_dn_syn": direct.get("dn_syn"),
        "direct_steering_frac": direct.get("steering_frac"),
        "n_relay_dns": relay.get("n_dns"),
        "relay_min_syn": d.get("relay_min_syn"),
        "dominant_motor_system": motor.get("dominant_motor_system") if motor.get("available") else None,
        "motor_system_pct": motor.get("motor_system_pct") if motor.get("available") else None,
    }
