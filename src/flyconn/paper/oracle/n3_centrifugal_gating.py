"""Stage-9 Channel N3 oracle: the centrifugal gating link and the soma-vs-arbor distinction."""

from __future__ import annotations

from flyconn.motif import compare as K


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []
    lat = d.get("gater_laterality", {})

    for t in ("VCH", "DCH"):
        g = lat.get(t, {})
        soma = g.get("soma_contra_pct")
        pos = g.get("position_cross_pct")
        # The soma-side measure calls the gater ~99% contralateral; the position measure shows it
        # does NOT cross (its whole arbor is in one lobe). Both are recorded; the result is that
        # the gater is a soma-displaced local cell, not an axonal bridge.
        out.append(K.compare_categorical(
            "N3.%s_not_axonal_bridge" % t,
            f"{t} is NOT an inter-hemispheric axonal bridge (soma {soma}%% contra but arbor does not cross; position {pos}%%)",
            False, bool(g.get("is_axonal_bridge")),
            refuted_note=f"{t} position-based crossing = {pos}% (>=50 would make it an axonal bridge)"))
        out.append(K.compare_categorical(
            "N3.%s_soma_arbor_swapped" % t,
            f"{t} soma and arbor are on opposite sides (displaced soma, contralateral arbor)",
            True, bool(g.get("soma_arbor_swapped")),
            refuted_note=f"{t} soma/arbor not consistently swapped"))

    # The two gaters do not couple directly.
    out.append(K.compare_categorical(
        "N3.gaters_not_directly_coupled",
        "The left and right gating cells do not synapse on each other directly",
        False, bool(d.get("gaters_directly_coupled")),
        refuted_note="a direct gater-gater synapse was found"))

    return out
