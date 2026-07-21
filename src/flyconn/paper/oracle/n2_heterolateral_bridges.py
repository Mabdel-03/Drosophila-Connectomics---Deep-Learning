"""Stage-9 Channel N2 oracle: heterolateral bridges and the Egelhaaf regressive-inhibition test.

The central biological validation of the stage. Egelhaaf (1985) showed FD1 receives
contralateral REGRESSIVE (back-to-front) inhibition. In the connectome this predicts that the
inhibitory heterolateral bridges onto the FD1/Nod1 figure pathway read lobula-plate layer-b
(regressive). This oracle confirms that prediction and that the established bridges (H1, H2)
are recovered.
"""

from __future__ import annotations

from flyconn.motif import compare as K

EXPECTED_BRIDGES = {"H1", "H2"}  # the established heterolateral bridges onto the figure pathway


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []

    bridges = {b["bridge_type"]: b for b in d.get("bridges", [])}

    # The discovery recovers the known heterolateral bridges onto the figure pathway.
    found = EXPECTED_BRIDGES & set(bridges)
    out.append(K.compare_categorical(
        "N2.bridges_found",
        f"Heterolateral bridges onto the figure pathway are present ({sorted(found)})",
        True, bool(len(found) >= 1),
        refuted_note=f"bridges found = {sorted(bridges)}; expected to include {sorted(EXPECTED_BRIDGES)}"))

    # H1 and H2 specifically, with their direction.
    for name in ("H1", "H2"):
        b = bridges.get(name)
        if b is None:
            continue
        lay = b.get("input_layer", {})
        dom = lay.get("dominant_layer")
        out.append(K.compare_categorical(
            "N2.%s_regressive" % name,
            f"{name} (the contralateral bridge) reads REGRESSIVE motion (layer-b)",
            "b", dom,
            refuted_note=f"{name} dominant input layer = {dom} (expected b/regressive)"))

    # THE EGELHAAF TEST: the DOMINANT horizontal-motion inhibitory bridge onto the FD1/Nod1
    # pathway is regressive-tuned, as Egelhaaf (1985) inferred for FD1's contralateral inhibition.
    dom = d.get("dominant_horizontal_inhibitory_bridge")
    dom_dir = d.get("dominant_horizontal_inhibitory_direction")
    dom_n = d.get("dominant_horizontal_inhibitory_cross_syn")
    out.append(K.compare_categorical(
        "N2.egelhaaf_regressive_inhibition",
        f"The dominant horizontal-motion inhibitory bridge onto the FD1/Nod1 pathway is REGRESSIVE "
        f"(Egelhaaf 1985): {dom} ({dom_dir}, {dom_n} synapses)",
        True, bool(d.get("egelhaaf_regressive_inhibition_holds")),
        refuted_note=f"dominant horizontal inhibitory bridge {dom} is {dom_dir}, not regressive"))

    # The connectome extends Egelhaaf: weaker VERTICAL-motion and central/neuromodulatory
    # contralateral inputs onto the figure pathway, undetectable in a horizontal-motion paradigm.
    vert = d.get("inhibitory_vertical_bridges", [])
    out.append(K.unverifiable(
        "N2.vertical_inhibition_extension",
        f"Additional contralateral inhibitory bridges reading VERTICAL motion ({vert})",
        bool(vert),
        f"beyond the horizontal regressive inhibition Egelhaaf described, the connectome shows "
        f"weaker contralateral inhibitory bridges tuned to vertical motion: {vert}. A "
        f"horizontal-motion paradigm could not have detected these."))

    # Report the full bridge inventory with sign and direction.
    inventory = {b["bridge_type"]: {"nt": b["nt"], "dir": (b.get("input_layer") or {}).get("dominant_direction"),
                                    "cross_syn": b["cross_syn_onto_machinery"]}
                 for b in d.get("bridges", [])}
    out.append(K.unverifiable(
        "N2.bridge_inventory", "Inventory of heterolateral bridges onto the figure pathway",
        len(inventory), f"bridges (type -> nt/direction/crossing synapses): {inventory}"))

    return out
