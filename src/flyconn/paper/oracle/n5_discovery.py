"""Stage-9 Channel N5 oracle: systematic discovery + the soma-vs-position artifact contrast."""

from __future__ import annotations

from flyconn.motif import compare as K

# The dominant genuine axonal bridge onto the figure cells (the largest contralateral input).
EXPECTED_GENUINE = {"H1"}
# The soma-displaced cells the position test must remove: the centrifugal gaters appear high in
# the soma-side ranking (their displaced soma makes their local arbor look "contralateral") but
# vanish from the position-based ranking, because their arbor does not actually cross.
EXPECTED_ARTIFACT = {"VCH", "DCH"}


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []

    genuine = set(d.get("genuine_bridge_types", []))
    artifact = set(d.get("artifact_types_soma_only", []))

    # The discovery recovers the dominant genuine bridge (H1) among a broader bridge set.
    found_known = EXPECTED_GENUINE & genuine
    out.append(K.compare_categorical(
        "N5.recovers_known_bridges",
        f"Systematic discovery recovers the dominant axonal bridge H1 among {len(genuine)} genuine bridges",
        True, bool(found_known),
        refuted_note=f"genuine bridges found = {sorted(genuine)}; expected to include {sorted(EXPECTED_GENUINE)}"))

    # The position-based definition removes the centrifugal-gater soma-side artifact.
    artifact_removed = EXPECTED_ARTIFACT & artifact
    out.append(K.compare_categorical(
        "N5.artifact_removed_by_position",
        f"The centrifugal gaters (VCH/DCH) are a soma-side artifact removed by the position test ({sorted(artifact_removed)})",
        True, bool(len(artifact_removed) >= 1),
        refuted_note=f"VCH/DCH not in the soma-only artifact set; found {sorted(artifact)}"))

    # Report the magnitude of inter-hemispheric input.
    out.append(K.unverifiable(
        "N5.position_cross_fraction",
        f"Position-based inter-hemispheric input fraction onto the circuit ({d.get('position_cross_frac')}%)",
        d.get("position_cross_frac"),
        f"of {d.get('total_input_syn')} input synapses onto the circuit, "
        f"{d.get('position_cross_syn')} ({d.get('position_cross_frac')}%) are genuine "
        f"position-based inter-hemispheric crossings. Top genuine bridges: "
        f"{d.get('bridges_position', [])[:8]}"))

    return out
