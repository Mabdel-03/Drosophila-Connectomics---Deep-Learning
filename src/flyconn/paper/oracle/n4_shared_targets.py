"""Stage-9 Channel N4 oracle: shared downstream convergence (bilateral integration locus)."""

from __future__ import annotations

from flyconn.motif import compare as K


def build_claims(d: dict) -> list[K.ClaimResult]:
    out: list[K.ClaimResult] = []

    out.append(K.compare_categorical(
        "N4.convergence_exists",
        f"Left and right figure readouts converge on shared downstream cells ({d.get('n_shared')})",
        True, bool((d.get("n_shared") or 0) > 0),
        refuted_note=f"shared targets = {d.get('n_shared')}"))

    out.append(K.compare_categorical(
        "N4.nod1_crosstalk",
        f"The two readouts cross-talk (Nod1 among shared targets: {[c['cell_type'] for c in d.get('nod1_crosstalk_cells', [])]})",
        True, bool(d.get("has_nod1_crosstalk")),
        refuted_note="no Nod1<->Nod1 cross-talk among shared targets"))

    out.append(K.compare_categorical(
        "N4.bridge_feedback",
        "A heterolateral bridge (H1) receives from both readouts (feedback loop)",
        True, bool(d.get("has_bridge_feedback")),
        refuted_note="no bridge-feedback cell among shared targets"))

    # neuromodulatory convergence is reported, not gated (it is a discovery, not a prediction)
    nm = d.get("neuromodulatory_convergence", [])
    out.append(K.unverifiable(
        "N4.neuromodulatory_convergence",
        f"Neuromodulatory convergence onto both readouts' shared pool ({[c['cell_type'] for c in nm]})",
        bool(nm),
        f"shared neuromodulatory cells: {[c['cell_type'] for c in nm]}; reported as a discovery "
        f"(octopaminergic convergence onto the bilateral figure pool), not a prediction."))

    # role breakdown summary
    roles = d.get("shared_by_role", {})
    out.append(K.unverifiable(
        "N4.role_breakdown", "Roles of the shared convergence cells", roles,
        f"shared-target roles: {roles}"))

    return out
