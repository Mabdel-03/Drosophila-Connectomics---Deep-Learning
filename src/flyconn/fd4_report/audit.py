"""Evidence audit for the FD4 search report.

Ties each reader-facing FD4 claim to its verdict and the derived-JSON quantity that supports it,
so a reviewer can check the narrative against the evidence. Written beside the generated TeX.
"""

from __future__ import annotations

from typing import Any


def build_audit(results: dict) -> dict[str, Any]:
    fam = results["families"]["FD4"]
    D = fam["derived"]
    claims = fam["claims"]
    meta = results.get("meta", {})
    track = meta.get("flywire_track", D.get("track", "unknown"))

    sp = D.get("nod1_split", {}); sc = D.get("candidate_screen", {}); cf = D.get("confidence", {})
    entries = []
    for c in claims:
        entries.append({
            "claim_id": c.get("id"), "claim": c.get("description"),
            "verdict": c.get("verdict"), "measured": c.get("computed_primary"),
            "notes": c.get("notes", ""),
        })

    headline = {
        "result": "HONEST_NULL",
        "statement": ("FD4 has no cleanly individually resolved connectome correlate in FlyWire "
                      "v783; it shares FD1's entire progressive, heterolateral, cholinergic, "
                      "noduli-group output class, and no separable FD4 cell or Nod1 sub-pair exists."),
        "confidence_point": cf.get("point_estimate"),
        "confidence_interval": cf.get("interval"),
        "identity_verdict": cf.get("identity_verdict"),
        "candidate_screen_survivors": sc.get("survivors"),
        "only_survivor_is_Nod1": sc.get("only_survivor_is_prog_type"),
        "nod1_homogeneous": sp.get("homogeneous"),
        "nod1_best_pairing_silhouette": sp.get("best_pairing", {}).get("silhouette"),
        "same_side_input_jaccard": sp.get("same_side_input_jaccard"),
        "cross_side_input_jaccard": sp.get("cross_side_input_jaccard"),
        "split_tracks": D.get("split_tracks"),
        "track": track,
    }
    from collections import Counter
    verdict_counts = dict(Counter(c.get("verdict") for c in claims))
    return {"headline": headline, "verdict_counts": verdict_counts,
            "n_claims": len(claims), "claims": entries}


def audit_markdown(audit: dict) -> str:
    h = audit["headline"]
    lines = ["# FD4 search evidence audit", "",
             f"**Result: {h['result']}.** {h['statement']}", "",
             f"- Confidence FD4 is individually resolved: {h['confidence_point']} "
             f"(range {h['confidence_interval']}); verdict {h['identity_verdict']}",
             f"- Candidate-screen survivors: {h['candidate_screen_survivors']} "
             f"(only Nod1 = {h['only_survivor_is_Nod1']})",
             f"- Nod1 homogeneous (no FD1+FD4 split): {h['nod1_homogeneous']} "
             f"(best pairing silhouette {h['nod1_best_pairing_silhouette']})",
             f"- Same-side / cross-side input Jaccard: {h['same_side_input_jaccard']} / "
             f"{h['cross_side_input_jaccard']}",
             f"- Homogeneity confirmed across tracks: {h['split_tracks']}",
             f"- Primary track: {h['track']}", "",
             f"## Claim ledger ({audit['n_claims']} claims: {audit['verdict_counts']})", ""]
    for e in audit["claims"]:
        lines.append(f"- **{e['claim_id']}** [{e['verdict']}]: {e['claim']}")
        if e.get("measured"):
            lines.append(f"    - measured: {e['measured']}")
        if e.get("notes"):
            lines.append(f"    - note: {e['notes']}")
    return "\n".join(lines)
