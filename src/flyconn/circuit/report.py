"""Generic report emitters for a verified circuit.

Wraps ``motif.compare``'s JSON + Markdown emitters and adds the two things the closed-loop
campaign gate needs that the Stage-4 schema didn't surface:

  * flat top-level lists ``refuted_claims`` / ``unverifiable_claims`` (claim ids), so the
    campaign's JSON ``artifact_validators`` (``json_max_list_length: {refuted_claims: 0}``)
    can act without parsing ``claims[]``;
  * a ``coverage.json`` writer for the bilateral / chain-completeness gate.

Reuses ``figures._grouped_bar`` where a circuit wants a report-vs-computed bar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import write_json
from ..motif import compare as K


def ids_with_verdict(claims: list[K.ClaimResult], verdict: str) -> list[str]:
    return [c.id for c in claims if c.verdict == verdict]


def build_results(meta: dict[str, Any], claims: list[K.ClaimResult]) -> dict[str, Any]:
    """The verification_results.json payload, with the flat gate lists added."""
    payload = K.results_to_json(meta, claims)
    vc = K.verdict_counts(claims)
    payload["meta"]["verdict_counts"] = vc
    # Also surface verdict_counts + the flat lists at the TOP level so the campaign's
    # JSON artifact_validators (json_required_keys / json_max_list_length) can read them
    # without descending into meta.
    payload["verdict_counts"] = vc
    payload["refuted_claims"] = ids_with_verdict(claims, K.REFUTED)
    payload["unverifiable_claims"] = ids_with_verdict(claims, K.UNVERIFIABLE)
    payload["caveat_claims"] = ids_with_verdict(claims, K.CONFIRMED_WITH_CAVEAT)
    return payload


def write_results_json(path: str | Path, meta: dict[str, Any],
                       claims: list[K.ClaimResult], details: dict | None = None) -> dict:
    payload = build_results(meta, claims)
    if details is not None:
        payload["details"] = details
    write_json(path, payload)
    return payload


def write_coverage_json(path: str | Path, *, bilateral_coverage: float,
                        muscle_chain_complete: float, dn_to_mn_edges: int,
                        mn_to_muscle_edges: int, missing: list[dict] | None = None,
                        extra: dict | None = None) -> dict:
    """The coverage.json gate file (numeric thresholds the heartbeat validates)."""
    payload = {
        "bilateral_coverage": float(bilateral_coverage),
        "muscle_chain_complete": float(muscle_chain_complete),
        "dn_to_mn_edges": int(dn_to_mn_edges),
        "mn_to_muscle_edges": int(mn_to_muscle_edges),
        "missing": missing or [],
    }
    if extra:
        payload.update(extra)
    write_json(path, payload)
    return payload


def markdown_report(name: str, meta: dict[str, Any], claims: list[K.ClaimResult],
                    sections: list[tuple[str, str]] | None = None) -> str:
    """A generic VERIFICATION.md: header + verdict tally + headline table + sections."""
    vc = K.verdict_counts(claims)
    lines = [
        f"# Verification — {name}",
        "",
        f"- Datasets: {meta.get('datasets', {})}",
        f"- Seeds: {meta.get('seeds', [])}",
        f"- Tracks: {meta.get('tracks', {})}",
        "",
        "## Verdict tally",
        "",
        " · ".join(f"{k}: {v}" for k, v in sorted(vc.items())),
        "",
        "## Headline claims",
        "",
        K.summary_table_md(claims),
        "",
    ]
    for title, body in sections or []:
        lines += [f"## {title}", "", body, ""]
    return "\n".join(lines)


def write_markdown(path: str | Path, name: str, meta: dict[str, Any],
                   claims: list[K.ClaimResult],
                   sections: list[tuple[str, str]] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(markdown_report(name, meta, claims, sections))
