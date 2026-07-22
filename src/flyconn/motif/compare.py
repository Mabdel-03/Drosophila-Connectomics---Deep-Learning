"""Tolerance / verdict policy + JSON & Markdown emitters for the verification.

The scientific point of this module: distinguish "the report is wrong" from "our
offline data source differs from the live API the report used". Every numeric claim
is compared on a PRIMARY track (raw synapse table, the API analog) and, where it
exists, a SECONDARY track (the repo's proofread-only derived graph). A gap that is
small and in the expected direction is a data-source difference (MINOR_DIFF); a gap
that no source difference can explain, or a categorical mismatch, is a real
discrepancy (MISMATCH / REFUTED).

Verdict vocabulary used in the human report:
  CONFIRMED              - within tolerance on the appropriate track
  CONFIRMED_WITH_CAVEAT  - matches only after a named, demonstrated source adjustment
  REFUTED                - categorical mismatch, or numeric mismatch the report got wrong
  UNVERIFIABLE           - no ground truth in the offline dump

compare_* return the lower-level MATCH / MINOR_DIFF / MISMATCH on `verdict`; the
caller maps those to the report-level vocabulary with context (e.g. a MINOR_DIFF
attributable to proofreading drift becomes CONFIRMED_WITH_CAVEAT in the narrative).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Report-level verdicts (used in VERIFICATION.md headline table).
CONFIRMED = "CONFIRMED"
CONFIRMED_WITH_CAVEAT = "CONFIRMED_WITH_CAVEAT"
REFUTED = "REFUTED"
UNVERIFIABLE = "UNVERIFIABLE"

# Low-level numeric outcomes (from compare_count / compare_pct).
MATCH = "MATCH"
MINOR_DIFF = "MINOR_DIFF"
MISMATCH = "MISMATCH"


@dataclass
class ClaimResult:
    """One verified claim: the report value vs what we computed, with a verdict."""

    id: str
    description: str
    report_value: Any
    computed_primary: Any
    computed_secondary: Any = None
    tolerance: str = ""
    verdict: str = ""           # report-level (CONFIRMED / ... / UNVERIFIABLE)
    numeric_outcome: str = ""   # low-level (MATCH / MINOR_DIFF / MISMATCH), if applicable
    drift_explains: bool = False
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pct_diff(report: float, computed: float) -> float:
    if report == 0:
        return 0.0 if computed == 0 else float("inf")
    return 100.0 * (computed - report) / abs(report)


def compare_count(
    cid: str,
    desc: str,
    report: float,
    primary: float,
    secondary: float | None = None,
    *,
    rel: float = 0.05,
    abs_floor: int = 5,
    drift_dir: str = "down",
) -> ClaimResult:
    """Compare an integer-ish count. MATCH within max(rel*report, abs_floor).

    drift_dir: the direction a known data-source difference would push the offline
    value relative to the report. "down" = offline expected lower (frozen snapshot +
    proofread filtering both remove partners/synapses). A diff in that direction but
    out of tolerance -> MINOR_DIFF (data source differs); the opposite direction or a
    huge gap -> MISMATCH.
    """
    tol = max(rel * abs(report), abs_floor)
    diff = primary - report
    pdiff = _pct_diff(report, primary)
    drift_explains = (drift_dir == "down" and diff < 0) or (drift_dir == "up" and diff > 0)
    if abs(diff) <= tol:
        outcome, verdict = MATCH, CONFIRMED
    elif drift_explains:
        outcome, verdict = MINOR_DIFF, CONFIRMED_WITH_CAVEAT
    else:
        outcome, verdict = MISMATCH, REFUTED
    return ClaimResult(
        id=cid, description=desc, report_value=report,
        computed_primary=primary, computed_secondary=secondary,
        tolerance=f"+/-{rel:.0%} or +/-{abs_floor} (drift {drift_dir})",
        verdict=verdict, numeric_outcome=outcome, drift_explains=drift_explains,
        notes=f"primary diff {diff:+.0f} ({pdiff:+.1f}%)"
              + (f"; secondary={secondary}" if secondary is not None else ""),
    )


def compare_pct(
    cid: str,
    desc: str,
    report: float,
    primary: float,
    secondary: float | None = None,
    *,
    pp: float = 3.0,
) -> ClaimResult:
    """Compare two percentages; MATCH within ``pp`` percentage points."""
    diff = primary - report
    if abs(diff) <= pp:
        outcome, verdict = MATCH, CONFIRMED
    else:
        outcome, verdict = MISMATCH, REFUTED
    return ClaimResult(
        id=cid, description=desc, report_value=report,
        computed_primary=round(primary, 2), computed_secondary=secondary,
        tolerance=f"+/-{pp} pp", verdict=verdict, numeric_outcome=outcome,
        notes=f"diff {diff:+.1f} pp",
    )


def compare_ratio(
    cid: str, desc: str, report_lo: float, report_hi: float, primary: float,
) -> ClaimResult:
    """Confirm a ratio falls in the report's stated band (with +/-0.5x slack)."""
    ok = (report_lo - 0.5) <= primary <= (report_hi + 0.5)
    return ClaimResult(
        id=cid, description=desc, report_value=f"{report_lo}-{report_hi}x",
        computed_primary=round(primary, 2),
        tolerance=f"[{report_lo}, {report_hi}] +/-0.5x",
        verdict=CONFIRMED if ok else REFUTED,
        numeric_outcome=MATCH if ok else MISMATCH,
    )


def compare_categorical(
    cid: str, desc: str, report: Any, computed: Any, *, refuted_note: str = "",
) -> ClaimResult:
    """Exact categorical match (NT sign, hemisphere, dominant subtype identity)."""
    ok = report == computed
    return ClaimResult(
        id=cid, description=desc, report_value=report, computed_primary=computed,
        tolerance="exact", verdict=CONFIRMED if ok else REFUTED,
        numeric_outcome=MATCH if ok else MISMATCH,
        notes="" if ok else refuted_note,
    )


def check_table_sum(
    cid: str, desc: str, rows_total: float, stated_total: float, *, abs_floor: int = 5,
    rel: float = 0.05,
) -> ClaimResult:
    """Internal-consistency: do a table's rows sum to its stated total?

    Independent of any data source — an inconsistency here is unambiguously a report
    error. (The report's Table 3 uses "~" approx synapse counts, so allow tolerance.)
    """
    tol = max(rel * abs(stated_total), abs_floor)
    diff = rows_total - stated_total
    ok = abs(diff) <= tol
    return ClaimResult(
        id=cid, description=desc, report_value=stated_total, computed_primary=rows_total,
        tolerance=f"rows sum +/-{rel:.0%} or +/-{abs_floor}",
        verdict=CONFIRMED if ok else REFUTED,
        numeric_outcome=MATCH if ok else MISMATCH,
        notes=f"rows sum to {rows_total}, stated {stated_total} (diff {diff:+.0f})",
    )


def unverifiable(cid: str, desc: str, report_value: Any, why: str) -> ClaimResult:
    return ClaimResult(
        id=cid, description=desc, report_value=report_value, computed_primary=None,
        tolerance="n/a", verdict=UNVERIFIABLE, notes=why,
    )


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------
_VERDICT_ICON = {
    CONFIRMED: "PASS",
    CONFIRMED_WITH_CAVEAT: "PASS*",
    REFUTED: "FAIL",
    UNVERIFIABLE: "N/A",
}


def results_to_json(meta: dict[str, Any], claims: list[ClaimResult]) -> dict[str, Any]:
    return {"meta": meta, "claims": [c.to_dict() for c in claims]}


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def summary_table_md(claims: list[ClaimResult]) -> str:
    """Compact headline pass/fail table for the top of VERIFICATION.md."""
    lines = [
        "| Claim | Report | Computed (primary) | Secondary | Verdict |",
        "|---|---|---|---|---|",
    ]
    for c in claims:
        v = f"{_VERDICT_ICON.get(c.verdict, '?')} {c.verdict}"
        lines.append(
            f"| {c.description} | {_fmt(c.report_value)} | "
            f"{_fmt(c.computed_primary)} | {_fmt(c.computed_secondary)} | {v} |"
        )
    return "\n".join(lines)


def verdict_counts(claims: list[ClaimResult]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in claims:
        out[c.verdict] = out.get(c.verdict, 0) + 1
    return out
