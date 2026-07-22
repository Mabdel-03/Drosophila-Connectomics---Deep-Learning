"""Stage-6 CLI: resolve the three open anatomical questions end to end.

  python -m flyconn.openq run [--only q1,q2,q3] [--offline] [--n-perm N]

Each question runs its own derivation (``flyconn.openq.q{1,2,3}_*``), the oracle turns
the resulting number dict into ``K.ClaimResult`` verdicts, and the runner writes the
campaign gate files into '6 - Open Questions/'. A question that raises does NOT sink the
others: it becomes a single UNVERIFIABLE claim carrying its traceback, and the CLI still
exits 0 (only a true infra error -- e.g. an unwritable data root -- is fatal), so the
campaign gate can always inspect the verdict counts.

The q-modules are imported INSIDE the runner, never at module load: Agent A authors them
in parallel, so deferring the import keeps the two files' edits from racing.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from ..circuit import report as R
from ..io import write_json
from ..motif import compare as K
from . import oracle as O

REPO = Path(__file__).resolve().parents[3]

# (question id, builder, run_* attr, module path, which source(s) the run_* expects).
# The three questions live in different animals: Q1 reads the male MCNS client (M); Q2 is
# pure-FlyWire (the fw source); Q3 spans both (FlyWire brain-side census -> MCNS muscle).
_QUESTIONS = (
    ("q1", O.build_claims_q1, "run_q1", ".q1_single_animal", "mcns"),
    ("q2", O.build_claims_q2, "run_q2", ".q2_bilateral", "fafb"),
    ("q3", O.build_claims_q3, "run_q3", ".q3_escape", "both"),
)


def _stage_dir() -> Path:
    d = REPO / "6 - Open Questions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _check_data_root_writable() -> None:
    """Fail fast (true infra error) if the resolved data root is not writable.

    Like Stage 5: the q-modules populate the CAVE cache + intermediates under
    FLYCONN_DATA_ROOT (group volume; personal scratch is over quota). Catch a quota /
    permission problem here with an actionable message instead of deep in a parquet write.
    """
    from ..paths import data_root  # lazy
    root = data_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        raise SystemExit(
            f"[flyconn.openq] data root not writable: {root}\n"
            f"  ({type(exc).__name__}: {exc})\n"
            "  Set FLYCONN_DATA_ROOT to a writable location (Stage 6 uses the group volume\n"
            "  /orcd/data/tpoggio/001/mabdel03/connectome_data; personal scratch is over quota)."
        ) from exc


def _call_run(fn, args: tuple, *, n_perm: int):
    """Call a run_* with its positional source(s), tolerating signature variants.

    Each agent's run_* may or may not accept ``n_perm`` and may take one source object or
    two (Q3 spans FlyWire + MCNS). We try the richest signature first and degrade.
    """
    for kw in ({"n_perm": n_perm}, {}):
        for a in (args, args[:1]):  # full source tuple, then just the first source
            try:
                return fn(*a, **kw)
            except TypeError:
                continue
    # last resort: surface the real TypeError instead of swallowing it
    return fn(*args, n_perm=n_perm)


def _run_one(qid: str, run_attr: str, mod_path: str, sources: tuple, *, n_perm: int) -> dict:
    """Invoke one run_qN, returning its raw dict (or an ``{'error': traceback}`` dict).

    The q-module is imported here (not at top level). Any failure -- missing module,
    bad signature, or an exception inside the derivation -- is captured as an error dict
    so the oracle can emit a single UNVERIFIABLE claim and the other questions proceed.
    """
    try:
        from importlib import import_module
        mod = import_module(mod_path, package=__package__)
        fn = getattr(mod, run_attr)
        res = _call_run(fn, sources, n_perm=n_perm)
        return res if isinstance(res, dict) else {"value": res}
    except Exception:  # noqa: BLE001 -- isolate per-question failures
        return {"error": traceback.format_exc()}


def _sources_for(kind: str, src, M) -> tuple:
    """The positional source(s) a question's run_* expects, by animal."""
    if kind == "mcns":
        return (M,)
    if kind == "both":
        return (src, M)
    return (src,)  # "fafb"


def cmd_run(args) -> int:
    from ..paper.fw_access import make_source
    from ..paper.malecns import client as M  # offline MCNS flat-file access (module = M)

    t0 = time.time()
    only = None
    if args.only:
        only = {q.strip().lower() for q in args.only.split(",") if q.strip()}
    prefer = "offline" if args.offline else "auto"
    src = make_source(prefer=prefer)

    stage = _stage_dir()
    claims: list[K.ClaimResult] = []
    raw: dict[str, dict] = {}
    details: dict[str, dict] = {}

    for qid, builder, run_attr, mod_path, kind in _QUESTIONS:
        if only is not None and qid not in only:
            continue
        res = _run_one(qid, run_attr, mod_path, _sources_for(kind, src, M), n_perm=args.n_perm)
        raw[qid] = res
        claims += builder(res)
        details[qid] = res

    # Coverage metrics pulled from the question dicts (UNVERIFIABLE-safe defaults).
    q2 = raw.get("q2", {}) or {}
    q3 = raw.get("q3", {}) or {}
    sides = q2.get("sides_resolved")
    if sides is None:
        # fraction of {left,right} sides that produced a usable sheet block
        resolved = sum(1 for k in ("right", "left") if isinstance(q2.get(k), dict) and q2.get(k))
        sides = resolved / 2.0
    bilateral_coverage = float(sides)
    # Q3 nests the MCNS muscle map; the escape route "completes" iff it reaches >=1 escape
    # muscle (tergotrochanter STTMm/TTMn) AND the separability block was computed.
    q3_mcns = (q3.get("mcns") or {}) if isinstance(q3, dict) else {}
    q3_sep = (q3.get("separability") or {}) if isinstance(q3, dict) else {}
    escape_muscles = q3_mcns.get("escape_muscles")
    reaches = (q3.get("reaches_jump_ttm", q3.get("escape_muscle_reached"))
               if isinstance(q3, dict) else None)
    if reaches is None and escape_muscles is not None:
        reaches = len(escape_muscles) >= 1
    muscle_chain_complete = 1.0 if reaches else 0.0
    dn_to_mn_edges = int(q3_mcns.get("n_escape_motor_edges",
                                     q3.get("dn_to_mn_edges", q3.get("n_dn_mn_edges", 0))) or 0)
    mn_to_muscle_edges = int(len(escape_muscles) if escape_muscles is not None
                             else q3.get("mn_to_muscle_edges", q3.get("n_mn_muscle_edges", 0)) or 0)
    missing = list(q3.get("missing", []) or []) if isinstance(q3, dict) else []

    vc = K.verdict_counts(claims)
    meta = {
        "stage": "6 - Open Questions",
        "datasets": {"fafb": {"materialization": 783}, "mcns": {"materialization": "v1.0"}},
        "seeds": ["LLPC1", "Nod1", "DNp26", "LPLC2", "LC4", "DNp01", "DNp03", "DNp04", "DNp06"],
        "tracks": {"q1": "single-animal MCNS", "q2": "bilateral FlyWire (left vs right control)",
                   "q3": "escape census FlyWire -> MCNS muscle"},
        "n_perm": int(args.n_perm),
        "source": getattr(type(src), "__name__", str(type(src))),
        "verdict_counts": vc,
        "questions_run": [q for q in raw],
        "queried_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t0, 1),
    }

    payload = R.write_results_json(stage / "openq_results.json", meta, claims, details=details)
    R.write_coverage_json(stage / "coverage.json",
                          bilateral_coverage=bilateral_coverage,
                          muscle_chain_complete=muscle_chain_complete,
                          dn_to_mn_edges=dn_to_mn_edges,
                          mn_to_muscle_edges=mn_to_muscle_edges,
                          missing=missing)
    write_json(stage / "openq_raw.json", raw)

    sections = [
        ("Q1 — single-animal closure (MCNS)", _q_section_md(raw.get("q1", {}), claims, "Q1")),
        ("Q2 — bilateral / 4-direction generalization", _q_section_md(raw.get("q2", {}), claims, "Q2")),
        ("Q3 — escape-route census to muscle", _q_section_md(raw.get("q3", {}), claims, "Q3")),
    ]
    R.write_markdown(stage / "OPENQ.md", "Open Questions", meta, claims, sections=sections)

    print(f"[openq run] verdicts={vc} refuted={payload['refuted_claims']} "
          f"bilateral={bilateral_coverage:.2f} chain={muscle_chain_complete:.2f} "
          f"-> {stage}")
    return 0  # exit 0 even with UNVERIFIABLE claims; only infra errors are fatal


def _q_section_md(raw: dict, claims: list[K.ClaimResult], prefix: str) -> str:
    """One Markdown block per question: its verdicts plus the raw numbers."""
    rows = [c for c in claims if c.id.startswith(prefix + ".")]
    lines = [K.summary_table_md(rows) if rows else "_no claims_", ""]
    if raw.get("error"):
        lines += ["", "**question errored:**", "", "```", raw["error"].strip(), "```"]
    elif raw:
        lines += ["", "Raw fields:", ""]
        for k, v in raw.items():
            if isinstance(v, dict):
                v = "{" + ", ".join(f"{kk}={vv}" for kk, vv in v.items()) + "}"
            lines.append(f"- `{k}` = {v}")
    return "\n".join(lines)


def cmd_report(args) -> int:
    """Render + compile the LaTeX report from the existing ledgers in the stage dir."""
    from . import report as RP
    out = RP.build_report(_stage_dir())
    print(f"[openq report] tex={out['tex']} pdf={out['pdf']} compiled={out['compiled']}")
    return 0  # report failure to compile is not fatal (the .tex is still emitted)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="flyconn.openq", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="resolve Q1/Q2/Q3, emit verdicts + coverage + OPENQ.md")
    pr.add_argument("--only", type=str, default=None,
                    help="comma list of questions to run (e.g. q1,q3); default all")
    pr.add_argument("--offline", action="store_true",
                    help="force the offline FlyWire source (no live CAVE)")
    pr.add_argument("--n-perm", type=int, default=500,
                    help="permutations for any null model inside the questions")
    pr.set_defaults(func=cmd_run)

    prep = sub.add_parser("report", help="render the LaTeX report from the ledgers and compile to PDF")
    prep.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    _check_data_root_writable()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
