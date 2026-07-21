"""Stage-6 LaTeX report generator for the three open anatomical questions.

The CLI (``flyconn.openq`` run step) writes two JSON artifacts the verifier produced:

  * ``openq_results.json``  -- the claim ledger: ``meta`` (datasets, seeds, env, command,
    ``verdict_counts``) + ``claims[]`` (one ``ClaimResult`` dict each, with a ``q`` tag in
    ``extra`` so we can group rows under their question) + the flat gate lists.
  * ``openq_raw.json``      -- the per-question raw number dicts returned by ``run_q1`` /
    ``run_q2`` / ``run_q3`` (keys ``q1`` / ``q2`` / ``q3``), each a flat mapping of headline
    numbers we render as the RESULTS table.

This module turns those into a self-contained ``article`` (no project ``.sty``) and compiles
it to PDF. The WHY: the audit must be reproducible from the JSON alone -- a reviewer with the
two files (and no CAVE token / no MCNS feathers) can regenerate the exact PDF -- so the
generator reads only the JSON, never the live sources, and the compile is best-effort
(tectonic preferred, pdflatex fallback) so a missing TeX engine downgrades to "tex written,
not compiled" rather than crashing the run.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..io import read_json

# The miniforge bin that ships tectonic + pdflatex in this environment; added to the
# subprocess PATH as a fallback so the compile works even when the caller's PATH is bare.
_MINIFORGE_BIN = "/orcd/data/lhtsai/001/om2/mabdel03/miniforge3/bin"

# xcolor names -> the {HTML} definitions emitted in the preamble, matched to the verdict
# vocabulary (CONFIRMED green / REFUTED red / caveat orange / unverifiable gray).
_VERDICT_COLOR = {
    "CONFIRMED": "vgreen",
    "CONFIRMED_WITH_CAVEAT": "vorange",
    "REFUTED": "vred",
    "UNVERIFIABLE": "vgray",
}

# Human titles for each question block (the JSON only carries the ``q1``/``q2``/``q3`` tag).
_Q_TITLES = {
    "q1": "Q1 -- Single-Animal Closure of the LLPC1$\\rightarrow$Nod1$\\rightarrow$DNp26 Readout",
    "q2": "Q2 -- Bilateral / Four-Direction Generalization of the Sheet",
    "q3": "Q3 -- Escape-Route Census to Muscle and Its Separability",
}
_Q_QUESTIONS = {
    "q1": ("Does the figure-ground readout chain close inside a SINGLE animal? The published "
           "circuit is split across two specimens of different sex (female FlyWire FAFB for the "
           "optic-lobe/DN side, male MCNS for the motor side). We test whether LLPC1, Nod1 and "
           "DNp26 exist in MCNS and whether the LLPC1$\\rightarrow$Nod1$\\rightarrow$DNp26 "
           "wiring reproduces within MCNS alone, and report the fraction of the chain recovered "
           "in one animal."),
    "q2": ("Do the headline numbers GENERALIZE across hemisphere and direction, or are they "
           "specific to the shown exemplar (right LLPC1 sheet, front-to-back layer-a T4a)? We "
           "derive the LEFT LLPC1 sheet (and, where feasible, the other directional T4/T5 "
           "channels) and test, per hemisphere, whether (a) VCH presynaptically gates the "
           "driving terminals, (b) pooling is spatially local/retinotopic, and (c) Nod1 "
           "dominates the readout."),
    "q3": ("Is the PLP/PVLP$\\rightarrow$LPLC2/LC4 escape arm ANATOMICALLY SEPARABLE from the "
           "LLPC1 course-control / wing-steering output? We enumerate the full descending-neuron "
           "census downstream of LPLC2/LC4 and the broadcast cells (giant-fibre DNp01 plus "
           "DNp02/03/04/06), trace them to the muscle level in MCNS with the same "
           "DN$\\rightarrow$motor-neuron$\\rightarrow$muscle method as the Nod1 arm, and test "
           "whether the two outputs use distinct DNs and distinct muscles rather than converging."),
}
_Q_METHODS = {
    "q1": ("Both connectome layers are queried offline from the MCNS v1.0 public bulk feathers "
           "(\\texttt{body-annotations.feather}, \\texttt{connectome-weights.feather}; synapse "
           "counts as edge weights). Cell identity is resolved through the \\texttt{type} column; "
           "the LLPC1$\\rightarrow$Nod1 and Nod1$\\rightarrow$DNp26 edges are summed over all "
           "bodies of each type. The within-MCNS motif is compared row-for-row against the "
           "paper's FlyWire numbers (Nod1 the dominant excitatory readout; Nod1$\\rightarrow$"
           "DNp26 the strongest convergent steering target)."),
    "q2": ("The brain side is queried live from FlyWire FAFB v783 via CAVE "
           "(\\texttt{synapses\\_nt\\_v1}, no cleft threshold, reproducing the paper counts "
           "exactly). Positions are in nanometres. The LEFT sheet uses the RIGHT-soma "
           "VCH/DCH centrifugal cells (centrifugal cells cross, so the right VCH gates the left "
           "sheet). Retinotopic locality is assessed on synapse positions; each headline number "
           "is computed per hemisphere and compared to its right-sheet analogue."),
    "q3": ("DN identity and the broadcast/escape cluster are resolved in FlyWire FAFB v783 "
           "(live CAVE); the DN$\\rightarrow$motor-neuron$\\rightarrow$muscle trace is read "
           "offline from the MCNS v1.0 feathers using the same tracer as the Nod1 arm (motor "
           "neurons annotated by innervated muscle and \\texttt{somaSide}; wing-steering = "
           "subclass \\texttt{wm} excluding DLM/DVM power muscles). Separability is the overlap "
           "of the escape DN/muscle sets with the LLPC1$\\rightarrow$Nod1$\\rightarrow$DNp26 "
           "sets."),
}


# ---------------------------------------------------------------------------
# LaTeX escaping
# ---------------------------------------------------------------------------
# Order matters: backslash first so we don't re-escape the backslashes we just inserted.
_LATEX_SUB = [
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
]


def _esc(value: Any) -> str:
    """Escape LaTeX special chars in any interpolated value (numbers pass through)."""
    s = "" if value is None else str(value)
    for raw, rep in _LATEX_SUB:
        s = s.replace(raw, rep)
    return s


def _fmt_num(value: Any) -> str:
    """Render a raw-number-table value: thousands separators for ints, escaped otherwise."""
    if isinstance(value, bool):
        return _esc(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.3g}" if abs(value) >= 1 else f"{value:.3g}"
    return _esc(value)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def _verdict_tex(verdict: str) -> str:
    color = _VERDICT_COLOR.get(verdict, "vgray")
    return f"\\textcolor{{{color}}}{{\\textbf{{{_esc(verdict)}}}}}"


def _claims_for_q(claims: list[dict], q: str) -> list[dict]:
    """Claims tagged for question ``q`` (via ``extra.q``), falling back to an id prefix."""
    out = [c for c in claims if str(c.get("extra", {}).get("q", "")).lower() == q]
    if not out:
        out = [c for c in claims if str(c.get("id", "")).lower().startswith(q)]
    return out


def _truncate(s: str, limit: int = 90) -> str:
    """Cap a summary string at ``limit`` chars with a trailing ellipsis if it overflows."""
    return s if len(s) <= limit else s[: max(0, limit - 1)].rstrip(", ") + "…"


def _summarize_dict(val: dict[str, Any], limit: int = 90) -> str:
    """Flatten one level of a nested dict to a compact ``key=val, key=val, ...`` line."""
    return _truncate(", ".join(f"{k}={v}" for k, v in val.items()), limit)


def _summarize_list(val: list[Any], limit: int = 90, head: int = 3) -> str:
    """Render a list as ``[N items: a, b, c, ...]`` with the first few elements shown."""
    n = len(val)
    if n == 0:
        return "[0 items]"
    shown = ", ".join(str(x) for x in val[:head])
    suffix = ", ..." if n > head else ""
    return _truncate(f"[{n} items: {shown}{suffix}]", limit)


def _raw_table(raw_q: dict[str, Any]) -> str:
    """A two-column longtable of the question's headline numbers (key / value).

    Nested ``dict``/``list`` values are collapsed to a compact one-line summary (and the
    redundant ``claims`` ledger -- already shown in colour in the Verdict table -- is dropped)
    so the value column never overflows the page width.
    """
    if not raw_q:
        return "\\emph{No raw numbers were recorded for this question.}\n"
    lines = [
        "\\begin{longtable}{@{}p{0.30\\textwidth}p{0.62\\textwidth}@{}}",
        "\\toprule",
        "\\textbf{Quantity} & \\textbf{Value} \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for key, val in raw_q.items():
        if key == "claims":  # redundant with the colour-coded Verdict table; it overflows
            continue
        if isinstance(val, dict):
            cell = _esc(_summarize_dict(val))
        elif isinstance(val, list):
            cell = _esc(_summarize_list(val))
        else:
            cell = _fmt_num(val)
        lines.append(f"{_esc(key)} & {cell} \\\\")
    lines += ["\\bottomrule", "\\end{longtable}"]
    return "\n".join(lines)


def _claims_table(claims: list[dict]) -> str:
    """The colour-coded ClaimResult ledger for one question."""
    if not claims:
        return "\\emph{No claim rows were recorded for this question.}\n"
    lines = [
        "\\begin{longtable}{@{}llp{0.30\\textwidth}rrl@{}}",
        "\\toprule",
        "\\textbf{ID} & & \\textbf{Description} & \\textbf{Report} & "
        "\\textbf{Computed} & \\textbf{Verdict} \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for c in claims:
        comp = c.get("computed_primary")
        sec = c.get("computed_secondary")
        comp_s = _fmt_num(comp) if comp is not None else "--"
        if sec is not None:
            comp_s += f" ({_fmt_num(sec)})"
        lines.append(
            f"\\texttt{{{_esc(c.get('id'))}}} & & {_esc(c.get('description'))} & "
            f"{_fmt_num(c.get('report_value'))} & {comp_s} & "
            f"{_verdict_tex(str(c.get('verdict', '')))} \\\\"
        )
    lines += ["\\bottomrule", "\\end{longtable}"]
    return "\n".join(lines)


def _question_section(q: str, raw: dict, claims: list[dict]) -> str:
    """One full per-question section: question, methods, results, verdicts."""
    qclaims = _claims_for_q(claims, q)
    verdicts = sorted({str(c.get("verdict", "")) for c in qclaims if c.get("verdict")})
    verdict_line = (
        " ".join(_verdict_tex(v) for v in verdicts)
        if verdicts else "\\textcolor{vgray}{\\textbf{UNVERIFIABLE}}"
    )
    return "\n".join([
        f"\\section{{{_Q_TITLES.get(q, _esc(q))}}}",
        "",
        "\\subsection*{Question}",
        _Q_QUESTIONS.get(q, ""),
        "",
        "\\subsection*{Methods}",
        _Q_METHODS.get(q, ""),
        "",
        "\\subsection*{Results}",
        _raw_table(raw.get(q, {})),
        "",
        "\\subsection*{Verdict}",
        f"\\noindent Overall: {verdict_line}\\par\\medskip",
        _claims_table(qclaims),
        "",
    ])


def _tally_table(verdict_counts: dict[str, int]) -> str:
    """The final verdict-tally summary table (colour-coded count per verdict)."""
    order = ["CONFIRMED", "CONFIRMED_WITH_CAVEAT", "REFUTED", "UNVERIFIABLE"]
    rows = [v for v in order if v in verdict_counts]
    rows += [v for v in verdict_counts if v not in order]
    lines = [
        "\\begin{longtable}{@{}lr@{}}",
        "\\toprule",
        "\\textbf{Verdict} & \\textbf{Count} \\\\",
        "\\midrule",
        "\\endhead",
    ]
    total = 0
    for v in rows:
        n = int(verdict_counts.get(v, 0))
        total += n
        lines.append(f"{_verdict_tex(v)} & {n} \\\\")
    lines += ["\\midrule", f"\\textbf{{Total}} & \\textbf{{{total}}} \\\\",
              "\\bottomrule", "\\end{longtable}"]
    return "\n".join(lines)


def _abstract(meta: dict, claims: list[dict]) -> str:
    """One-paragraph abstract summarising the three verdicts."""
    parts = []
    for q in ("q1", "q2", "q3"):
        verdicts = sorted({str(c.get("verdict", "")) for c in _claims_for_q(claims, q)
                           if c.get("verdict")})
        v = ", ".join(verdicts) if verdicts else "UNVERIFIABLE"
        label = {"q1": "single-animal closure", "q2": "bilateral generalization",
                 "q3": "escape-route separability"}[q]
        parts.append(f"{label}: {_esc(v)}")
    vc = meta.get("verdict_counts", {})
    tally = ", ".join(f"{_esc(k)}={v}" for k, v in sorted(vc.items())) or "n/a"
    return (
        "We resolve three open anatomical questions left by the figure-ground circuit report "
        "(VCH$\\rightarrow$T4/T5$\\rightarrow$LLPC1$\\rightarrow$Nod1$\\rightarrow$DNp26"
        "$\\rightarrow$wing-steering) end to end from connectome data, each yielding a neutral, "
        "falsifiable verdict. "
        + "; ".join(parts) + ". "
        f"Across all claims the tally is {tally}. "
        "Every number is re-derived from the FlyWire FAFB v783 live CAVE synapse table "
        "(\\texttt{synapses\\_nt\\_v1}, no cleft threshold) and the male whole-CNS (MCNS) v1.0 "
        "offline bulk feathers; negatives are reported faithfully."
    )


def _provenance_section(meta: dict) -> str:
    ds = meta.get("datasets", {})
    seeds = meta.get("seeds", [])
    command = meta.get("command", "python -m flyconn.openq run && python -m flyconn.openq report")
    env = meta.get("env", {})
    materializations = meta.get("materializations", ds)
    rows = [
        ("FlyWire FAFB", str(materializations.get("fafb", "v783 live CAVE, synapses_nt_v1, "
                                                  "no cleft threshold"))),
        ("Male CNS (MCNS)", str(materializations.get("mcns", "v1.0 offline bulk feathers"))),
        ("Seeds", ", ".join(map(str, seeds)) if seeds else "see per-question methods"),
        ("Python env", str(env.get("python", "flyconn_cave conda env"))),
        ("PYTHONPATH", str(env.get("pythonpath", "src"))),
        ("FLYCONN_DATA_ROOT", str(env.get("data_root",
                                          "/orcd/data/tpoggio/001/mabdel03/connectome_data"))),
    ]
    table = ["\\begin{longtable}{@{}lp{0.66\\textwidth}@{}}", "\\toprule",
             "\\textbf{Item} & \\textbf{Value} \\\\", "\\midrule", "\\endhead"]
    for k, v in rows:
        table.append(f"{_esc(k)} & {_esc(v)} \\\\")
    table += ["\\bottomrule", "\\end{longtable}"]
    return "\n".join([
        "\\section{Provenance \\& Reproducibility}",
        "",
        "All materializations and the run environment:",
        "",
        "\n".join(table),
        "",
        "\\noindent\\textbf{Exact command:}",
        f"\\begin{{quote}}\\ttfamily {_esc(command)}\\end{{quote}}",
        "",
        "\\noindent\\textbf{Token-free re-run.} This PDF is regenerated from "
        "\\texttt{openq\\_results.json} and \\texttt{openq\\_raw.json} alone "
        "(\\texttt{python -m flyconn.openq report}); the report step reads neither the live "
        "CAVE API nor the MCNS feathers, so a reviewer with only the two JSON files and a TeX "
        "engine reproduces the identical document with no FlyWire token and no network.",
        "",
    ])


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------
_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{xcolor}
\usepackage{amsmath}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55em}

\definecolor{vgreen}{HTML}{1B7837}
\definecolor{vorange}{HTML}{C97A0A}
\definecolor{vred}{HTML}{C0392B}
\definecolor{vgray}{HTML}{555555}

\title{Closing Three Open Anatomical Questions in the Figure-Ground Circuit}
\author{Stage 6 -- Open Questions}
\date{\today}
"""


def build_tex(results: dict, raw: dict) -> str:
    """Render the full standalone LaTeX document string from the two JSON payloads."""
    meta = results.get("meta", {})
    claims = results.get("claims", [])
    vc = meta.get("verdict_counts") or {}
    if not vc:  # recompute defensively if the runner didn't stamp it into meta
        for c in claims:
            v = str(c.get("verdict", ""))
            vc[v] = vc.get(v, 0) + 1

    body = [
        _PREAMBLE,
        "\\begin{document}",
        "\\maketitle",
        "",
        "\\begin{abstract}",
        _abstract(meta, claims),
        "\\end{abstract}",
        "",
    ]
    for q in ("q1", "q2", "q3"):
        body.append(_question_section(q, raw, claims))
    body.append("\\section{Verdict Tally}\n")
    body.append(_tally_table(vc))
    body.append("")
    body.append(_provenance_section(meta))
    body.append("\\end{document}")
    return "\n".join(body)


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
def _engine() -> tuple[str, str] | None:
    """Locate a TeX engine: tectonic preferred, else pdflatex. Returns (kind, path)."""
    for kind in ("tectonic", "pdflatex"):
        path = shutil.which(kind)
        if not path:
            cand = Path(_MINIFORGE_BIN) / kind
            path = str(cand) if cand.exists() else None
        if path:
            return kind, path
    return None


def _compile(tex_path: Path) -> tuple[bool, str]:
    """Compile the .tex to PDF (best effort). Returns (compiled, combined_log)."""
    eng = _engine()
    if eng is None:
        return False, "no TeX engine found (tried tectonic, pdflatex; miniforge bin fallback)"
    kind, path = eng
    env_path = f"{_MINIFORGE_BIN}:{__import__('os').environ.get('PATH', '')}"
    env = {**__import__('os').environ, "PATH": env_path}
    out_dir = tex_path.parent
    # Run FROM the tex's directory and pass the BARE filename, so the stage-dir name
    # containing spaces ("6 - Open Questions") never has to survive arg-splitting or a
    # relative --outdir. Output (.pdf/.log) lands in cwd == out_dir, which is what we want.
    tex_name = tex_path.name
    logs: list[str] = []
    try:
        if kind == "tectonic":
            cmd = [path, "--keep-logs", tex_name]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                  cwd=str(out_dir), timeout=300)
            logs.append(f"$ (cd {out_dir!s}) {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
            ok = proc.returncode == 0
        else:  # pdflatex twice for refs/longtable widths
            ok = True
            for _ in range(2):
                cmd = [path, "-interaction=nonstopmode", "-halt-on-error", tex_name]
                proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                      cwd=str(out_dir), timeout=300)
                logs.append(f"$ (cd {out_dir!s}) {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
                ok = ok and proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}\n" + "\n".join(logs)
    pdf = tex_path.with_suffix(".pdf")
    return (ok and pdf.exists()), "\n\n".join(logs)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_report(stage_dir: str | Path = "6 - Open Questions") -> dict[str, Any]:
    """Read the JSON ledgers, emit ``open_questions_report.tex``, compile to PDF.

    Best effort: if no TeX engine is available or the compile fails, the .tex is still
    written and a ``.compile_error.txt`` is left next to it, but no exception is raised --
    the caller gets ``compiled=False`` and ``pdf=None`` so a missing engine degrades the
    artifact rather than the run.
    """
    stage = Path(stage_dir)
    results = read_json(stage / "openq_results.json")
    try:
        raw = read_json(stage / "openq_raw.json")
    except FileNotFoundError:
        raw = {}

    tex_path = stage / "open_questions_report.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(build_tex(results, raw))

    compiled, log = _compile(tex_path)
    pdf_path = tex_path.with_suffix(".pdf")
    err_path = tex_path.with_suffix(".compile_error.txt")
    if not compiled:
        err_path.write_text(log)
    elif err_path.exists():
        err_path.unlink()

    return {
        "tex": str(tex_path),
        "pdf": str(pdf_path) if compiled else None,
        "compiled": bool(compiled),
    }
