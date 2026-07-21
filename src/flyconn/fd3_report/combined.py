"""Build the single combined FD3 circuit report: Identity + Inputs + Outputs in one PDF.

Stitches the three FD3 reports, ``builder`` (Family K, identity), ``input`` (Family P, the
afferent pathway), and ``descending`` (Family L, the output arm), into one document with a
\\part divider per section, under a single preamble. Reuses each module's ``build_tex`` so the
prose and figures stay in one place, and the shared ``_compile`` engine. Writes
``7 - FD3 Identification/fd3_full_circuit_report.{tex,pdf}``.

The body of each report (everything between ``\\begin{document}`` and ``\\end{document}``, with
that report's own ``\\title/\\author/\\date/\\maketitle`` block removed) is extracted and placed
under a ``\\part``. Figures are shared: each part's ``\\includegraphics`` paths already point at
``figures/`` in the same stage directory, so the combined document reuses the PNGs the three
individual builds render.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..io import read_json
from . import audit
from . import builder, descending, functional, input as input_report
from .builder import _PREAMBLE, _compile


def _drop_braced_command(text: str, command: str) -> str:
    """Remove ``\\command{...}`` including a BALANCED (possibly nested) brace group.

    A plain non-greedy regex breaks on nested braces (e.g. ``\\title{\\textbf{...}}`` stops at
    the first ``}``), so we scan for the matching close brace by depth counting.
    """
    marker = "\\" + command + "{"
    while (i := text.find(marker)) != -1:
        j = i + len(marker)
        depth = 1
        while j < len(text) and depth:
            depth += (text[j] == "{") - (text[j] == "}")
            j += 1
        text = text[:i] + text[j:]
    return text


def _body(full_tex: str) -> str:
    """Extract the document body, dropping the preamble and the per-report title block."""
    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", full_tex, flags=re.S)
    body = m.group(1) if m else full_tex
    # Remove the report's own title/author/date block (one \part header replaces it). Use a
    # balanced-brace remover so nested braces inside \title{...} do not leave dangling braces.
    for cmd in ("title", "author", "date"):
        body = _drop_braced_command(body, cmd)
    body = body.replace(r"\maketitle", "").replace(r"\tableofcontents", "")
    body = re.sub(r"\\begin\{thebibliography\}\{.*?\}.*?\\end\{thebibliography\}",
                  "", body, flags=re.S)
    return body.strip()


def _bibliography() -> str:
    items = [
        r"\bibitem{egelhaaf1985a} Egelhaaf M. (1985) On the neuronal basis of "
        r"figure-ground discrimination by relative motion in the visual system of the fly. "
        r"I. Behavioural constraints imposed on the neuronal network and the role of the "
        r"optomotor system. \textit{Biological Cybernetics} 52:123--140. "
        r"doi:10.1007/BF00364003.",
        r"\bibitem{egelhaaf1985b} Egelhaaf M. (1985) On the neuronal basis of "
        r"figure-ground discrimination by relative motion in the visual system of the fly. "
        r"II. Figure-detection cells, a new class of visual interneurones. "
        r"\textit{Biological Cybernetics} 52:195--209. doi:10.1007/BF00339948.",
        r"\bibitem{egelhaaf1985c} Egelhaaf M. (1985) On the neuronal basis of "
        r"figure-ground discrimination by relative motion in the visual system of the fly. "
        r"III. Possible input circuitries and behavioural significance of the FD-cells. "
        r"\textit{Biological Cybernetics} 52:267--280. doi:10.1007/BF00336983.",
        r"\bibitem{rp1979} Reichardt W., Poggio T. (1979) Figure-ground discrimination by "
        r"relative movement in the visual system of the fly. Part I. Experimental results. "
        r"\textit{Biological Cybernetics} 35:81--100. doi:10.1007/BF00337434.",
        r"\bibitem{fd1989} Fischbach K.-F., Dittrich A.P.M. (1989) The optic lobe of "
        r"\textit{Drosophila melanogaster}. I. A Golgi analysis of wild-type structure. "
        r"\textit{Cell and Tissue Research} 258:441--475. doi:10.1007/BF00218858.",
        r"\bibitem{maisak2013} Maisak M.S. et al. (2013) A directional tuning map of "
        r"\textit{Drosophila} elementary motion detectors. \textit{Nature} 500:212--216. "
        r"doi:10.1038/nature12320.",
        r"\bibitem{hardie1989} Hardie R.C. (1989) A histamine-activated chloride channel "
        r"involved in neurotransmission at a photoreceptor synapse. \textit{Nature} "
        r"339:704--706. doi:10.1038/339704a0.",
        r"\bibitem{namiki2018} Namiki S., Dickinson M.H., Wong A.M., Korff W., Card G.M. "
        r"(2018) The functional organization of descending sensory-motor pathways in "
        r"\textit{Drosophila}. \textit{eLife} 7:e34272. doi:10.7554/eLife.34272.",
        r"\bibitem{dorkenwald2024} Dorkenwald S. et al. (2024) Neuronal wiring diagram of "
        r"an adult brain. \textit{Nature} 634:124--138. doi:10.1038/s41586-024-07558-y.",
        r"\bibitem{schlegel2024} Schlegel P. et al. (2024) Whole-brain annotation and "
        r"multi-connectome cell typing of \textit{Drosophila}. \textit{Nature} "
        r"634:139--152. doi:10.1038/s41586-024-07686-5.",
        r"\bibitem{codex} FlyWire Codex. Cell-type and annotation portal. "
        r"\url{https://codex.flywire.ai}.",
    ]
    return (r"\begin{thebibliography}{11}" "\n" + "\n".join(items) + "\n"
            r"\end{thebibliography}")


def build_tex(k_results: dict, p_results: dict, l_results: dict,
              func_results: dict | None = None) -> str:
    parts = [
        _PREAMBLE,
        r"\begin{document}",
        r"\title{\textbf{Connectomic identification and sensorimotor partners of the FD3 "
        r"figure-detection cell "
        r"(\ttt{LPT42\_Nod4})}}",
        r"\author{Stage 7 FD3 Identification}",
        r"\date{\today}",
        r"\maketitle",
        r"\tableofcontents",
        r"\part{Identity: FD3 is \texttt{LPT42\_Nod4}}",
        _body(builder.build_tex(k_results, None)),
        r"\clearpage\part{Inputs: what drives FD3}",
        _body(input_report.build_tex(p_results)),
        r"\clearpage\part{Outputs: descending and motor-system partners}",
        _body(descending.build_tex(l_results)),
    ]
    # Optional 4th part: the comprehensive census + the named functional circuit (Families Q/R/S).
    if func_results is not None:
        parts += [
            r"\clearpage\part{The functional figure-ground circuit within FD3}",
            _body(functional.build_tex(func_results)),
        ]
    parts += [r"\clearpage", _bibliography(), r"\end{document}"]
    return "\n\n".join(parts)


def build_report(stage_dir: str | Path = "7 - FD3 Identification",
                 results_path: str | Path = "5 - Paper Verification/verification_results.json",
                 *, k_path: str | Path | None = None, p_path: str | Path | None = None,
                 l_path: str | Path | None = None, src=None, meta=None) -> dict[str, Any]:
    """Render every part's figures, then compile the stitched document.

    The three families may live in separate JSON files (each ``--only`` run overwrites the shared
    ``verification_results.json`` with just its family), so ``k_path``/``p_path``/``l_path`` can
    point at per-family copies; each defaults to ``results_path``. A file missing a family is
    skipped with a note rather than aborting the whole report.
    """
    stage = Path(stage_dir)
    fig_dir = stage / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def _load(path, fam):
        try:
            r = read_json(Path(path or results_path))
        except FileNotFoundError:
            return None
        if "families" in r and fam not in r["families"]:
            return None
        return r

    k_results = _load(k_path, "K")
    p_results = _load(p_path, "P")
    l_results = _load(l_path, "L")
    missing = [f for f, r in (("K", k_results), ("P", p_results), ("L", l_results)) if r is None]
    if missing:
        print(f"[combined] missing families {missing}; run paper_verify for them first")
        return {"tex": None, "pdf": None, "compiled": False, "missing": missing}

    # Optional 4th part: the functional circuit (Families Q/R/S). Included only if Q,R,S are all
    # present in the results JSON; otherwise the combined report is the original 3 parts.
    func_results = None
    try:
        r = read_json(Path(results_path))
        if "families" in r and all(f in r["families"] for f in ("Q", "R", "S")):
            func_results = functional._augment_with_census(r, src, meta)
    except FileNotFoundError:
        pass

    # Render each part's figures into the shared figures dir (reused by the combined doc).
    builder._render_figures(k_results, None, fig_dir)
    input_report._render_figures(p_results, fig_dir, src=src, meta=meta)
    descending._render_figures(l_results, fig_dir)
    if func_results is not None:
        functional._render_figures(func_results, fig_dir, src=src, meta=meta)
    audit.write_audit(k_results, p_results, l_results, stage)

    tex_path = stage / "fd3_full_circuit_report.tex"
    tex_path.write_text(build_tex(k_results, p_results, l_results, func_results))
    compiled, log = _compile(tex_path)
    err = tex_path.with_suffix(".compile_error.txt")
    if not compiled:
        err.write_text(log)
    elif err.exists():
        err.unlink()
    return {"tex": str(tex_path),
            "pdf": str(tex_path.with_suffix(".pdf")) if compiled else None,
            "compiled": bool(compiled)}
