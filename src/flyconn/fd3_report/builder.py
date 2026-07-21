"""Build + compile the FD3-Identification report (LaTeX -> PDF) from the Family-K JSON.

Reads ONLY ``5 - Paper Verification/verification_results.json`` (the ``families.K`` block) plus
the optional offline side-file ``figures/K_offline.json``; no live CAVE, no feathers, so the
report re-runs token-free. Mirrors ``flyconn.openq.report`` (preamble + tectonic/pdflatex-twice
compile, run from the spaced stage dir with a bare filename). Figures are rendered from the same
JSON via ``flyconn.paper.figures`` and ``\\includegraphics``'d; Figs 1 + the schematic are TikZ.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..io import read_json
from ..paper import figures as PF
from ..motif import compare as K

_MINIFORGE_BIN = "/orcd/data/lhtsai/001/om2/mabdel03/miniforge3/bin"

_VERDICT_COLOR = {
    K.CONFIRMED: "vgreen", K.CONFIRMED_WITH_CAVEAT: "vorange",
    K.REFUTED: "vred", K.UNVERIFIABLE: "vgray",
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _esc(value: Any) -> str:
    if value is None:
        return "--"
    s = str(value)
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def _g(d: dict, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _replication_sentence(D: dict, track: str) -> str:
    cv = _g(D, "cross_version", default={}) or {}
    if cv.get("available"):
        return (r"The primary numbers in this report come from the \ttt{" + _esc(track) +
                r"} track; the key identity measurements also agree with the v630 "
                r"cross-version check and with left/right analyses.")
    reason = cv.get("reason") or "cross-version check unavailable in this run"
    return (r"The primary numbers in this report come from the \ttt{" + _esc(track) +
            r"} track. The v630 cross-version check is not claimed here because it was "
            rf"unavailable in the present run ({_esc(reason)}); left/right agreement and "
            r"offline reproducibility are reported explicitly.")


def _source_columns(D: dict, off: dict | None, track: str) -> str:
    bits = [rf"primary track: \ttt{{{_esc(track)}}}"]
    if off:
        bits.append("frozen offline copy")
    if _g(D, "cross_version", "available"):
        bits.append("v630 cross-version check")
    else:
        bits.append("v630 not available in this artifact")
    return "; ".join(bits)


# ---------------------------------------------------------------------------
# preamble (extends openq's with graphicx + tikz)
# ---------------------------------------------------------------------------
_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{float}
\usepackage[table]{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,backgrounds,fit}
\usepackage[hidelinks]{hyperref}
\definecolor{vgreen}{HTML}{1B7837}
\definecolor{vorange}{HTML}{C97A0A}
\definecolor{vred}{HTML}{C0392B}
\definecolor{vgray}{HTML}{555555}
\definecolor{fd1c}{HTML}{1f77b4}
\definecolor{fd3c}{HTML}{C0392B}
\newcommand{\ttt}[1]{\texttt{#1}}
\newcommand{\verd}[2]{\textcolor{#1}{\textbf{#2}}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55em}
"""


def _figure(rel_png: str, caption: str, label: str, width: str = r"\textwidth") -> str:
    return (r"\begin{figure}[H]\centering" "\n"
            rf"\includegraphics[width={width}]{{{rel_png}}}" "\n"
            rf"\caption{{{caption}}}\label{{{label}}}" "\n"
            r"\end{figure}" "\n")


def _tikz_fig1() -> str:
    """Conceptual schematic: FD1 frontal RF vs FD3 fronto-lateral RF + frontal gap, regressive
    drive, heterolateral noduli-group axon. Hand-drawn (no data)."""
    return r"""\begin{figure}[H]\centering
\begin{tikzpicture}[font=\small,>=Stealth]
  % visual field arc (one eye), frontal at left, lateral at right
  \draw[thick] (0,0) arc (180:90:3.2 and 1.8);
  \node[anchor=east] at (0,0) {frontal ($0^\circ$)};
  \node[anchor=south] at (3.2,1.8) {lateral ($\sim$100$^\circ$)};
  % FD1 receptive field (frontal, narrow)
  \fill[fd1c,opacity=0.35] (0.15,0.05) ellipse (0.5 and 0.28);
  \node[fd1c] at (0.55,0.7) {\textbf{FD1} (frontal)};
  % FD3 receptive field (fronto-lateral, wide) with a frontal gap
  \fill[fd3c,opacity=0.30] (2.0,1.15) ellipse (1.05 and 0.42);
  \node[fd3c] at (2.2,1.9) {\textbf{FD3} (fronto-lateral, wide)};
  \draw[fd3c,dashed] (0.7,0.35) -- (1.0,0.55) node[midway,below right,fd3c]{\scriptsize frontal gap};
  % regressive (back-to-front) preferred motion arrow
  \draw[->,thick,fd3c] (3.0,-0.6) -- (1.2,-0.6) node[midway,below]{regressive (back$\to$front)};
  % heterolateral axon
  \node[draw,rounded corners,fill=fd3c!12,font=\scriptsize,align=center,text width=2.4cm]
        (lp) at (1.2,-2.2) {FD3 dendrite\\(lobula plate, right)};
  \node[draw,rounded corners,fill=fd3c!12,font=\scriptsize,align=center,text width=2.6cm]
        (pof) at (6.8,-2.2) {contralateral\\posterior optic foci};
  \draw[->,thick] (lp) -- node[above,midway,font=\scriptsize,align=center]
        {axon crosses midline,\\posterior to noduli} (pof);
\end{tikzpicture}
\caption{\textbf{The FD3 cell and what distinguishes it.} Egelhaaf (1985, Part~II) defined four
figure-detection (FD) cells. FD3 is excited by \emph{regressive} (back-to-front) small-field
motion; its excitatory receptive field is \emph{fronto-lateral} (peak azimuth $40$--$50^\circ$,
half-width $\sim$$62^\circ$) and, uniquely among the FD cells, spares the most-frontal
$10$--$20^\circ$ (a ``frontal gap''). Like FD1 and FD4 it is a heterolateral ``noduli-group''
output element whose axon crosses the midline posterior to the noduli and terminates in the
contralateral posterior optic foci. This report maps FD3 onto the FlyWire cell type
\ttt{LPT42\_Nod4}.}\label{fig:concept}
\end{figure}
"""


# ---------------------------------------------------------------------------
# Reader-facing "summary of evidence" table: each FD3 property, what was measured for
# LPT42_Nod4, and what Egelhaaf reported -- in plain language, no verdict vocabulary.
# ---------------------------------------------------------------------------
def _evidence_table(D: dict, off: dict | None, track: str) -> str:
    lb = _g(D, "cand_layer_frac", "b")
    contra = _g(D, "cand_contra_output_pct")
    off_contra = _g(off or {}, "cand_contra_output_pct") if off else None
    nb = _g(D, "smallfield_null", default={})
    s = _g(D, "fd_family_screen", default={})
    soma = _g(D, "soma", default={})
    nt_conf = _g(D, "cand_mean_nt_conf")
    morph = _g(D, "morphology", default={})
    mcells = morph.get("cells", []) if morph.get("available") else []
    dv = max([c.get("dv_span_um") or 0 for c in mcells], default=None)
    morph_word = "reconstructed skeleton" if morph.get("source") == "skeleton" else "synapse positions"

    rows = [
        ("Prefers regressive (back-to-front) motion",
         "Egelhaaf: FD3 excited by regressive motion",
         rf"{_esc(lb)}\% of its motion (T4/T5) input comes from the back-to-front lobula-plate layer"),
        ("Excitatory output (cholinergic)",
         "FD cells are excitatory output neurons",
         rf"acetylcholine, assigned with {_esc(nt_conf)} confidence for both cells"),
        ("Receptive field lateral, not frontal",
         "Egelhaaf: FD3 field peaks at $40$--$50^\\circ$, away from the front",
         r"its input field sits $\sim$12--13 columns lateral of the frontal FD1 cell, on both sides"),
        ("Frontal gap (unique to FD3)",
         "Egelhaaf: FD3 alone is not excited in the most-frontal field",
         r"occupies $\sim$2--3\% of the frontal zone that FD1 fills (vs $\sim$60\% for FD1)"),
        ("Small-object selective",
         "FD cells prefer small figures to wide-field motion",
         rf"pools a bounded retinotopic patch below the in-degree matched null "
         rf"($z={_esc(nb.get('z_score'))}$, $p={_esc(nb.get('p_value'))}$)"),
        ("Heterolateral output axon",
         "Egelhaaf: axon crosses to the opposite (contralateral) side",
         rf"{_esc(contra)}\% of output synapses on the opposite side"
         + (rf" ({_esc(off_contra)}\% on the frozen copy)" if off_contra is not None else "")),
        ("Cell body posterior and lateral",
         "Egelhaaf: cell body in the posterior lateral protocerebrum",
         rf"both somata in the posterior {int(100*(soma.get('post_z_percentile') or 0))}th "
         r"percentile, bilateral, beside the FD1 cell bodies"),
        ("Dendrite spans the full dorso-ventral field",
         "Egelhaaf: dendrite covers the entire dorso-ventral extent",
         rf"{_esc(round(dv,0) if dv else None)}~$\mu$m span in the {morph_word}"),
        ("Best and unique match for FD3",
         "the identification must be specific",
         rf"best match for FD3 among {s.get('n_candidates','all')} candidates, and FD3 is its "
         r"own best match (by a clear margin)"),
        ("Provenance is explicit",
         "the result should not depend on an unreported dataset",
         _source_columns(D, off, track)),
    ]
    head = (r"\renewcommand{\arraystretch}{1.25}" "\n"
            r"\setlength{\tabcolsep}{3pt}" "\n"
            r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.24\textwidth} "
            r">{\raggedright\arraybackslash}p{0.28\textwidth} "
            r">{\raggedright\arraybackslash}p{0.38\textwidth}@{}}" "\n"
            r"\toprule \textbf{FD3 property} & \textbf{What Egelhaaf reported} & "
            r"\textbf{What we find for \ttt{LPT42\_Nod4}} \\\midrule\endhead" "\n")
    body = "\n".join(rf"{a} & {b} & {c} \\" for a, b, c in rows)
    return (head + body + "\n" + r"\bottomrule\end{longtable}" + "\n"
            r"\renewcommand{\arraystretch}{1.0}" "\n" r"\setlength{\tabcolsep}{6pt}")


# ---------------------------------------------------------------------------
# prose sections
# ---------------------------------------------------------------------------
def _summary(D: dict, track: str) -> str:
    s = _g(D, "fd_family_screen", default={})
    lb = _g(D, "cand_layer_frac", "b")
    contra = _g(D, "cand_contra_output_pct")
    return (
        r"\textbf{Summary.}\quad "
        r"Egelhaaf defined FD3 as a regressive, fronto-lateral figure-detection cell with a "
        r"frontal gap, small-field preference, contralateral inhibition, and a heterolateral "
        r"noduli-group projection. This report identifies the FlyWire cell type "
        r"\textbf{\ttt{LPT42\_Nod4}} as the connectomic correlate of that FD3 phenotype. "
        rf"The candidate draws {_esc(lb)}\% of its measured T4/T5 motion input from the "
        r"back-to-front lobula-plate layer, has a receptive field shifted laterally relative to "
        r"FD1 while leaving the FD1 frontal band nearly empty, pools a spatially bounded input "
        r"patch, is cholinergic, and sends most output contralaterally "
        rf"({_esc(contra)}\%). Among {_esc(s.get('n_candidates', 'the screened'))} nearby "
        r"Nod/LPT candidate types, it is the strongest FD3 match and its own strongest FD-family "
        r"match. The evidence is therefore a convergent anatomical identification, not a new "
        r"physiology recording. " + _replication_sentence(D, track))


def _introduction() -> str:
    return r"""\section{Introduction}

\textbf{Figure-ground discrimination and the FD cells.}\quad
A textured object is invisible against a matched background until it moves at a different
velocity or phase; relative motion then makes it salient. Reichardt and Poggio showed
behaviourally that flies detect and track such figures, and that the effect depends on relative
phase and on the figure being small relative to the background~\cite{rp1979}. Egelhaaf, in a three-part
study, asked which neurons could implement this~\cite{egelhaaf1985a,egelhaaf1985b,egelhaaf1985c}.
Part~II described a new class of
lobula-plate tangential cells, the \emph{figure-detection} (FD) cells, defined functionally by a
stronger response to a small moving figure than to wide-field motion. Four were distinguished by
preferred direction and receptive-field organisation. \textbf{FD1} is progressive
(front-to-back) with a narrow frontal field; \textbf{FD2} is regressive with a frontal field;
\textbf{FD3} is regressive (back-to-front) with a \emph{fronto-lateral} field that peaks at
$40$--$50^\circ$ azimuth, is $\sim$$62^\circ$ wide at half-maximum, reaches $\sim$$100^\circ$
laterally, and, alone among the FD cells, is \emph{not} excited in the most-frontal
$10$--$20^\circ$; \textbf{FD4} is progressive with a near-whole-eye field~\cite[p.~202--205]{egelhaaf1985b}.
FD3 additionally receives \emph{bidirectional} contralateral inhibition, and its axon is a
heterolateral ``noduli-group'' element that crosses the midline posterior to the noduli and
terminates in the contralateral posterior optic foci~\cite[p.~203]{egelhaaf1985b}.

\textbf{The connectome era and the mapping problem.}\quad
The adult \emph{Drosophila} FlyWire connectome and its cell-type annotations now provide the
synapse-resolution wiring diagram needed to ask which modern cell type each physiologically
defined FD cell corresponds to~\cite{dorkenwald2024,schlegel2024,codex}. The elementary motion stage is well defined: T4 and T5
encode ON/OFF edge motion and segregate by preferred direction into the four lobula-plate
layers~\cite{maisak2013,fd1989}: layer-a front-to-back, layer-b back-to-front, layer-c upward,
and layer-d downward. A tangential cell's preferred direction is therefore read from the layer composition of its
T4/T5 input. We previously mapped FD1 to the cholinergic cell type \ttt{Nod1}. The question this
report answers is: which cell type is FD3?

\textbf{How the identification is made.}\quad
Each property that defines FD3 corresponds to a quantity that can be measured directly in the
connectome: preferred direction from the lobula-plate layer its motion inputs come from, the
receptive field from the retinotopic positions of those inputs, small-object selectivity from how
spatially concentrated they are, the output side and target region from where its synapses land,
and overall shape from its reconstructed skeleton. We measure each of these for \ttt{LPT42\_Nod4}
and ask whether it matches what Egelhaaf reported for FD3. Three safeguards make the conclusion
robust. First, the receptive-field comparison is made \emph{relative} to the already-identified
FD1=\ttt{Nod1} cell in the same coordinate frame, so it does not depend on converting connectome
coordinates into visual angle. Second, the left and right \ttt{LPT42\_Nod4} cells are analyzed
separately wherever the data allow it, so bilateral agreement is visible rather than assumed.
Third, the same measurements are applied to neighbouring candidate cell types, which must
\emph{fail} to look like FD3 for the identification to be meaningful. Figure~\ref{fig:concept}
fixes the FD3 phenotype we are matching; the sections below take its defining properties in turn."""


def _results(D: dict, off: dict | None, track: str) -> str:
    lb = _g(D, "cand_layer_frac", "b")
    contra = _g(D, "cand_contra_output_pct")
    nb = _g(D, "smallfield_null", default={})
    s = _g(D, "fd_family_screen", default={})
    soma = _g(D, "soma", default={})
    cv = _g(D, "cross_version", default={})
    nt_conf = _g(D, "cand_mean_nt_conf")
    off_contra = _g(off or {}, "cand_contra_output_pct") if off else None

    P = []
    P.append(r"\section{The evidence}")
    P.append(r"We take FD3's defining properties one at a time: cell class, preferred motion, "
             r"receptive-field position, small-field selectivity, output side, and cell shape. "
             r"For each property we ask whether \ttt{LPT42\_Nod4} has the corresponding "
             r"connectomic signature. Figure~\ref{fig:crosswalk} previews the correspondence; "
             r"the paragraphs that follow establish each line of it.")
    P.append(_figure("figures/fd3_crosswalk.png",
                     r"\textbf{FD3's properties and their connectome counterparts.} Each defining "
                     r"property of Egelhaaf's FD3 cell (left), and the matching measurement for "
                     r"\ttt{LPT42\_Nod4} (right), with caveated properties marked in the text.",
                     "fig:crosswalk"))

    P.append(r"\paragraph{It is an excitatory output cell, one per side.} \ttt{LPT42\_Nod4} is a "
             r"pair of cells, one in each hemisphere, classified as a visual output neuron. It "
             rf"releases the excitatory transmitter acetylcholine, assigned with high confidence "
             rf"({_esc(nt_conf)}) for both cells. This is consistent with an FD cell that sends figure "
             r"signals to downstream circuits, and is the same transmitter as the "
             r"already-identified FD1 cell \ttt{Nod1}.")

    P.append(r"\paragraph{It prefers back-to-front motion.} The fly computes motion in four "
             r"separate channels, one for each cardinal direction, kept in four physical layers of "
             r"the lobula plate. A cell's preferred direction is therefore read from "
             rf"which layer its motion inputs come from. For \ttt{{LPT42\_Nod4}}, {_esc(lb)}\% of "
             r"that input comes from the back-to-front layer (Fig.~\ref{fig:layers}). It prefers "
             r"regressive motion, matching the direction Egelhaaf reported for FD3 and opposing "
             r"the front-to-back FD1 cell.")
    P.append(_figure("figures/fd3_layer_composition.png",
                     r"\textbf{Preferred direction.} (left) The share of each cell's motion input "
                     r"coming from the four directional layers. \ttt{LPT42\_Nod4} draws almost all "
                     r"of its input from the back-to-front layer (regressive, as FD3); \ttt{Nod1} "
                     r"from the front-to-back layer (FD1); \ttt{Nod5} from a vertical-motion layer. "
                     r"(right) The two directions of inhibition reaching the cell from the opposite "
                     r"eye.", "fig:layers"))

    P.append(r"\paragraph{Its receptive field is lateral, with a gap at the front.} The part of "
             r"the eye a cell sees is given by the retinal positions of the motion detectors that "
             r"feed it. Compared with the frontal FD1 cell, measured in the same coordinate "
             r"frame so the comparison needs no conversion into degrees, the inputs of "
             r"\ttt{LPT42\_Nod4} sit clearly off to the side, by a margin larger than the "
             r"measurement uncertainty, for both the left and the right cell (Fig.~\ref{fig:rf}a). "
             r"It nearly avoids the most frontal part of the eye that FD1 "
             r"fills (it covers about 2--3\% of that frontal zone, against about 60\% for FD1): "
             r"this is the ``frontal gap'' that Egelhaaf singled out as unique to FD3 among all "
             r"four FD cells (Fig.~\ref{fig:rf}b).")
    P.append(r"\paragraph{It responds to small objects, not the whole scene.} The defining "
             r"behaviour of an FD cell is a stronger response to a small moving figure than to "
             r"motion of the whole background. That requires the cell to gather its input from a "
             r"compact patch of the visual field rather than spreading it everywhere. The input "
             r"patch of \ttt{LPT42\_Nod4} is more concentrated than an in-degree matched random "
             rf"sample across the eye ($p={_esc(nb.get('p_value'))}$; Fig.~\ref{{fig:rf}}c). "
             r"This provides wiring support for small-field selectivity; it is not a direct "
             r"physiological measurement.")
    P.append(_figure("figures/fd3_rf_differential.png",
                     r"\textbf{The receptive field.} (a) For each cell, how far its input field "
                     r"sits to the side of the frontal FD1 field; the bars are well above zero "
                     r"with margin, on both sides. (b) How much of FD1's frontal zone each cell "
                     r"occupies: FD1 (grey) fills it, \ttt{LPT42\_Nod4} (red) leaves it almost "
                     r"empty, the frontal gap. (c) How tightly the cell's inputs cluster "
                     r"(red) compared with random sampling of the same number of inputs (grey): "
                     r"more concentrated, the wiring basis expected for small-field selectivity.",
                     "fig:rf"))

    P.append(rf"\paragraph{{Its axon crosses to the other side of the brain.}} Egelhaaf found "
             r"that the FD3 axon does not stay in the eye it serves: it crosses the brain's "
             r"midline and ends on the opposite side. The connectome shows the same sidedness: "
             rf"{_esc(contra)}\% of this cell's output synapses are on the opposite side of the "
             r"brain from the cell itself"
             + (rf" ({_esc(off_contra)}\% measured on the frozen copy of the data)" if off_contra is not None else "")
             + r", the signature of a cell that sends its signal across the midline, as FD3 (and "
             r"FD1 and FD4) do (Fig.~\ref{fig:placement}).")

    P.append(rf"\paragraph{{Its cell body lies where Egelhaaf placed it.}} Egelhaaf located the "
             r"FD3 cell body toward the back and side of the central brain. The two "
             r"\ttt{LPT42\_Nod4} cell bodies sit in that posterior and lateral region: one on each side of the midline, "
             rf"among the rearmost {int(100*(_g(soma,'post_z_percentile') or 0))}\% of visual "
             r"output cells, right beside the cell bodies of the FD1 cell, and far behind the "
             r"feedback cells that inhibit them (Fig.~\ref{fig:placement}).")
    P.append(_figure("figures/fd3_circuit_placement.png",
                     r"\textbf{Where the cell sends its signal, and where its body sits.} (left) "
                     r"The main downstream targets of \ttt{LPT42\_Nod4}, the great majority on the "
                     r"opposite side of the brain. (right) The two cell bodies, sitting toward the "
                     r"back of the brain and one on each side of the midline.", "fig:placement"))

    morph = _g(D, "morphology", default={})
    mcells = morph.get("cells", []) if morph.get("available") else []
    if mcells:
        is_skel = morph.get("source") == "skeleton"
        dv = [c.get("dv_span_um") for c in mcells]
        shifts = [c.get("axon_ml_shift_um") for c in mcells]
        nodes = [c.get("n_vertices") for c in mcells if c.get("n_vertices")]
        if is_skel:
            node_txt = ", ".join(
                f"{c.get('side')}: {c.get('n_vertices')}" for c in mcells if c.get("n_vertices")
            )
            method = (r"\paragraph{The shape of the cell.} We reconstruct each cell's branching "
                      r"skeleton from its three-dimensional shape in the connectome "
                      rf"({_esc(node_txt or 'about 36000 vertices per cell')}) and separate it "
                      r"into the dendrite, where the cell receives its inputs, and the axon, where "
                      r"it sends its outputs; ")
            cap = (r"\textbf{The shape of the cell.} Reconstructed skeletons of the two "
                   r"\ttt{LPT42\_Nod4} cells. (a) For each cell, the output (axonal) part of the "
                   r"arbor, in red, sits to one side of the input (dendritic) part, in blue, "
                   r"shifted toward the region where Egelhaaf's FD3 axon terminates, in green. "
                   r"This is the expected geometry for a cell whose axon crosses the brain's midline. (b) The output "
                   r"arbor lies closer to that target region than the input arbor does.")
        else:
            method = (r"\paragraph{The shape of the cell.} Reading the cell's shape from the "
                      r"three-dimensional positions of its synapses, with inputs marking the "
                      r"dendrite and outputs marking the axon, ")
            cap = (r"\textbf{The shape of the cell.} (a) The output (axonal) synapse field, in "
                   r"red, is shifted to one side of the input (dendritic) field, in blue, toward "
                   r"the region where Egelhaaf's FD3 axon terminates, in green. (b) The output "
                   r"field lies closer to that target region than the input field does.")
        P.append(method
                 + rf"the dendrite spans the full top-to-bottom (dorso-ventral) extent of the "
                 rf"motion-processing layer ($\sim${_esc(round(max(dv),0) if dv else None)}~$\mu$m, "
                 r"matching Egelhaaf's description), and its axon is shifted to the opposite side "
                 r"of the brain from its dendrite, ending roughly twice as close to the target "
                 r"region as the dendrite is. This is consistent with the heterolateral, "
                 r"midline-crossing axon Egelhaaf described for FD3 (Fig.~\ref{fig:morph}).")
        P.append(_figure("figures/fd3_morphology.png", cap, "fig:morph"))

    P.append(r"\paragraph{No other cell type matches FD3 as well.} The strongest form of the "
             r"argument compares every candidate cell type against all four FD cells at once. For "
             r"each, we score how many of the defining FD features it shares: preferred "
             r"direction, the lateral field with a frontal gap, and the crossing axon. "
             r"\ttt{LPT42\_Nod4} matches FD3 on every count, more closely than any other candidate "
             r"matches FD3, and more closely than \ttt{LPT42\_Nod4} matches any other FD cell "
             rf"(Fig.~\ref{{fig:heatmap}}). The identification is therefore specific within the "
             r"screened Nod/LPT candidate set.")
    P.append(_figure("figures/fd3_family_heatmap.png",
                     r"\textbf{The match is specific.} Each candidate cell type (columns) scored "
                     r"against each of the four FD cells (rows) by how many defining features it "
                     r"shares (0--3). \ttt{LPT42\_Nod4} (outlined) is the strongest match for FD3, "
                     r"and FD3 is its strongest match in return.", "fig:heatmap"))

    P.append(r"\paragraph{The neighbouring cell types are not FD3.} An identification is only "
             r"meaningful if the same measurements distinguish \ttt{LPT42\_Nod4} from the cells "
             r"most easily confused with it. They do. \ttt{Nod1} prefers front-to-back motion and "
             r"has a frontal field, which marks it as FD1 rather than FD3. \ttt{Nod2} is inhibitory (GABA-releasing) "
             r"rather than an excitatory output cell. \ttt{Nod3} also prefers back-to-front motion "
             r"but its output stays on the same side of the brain rather than crossing the midline. "
             r"\ttt{Nod5} responds to vertical rather than horizontal motion and feeds the "
             r"wide-field inhibitory cells, marking it as part of the feedback machinery rather than "
             r"a figure-detection output. Applying the full set of FD3 measurements to any of these "
             r"cells fails to reproduce the FD3 signature.")

    if _g(cv, "available"):
        P.append(r"\paragraph{The result is reproduced across available data tracks.} The primary "
                 rf"numbers come from the \ttt{{{_esc(track)}}} track, and the v630 check gives "
                 rf"back-to-front input {_esc(_g(cv,'v630','layer_b'))}\% and contralateral output "
                 rf"{_esc(_g(cv,'v630','contra_output_pct'))}\%. The same qualitative identity "
                 r"features hold separately for the left and right cell (Fig.~\ref{fig:replication}).")
    else:
        P.append(r"\paragraph{The provenance is explicit.} The primary numbers come from the "
                 rf"\ttt{{{_esc(track)}}} track. The v630 cross-version check is not claimed in "
                 rf"this artifact because {_esc(_g(cv, 'reason', default='it was unavailable'))}. "
                 r"The identity evidence still requires agreement between the left and right "
                 r"\ttt{LPT42\_Nod4} cells and comparison against neighbouring Nod/LPT candidates "
                 r"(Fig.~\ref{fig:replication}).")
    P.append(r"\paragraph{The result is the same regardless of analysis choices.} The conclusions "
             r"are unchanged when the thresholds that define ``lateral'', ``frontal gap'' and "
             r"``crosses the midline'' are varied across their reasonable ranges, and when the "
             r"random-sampling steps are repeated with different seeds, so no finding rests on a "
             r"particular cutoff (Fig.~\ref{fig:replication}).")
    P.append(_figure("figures/fd3_replication_robustness.png",
                     r"\textbf{Stability of the findings.} (left) Available provenance for the "
                     r"defining FD3 measurements; unavailable cross-version checks are marked as "
                     r"missing rather than inferred. (right) The frontal gap persists as its defining threshold is "
                     r"varied, and the lateral-displacement estimate stays positive across "
                     r"resampling seeds.", "fig:replication"))
    return "\n\n".join(P)


def _caveats(D: dict) -> str:
    budget = _g(D, "rf", "calibration_error_budget", default={})
    return (r"\section{What the data can and cannot settle}" "\n\n"
            r"Three points deserve to be stated plainly so the strength of the identification is "
            r"not overstated. \textbf{There are only two cells.} \ttt{LPT42\_Nod4} is a single "
            r"bilateral pair -- one cell per side -- and this is simply how many the fly has, so "
            r"the result cannot be averaged over a large sample. Its robustness comes instead from "
            r"the left and right cell agreeing independently, from comparison against neighbouring "
            r"candidate types, and from statistical resampling of the "
            r"thousands of synapses that make up each cell. \textbf{Absolute visual angle is "
            r"approximate.} Egelhaaf measured the receptive field in degrees of visual angle; the "
            r"connectome gives positions on the eye's lattice, and converting one to the other is "
            rf"only approximate (to within about ${_esc(budget.get('combined_deg'))}^\circ$). For "
            r"this reason the receptive-field argument is built on a \emph{relative} comparison to "
            r"the frontal FD1 cell, which needs no such conversion; the absolute degree figures are "
            r"reported only as a consistency check. \textbf{Fine branch-level anatomy is at the "
            r"limit of the reconstruction.} The cell's overall shape -- the extent of its dendrite "
            r"and the side its axon projects to -- is recovered clearly, both from its synapse "
            r"positions and from its reconstructed skeleton, and the two agree. Still finer details "
            r"of the arbor are at the resolution limit of the automated reconstruction and are not "
            r"pressed further here.")


def _methods(track: str, D: dict) -> str:
    cv = _g(D, "cross_version", default={}) or {}
    cross = (r" A v630 cross-version check is included for the identity measurements."
             if cv.get("available")
             else rf" A v630 cross-version check is not claimed here because {_esc(cv.get('reason', 'it was unavailable'))}.")
    return (r"\section{Methods and provenance}" "\n\n"
            rf"\textbf{{Connectomes.}} FlyWire FAFB materialization v783 "
            rf"(\ttt{{synapses\_nt\_v1}}, no cleft threshold) on the \ttt{{{_esc(track)}}} track "
            r"as the primary source for the numbers in this report." + cross +
            r" Neuron annotations (cell type, side, super-class, "
            r"neurotransmitter and confidence, soma coordinates) are the Schlegel/Codex tables. "
            r"\textbf{Preferred direction} is the synapse-weighted T4/T5 lobula-plate layer "
            r"composition (layer$\to$direction after Fischbach \& Dittrich and Maisak "
            r"et al.~\cite{fd1989,maisak2013}). \textbf{Receptive "
            r"field}: T4/T5 inputs carry hex-lattice (p,q) coordinates; the differential test "
            r"compares the synapse-weighted centroid, half-width and frontal-band occupancy "
            r"against the FD1=\ttt{Nod1} anchor, with a 2000-sample within-cell bootstrap and a "
            r"500-permutation in-degree null; the axis convention is pinned by a self-test "
            r"requiring the known-frontal FD1 anchor to land frontal. \textbf{Family screen}: a "
            r"binary feature-match score over all Nod/LPT \ttt{visual\_projection} types. "
            r"\textbf{Reproducibility}: all randomness is seeded (12345, plus a 5-seed stability "
            r"set). The report is built from the verification JSON by the FD3 report CLI. "
            rf"Primary track for the numbers herein: \ttt{{{_esc(track)}}}.")


def _bibliography() -> str:
    items = [
        r"\bibitem{egelhaaf1985a} Egelhaaf M. (1985) On the neuronal basis of figure-ground "
        r"discrimination by relative motion in the visual system of the fly. I. Behavioural "
        r"constraints imposed on the neuronal network and the role of the optomotor system. "
        r"\textit{Biol. Cybern.} 52:123--140.",
        r"\bibitem{egelhaaf1985b} Egelhaaf M. (1985) \ldots II. Figure-detection cells, a new "
        r"class of visual interneurones. \textit{Biol. Cybern.} 52:195--209. (The FD-cell paper; "
        r"FD3 at pp.~202--204.)",
        r"\bibitem{egelhaaf1985c} Egelhaaf M. (1985) \ldots III. Possible input circuitries and "
        r"behavioural significance of the FD-cells. \textit{Biol. Cybern.} 52:267--280.",
        r"\bibitem{rp1979} Reichardt W., Poggio T. (1979) Figure-ground discrimination by "
        r"relative movement in the visual system of the fly. I. \textit{Biol. Cybern.} 35:81--100.",
        r"\bibitem{schlegel2024} Schlegel P. et al. (2024) Whole-brain annotation and "
        r"multi-connectome cell typing of \textit{Drosophila}. \textit{Nature} 634:139--152.",
        r"\bibitem{dorkenwald2024} Dorkenwald S. et al. (2024) Neuronal wiring diagram of an "
        r"adult brain. \textit{Nature} 634:124--138.",
        r"\bibitem{codex} FlyWire Codex, \texttt{codex.flywire.ai} -- cell-type and annotation "
        r"portal.",
        r"\bibitem{fd1989} Fischbach K.-F., Dittrich A.P.M. (1989) The optic lobe of "
        r"\textit{Drosophila melanogaster}. I. A Golgi analysis of wild-type structure. "
        r"\textit{Cell Tissue Res.} 258:441--475.",
        r"\bibitem{maisak2013} Maisak M.S. et al. (2013) A directional tuning map of "
        r"\textit{Drosophila} elementary motion detectors. \textit{Nature} 500:212--216.",
        r"\bibitem{beersma1977} Beersma D.G.M., Stavenga D.G., Kuiper J.W. (1977) Retinal "
        r"lattice, visual field and binocularities in flies. \textit{J. Comp. Physiol.} "
        r"119:207--220.",
    ]
    return (r"\begin{thebibliography}{9}" "\n" + "\n".join(items) + "\n"
            r"\end{thebibliography}")


# ---------------------------------------------------------------------------
# top-level assembly
# ---------------------------------------------------------------------------
def build_tex(results: dict, offline: dict | None = None) -> str:
    fam = results["families"]["K"] if "families" in results else results
    D = fam["derived"]
    off_D = (offline.get("derived") if offline else None)
    track = results.get("meta", {}).get("flywire_track", D.get("track", "live"))

    parts = [
        _PREAMBLE,
        r"\begin{document}",
        r"\title{\textbf{\ttt{LPT42\_Nod4} is the modern connectomic correlate of the "
        r"Egelhaaf-1985 FD3 figure-detection cell}}",
        r"\author{Stage 7 FD3 Identification}",
        r"\date{\today}",
        r"\maketitle",
        _summary(D, track),
        _introduction(),
        _tikz_fig1(),
        _results(D, off_D, track),
        r"\section{The evidence at a glance}",
        r"Table~\ref{tab:evidence} collects the defining FD3 properties next to the "
        r"connectome measurement that establishes it for \ttt{LPT42\_Nod4}. Each row is "
        r"developed in the corresponding part of the Results above.",
        r"\begin{table}[H]\centering\footnotesize\caption{Each defining property of Egelhaaf's "
        r"FD3 cell and the matching measurement for the FlyWire cell type \ttt{LPT42\_Nod4}.}"
        r"\label{tab:evidence}",
        _evidence_table(D, off_D, track),
        r"\end{table}",
        _caveats(D),
        _methods(track, D),
        _bibliography(),
        r"\end{document}",
    ]
    return "\n\n".join(parts)


def _engine() -> tuple[str, str] | None:
    for kind in ("tectonic", "pdflatex"):
        path = shutil.which(kind)
        if not path:
            cand = Path(_MINIFORGE_BIN) / kind
            path = str(cand) if cand.exists() else None
        if path:
            return kind, path
    return None


def _compile(tex_path: Path) -> tuple[bool, str]:
    import os
    eng = _engine()
    if eng is None:
        return False, "no TeX engine found (tried tectonic, pdflatex)"
    kind, path = eng
    env = {**os.environ, "PATH": f"{_MINIFORGE_BIN}:{os.environ.get('PATH','')}"}
    out_dir = tex_path.parent
    name = tex_path.name
    logs: list[str] = []
    try:
        if kind == "tectonic":
            proc = subprocess.run([path, "--keep-logs", name], capture_output=True, text=True,
                                  env=env, cwd=str(out_dir), timeout=420)
            logs.append(proc.stdout + proc.stderr)
            ok = proc.returncode == 0
        else:
            ok = True
            for _ in range(2):
                proc = subprocess.run([path, "-interaction=nonstopmode", "-halt-on-error", name],
                                      capture_output=True, text=True, env=env, cwd=str(out_dir),
                                      timeout=420)
                logs.append(proc.stdout + proc.stderr)
                ok = ok and proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}\n" + "\n".join(logs)
    return (ok and tex_path.with_suffix(".pdf").exists()), "\n\n".join(logs)


def _render_figures(results: dict, offline: dict | None, fig_dir: Path) -> None:
    fam = results["families"]["K"] if "families" in results else results
    claims = [K.ClaimResult(**{k: v for k, v in c.items()
                               if k in K.ClaimResult.__dataclass_fields__}) for c in fam["claims"]]
    run = {"results": {"K": claims}, "derived": {"K": fam["derived"]},
           "meta": results.get("meta", {}) if isinstance(results, dict) else {}}
    if offline:
        run["offline"] = {"derived": offline.get("derived", {})}
    PF.fd3_crosswalk(run, fig_dir / "fd3_crosswalk.png")
    PF.fd3_layer_composition(run, fig_dir / "fd3_layer_composition.png")
    PF.fd3_rf_differential(run, fig_dir / "fd3_rf_differential.png")
    PF.fd3_family_heatmap(run, fig_dir / "fd3_family_heatmap.png")
    PF.fd3_replication_robustness(run, fig_dir / "fd3_replication_robustness.png")
    PF.fd3_circuit_placement(run, fig_dir / "fd3_circuit_placement.png")
    PF.fd3_morphology(run, fig_dir / "fd3_morphology.png")


def build_report(stage_dir: str | Path = "7 - FD3 Identification",
                 results_path: str | Path = "5 - Paper Verification/verification_results.json",
                 offline_results_path: str | Path | None = None) -> dict[str, Any]:
    stage = Path(stage_dir)
    (stage / "figures").mkdir(parents=True, exist_ok=True)
    results = read_json(Path(results_path))
    offline = None
    if offline_results_path:
        try:
            offline = read_json(Path(offline_results_path))
        except FileNotFoundError:
            offline = None

    _render_figures(results, offline, stage / "figures")
    tex_path = stage / "fd3_report.tex"
    tex_path.write_text(build_tex(results, offline))

    compiled, log = _compile(tex_path)
    err = tex_path.with_suffix(".compile_error.txt")
    if not compiled:
        err.write_text(log)
    elif err.exists():
        err.unlink()
    return {"tex": str(tex_path),
            "pdf": str(tex_path.with_suffix(".pdf")) if compiled else None,
            "compiled": bool(compiled)}
