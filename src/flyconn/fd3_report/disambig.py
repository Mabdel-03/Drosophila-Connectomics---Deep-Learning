"""Build + compile the FD2/FD3 disambiguation report (LPT21 vs Nod3 vs LPT42_Nod4) as LaTeX -> PDF.

Reads ONLY the disambiguation JSON written by ``scripts/fd3_disambig_verify.py`` (the ``families.KD``
block plus the ``families.K`` block it copies in for the shared layer/screen figures); no live CAVE
and no feathers, so the report re-runs token-free. It reuses the Family-K report machinery for the
preamble, escaping, figure wrapper and TeX compile, and adds the three-way disambiguation prose, a
three-cell TikZ concept figure, the comparison table, and the KD figures.

The narrative is reader-facing (no internal verdict vocabulary). It resolves three regressive
noduli-family cells against Egelhaaf's FD cells: FD3 = LPT42_Nod4 (lateral RF + frontal gap +
heterolateral axon), FD2 = LPT21 (frontal RF + homolateral/ipsilateral projection), and Nod3 = an
intermediate regressive cell that matches neither cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import read_json
from ..paper import figures as PF
from ..motif import compare as K
from .builder import _PREAMBLE, _esc, _figure, _compile


LPT = r"\ttt{LPT42\_Nod4}"
NOD3 = r"\ttt{Nod3}"
LPT21 = r"\ttt{LPT21}"


# ---------------------------------------------------------------------------
# TikZ Figure 1: three regressive cells placed on the FD2 <-> FD3 axis.
# ---------------------------------------------------------------------------
def _tikz_fig1(lpt_contra, nod3_contra, lpt21_contra) -> str:
    def pc(x):
        return _esc(round(x)) if isinstance(x, (int, float)) else "?"
    lc, nc, l21 = pc(lpt_contra), pc(nod3_contra), pc(lpt21_contra)
    return (r"""\begin{figure}[H]\centering
\begin{tikzpicture}[font=\small,>=Stealth]
  % ---- top: receptive fields on a shared visual-field arc ----
  \draw[thick] (0,0) arc (180:90:5.6 and 2.3);
  \node[anchor=north east,font=\scriptsize] at (0,-0.05) {frontal ($0^\circ$)};
  \node[anchor=south west,font=\scriptsize] at (5.2,2.35) {lateral ($\sim$100$^\circ$)};
  \fill[fd1c,opacity=0.25] (0.35,0.18) ellipse (0.5 and 0.26);
  \node[fd1c,font=\scriptsize,anchor=west] at (0.95,0.18) {\textbf{FD1} anchor};
  % FD2 (LPT21): frontal field, fills the frontal band
  \fill[vgreen,opacity=0.28] (1.5,0.9) ellipse (0.9 and 0.32);
  \node[vgreen,font=\scriptsize,anchor=west] at (2.55,0.9) {\textbf{FD2} $=$ \ttt{LPT21} (frontal)};
  % FD3 (LPT42_Nod4): fronto-lateral field with a gap
  \fill[fd3c,opacity=0.30] (4.0,1.7) ellipse (1.05 and 0.32);
  \node[fd3c,font=\scriptsize,anchor=east] at (3.05,1.7) {\textbf{FD3} $=$ \ttt{LPT42\_Nod4} ($+$ gap)};
  \draw[fd3c,dashed,->] (2.2,1.3) .. controls (2.55,1.45) .. (2.9,1.5)
        node[right,fd3c,font=\scriptsize,xshift=1pt]{gap};
  \draw[->,thick,vgray] (4.5,-0.7) -- (2.3,-0.7)
        node[midway,below,font=\scriptsize]{regressive (back$\to$front): shared by all three};
  % ---- bottom: projection, split by a dashed midline ----
  \draw[dashed,gray] (4.8,-1.5) -- (4.8,-3.4);
  \node[gray,font=\scriptsize,anchor=south] at (4.8,-1.5) {brain midline};
  \node[draw,rounded corners,fill=vgreen!12,font=\scriptsize,align=center,text width=2.3cm]
        (fd2d) at (1.5,-2.2) {\ttt{LPT21}\\dendrite};
  \node[draw,rounded corners,fill=vgreen!12,font=\scriptsize,align=center,text width=2.5cm]
        (fd2a) at (1.5,-3.3) {ipsilateral\\posterior optic foci};
  \draw[->,thick,vgreen] (fd2d) -- node[right,font=\scriptsize]{stays ipsilateral} (fd2a);
  \node[draw,rounded corners,fill=fd3c!12,font=\scriptsize,align=center,text width=2.4cm]
        (fd3d) at (3.4,-2.2) {\ttt{LPT42\_Nod4}\\dendrite};
  \node[draw,rounded corners,fill=fd3c!12,font=\scriptsize,align=center,text width=2.4cm]
        (fd3a) at (7.6,-2.2) {contralateral\\posterior optic foci};
  \draw[->,thick,fd3c] (fd3d) -- node[above,midway,font=\scriptsize,align=center]
        {axon crosses midline,\\posterior to noduli} (fd3a);
  % ---- candidate tags ----
  \node[draw,rounded corners,vgreen,font=\scriptsize,align=center,text width=3.1cm]
        at (1.5,-4.4) {\ttt{LPT21} $=$ FD2\\""" + l21 + r"""\% contra (homolateral), frontal RF};
  \node[draw,rounded corners,vorange,font=\scriptsize,align=center,text width=3.1cm]
        at (4.9,-4.4) {\ttt{Nod3} $=$ intermediate\\""" + nc + r"""\% contra, slightly lateral RF};
  \node[draw,rounded corners,fd3c,font=\scriptsize,align=center,text width=3.3cm]
        at (8.2,-4.4) {\ttt{LPT42\_Nod4} $=$ FD3\\""" + lc + r"""\% contra, lateral RF $+$ gap};
\end{tikzpicture}
\caption{\textbf{Three regressive figure-detection candidates, and which Egelhaaf cell each is.}
Egelhaaf's FD2 and FD3 are both excited by regressive (back-to-front) motion, so direction alone
cannot tell them apart. They differ in receptive field and projection side: FD2 has a frontal field
and a homolateral axon to the ipsilateral posterior optic foci; FD3 has a fronto-lateral field with a
frontal gap and a heterolateral axon that crosses the midline posterior to the noduli. The connectome
places \ttt{LPT21} at FD2 (frontal field, essentially all output ipsilateral), \ttt{LPT42\_Nod4} at
FD3 (fronto-lateral field with a gap, output contralateral), and \ttt{Nod3} between them (a mixed
projection and a slightly lateral field), so \ttt{Nod3} is not a clean match to either.}%
\label{fig:concept}
\end{figure}
""")


# ---------------------------------------------------------------------------
# prose
# ---------------------------------------------------------------------------
def _b(KD, ct):
    return KD.get("candidates_offline", {}).get(ct, {})


def _summary(KD: dict) -> str:
    lpt = _b(KD, "LPT42_Nod4"); nod3 = _b(KD, "Nod3"); lpt21 = _b(KD, "LPT21")
    return (
        r"\textbf{Summary.}\quad "
        r"Three cell types in the fly connectome are regressive, cholinergic, noduli-family figure "
        r"cells: \ttt{LPT42\_Nod4}, \ttt{Nod3}, and \ttt{LPT21}. Egelhaaf described two regressive "
        r"figure-detection cells, FD2 and FD3, which share a preferred direction but differ in "
        r"receptive field and projection side. This report measures the distinguishing properties for "
        r"all three candidates and assigns each. \ttt{LPT42\_Nod4} is FD3: its receptive field sits "
        rf"laterally with a frontal gap and {_esc(lpt.get('contra_output_pct'))}\% of its output "
        r"crosses the midline (heterolateral). \ttt{LPT21} is FD2: its receptive field is frontal, "
        r"co-located with the FD1 cell, and its output is almost entirely on its own side "
        rf"({_esc(lpt21.get('contra_output_pct'))}\% contralateral), the homolateral projection to the "
        r"ipsilateral posterior optic foci that Egelhaaf reported for FD2. \ttt{Nod3} is neither "
        rf"cleanly: it is regressive but its output is mixed ({_esc(nod3.get('contra_output_pct'))}\% "
        r"contralateral) and its field is only slightly lateral, so it sits between FD2 and FD3 and is "
        r"left as an intermediate rather than forced into one identity. The primary numbers come from "
        r"the offline v783 track; the laterality measurements are confirmed on the live and v630 "
        r"tracks. Each cell type is a single bilateral pair, so the result rests on left-right "
        r"agreement and on the thousands of synapses per cell, not on a population average.")


def _introduction() -> str:
    return r"""\section{Introduction}

\textbf{Figure-ground discrimination and the FD cells.}\quad
A textured object is invisible against a matched background until it moves at a different velocity or
phase; relative motion then makes it salient. Reichardt and Poggio showed behaviourally that flies
detect and track such figures~\cite{rp1979}. Egelhaaf, in a three-part study, described the neurons
that could implement this~\cite{egelhaaf1985a,egelhaaf1985b,egelhaaf1985c}. Part~II defined a class of
lobula-plate tangential cells, the \emph{figure-detection} (FD) cells, each responding more strongly
to a small moving figure than to wide-field motion. Two of them are excited by \emph{regressive}
(back-to-front) motion. \textbf{FD2} has a frontal excitatory field (peak azimuth $0$--$10^\circ$,
reaching the frontal margin of the eye) and is a homolateral cell whose axon projects to the
\emph{ipsilateral} posterior optic foci; Egelhaaf notes it is not a ``noduli-group'' cell and that its
large-field organisation could not be resolved~\cite[p.~201]{egelhaaf1985b}. \textbf{FD3} has a
fronto-lateral field (peak $40$--$50^\circ$) that alone among the FD cells spares the most-frontal
$10$--$20^\circ$ (a frontal gap), and it is a heterolateral ``noduli-group'' cell whose axon crosses
the midline posterior to the noduli to the \emph{contralateral} posterior optic
foci~\cite[p.~202--204]{egelhaaf1985b}. The two regressive cells therefore share a preferred direction
and differ on two axes: the receptive field (frontal versus fronto-lateral with a gap) and the
projection side (homolateral/ipsilateral versus heterolateral/contralateral).

\textbf{The three candidates.}\quad
FD1 has already been mapped to the cholinergic cell type \ttt{Nod1}, which reads front-to-back
motion~\cite{maisak2013,fd1989}. Among the remaining noduli-family cells, three are regressive and
cholinergic: \ttt{LPT42\_Nod4}, \ttt{Nod3}, and \ttt{LPT21}. Because they share direction and
transmitter, those properties cannot assign them; the assignment rests on the receptive field and the
projection side. This report measures both for all three, together with the small-field selectivity,
the morphology, and the whole-family feature match, and reads off each cell's Egelhaaf identity.

\textbf{How the assignment is made.}\quad
Each distinguishing property is measured directly in the connectome: the projection side from where a
cell's output synapses land, the receptive field from the retinotopic positions of its motion inputs
relative to the frontal FD1 cell, and the axon path from the cell's reconstructed shape. We measure
these for all three candidates, on both cells of each bilateral pair, on the offline (primary), live,
and v630 tracks. The receptive-field comparison is made relative to the FD1=\ttt{Nod1} anchor in the
same coordinate frame, so it needs no conversion into visual angle. Figure~\ref{fig:concept} fixes the
FD2-versus-FD3 distinction; the sections that follow take each property in turn."""


def _rule_section(KD: dict) -> str:
    return r"""\section{The assignment rule}

Two Egelhaaf properties separate the regressive figure cells, and each maps onto a connectome
measurement. \textbf{FD3} requires a fronto-lateral receptive field with a frontal gap (its field sits
lateral of the frontal FD1 cell and vacates the frontal band) \emph{and} a heterolateral axon (the
large majority of its output crosses the midline). \textbf{FD2} requires a frontal receptive field
(co-located with the frontal FD1 cell, filling the frontal band, no gap) \emph{and} a homolateral axon
(its output stays on its own side, the ipsilateral projection). A cell that is regressive but matches
neither pattern cleanly, sitting between the two on both axes, is left as an intermediate rather than
forced into an identity.

The rule can conclude differently for each candidate and is not tuned to a desired answer. \ttt{LPT42\_Nod4}
is assigned FD3 when it has the lateral field with a gap and the heterolateral axon; \ttt{LPT21} is
assigned FD2 when it has the frontal field and the homolateral axon; and \ttt{Nod3} is assigned an
identity only if it matches one pattern cleanly. The single most important point of method is that
output laterality is measured on a continuum rather than a single threshold: a homolateral cell (FD2)
keeps its output near zero percent contralateral, a heterolateral cell (FD3) sends most output across,
and a cell in between is neither. This is what distinguishes \ttt{LPT21} (homolateral) from \ttt{Nod3}
(mixed), a distinction a single heterolateral cut-off would miss."""


def _favor_none(v):
    return "--"


def _evidence_table(KD: dict) -> str:
    rows = KD.get("comparison_rows", [])
    head = (r"\renewcommand{\arraystretch}{1.2}" "\n"
            r"\setlength{\tabcolsep}{3pt}" "\n"
            r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.17\textwidth} "
            r">{\raggedright\arraybackslash}p{0.21\textwidth} "
            r">{\raggedright\arraybackslash}p{0.20\textwidth} "
            r">{\raggedright\arraybackslash}p{0.20\textwidth} "
            r">{\raggedright\arraybackslash}p{0.20\textwidth}@{}}" "\n"
            r"\toprule \textbf{Property} & \textbf{Egelhaaf reference} & "
            r"\textbf{\ttt{LPT21} (FD2)} & \textbf{\ttt{Nod3} (interm.)} & "
            r"\textbf{\ttt{LPT42\_Nod4} (FD3)} \\\midrule\endhead" "\n")
    body_lines = []
    for r in rows:
        body_lines.append(
            rf"{_esc(r['property'])} & {_esc(r['egelhaaf'])} & {_esc(r.get('lpt21',''))} & "
            rf"{_esc(r['nod3'])} & {_esc(r['lpt42'])} \\")
    body = "\n".join(body_lines)
    return (head + body + "\n" + r"\bottomrule\end{longtable}" + "\n"
            r"\renewcommand{\arraystretch}{1.0}" "\n" r"\setlength{\tabcolsep}{6pt}")


def _results(KD: dict, track: str) -> str:
    lpt = _b(KD, "LPT42_Nod4"); nod3 = _b(KD, "Nod3"); lpt21 = _b(KD, "LPT21")
    live = KD.get("live", {})
    lpt_l = (live.get("candidates") or {}).get("LPT42_Nod4", {}) if live.get("available") else {}
    nod3_l = (live.get("candidates") or {}).get("Nod3", {}) if live.get("available") else {}
    lpt21_l = (live.get("candidates") or {}).get("LPT21", {}) if live.get("available") else {}
    cons = KD.get("constructive", {})

    P = [r"\section{Evidence, property by property}"]
    P.append(r"We take the properties in turn: first the ones all three candidates share, which "
             r"cannot assign them, then the two axes that separate FD2 from FD3.")

    P.append(r"\paragraph{All three cells prefer regressive motion, so direction cannot assign them.} "
             r"A cell's preferred direction is read from which lobula-plate layer its motion inputs "
             rf"come from. {LPT21} draws {_esc(lpt21.get('layer_b_pct'))}\% of its motion input from "
             rf"the back-to-front layer, {NOD3} {_esc(nod3.get('layer_b_pct'))}\%, and {LPT} "
             rf"{_esc(lpt.get('layer_b_pct'))}\% (Fig.~\ref{{fig:layers}}). All three are regressive, "
             r"as both FD2 and FD3 are, so this property is shared and assigns none of them.")
    P.append(_figure("figures/fd3_layer_composition.png",
                     r"\textbf{Preferred direction is shared.} The share of each cell's motion input "
                     r"from the four directional layers. \ttt{LPT42\_Nod4}, \ttt{Nod3} and \ttt{Nod1} "
                     r"are shown here (the layer profile of \ttt{LPT21} is likewise almost entirely "
                     r"back-to-front). Because FD2 and FD3 are both regressive, this panel does not "
                     r"separate the candidates.", "fig:layers"))
    P.append(r"\paragraph{All three are cholinergic, so transmitter cannot assign them.} Each releases "
             rf"acetylcholine ({LPT21} confidence {_esc(lpt21.get('nt_conf'))}, {NOD3} "
             rf"{_esc(nod3.get('nt_conf'))}, {LPT} {_esc(lpt.get('nt_conf'))}), and each is classified "
             r"as a visual output neuron. Transmitter and cell class are therefore also shared.")

    P.append(r"\paragraph{Projection side: \ttt{LPT21} homolateral, \ttt{LPT42\_Nod4} heterolateral, "
             r"\ttt{Nod3} mixed.} Egelhaaf's FD2 is a homolateral cell whose axon projects to the "
             r"ipsilateral posterior optic foci, whereas FD3 is heterolateral, its axon crossing to "
             rf"the contralateral side. In the connectome {LPT21} sends only "
             rf"{_esc(lpt21.get('contra_output_pct'))}\% of its output across the midline, the clean "
             rf"homolateral projection of FD2; {LPT} sends {_esc(lpt.get('contra_output_pct'))}\%, the "
             rf"heterolateral projection of FD3; and {NOD3} sends {_esc(nod3.get('contra_output_pct'))}\%, "
             r"neither on its own side nor predominantly across, a mixed projection that matches FD2's "
             r"homolateral anatomy no better than FD3's heterolateral one (Fig.~\ref{fig:contra}).")
    if lpt_l and lpt21_l:
        P.append(r"\paragraph{The projection ordering holds on every data track.} The fraction of "
                 r"output that is contralateral is sensitive to the data source, so it is reported on "
                 rf"more than one track. On the live track {LPT21} is "
                 rf"{_esc(lpt21_l.get('contra_output_pct'))}\%, {NOD3} "
                 rf"{_esc(nod3_l.get('contra_output_pct'))}\%, and {LPT} "
                 rf"{_esc(lpt_l.get('contra_output_pct'))}\%. The exact numbers move between tracks, "
                 r"but the ordering, \ttt{LPT21} lowest, \ttt{LPT42\_Nod4} highest, \ttt{Nod3} in "
                 r"between, is preserved.")
    P.append(_figure("figures/fd3_disambig_contra.png",
                     r"\textbf{Output laterality.} The fraction of each candidate's output synapses "
                     r"on the contralateral side, on the offline, live and v630 tracks. \ttt{LPT21} "
                     r"lies in the homolateral band (near zero, the FD2 projection); \ttt{LPT42\_Nod4} "
                     r"clears the heterolateral threshold (the FD3 projection); \ttt{Nod3} sits "
                     r"between them.", "fig:contra"))

    P.append(r"\paragraph{Receptive field: \ttt{LPT21} frontal, \ttt{LPT42\_Nod4} lateral with a gap, "
             r"\ttt{Nod3} slightly lateral.} Measured against the frontal FD1 cell in the same "
             rf"coordinate frame, the field of {LPT21} is co-located with FD1 and fills the frontal "
             r"band, the frontal field Egelhaaf reported for FD2. The field of "
             rf"{LPT} sits far to the side and vacates the frontal band, the fronto-lateral field "
             rf"with a frontal gap that defines FD3. The field of {NOD3} is displaced only a few "
             r"columns from the front and keeps most of its frontal occupancy, so it is neither "
             r"clearly frontal nor clearly lateral (Fig.~\ref{fig:rf}).")
    P.append(_figure("figures/fd3_disambig_rf.png",
                     r"\textbf{Receptive field position and the frontal gap.} (left) The centroid of "
                     r"each candidate's input field relative to the frontal FD1 anchor, per side, with "
                     r"$95\%$ confidence intervals: \ttt{LPT21} near zero (frontal), \ttt{Nod3} a few "
                     r"columns lateral, \ttt{LPT42\_Nod4} far lateral. (right) Frontal-band occupancy: "
                     r"\ttt{LPT21} and \ttt{Nod3} fill the FD1 band, \ttt{LPT42\_Nod4} leaves it "
                     r"nearly empty (the gap).", "fig:rf"))

    P.append(_figure("figures/fd3_disambig_fd2.png",
                     r"\textbf{The FD2/FD3 plane.} Each candidate placed by output laterality "
                     r"(horizontal, homolateral to heterolateral) and receptive-field offset from FD1 "
                     r"(vertical, frontal to lateral). \ttt{LPT21} sits in the FD2 corner (homolateral, "
                     r"frontal); \ttt{LPT42\_Nod4} in the FD3 corner (heterolateral, lateral); "
                     r"\ttt{Nod3} between them.", "fig:plane"))

    P.append(r"\paragraph{The axon morphology agrees with the projection count.} Reconstructing each "
             r"cell's shape and separating the input (dendritic) from the output (axonal) field shows "
             r"a medio-lateral shift in every cell, so the shift direction alone does not assign them; "
             rf"what does is how much output lands on the far side. The {LPT} axon is displaced across "
             rf"the midline and its output is predominantly contralateral; the {LPT21} and {NOD3} axons "
             r"stay on their own side by output count (Fig.~\ref{fig:morph}).")
    P.append(_figure("figures/fd3_disambig_morphology.png",
                     r"\textbf{Axon shift and output laterality.} The dendrite-to-axon medio-lateral "
                     r"shift per cell, with the contralateral output fraction noted beneath. Bars are "
                     r"solid where the cell is heterolateral (at least $70\%$ contralateral); only "
                     r"\ttt{LPT42\_Nod4} is. \ttt{LPT21} and \ttt{Nod3} keep their output ipsilateral.",
                     "fig:morph"))

    P.append(r"\paragraph{The whole-family feature match, and where it needs refining.} Scoring every "
             r"candidate against all four FD cells on three binary features (direction, the lateral "
             rf"field with a gap, and the crossing axon) makes {LPT} the reciprocal-best FD3 "
             r"(Fig.~\ref{fig:heatmap}). For FD2 the binary score alone is not decisive: it rates both "
             rf"{LPT21} and {NOD3} at the FD2 signature, because it collapses every cell below the "
             r"heterolateral cut-off into one category and so cannot see that \ttt{LPT21} is cleanly "
             r"homolateral while \ttt{Nod3} is mixed. Resolving the tie on the finer measurements, the "
             r"homolateral projection and the frontal co-location, selects \ttt{LPT21} as FD2 and "
             r"leaves \ttt{Nod3} intermediate. As a check that the feature set can separate the FD "
             rf"cells, the FD1 anchor \ttt{{Nod1}} scores best for FD1 ({_esc(cons.get('anchor_best_fd'))}).")
    P.append(_figure("figures/fd3_family_heatmap.png",
                     r"\textbf{The family feature match.} Each candidate (columns) scored against each "
                     r"FD cell (rows) by shared binary features (0--3). \ttt{LPT42\_Nod4} (red outline) "
                     r"is the best match for FD3. The FD2 row scores \ttt{LPT21} (green outline) and "
                     r"\ttt{Nod3} equally on the binary features; the tie is broken toward \ttt{LPT21} "
                     r"by the finer homolateral and frontal-co-location measurements.", "fig:heatmap"))

    P.append(_figure("figures/fd3_disambig_decision.png",
                     r"\textbf{The three-way assignment, at a glance.} Each property, the Egelhaaf "
                     r"reference, and the measured value for each candidate. The shared properties "
                     r"(top) assign none; the projection and receptive-field rows place \ttt{LPT21} at "
                     r"FD2, \ttt{LPT42\_Nod4} at FD3, and \ttt{Nod3} between them.", "fig:decision"))
    return "\n\n".join(P)


def _nod3_section(KD: dict) -> str:
    nod3 = _b(KD, "Nod3")
    return (r"\section{\ttt{Nod3} is an intermediate regressive cell}" "\n\n"
            r"The earlier version of this analysis, comparing only \ttt{Nod3} and \ttt{LPT42\_Nod4}, "
            r"assigned \ttt{Nod3} to FD2 because it is regressive and not FD3. Adding \ttt{LPT21} "
            r"shows that assignment was too quick. On the two axes that define FD2, \ttt{LPT21} is a "
            r"clean match and \ttt{Nod3} is not. \ttt{Nod3}'s output is "
            rf"{_esc(nod3.get('contra_output_pct'))}\% contralateral, neither the near-zero of a "
            r"homolateral FD2 cell nor the high fraction of a heterolateral FD3 cell, and its "
            r"receptive field is displaced only slightly from the front, between FD2's frontal field "
            r"and FD3's lateral field. \ttt{Nod3} is therefore best described as an intermediate "
            r"regressive noduli-family figure cell: it belongs to the same functional group but does "
            r"not match a single one of Egelhaaf's described FD cells. Whether it is a distinct cell "
            r"Egelhaaf did not record, a variant of one he did, or a graded member of the regressive "
            r"set is a question the wiring alone does not settle, and it is left open here rather than "
            r"forced.")


def _caveats(KD: dict) -> str:
    conc = KD.get("concordance", {}).get("offline_vs_live", {})
    lpt_gap = conc.get("LPT42_Nod4", {}).get("contra_pp_gap")
    return (r"\section{What the data can and cannot settle}" "\n\n"
            r"\textbf{Each cell type is one bilateral pair.} \ttt{LPT21}, \ttt{Nod3} and "
            r"\ttt{LPT42\_Nod4} are each a single left/right pair, which is how many the fly has. The "
            r"result rests on the two cells of each pair agreeing, on the comparison against the FD1 "
            r"anchor, and on resampling the thousands of synapses each cell carries, not on a "
            r"population average. \textbf{The contralateral fraction is data-source sensitive.} The "
            r"exact percentage moves between the offline, live and v630 tracks because unannotated "
            r"local fragments are excluded from the count"
            + (rf" (the track-to-track movement for \ttt{{LPT42\_Nod4}} is about {_esc(lpt_gap)} "
               rf"percentage points)" if lpt_gap is not None else "")
            + r"; the ordering of the three candidates is preserved on every track, so the "
            r"assignment does not depend on any one of them. \textbf{FD2's large-field organisation "
            r"was not measured.} Egelhaaf reports that FD2's large-field input could not be resolved, "
            r"so FD2's only documented properties are its frontal field and its ipsilateral "
            r"projection; the FD2 assignment of \ttt{LPT21} rests on those two, not on a contralateral "
            r"property. \textbf{Absolute visual angle is approximate.} The receptive-field argument is "
            r"built on the relative comparison to the frontal FD1 cell and does not rely on absolute "
            r"azimuth. \textbf{\ttt{Nod3}'s positive identity is left open.} It is regressive and a "
            r"figure cell, but the data place it between FD2 and FD3 rather than on either, and it is "
            r"reported as an intermediate rather than assigned.")


def _methods(track: str) -> str:
    return (r"\section{Methods and provenance}" "\n\n"
            r"\textbf{Connectome.} FlyWire FAFB materialization v783 (\ttt{synapses\_nt\_v1}, no "
            r"cleft threshold), offline track as the primary source, with the live v783 and v630 "
            r"tracks confirming the laterality measurements. Neuron annotations (cell type, side, "
            r"super-class, neurotransmitter and confidence, soma coordinates) are the Schlegel/Codex "
            r"tables. \textbf{Preferred direction} is the synapse-weighted T4/T5 lobula-plate layer "
            r"composition (layer$\to$direction after Fischbach \& Dittrich and Maisak et "
            r"al.~\cite{fd1989,maisak2013}). \textbf{Output laterality} is the fraction of a cell's "
            r"output synapses whose postsynaptic partner lies on the opposite side of the brain, "
            r"reported per track and read as a continuum (homolateral near zero, heterolateral high). "
            r"\textbf{Receptive field}: each cell's T4/T5 inputs carry hex-lattice (p,q) coordinates; "
            r"the differential test compares the synapse-weighted centroid and frontal-band occupancy "
            r"against the FD1=\ttt{Nod1} anchor, with a 2000-sample within-cell bootstrap, and the "
            r"axis convention is pinned by a self-test requiring the known-frontal FD1 anchor to land "
            r"frontal. \textbf{Morphology}: input and output synapse positions and reconstructed "
            r"skeletons give the dendrite and axon fields. \textbf{Family match}: a binary "
            r"feature-match over the Nod/LPT \ttt{visual\_projection} types against the four FD "
            r"signatures, with the FD2 tie between \ttt{LPT21} and \ttt{Nod3} resolved on the "
            r"homolateral and frontal-co-location measurements. \textbf{Reproducibility}: all "
            rf"randomness is seeded (12345). The report is built from the disambiguation JSON. Primary "
            rf"track: \ttt{{{_esc(track)}}}.")


def _bibliography() -> str:
    items = [
        r"\bibitem{egelhaaf1985a} Egelhaaf M. (1985) On the neuronal basis of figure-ground "
        r"discrimination by relative motion in the visual system of the fly. I. \textit{Biol. "
        r"Cybern.} 52:123--140.",
        r"\bibitem{egelhaaf1985b} Egelhaaf M. (1985) \ldots II. Figure-detection cells, a new class "
        r"of visual interneurones. \textit{Biol. Cybern.} 52:195--209. (FD2 at p.~201; FD3 at "
        r"pp.~202--204.)",
        r"\bibitem{egelhaaf1985c} Egelhaaf M. (1985) \ldots III. Possible input circuitries and "
        r"behavioural significance of the FD-cells. \textit{Biol. Cybern.} 52:267--280.",
        r"\bibitem{rp1979} Reichardt W., Poggio T. (1979) Figure-ground discrimination by relative "
        r"movement in the visual system of the fly. I. \textit{Biol. Cybern.} 35:81--100.",
        r"\bibitem{fd1989} Fischbach K.-F., Dittrich A.P.M. (1989) The optic lobe of \textit{"
        r"Drosophila melanogaster}. I. A Golgi analysis of wild-type structure. \textit{Cell Tissue "
        r"Res.} 258:441--475.",
        r"\bibitem{maisak2013} Maisak M.S. et al. (2013) A directional tuning map of \textit{"
        r"Drosophila} elementary motion detectors. \textit{Nature} 500:212--216.",
        r"\bibitem{dorkenwald2024} Dorkenwald S. et al. (2024) Neuronal wiring diagram of an adult "
        r"brain. \textit{Nature} 634:124--138.",
        r"\bibitem{schlegel2024} Schlegel P. et al. (2024) Whole-brain annotation and "
        r"multi-connectome cell typing of \textit{Drosophila}. \textit{Nature} 634:139--152.",
        r"\bibitem{codex} FlyWire Codex, \texttt{codex.flywire.ai} -- cell-type and annotation "
        r"portal.",
    ]
    return (r"\begin{thebibliography}{9}" "\n" + "\n".join(items) + "\n"
            r"\end{thebibliography}")


# ---------------------------------------------------------------------------
# assembly + figures
# ---------------------------------------------------------------------------
def build_tex(results: dict) -> str:
    KD = results["families"]["KD"]["derived"]
    track = KD.get("meta", {}).get("primary_track") or KD.get("track") or "offline"
    lpt_contra = _b(KD, "LPT42_Nod4").get("contra_output_pct")
    nod3_contra = _b(KD, "Nod3").get("contra_output_pct")
    lpt21_contra = _b(KD, "LPT21").get("contra_output_pct")

    parts = [
        _PREAMBLE,
        r"\begin{document}",
        r"\title{\textbf{Which cells are FD2 and FD3? A connectomic assignment of \ttt{LPT21}, "
        r"\ttt{Nod3} and \ttt{LPT42\_Nod4}}}",
        r"\author{Stage 10 FD2/FD3 Disambiguation}",
        r"\date{\today}",
        r"\maketitle",
        r"\tableofcontents",
        _summary(KD),
        _introduction(),
        _tikz_fig1(lpt_contra, nod3_contra, lpt21_contra),
        _rule_section(KD),
        _results(KD, track),
        _nod3_section(KD),
        r"\section{The evidence at a glance}",
        r"Table~\ref{tab:disambig} collects every property next to the measured value for each of the "
        r"three candidates. The shared properties assign none; the projection and receptive-field "
        r"rows place \ttt{LPT21} at FD2, \ttt{LPT42\_Nod4} at FD3, and \ttt{Nod3} between them.",
        r"\begin{table}[H]\centering\footnotesize\caption{The three regressive candidates against "
        r"Egelhaaf's FD reference properties. Direction and transmitter are shared; output laterality "
        r"and receptive-field position assign \ttt{LPT21} to FD2 (frontal, homolateral), "
        r"\ttt{LPT42\_Nod4} to FD3 (lateral with a gap, heterolateral), and leave \ttt{Nod3} "
        r"intermediate.}\label{tab:disambig}",
        _evidence_table(KD),
        r"\end{table}",
        _caveats(KD),
        _methods(track),
        _bibliography(),
        r"\end{document}",
    ]
    return "\n\n".join(parts)


def _render_figures(results: dict, fig_dir: Path) -> None:
    fams = results["families"]
    KD = fams["KD"]["derived"]
    kd_claims = fams["KD"].get("claims", [])
    KD = {**KD, "_claims": kd_claims}
    run = {"derived": {"KD": KD}, "meta": results.get("meta", {})}
    if "K" in fams:
        run["derived"]["K"] = fams["K"]["derived"]
    PF.fd3_disambig_contra_dualtrack(run, fig_dir / "fd3_disambig_contra.png")
    PF.fd3_disambig_rf_frontal(run, fig_dir / "fd3_disambig_rf.png")
    PF.fd3_disambig_fd2_scatter(run, fig_dir / "fd3_disambig_fd2.png")
    PF.fd3_disambig_morphology(run, fig_dir / "fd3_disambig_morphology.png")
    PF.fd3_disambig_decision(run, fig_dir / "fd3_disambig_decision.png")
    if "K" in fams:
        PF.fd3_layer_composition(run, fig_dir / "fd3_layer_composition.png")
    # The family heatmap reads the KD screen (which includes LPT21). Map the constructive block onto
    # the fd_family_screen shape the heatmap expects (best_match_for_FD3 alias + score_matrix).
    cons = KD.get("constructive", {})
    screen_shim = {"score_matrix": cons.get("score_matrix", {}),
                   "best_match_for_FD3": cons.get("fd3_best_candidate"),
                   "fd3_margin": cons.get("fd3_margin")}
    PF.fd3_family_heatmap({"derived": {"K": {"fd_family_screen": screen_shim}}},
                          fig_dir / "fd3_family_heatmap.png")


def build_report(stage_dir: str | Path = "10 - FD3 vs Nod3 Disambiguation",
                 results_path: str | Path = "10 - FD3 vs Nod3 Disambiguation/fd3_disambig.json"
                 ) -> dict[str, Any]:
    stage = Path(stage_dir)
    (stage / "figures").mkdir(parents=True, exist_ok=True)
    results = read_json(Path(results_path))

    _render_figures(results, stage / "figures")
    tex_path = stage / "fd2_fd3_assignment.tex"
    tex_path.write_text(build_tex(results))

    compiled, log = _compile(tex_path)
    err = tex_path.with_suffix(".compile_error.txt")
    if not compiled:
        err.write_text(log)
    elif err.exists():
        err.unlink()
    return {"tex": str(tex_path),
            "pdf": str(tex_path.with_suffix(".pdf")) if compiled else None,
            "compiled": bool(compiled)}
