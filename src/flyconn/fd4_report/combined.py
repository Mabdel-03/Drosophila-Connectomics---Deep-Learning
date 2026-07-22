"""Build + compile the FD4 search report (LaTeX -> PDF) from the FD4 JSON.

Reads ONLY ``12 - FD4 Identification/fd4_circuit.json`` (the ``families.FD4`` block); no live CAVE
and no feathers, so the report re-runs token-free. Reuses the FD3 report machinery for the
preamble, escaping, figure wrapper and TeX compile.

The report documents a NEGATIVE result: Egelhaaf's FD4 shares FD1's entire progressive,
heterolateral, cholinergic, noduli-group output class, and no separable FD4 cell (or Nod1
sub-pair) exists in FlyWire v783. It is a four-part document parallel to the FD2/FD3 reports:
Part I the identity search and its null result (with a decomposed, null-aware confidence),
Part II the afferent inputs of the progressive figure arm any FD4 correlate would use, Part III
its efferent outputs, and Part IV the functional circuit. Prose is reader-facing, uses no verdict
vocabulary and no em dashes, and never overstates the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import read_json
from ..paper import fd4_figures as FF
from ..fd3_report.builder import _PREAMBLE, _esc, _g, _figure, _compile

NOD1 = r"\ttt{Nod1}"


def _d(results: dict) -> dict:
    return results["families"]["FD4"]["derived"]


# ---------------------------------------------------------------------------
# TikZ concept figure: the four FD receptive fields, with FD4's whole-eye lateral field marked as
# the one with no resolved connectome correlate.
# ---------------------------------------------------------------------------
def _tikz_fig1() -> str:
    return (r"""\begin{figure}[H]\centering
\begin{tikzpicture}[font=\small,>=Stealth]
  \draw[thick] (0,0) arc (180:90:5.6 and 2.2);
  \node[anchor=north east,font=\scriptsize] at (0,-0.05) {frontal ($0^\circ$)};
  \node[anchor=south west,font=\scriptsize] at (5.2,2.25) {lateral ($\sim$120$^\circ$)};
  % FD1 frontal (progressive)
  \fill[fd1c,opacity=0.28] (0.45,0.2) ellipse (0.55 and 0.28);
  \node[fd1c,font=\scriptsize,anchor=west] at (1.1,0.2) {\textbf{FD1} frontal, progressive $=$ \ttt{Nod1}};
  % FD3 fronto-lateral + gap (regressive)
  \fill[fd3c,opacity=0.22] (3.6,1.5) ellipse (1.0 and 0.30);
  \node[fd3c,font=\scriptsize,anchor=east] at (2.7,1.5) {\textbf{FD3} fronto-lateral $+$ gap $=$ \ttt{LPT42\_Nod4}};
  % FD4 whole-eye, lateral-weighted (dashed = not resolved)
  \draw[vorange,thick,dashed] (0.2,0.75) .. controls (2.5,0.95) and (4.0,1.15) .. (5.2,1.65);
  \fill[vorange,opacity=0.10] (0.2,0.75) .. controls (2.5,0.95) and (4.0,1.15) .. (5.2,1.65)
        -- (5.2,0.4) -- (0.2,0.4) -- cycle;
  \node[vorange,font=\scriptsize,anchor=west] at (2.3,2.0)
        {\textbf{FD4} whole-eye, lateral-weighted, progressive (\emph{no resolved correlate})};
  \draw[->,thick,fd1c] (4.4,-0.55) -- (2.3,-0.55)
        node[midway,below,font=\scriptsize]{progressive (front$\to$back)};
  % output class: FD1 and FD4 share the same heterolateral noduli-group axon
  \node[draw,rounded corners,fill=fd1c!12,font=\scriptsize,align=center,text width=2.7cm]
        (d) at (1.3,-2.3) {progressive figure cell\\(lobula plate)};
  \node[draw,rounded corners,fill=fd1c!12,font=\scriptsize,align=center,text width=2.9cm]
        (pof) at (6.6,-2.3) {contralateral\\posterior optic foci};
  \draw[->,thick,fd1c] (d) -- node[above,midway,font=\scriptsize,align=center]
        {heterolateral noduli-group axon\\(shared by FD1, FD3, FD4)} (pof);
  \node[draw,rounded corners,vorange,font=\scriptsize,align=center,text width=8.8cm]
        at (3.9,-3.7) {FD4 shares FD1's entire progressive, heterolateral, cholinergic,
        noduli-group output class. In FlyWire v783 that class is occupied by a single homogeneous
        type, \ttt{Nod1} (= FD1); no separable FD4 cell is resolved.};
\end{tikzpicture}
\caption{\textbf{The FD4 cell and why it is hard to identify.} Egelhaaf (1985, Part II, pp.~204--206)
described FD4 as a progressive (front-to-back) figure-detection cell whose excitatory receptive
field covers the whole horizontal extent of the ipsilateral eye and is most sensitive laterally,
with no frontal gap, bidirectional contralateral inhibition, and a restricted dorso-ventral
dendrite. Its axon is a heterolateral noduli-group element using the same pathway as the FD1nod and
FD3 cells. FD4 therefore shares FD1's entire output class and is distinguished from FD1 only by its
receptive field and dendrite. This report searches the FlyWire connectome for that cell and reports
the result.}\label{fig:concept}
\end{figure}
""")


# ---------------------------------------------------------------------------
# Part I: the identity search and its null result.
# ---------------------------------------------------------------------------
def _summary(D: dict) -> str:
    sc = D.get("candidate_screen", {}); sp = D.get("nod1_split", {}); cf = D.get("confidence", {})
    return (
        r"\textbf{Summary.}\quad "
        r"Egelhaaf defined FD4 as a progressive (front-to-back) figure-detection cell whose excitatory "
        r"receptive field spans the whole horizontal extent of the ipsilateral eye and is most sensitive "
        r"laterally, with no frontal gap, bidirectional contralateral inhibition, a cholinergic "
        r"heterolateral noduli-group axon to the contralateral posterior optic foci, and a dendrite that "
        r"spans the full horizontal but a restricted dorso-ventral extent of the lobula plate. This "
        r"report searches the FlyWire connectome for that cell. The search returns a negative result. "
        r"FD4 shares the entire output class of the already-identified FD1 cell "
        rf"(\ttt{{Nod1}}): progressive, heterolateral, cholinergic, and noduli-group. A funnel over all "
        rf"{_esc(sc.get('universe_size'))} FlyWire cell types finds \ttt{{Nod1}} to be the "
        rf"{'sole' if sc.get('only_survivor_is_prog_type') else 'leading'} progressive figure-output "
        r"type; every other progressive midline-crossing cell is an inhibitory or centrifugal feedback "
        r"element. The \ttt{Nod1} type contains four cells, but they form one homogeneous frontal "
        rf"population (no partition isolates a lateral, wide, dorso-ventrally restricted FD4 pair; "
        rf"homogeneous = {_esc(sp.get('homogeneous'))}), most consistent with Egelhaaf's two "
        rf"anatomical FD1 variants, "
        r"and none of them carries FD4's whole-eye lateral receptive field or restricted dorso-ventral "
        r"dendrite. FD4 is therefore not individually resolved in FlyWire v783. The decomposed, "
        rf"null-aware confidence that FD4 is individually resolved is {_esc(cf.get('point_estimate'))} "
        rf"(plausible range {_esc(cf.get('interval'))}). The report states this negative result, its full "
        r"evidentiary basis, and the complete progressive figure arm that any FD4 correlate would use.")


def _introduction() -> str:
    return r"""\section{Introduction}

\textbf{Figure-ground discrimination and the FD cells.}\quad
A textured object is invisible against a matched background until it moves at a different velocity or
phase; relative motion then makes it salient. Reichardt and Poggio showed behaviourally that flies
detect and track such figures~\cite{rp1979}. Egelhaaf, in a three-part study, described a class of
lobula-plate tangential cells, the figure-detection (FD) cells, each more sensitive to a small moving
figure than to wide-field motion~\cite{egelhaaf1985b}. Four were distinguished by preferred direction
and receptive-field organisation. \textbf{FD1} is progressive with a narrow frontal field; \textbf{FD2}
is regressive with a frontal field; \textbf{FD3} is regressive with a fronto-lateral field and a
frontal gap; \textbf{FD4} is progressive with a receptive field that covers the whole horizontal
extent of the ipsilateral eye, is most sensitive laterally (peaks at azimuth $50$--$80^\circ$,
half-maximum width $80$--$110^\circ$, reaching beyond $120^\circ$), has no frontal gap, receives
bidirectional contralateral inhibition, and, like FD1 and FD3, is a heterolateral noduli-group output
element whose axon crosses to the contralateral posterior optic foci. Its dendrite spans the full
horizontal extent of the lobula plate but not its full dorso-ventral extent, and it has no second
arborisation in the lateral protocerebrum~\cite[p.~204--206]{egelhaaf1985b}.

\textbf{The connectome era and the mapping problem.}\quad
The adult \emph{Drosophila} FlyWire connectome and its cell-type annotations provide the
synapse-resolution wiring needed to ask which modern cell type corresponds to each FD
cell~\cite{dorkenwald2024,schlegel2024,codex}. T4 and T5 encode ON/OFF edge motion and segregate by
preferred direction into the four lobula-plate layers~\cite{maisak2013,fd1989}: layer-a front-to-back
(progressive), layer-b back-to-front (regressive). FD1 has been mapped to \ttt{Nod1}, FD2 to
\ttt{LPT21}, and FD3 to \ttt{LPT42\_Nod4}. This report asks which cell type is FD4.

\textbf{Why FD4 is the hard case.}\quad
The defining connectome signature of an FD output cell is its preferred direction (from its T4/T5
layer), its output side (from where its synapses land), and its transmitter. On all three, FD4 is
identical to FD1: both are progressive (layer-a), both are heterolateral (their axon crosses to the
contralateral posterior optic foci), and both are cholinergic. Egelhaaf states this directly: FD4
uses the same axonal pathway as the FD1nod and FD3 cells. FD4 is separated from FD1 only by its
receptive field (whole-eye and lateral, versus FD1's frontal and narrow) and by its dendrite (a
restricted dorso-ventral extent, versus FD1's full extent). The search for FD4 is therefore a search
for a progressive, heterolateral, cholinergic cell that is distinct from FD1 in receptive field and
dendrite. Figure~\ref{fig:concept} fixes the FD4 phenotype and the difficulty."""


def _identity_results(D: dict) -> str:
    sc = D.get("candidate_screen", {}); sp = D.get("nod1_split", {}); ph = D.get("phenotype", {})
    fd3r = ph.get("fd3_ruled_out", {})
    P = [r"\section{The search, step by step}"]
    P.append(r"We take the search in four steps: eliminate every candidate for FD4's output class, "
             r"test whether the one surviving type resolves into an FD1 and an FD4 population, measure "
             r"the FD4-defining properties on the best available candidate, and exclude the regressive "
             r"cell whose absolute receptive field happens to be FD4-shaped.")

    P.append(r"\paragraph{No progressive figure-output cell exists besides \ttt{Nod1}.} An FD4 cell "
             r"would be progressive (dominant lobula-plate layer a), heterolateral (its axon crossing to "
             r"the contralateral posterior optic foci), cholinergic, and a small, individually "
             rf"identifiable population. Screening all {_esc(sc.get('universe_size'))} FlyWire cell types "
             rf"for this conjunction leaves {_esc(sc.get('n_survivors'))} survivor: "
             rf"\ttt{{{', '.join(sc.get('survivors', [])) or 'none'}}} (Fig.~\ref{{fig:elim}}). Every "
             r"other progressive, midline-crossing cell fails on transmitter or class: they are "
             r"GABAergic or glutamatergic wide-field inhibitors, or centrifugal feedback cells, not "
             r"excitatory figure outputs. \ttt{Nod1} is the already-identified FD1 cell. There is thus "
             r"no separate progressive figure-output cell in the connectome for FD4 to be.")
    P.append(_figure("figures/fd4_candidate_elimination.png",
                     r"\textbf{The candidate elimination.} The FD4 output-class funnel over every "
                     r"FlyWire cell type: progressive (layer-a) and low-copy, then heterolateral, then "
                     r"cholinergic, then a visual projection output. The single survivor is \ttt{Nod1} "
                     r"(= FD1). Each layer-a low-copy type and the reason it fails are listed at right.",
                     "fig:elim"))

    P.append(r"\paragraph{The \ttt{Nod1} type is one homogeneous population, not FD1 plus FD4.} The "
             r"FlyWire \ttt{Nod1} type contains four cells, two per hemisphere, where the other FD types "
             r"contain a single bilateral pair. Egelhaaf described two anatomical FD1 representatives "
             r"(FD1nod and FD1pof) that were both classed FD1, so a four-cell type is expected to pool "
             r"two same-side cells. We tested whether those two populations are an FD1 pair and an FD4 "
             r"pair. They are not. The two same-side cells share a large fraction of their input "
             rf"partners (Jaccard {_esc(sp.get('same_side_input_jaccard'))}, against "
             rf"{_esc(sp.get('cross_side_input_jaccard'))} across sides), the signature of two copies of "
             r"one cell type rather than two different cells. Scoring every two-against-two partition of "
             r"the four cells on the FD4-discriminating features (receptive-field laterality and width, "
             r"dorso-ventral dendrite extent, and second arbor) isolates no FD4-like pair: the only "
             r"partition that groups the cells at all is the left-versus-right hemisphere split, a "
             r"mirror-symmetry effect of the cells' absolute positions rather than a functional FD1 "
             r"versus FD4 axis, and the more-lateral pair has a larger, not a smaller, dorso-ventral "
             r"dendrite, the opposite of the FD4 signature. All four cells are frontal, span the full "
             r"dorso-ventral extent, and have a single arbor. There is no lateral, wide-field, "
             r"restricted-dendrite FD4 pair inside \ttt{Nod1} (Fig.~\ref{fig:homog}).")
    P.append(_figure("figures/fd4_nod1_homogeneity.png",
                     r"\textbf{The \ttt{Nod1} quartet is homogeneous.} (left) The four \ttt{Nod1} cells "
                     r"placed by receptive-field centroid (frontal to lateral) and width; all four sit "
                     r"frontally, none in the lateral, wide region an FD4 cell would occupy. (right) The "
                     r"two possible bilateral pairings and their separation scores; both are negative, so "
                     r"the four cells are one population, not a separable FD1 plus FD4 split.",
                     "fig:homog"))
    P.append(_figure("figures/fd4_nod1_quartet.png",
                     r"\textbf{The four \ttt{Nod1} cells, reconstructed.} Each \ttt{Nod1} cell in "
                     r"anatomical space. All four share one frontal, full-dorso-ventral, single-arbor "
                     r"morphology. An FD1 plus FD4 split would show one pair with a restricted "
                     r"dorso-ventral dendrite; it does not.", "fig:quartet"))

    P.append(r"\paragraph{What the best candidate does and does not match.} Measuring the "
             r"FD4-defining properties on the \ttt{Nod1} population "
             rf"({_esc(ph.get('n_matched'))} of {_esc(ph.get('n_properties'))} matched) shows the "
             r"pattern behind the null. The properties it matches are exactly the ones FD4 shares with "
             r"FD1: progressive direction, heterolateral axon, cholinergic transmitter, and a single "
             r"arbor. The properties it fails are exactly the ones that would distinguish FD4 from FD1: "
             r"a whole-eye lateral receptive field, and a restricted dorso-ventral dendrite. In other "
             r"words, the connectome contains the FD1 half of the FD4 phenotype but not the "
             r"FD4-specific half (Fig.~\ref{fig:phenotype} and Fig.~\ref{fig:rfcmp}).")
    P.append(_figure("figures/fd4_phenotype.png",
                     r"\textbf{The FD4 phenotype crosswalk.} Each FD4-defining property and whether the "
                     r"\ttt{Nod1} population matches it. Every match is a property FD4 shares with FD1; "
                     r"every FD4-discriminating property is unmatched.", "fig:phenotype"))
    P.append(_figure("figures/fd4_rf_comparison.png",
                     r"\textbf{The four FD receptive fields.} FD1 frontal, FD2 frontal, FD3 "
                     r"fronto-lateral with a gap, and FD4 whole-eye and lateral. The measured \ttt{Nod1} "
                     r"field sits frontally with FD1; FD4's whole-eye lateral field (dashed) is the one "
                     r"with no resolved connectome correlate.", "fig:rfcmp"))
    P.append(_figure("figures/fd4_input_hexmap.png",
                     r"\textbf{Receptive fields in eye coordinates.} The T4/T5 input columns of each "
                     r"\ttt{Nod1} cell (all frontal) next to \ttt{LPT42\_Nod4} (= FD3, lateral) for "
                     r"contrast. No \ttt{Nod1} cell carries the whole-eye lateral field of FD4.",
                     "fig:hexmap"))

    P.append(r"\paragraph{The regressive cell that looks FD4-shaped is FD3, not FD4.} The cell "
             r"\ttt{LPT42\_Nod4}, identified as FD3, has an absolute receptive field that reaches "
             r"laterally toward the caudal pole and is, on absolute geometry alone, at least as "
             r"FD4-shaped as FD3-shaped. It is nonetheless FD3, not FD4, because its preferred direction "
             rf"is regressive: {_esc(fd3r.get('fd3_layer_b_pct'))}\% of its T4/T5 input is the "
             r"back-to-front (layer-b) channel, whereas FD4 is progressive (layer-a). Direction is the "
             r"clean separator. \ttt{LPT42\_Nod4} also has the frontal gap that Egelhaaf certifies as "
             r"unique to FD3, which FD4 does not have. The FD4-shaped absolute geometry of the FD3 cell "
             r"is therefore a coincidence of receptive-field position, not an FD4 identity.")
    return "\n\n".join(P)


def _confidence_section(D: dict) -> str:
    cf = D.get("confidence", {})
    disc = cf.get("discounts", {})
    return (
        r"\section{Confidence in the result}" "\n\n"
        r"The result is expressed as a decomposed, null-aware confidence that FD4 is individually "
        r"resolved in the connectome, rather than as an assertion either way. It combines the fraction "
        r"of FD4-\emph{discriminating} properties the best candidate matches (the shared FD1/FD4 class "
        r"properties carry no power to separate FD4 from FD1 and are excluded from the numerator), a "
        r"within-\ttt{Nod1} separability factor (zero, because the four cells do not separate into an "
        r"FD1 and an FD4 population), and explicit discounts. The discounts are larger than for the "
        r"other FD cells and are stated so a reader can reconstruct the estimate: FD4 shares FD1's "
        rf"entire output class (a collinearity discount of {_esc(disc.get('fd1_collinearity_shared_class'))}), "
        r"FD4 is not an independently annotated FlyWire type "
        rf"({_esc(disc.get('not_a_distinct_type'))}), there is no independent FD4 anchor "
        rf"({_esc(disc.get('no_independent_anchor'))}), the modern cell-typing authority declined the "
        rf"one-for-one FD1/2/3/4 mapping ({_esc(disc.get('literature_declined_mapping'))}), and further "
        rf"FD cells cannot be excluded ({_esc(disc.get('further_fd_cells_possible'))}). The resulting "
        rf"point estimate is {_esc(cf.get('point_estimate'))}, with a plausible range of "
        rf"{_esc(cf.get('interval'))} and a ceiling of {_esc(cf.get('ceiling'))} set by the "
        r"FD1-collinearity cap (Fig.~\ref{fig:confidence}). The estimate is low and the interval wide "
        r"by construction: the connectome cannot resolve a cell that shares its entire measurable "
        r"output class with an already-identified cell and differs only in a receptive field and "
        r"dendrite extent that are not separately represented in the current annotation." "\n\n"
        + _figure("figures/fd4_confidence.png",
                  r"\textbf{The confidence, decomposed.} (left) The point estimate that FD4 is "
                  r"individually resolved, its wide interval, and the FD1-collinearity ceiling. (right) "
                  r"The discounts that set it, led by the shared-output-class term.", "fig:confidence"))


def _evidence_table(D: dict) -> str:
    sc = D.get("candidate_screen", {}); sp = D.get("nod1_split", {}); cf = D.get("confidence", {})
    ph = D.get("phenotype", {}); fd3r = ph.get("fd3_ruled_out", {})
    rows = [
        ("Prefers progressive (front-to-back) motion", "Egelhaaf: FD4 excited by progressive motion",
         r"the progressive figure-output class exists (dominant layer-a), shared with FD1"),
        ("Whole-eye, laterally-weighted receptive field", "Egelhaaf: whole horizontal eye, most sensitive laterally, no gap",
         r"not found: all \ttt{Nod1} cells are frontal, none is whole-eye lateral"),
        ("Restricted dorso-ventral dendrite", "Egelhaaf: dorso-proximal and ventro-proximal parts devoid of dendrite",
         r"not found: all \ttt{Nod1} cells span the full dorso-ventral extent"),
        ("Heterolateral noduli-group axon", "Egelhaaf: axon to the contralateral posterior optic foci",
         r"present in the \ttt{Nod1} population (shared with FD1 and FD3)"),
        ("Cholinergic output", "FD cells are excitatory output neurons",
         r"acetylcholine (shared with FD1)"),
        ("Bidirectional contralateral inhibition", "Egelhaaf: FD4 inhibited by contra motion in either direction (FD1 unidirectional)",
         r"the one FD1-versus-FD4 discriminator not exhausted within \ttt{Nod1} (see caveats)"),
        ("A separate progressive figure-output cell", "the identification requires a distinct cell",
         rf"none: \ttt{{Nod1}} is the {'sole' if sc.get('only_survivor_is_prog_type') else 'leading'} survivor of the output-class funnel"),
        ("A separable FD4 sub-pair of \\ttt{Nod1}", "a four-cell type could pool FD1 and FD4",
         rf"none: the four cells are homogeneous (no partition isolates an FD4-like pair; separable = {_esc(sp.get('separable'))})"),
        ("The FD4-shaped regressive cell is not FD4", "FD4 is progressive; the frontal gap is FD3-unique",
         rf"\ttt{{LPT42\_Nod4}} is {_esc(fd3r.get('fd3_layer_b_pct'))}\% layer-b (regressive) = FD3, not FD4"),
        ("Confidence FD4 is individually resolved", "quantify the result honestly",
         rf"point {_esc(cf.get('point_estimate'))}, range {_esc(cf.get('interval'))}, verdict {_esc(cf.get('identity_verdict'))}"),
    ]
    head = (r"\renewcommand{\arraystretch}{1.25}" "\n" r"\setlength{\tabcolsep}{3pt}" "\n"
            r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.27\textwidth} "
            r">{\raggedright\arraybackslash}p{0.33\textwidth} "
            r">{\raggedright\arraybackslash}p{0.34\textwidth}@{}}" "\n"
            r"\toprule \textbf{FD4 property or test} & \textbf{What Egelhaaf reported} & "
            r"\textbf{What we find in FlyWire v783} \\\midrule\endhead" "\n")
    body = "\n".join(rf"{a} & {b} & {c} \\" for a, b, c in rows)
    return (head + body + "\n" + r"\bottomrule\end{longtable}" + "\n"
            r"\renewcommand{\arraystretch}{1.0}" "\n" r"\setlength{\tabcolsep}{6pt}")


def _identity_caveats() -> str:
    return (r"\section{What the data can and cannot settle}" "\n\n"
            r"\textbf{The result is a negative one, stated as such.} No cell in FlyWire v783 carries the "
            r"FD4-specific receptive field and dendrite together with the progressive noduli-group "
            r"output class. This is reported as a null result rather than forced into an identification. "
            r"\textbf{FD4 shares FD1's entire measurable output class.} Preferred direction, output side, "
            r"and transmitter, the three connectome signatures that identify an FD output cell, are "
            r"identical for FD1 and FD4. The connectome can therefore locate the progressive figure-"
            r"output class but cannot, from these signatures alone, split it into FD1 and FD4. "
            r"\textbf{The distinguishing features are not separately represented.} FD4 differs from FD1 "
            r"in receptive-field width and laterality and in dorso-ventral dendrite extent; measured on "
            r"the \ttt{Nod1} cells, these do not separate the population, so either the FD4 cell is not "
            r"reconstructed as distinct in this annotation, or it is pooled within \ttt{Nod1} below the "
            r"resolution of these measurements. \textbf{The receptive-field axis is relative.} The "
            r"frontal reading of the \ttt{Nod1} cells is made relative to the FD3 cell in the same "
            r"coordinate frame and gives the correct FD1-frontal, FD3-lateral ordering, so it does not "
            r"depend on an absolute azimuth calibration. \textbf{The modern typing declined the FD "
            r"mapping.} The cell-typing authority that named these cells stated that the FD cells are "
            r"almost certainly homologous to noduli-group cells but did not establish one-for-one "
            r"FD1/2/3/4 matches; every FD assignment in this program, including the FD1 anchor this "
            r"search rests on, is a derived correlate, not a literature-endorsed identity. "
            r"\textbf{One FD1-versus-FD4 discriminator was not exhausted.} Beyond receptive field and "
            r"dendrite, Egelhaaf gives one further property that separates FD4 from FD1 and is in "
            r"principle connectome-measurable: FD4 is inhibited by contralateral motion in either "
            r"direction (bidirectional), whereas FD1 is reduced only by contralateral regressive motion "
            r"(unidirectional). This is an inhibitory-input property, not one of the shared output-class "
            r"signatures, and a targeted search for a bidirectionally contra-inhibited subset within "
            r"\ttt{Nod1} was not carried out here. It is the one avenue by which a distinct FD4 could yet "
            r"be resolved, and it is left open rather than claimed either way. \textbf{Every load-bearing "
            r"claim was tested adversarially.} The candidate elimination, the \ttt{Nod1} homogeneity, the "
            r"FD1-collinearity, the receptive-field axis, the exclusion of the FD4-shaped FD3 cell, the "
            r"Nod-family siblings, and the confidence interval were each subjected to an independent "
            r"attempt at refutation with fresh connectome queries; all seven survived, and the residual "
            r"caveats surfaced there, including the bidirectional-inhibition avenue above, are the ones "
            r"stated in this report.")


# ---------------------------------------------------------------------------
# Part II: the progressive figure arm any FD4 correlate would use.
# ---------------------------------------------------------------------------
def _afferent(D: dict) -> str:
    aff = D.get("afferent", {}); cen = aff.get("census", {}); casc = aff.get("upstream_cascade", {})
    sheet = D.get("sheet", {})
    la = (cen.get("t4t5_layer_frac") or {}).get("a")
    P = [r"\part{Inputs: the progressive figure arm}"]
    P.append(r"\textbf{Summary.}\quad Although FD4 is not individually resolved, the progressive figure "
             r"arm that any FD4 correlate would use is fully present in the connectome and is traced here "
             r"on the \ttt{Nod1} (= FD1) population. It reads front-to-back (layer-a) motion from the "
             rf"T4a and T5a detectors ({_esc(la)}\% of its T4/T5 drive is layer-a), relays it through a "
             r"columnar projection sheet, and is gated by wide-field inhibitory cells. This is the "
             r"progressive mirror of the regressive FD2 and FD3 arms.")
    P.append(r"\section{The motion detectors that drive the progressive arm}")
    P.append(rf"The progressive figure arm reads the front-to-back (layer-a) motion channel: "
             rf"{_esc(la)}\% of its T4/T5 input is layer-a, carried by both the ON detector \ttt{{T4a}} "
             rf"and the OFF detector \ttt{{T5a}} (ON/OFF split {_esc(cen.get('on_off_split'))}). This is "
             r"the opposite motion channel from FD2 and FD3, which read the regressive (layer-b) "
             r"detectors, and it matches the progressive direction Egelhaaf reported for FD1 and FD4 "
             r"(Fig.~\ref{fig:census}).")
    P.append(_figure("figures/fd4_input_census.png",
                     r"\textbf{What feeds the progressive figure cell.} The direct inputs of \ttt{Nod1}, "
                     r"coloured by class, with the front-to-back (layer-a) T4a/T5a motion detectors "
                     r"marked. This is the arm any FD4 correlate would read.", "fig:census"))
    P.append(r"\section{From the eye to the detectors}")
    P.append(rf"The T4a and T5a cells that drive the progressive arm are fed by the canonical ON and OFF "
             r"medulla cells, which are fed by the lamina and, before it, the photoreceptors. The whole "
             r"cascade from the eye to the detectors is present in the connectome and matches published "
             r"optic-lobe wiring (lamina present: "
             rf"{_esc(casc.get('lamina_present'))}; photoreceptors: "
             rf"{_esc(casc.get('photoreceptor_present'))}). This cascade is the general column-to-column "
             r"wiring of the optic lobe, the pathway that culminates in the progressive detectors, not "
             r"a per-object trace.")
    P.append(r"\section{Where the input lands, and the sheet and gate}")
    P.append(rf"Each motion detector looks at one point in the eye, so the detectors feeding the "
             r"progressive cell draw out its receptive field on the eye's lattice; for the \ttt{Nod1} "
             r"population this field is frontal (Fig.~\ref{fig:hexmap2}). The motion detectors also reach "
             rf"the cell through a columnar projection sheet, the layer-a sheet "
             rf"\ttt{{{_esc(sheet.get('named_sheet'))}}}, and the cell is gated by wide-field "
             rf"inhibitory cells led by \ttt{{{_esc(D.get('gate', {}).get('named_inhibitor'))}}}. This "
             r"is the same sheet-and-gate architecture as the FD1 arm, as expected for a progressive "
             r"figure cell (Fig.~\ref{fig:arbor}).")
    P.append(_figure("figures/fd4_input_hexmap.png",
                     r"\textbf{The progressive arm's input in eye coordinates.} The T4/T5 input columns "
                     r"of the \ttt{Nod1} cells (frontal) and \ttt{LPT42\_Nod4} (= FD3, lateral) for "
                     r"contrast.", "fig:hexmap2"))
    P.append(_figure("figures/fd4_arbor_inputs.png",
                     r"\textbf{Inputs on the progressive cell's arbor.} Each \ttt{Nod1} cell's "
                     r"reconstructed shape with its input synapses drawn at their real positions and "
                     r"coloured by the class of partner: front-to-back motion detectors, columnar "
                     r"sheets, and inhibitory cells.", "fig:arbor"))
    P.append(r"\section{What the data can and cannot settle}" "\n\n"
             r"\textbf{The arm is traced on FD1, not on a resolved FD4.} Because FD4 is not individually "
             r"resolved, the afferent trace is measured on the \ttt{Nod1} (= FD1) population, which uses "
             r"the same progressive channel, sheet, and gate any FD4 correlate would. It is shown as the "
             r"realized progressive figure arm, not as an FD4-specific trace. \textbf{The upstream "
             r"cascade is general wiring.} The photoreceptor-to-detector cascade is measured cell-type "
             r"by cell-type across the optic lobe and matches the known column wiring.")
    return "\n\n".join(P)


# ---------------------------------------------------------------------------
# Part III: efferent outputs of the progressive arm.
# ---------------------------------------------------------------------------
def _efferent(D: dict) -> str:
    eff = D.get("efferent", {}); direct = eff.get("direct", {}); mot = eff.get("motor", {})
    top = direct.get("ranking", [{}])[0].get("cell_type") if direct.get("ranking") else None
    P = [r"\part{Outputs: descending and motor-system partners}"]
    P.append(r"\textbf{Summary.}\quad The progressive figure arm reaches descending neurons that project "
             r"into the ventral nerve cord and contact motor-control networks. Traced on the \ttt{Nod1} "
             rf"population, its leading direct descending target is \ttt{{{_esc(top)}}}, the same "
             r"wing-steering command neuron that the FD2 and FD3 arms reach, so the figure cells "
             r"converge on a shared descending target. Any FD4 correlate, sharing this arm, would reach "
             r"the same steering pathway.")
    P.append(r"\section{The descending neurons the progressive arm contacts}")
    P.append(rf"\ttt{{Nod1}} makes {_esc(direct.get('dn_syn'))} output synapses onto "
             rf"{_esc(len(direct.get('ranking', [])))} descending-neuron types, led by "
             rf"\ttt{{{_esc(top)}}} (Fig.~\ref{{fig:dnrank}}). \ttt{{{_esc(top)}}} is the convergent "
             r"wing-steering command neuron shared with the FD2 and FD3 arms. Egelhaaf could name only "
             r"``descending neurones'' generically; the connectome names them.")
    P.append(_figure("figures/fd4_dn_ranking.png",
                     r"\textbf{The descending neurons the progressive arm contacts.} Each bar is one "
                     r"descending-neuron type, ranked by synapses received from \ttt{Nod1}. Red marks "
                     r"neurons known to drive wing-steering muscles.", "fig:dnrank"))
    P.append(r"\section{Motor systems contacted by those neurons}")
    P.append(rf"Following each descending neuron into the male nerve-cord connectome and weighting by "
             rf"how strongly the progressive cell drives it, the descending output is biased toward the "
             rf"wing-steering motor system (dominant system: "
             rf"\ttt{{{_esc(mot.get('dominant_motor_system'))}}}; Fig.~\ref{{fig:motorsys}}). As in the "
             r"FD2 and FD3 reports, the label is by descending-neuron identity; weighting by actual "
             r"muscle targets softens the picture toward a mixed steering and neck-gaze bias, and the "
             r"contact is sparse. The pathway supports a steering role, consistent with Egelhaaf's "
             r"proposal that the FD cells contribute to yaw-torque control, but does not by itself prove "
             r"a behaviour.")
    P.append(_figure("figures/fd4_motor_systems.png",
                     r"\textbf{Motor-system proxy.} The motor systems reached by the progressive arm's "
                     r"descending output, weighted by how strongly the cell drives each descending "
                     r"neuron. Wing-steering has the largest share.", "fig:motorsys"))
    P.append(r"\section{The circuit in the brain}")
    P.append(r"Seen alongside the cells it connects, the progressive figure cell sits anatomically "
             r"between the lobula-plate motion detectors and the descending neurons that contact "
             r"wing-steering motor systems (Fig.~\ref{fig:circuit3d}).")
    P.append(_figure("figures/fd4_circuit_3d.png",
                     r"\textbf{The progressive arm in the brain.} Reconstructed neurons of the "
                     r"progressive figure arm in their true anatomical positions: the \ttt{T4a}/\ttt{T5a} "
                     r"motion detectors, the layer-a sheet, the \ttt{Nod1} cells, and the descending "
                     r"neuron they lead to.", "fig:circuit3d"))
    P.append(r"\section{What the data can and cannot settle}" "\n\n"
             r"\textbf{The output is traced on FD1.} As on the input side, the efferent trace is measured "
             r"on the \ttt{Nod1} population because FD4 is not individually resolved; it is the realized "
             r"progressive-arm output, which any FD4 correlate would share. \textbf{The steering label is "
             r"by neuron identity, not muscle count}, and \textbf{the muscle map crosses two "
             r"connectomes} by shared descending-neuron name, as in the FD2 and FD3 reports.")
    return "\n\n".join(P)


# ---------------------------------------------------------------------------
# Part IV: functional circuit.
# ---------------------------------------------------------------------------
def _functional(D: dict) -> str:
    census = D.get("census", {}); sheet = D.get("sheet", {}); gate = D.get("gate", {})
    eff = D.get("efferent", {})
    top = eff.get("direct", {}).get("ranking", [{}])[0].get("cell_type") if eff.get("direct", {}).get("ranking") else None
    inp = census.get("input", {}); out = census.get("output", {})
    P = [r"\part{The functional figure-ground circuit of the progressive arm}"]
    P.append(r"\textbf{Summary.}\quad This part names the working progressive figure-ground circuit, in "
             r"the same terms as the FD1, FD2, and FD3 arms: front-to-back motion detectors to the "
             rf"layer-a sheet \ttt{{{_esc(sheet.get('named_sheet'))}}} to the progressive figure cell to "
             rf"the steering neuron \ttt{{{_esc(top)}}}, gated by the wide-field inhibitor "
             rf"\ttt{{{_esc(gate.get('named_inhibitor'))}}}. This is the circuit an FD4 correlate would "
             r"occupy.")
    P.append(r"\section{The progressive cell's connectivity, comprehensively}")
    P.append(rf"The \ttt{{Nod1}} population receives about {_esc(_g(inp, 'total_syn'))} input synapses "
             rf"from roughly {_esc(_g(inp, 'n_partners'))} partner cells, and makes about "
             rf"{_esc(_g(out, 'total_syn'))} output synapses onto roughly {_esc(_g(out, 'n_partners'))} "
             r"partners. The inputs are dominated by the motion detectors, the columnar sheets, and "
             r"inhibitory cells; the outputs reach central brain cells and, through the descending "
             r"neurons, the steering muscles (Fig.~\ref{fig:wheel}).")
    P.append(_figure("figures/fd4_connectivity_wheel.png",
                     r"\textbf{The progressive cell's comprehensive connectivity.} Every major input "
                     r"(left) and output (right) partner type of \ttt{Nod1}.", "fig:wheel"))
    P.append(r"\section{The circuit, named end to end}")
    P.append(rf"Putting the pieces together, the progressive figure-ground circuit reads: the "
             rf"\ttt{{T4a}}/\ttt{{T5a}} front-to-back motion detectors drive the layer-a sheet "
             rf"\ttt{{{_esc(sheet.get('named_sheet'))}}}, which relays to the progressive figure cell; "
             rf"the cell drives the steering command neuron \ttt{{{_esc(top)}}} and thence the wing; and "
             rf"the wide-field cell \ttt{{{_esc(gate.get('named_inhibitor'))}}} gates the circuit. This "
             r"is the same architecture as the FD1, FD2, and FD3 arms, built from the progressive "
             r"channel. FD4, if it were individually resolved, would be a second progressive figure cell "
             r"reading this same arm with a wider, more lateral receptive field. The connectome contains "
             r"the arm but not a second cell reading it (Fig.~\ref{fig:funcircuit}).")
    P.append(_figure("figures/fd4_rf_comparison.png",
                     r"\textbf{Where FD4 would sit.} The four FD receptive fields on one eye's azimuth "
                     r"axis. FD4's whole-eye lateral field (dashed) is the position a second progressive "
                     r"figure cell would occupy; it has no resolved connectome correlate.",
                     "fig:funcircuit"))
    P.append(r"\section{What the data can and cannot settle}" "\n\n"
             r"\textbf{Sign is not read from wiring.} That a GABAergic contact is inhibitory, and that "
             r"the gate shapes the figure response, are expectations from physiology; the connectome "
             r"supplies the connections, not the sign. \textbf{The circuit is the FD1 arm.} It is named "
             r"as the progressive figure arm any FD4 correlate would use, not as an FD4-specific circuit, "
             r"because no distinct FD4 cell is resolved.")
    return "\n\n".join(P)


def _methods(D: dict) -> str:
    track = D.get("track", "offline")
    st = D.get("split_tracks", {})
    return (r"\section{Methods and provenance}" "\n\n"
            r"\textbf{Connectome.} FlyWire FAFB materialization v783 (\ttt{synapses\_nt\_v1}, no cleft "
            rf"threshold), track \ttt{{{_esc(track)}}}. The \ttt{{Nod1}} homogeneity finding, the "
            r"load-bearing negative result, was checked across the offline, live, and v630 tracks "
            rf"(tracks: {_esc(list(st.keys()) if isinstance(st, dict) else st)}). Neuron annotations are "
            r"the Schlegel/Codex tables. \textbf{Candidate elimination} funnels every cell type by "
            r"dominant lobula-plate layer, output laterality, transmitter, and copy number, using the "
            r"per-type scan of all FlyWire types. \textbf{Preferred direction} is the synapse-weighted "
            r"T4/T5 lobula-plate layer composition~\cite{fd1989,maisak2013}. \textbf{Receptive field}: "
            r"T4/T5 inputs carry hex-lattice coordinates; each \ttt{Nod1} cell's field is measured per "
            r"cell and compared in the shared frame. \textbf{The \ttt{Nod1} split} enumerates the two "
            r"bilateral pairings of the four cells and scores each on receptive-field laterality and "
            r"width, dorso-ventral dendrite extent, and second-arbor bimodality; a positive split "
            r"requires a positive separation score and an FD4-featured pair. \textbf{Confidence} combines "
            r"the discriminating-property match fraction, the within-\ttt{Nod1} separability, and "
            r"explicit discounts led by the FD1-collinearity term. \textbf{Afferent, sheet, gate, "
            r"efferent} reuse the FD-circuit machinery parameterised on the progressive (layer-a) arm and "
            r"the \ttt{Nod1} population; the motor read-out crosses into the male CNS by shared "
            r"descending-neuron name. \textbf{Reproducibility}: built from the FD4 verification JSON.")


def _bibliography() -> str:
    items = [
        r"\bibitem{egelhaaf1985a} Egelhaaf M. (1985) On the neuronal basis of figure-ground "
        r"discrimination by relative motion in the visual system of the fly. I. \textit{Biol. "
        r"Cybern.} 52:123--140.",
        r"\bibitem{egelhaaf1985b} Egelhaaf M. (1985) \ldots II. Figure-detection cells, a new class "
        r"of visual interneurones. \textit{Biol. Cybern.} 52:195--209. (FD4 at pp.~204--206.)",
        r"\bibitem{egelhaaf1985c} Egelhaaf M. (1985) \ldots III. Possible input circuitries and "
        r"behavioural significance of the FD-cells. \textit{Biol. Cybern.} 52:267--280.",
        r"\bibitem{rp1979} Reichardt W., Poggio T. (1979) Figure-ground discrimination by relative "
        r"movement in the visual system of the fly. I. \textit{Biol. Cybern.} 35:81--100.",
        r"\bibitem{fd1989} Fischbach K.-F., Dittrich A.P.M. (1989) The optic lobe of \textit{"
        r"Drosophila melanogaster}. I. A Golgi analysis of wild-type structure. \textit{Cell Tissue "
        r"Res.} 258:441--475.",
        r"\bibitem{maisak2013} Maisak M.S. et al. (2013) A directional tuning map of \textit{"
        r"Drosophila} elementary motion detectors. \textit{Nature} 500:212--216.",
        r"\bibitem{namiki2018} Namiki S. et al. (2018) The functional organization of descending "
        r"sensory-motor pathways in \textit{Drosophila}. \textit{eLife} 7:e34272.",
        r"\bibitem{dorkenwald2024} Dorkenwald S. et al. (2024) Neuronal wiring diagram of an adult "
        r"brain. \textit{Nature} 634:124--138.",
        r"\bibitem{schlegel2024} Schlegel P. et al. (2024) Whole-brain annotation and "
        r"multi-connectome cell typing of \textit{Drosophila}. \textit{Nature} 634:139--152.",
        r"\bibitem{codex} FlyWire Codex, \texttt{codex.flywire.ai} -- cell-type and annotation portal.",
    ]
    return (r"\begin{thebibliography}{11}" "\n" + "\n".join(items) + "\n" r"\end{thebibliography}")


# ---------------------------------------------------------------------------
# Assembly.
# ---------------------------------------------------------------------------
def build_tex(results: dict) -> str:
    D = _d(results)
    has_circuit = bool(D.get("afferent"))
    parts = [
        _PREAMBLE,
        r"\begin{document}",
        r"\title{\textbf{The connectomic search for the FD4 figure-detection cell: a homogeneous "
        r"\ttt{Nod1}/FD1 population and an honest null}}",
        r"\author{Stage 12 FD4 Identification}",
        r"\date{\today}",
        r"\maketitle",
        r"\tableofcontents",
        r"\part{Identity: the search for FD4 and its result}",
        _summary(D),
        _introduction(),
        _tikz_fig1(),
        _identity_results(D),
        _confidence_section(D),
        r"\section{The evidence at a glance}",
        r"Table~\ref{tab:evidence} collects the FD4-defining properties and the search tests next to "
        r"the connectome measurement for each.",
        r"\begin{table}[H]\centering\footnotesize\caption{Each defining property of Egelhaaf's FD4 cell "
        r"and each search test, next to what we find in FlyWire v783.}\label{tab:evidence}",
        _evidence_table(D),
        r"\end{table}",
        _identity_caveats(),
    ]
    if has_circuit:
        parts += [
            r"\clearpage", _afferent(D),
            r"\clearpage", _efferent(D),
            r"\clearpage", _functional(D),
        ]
    parts += [_methods(D), r"\clearpage", _bibliography(), r"\end{document}"]
    return "\n\n".join(parts)


def _render_figures(results: dict, fig_dir: Path, src=None, meta=None) -> None:
    run = {"derived": {"FD4": _d(results)}, "meta": results.get("meta", {})}
    FF.render_all(run, fig_dir)
    # Circuit data figures (reuse the FD3 figure library on the progressive-arm blocks).
    if _d(results).get("afferent"):
        try:
            _render_circuit_figures(results, fig_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[fd4-report] circuit data figures skipped: {type(exc).__name__}: {exc}")
    # Anatomical renders (real skeletons). Best-effort: skip on failure.
    if src is not None and meta is not None:
        try:
            from ..paper import fd4_figures_anat as FA
            FA.render_all(src, meta, fig_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[fd4-report] anatomical figures skipped: {type(exc).__name__}: {exc}")


def _render_circuit_figures(results: dict, fig_dir: Path) -> None:
    """Reuse the FD3 data-figure library for the progressive-arm census / DN / motor / wheel figures."""
    from ..paper import figures as F3
    D = _d(results)
    run = {"derived": {"FD3": {  # the FD3 figure functions read a 'FD3'-shaped block
        "census": D.get("afferent", {}).get("census", {}),
        "input_census": D.get("afferent", {}).get("census", {}),
        "descending": D.get("efferent", {}),
        "motor": D.get("efferent", {}).get("motor", {}),
        "connectivity": D.get("census", {}),
    }}, "meta": results.get("meta", {})}
    # Best-effort: each figure may or may not match the block shape; skip individually on error.
    for name, fn in [("fd4_input_census", getattr(F3, "fd3_input_census", None)),
                     ("fd4_dn_ranking", getattr(F3, "fd3_dn_ranking", None)),
                     ("fd4_motor_systems", getattr(F3, "fd3_motor_systems", None)),
                     ("fd4_connectivity_wheel", getattr(F3, "fd3_connectivity_wheel", None)),
                     ("fd4_afferent_cascade", getattr(F3, "fd3_afferent_cascade", None))]:
        if fn is None:
            continue
        try:
            fn(run, fig_dir / f"{name}.png")
        except Exception:  # noqa: BLE001
            pass


def build_report(stage_dir: str | Path = "12 - FD4 Identification",
                 results_path: str | Path = "12 - FD4 Identification/fd4_circuit.json",
                 *, src=None, meta=None) -> dict[str, Any]:
    stage = Path(stage_dir)
    (stage / "figures").mkdir(parents=True, exist_ok=True)
    results = read_json(Path(results_path))
    _render_figures(results, stage / "figures", src=src, meta=meta)
    tex_path = stage / "fd4_full_circuit_report.tex"
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
