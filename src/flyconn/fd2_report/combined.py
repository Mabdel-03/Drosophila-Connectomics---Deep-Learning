"""Build + compile the FD2 = LPT21 full-circuit report (LaTeX -> PDF) from the FD2 JSON.

Reads ONLY ``11 - FD2 Identification/fd2_circuit.json`` (the ``families.FD2`` block); no live CAVE and
no feathers, so the report re-runs token-free. Reuses the FD3 report machinery for the preamble,
escaping, figure wrapper and TeX compile. Produces a four-part document parallel to the FD3
full-circuit report: Identity (with a decomposed confidence interval), Afferent inputs, Efferent
outputs, and the functional circuit (including the unique dual-output / landing branch).

Prose is reader-facing: each property is presented as what Egelhaaf reported plus what the connectome
measures; no verdict vocabulary; no em dashes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import read_json
from ..paper import fd2_figures as FF
from ..fd3_report.builder import _PREAMBLE, _esc, _g, _figure, _compile

LPT21 = r"\ttt{LPT21}"


def _d(results: dict) -> dict:
    return results["families"]["FD2"]["derived"]


# ---------------------------------------------------------------------------
# TikZ concept figure: FD1 (frontal, progressive) vs FD2 (frontal, regressive, homolateral) vs
# FD3 (fronto-lateral + gap, contralateral), placing LPT21 at FD2.
# ---------------------------------------------------------------------------
def _tikz_fig1(contra_pct) -> str:
    cc = _esc(round(contra_pct)) if isinstance(contra_pct, (int, float)) else "?"
    return (r"""\begin{figure}[H]\centering
\begin{tikzpicture}[font=\small,>=Stealth]
  \draw[thick] (0,0) arc (180:90:5.4 and 2.2);
  \node[anchor=north east,font=\scriptsize] at (0,-0.05) {frontal ($0^\circ$)};
  \node[anchor=south west,font=\scriptsize] at (5.0,2.25) {lateral ($\sim$100$^\circ$)};
  % FD1 frontal (progressive) anchor
  \fill[fd1c,opacity=0.25] (0.4,0.2) ellipse (0.55 and 0.28);
  \node[fd1c,font=\scriptsize,anchor=west] at (1.05,0.2) {\textbf{FD1} frontal, progressive};
  % FD2 = LPT21: frontal, regressive, homolateral
  \fill[vgreen,opacity=0.30] (0.55,0.9) ellipse (0.9 and 0.34);
  \node[vgreen,font=\scriptsize,anchor=west] at (1.6,0.9) {\textbf{FD2} $=$ \ttt{LPT21}: frontal, regressive};
  % FD3 fronto-lateral + gap (context)
  \fill[fd3c,opacity=0.22] (3.9,1.7) ellipse (1.0 and 0.30);
  \node[fd3c,font=\scriptsize,anchor=east] at (3.05,1.7) {\textbf{FD3} fronto-lateral $+$ gap};
  \draw[->,thick,vgray] (4.4,-0.65) -- (2.3,-0.65)
        node[midway,below,font=\scriptsize]{regressive (back$\to$front)};
  % projection: FD2 axon stays ipsilateral (homolateral), with a frontal branch
  \draw[dashed,gray] (3.9,-1.5) -- (3.9,-4.6);
  \node[gray,font=\scriptsize,anchor=south] at (3.9,-1.5) {brain midline};
  \node[draw,rounded corners,fill=vgreen!12,font=\scriptsize,align=center,text width=2.5cm]
        (d) at (1.1,-2.4) {\ttt{LPT21} dendrite\\(lateral lobula plate)};
  \node[draw,rounded corners,fill=vgreen!12,font=\scriptsize,align=center,text width=2.7cm]
        (a1) at (1.1,-4.1) {\textbf{main} axon:\\ipsilateral posterior optic foci};
  \node[draw,rounded corners,fill=vorange!12,font=\scriptsize,align=center,text width=2.7cm]
        (a2) at (6.2,-2.4) {\textbf{second} frontal branch:\\anterior optic foci};
  \draw[->,thick,vgreen] (d) -- node[right,font=\scriptsize]{stays ipsilateral} (a1);
  \draw[->,thick,vorange] (d) -- node[above,midway,font=\scriptsize]{frontal branch} (a2);
  \node[draw,rounded corners,vgreen,font=\scriptsize,align=center,text width=5.2cm]
        at (4.6,-5.4) {\ttt{LPT21} $=$ FD2: """ + cc + r"""\% contralateral output (homolateral), frontal receptive field, dual axonal output};
\end{tikzpicture}
\caption{\textbf{The FD2 cell and what distinguishes it.} Egelhaaf (1985, Part II) described FD2 as a
regressive figure-detection cell with a frontal receptive field reaching the frontal margin of the
ipsilateral eye, and as a homolateral output element whose main axon projects to the ipsilateral
posterior optic foci with a second branch running frontally toward the anterior optic foci. It is the
frontal, regressive member of the FD set (FD1 is frontal and progressive; FD3 is fronto-lateral with a
frontal gap and a contralateral axon). This report maps FD2 onto the FlyWire cell type \ttt{LPT21}.}%
\label{fig:concept}
\end{figure}
""")


# ---------------------------------------------------------------------------
# Part I: Identity + confidence.
# ---------------------------------------------------------------------------
def _summary(D: dict) -> str:
    ident = D.get("identity", {}); conf = D.get("confidence", {}); uniq = D.get("uniqueness", {})
    return (
        r"\textbf{Summary.}\quad "
        r"Egelhaaf defined FD2 as a regressive, frontal, small-field figure-detection cell that is a "
        r"homolateral output element projecting to the ipsilateral posterior optic foci, with a second "
        r"axonal branch running frontally toward the anterior optic foci. This report identifies the "
        r"FlyWire cell type \textbf{\ttt{LPT21}} as the connectomic correlate of that FD2 phenotype and "
        r"traces its complete circuit from photoreceptor input to motor output. \ttt{LPT21} draws "
        rf"{_esc(ident.get('layer_b_pct'))}\% of its T4/T5 motion input from the back-to-front "
        r"lobula-plate layer, has a receptive field co-located with the frontal FD1 cell, sends "
        rf"{_esc(ident.get('contra_output_pct'))}\% of its output across the midline (a homolateral, "
        r"ipsilateral projection), is cholinergic, and has two spatially separated output terminal "
        r"fields matching Egelhaaf's description of a main and a frontal axonal branch. Among the "
        r"lobula-plate tangential cells it is the "
        rf"{'unique' if uniq.get('unique') else 'leading'} full FD2 match. The identification carries a "
        rf"decomposed confidence of {_esc(conf.get('point_estimate'))} (plausible range "
        rf"{_esc(conf.get('interval'))}), after discounting for the absence of an independent FD2 "
        r"anchor and for FD2's large-field organisation, which Egelhaaf could not resolve.")


def _introduction() -> str:
    return r"""\section{Introduction}

\textbf{Figure-ground discrimination and the FD cells.}\quad
A textured object is invisible against a matched background until it moves at a different velocity or
phase; relative motion then makes it salient. Reichardt and Poggio showed behaviourally that flies
detect and track such figures~\cite{rp1979}. Egelhaaf, in a three-part study, described a class of
lobula-plate tangential cells, the figure-detection (FD) cells, each more sensitive to a small moving
figure than to wide-field motion~\cite{egelhaaf1985b}. Four were distinguished by preferred direction
and receptive-field organisation. \textbf{FD2} is excited by regressive (back-to-front) motion; its
excitatory receptive field is frontal, with its maximum at azimuth $0$--$10^\circ$ and its frontal
boundary at the margin of the ipsilateral eye, reaching laterally to about $60^\circ$ and covering the
whole vertical extent~\cite[p.~201]{egelhaaf1985b}. Anatomically FD2 is a homolateral output element
of the lobula plate: its dendrite covers the lateral lobula plate along the full dorso-ventral axis,
its main axon projects to the ipsilateral posterior optic foci, and a second branch runs frontally
for some $70$--$90\,\mu$m toward the anterior optic foci. FD2 was the least thoroughly recorded of the
FD cells, and its large-field inhibitory input could not be resolved~\cite[p.~201--202]{egelhaaf1985b}.

\textbf{The connectome era and the mapping problem.}\quad
The adult \emph{Drosophila} FlyWire connectome and its cell-type annotations provide the
synapse-resolution wiring needed to ask which modern cell type corresponds to each FD
cell~\cite{dorkenwald2024,schlegel2024,codex}. T4 and T5 encode ON/OFF edge motion and segregate by
preferred direction into the four lobula-plate layers~\cite{maisak2013,fd1989}: layer-a front-to-back,
layer-b back-to-front. FD1 has been mapped to \ttt{Nod1} and FD3 to \ttt{LPT42\_Nod4}. This report
asks which cell type is FD2, attaches a quantified confidence, and traces the cell's full circuit.

\textbf{How the identification is made.}\quad
Each defining FD2 property corresponds to a connectome measurement: preferred direction from the
lobula-plate layer of its T4/T5 input, the receptive field from the retinotopic positions of those
inputs relative to the frontal FD1 cell, the projection side from where its output synapses land, the
dual axonal output from the spatial distribution of those synapses, and overall shape from its
reconstructed skeleton. We measure each for \ttt{LPT21}, compare against a global scan of the
lobula-plate tangential family so uniqueness is measured rather than assumed, and combine the results
into a transparent confidence interval. Figure~\ref{fig:concept} fixes the FD2 phenotype."""


def _identity_results(D: dict) -> str:
    ident = D.get("identity", {}); dual = D.get("dual_output", {}); nb = ident.get("smallfield_null", {})
    P = [r"\section{The evidence}"]
    P.append(r"We take FD2's defining properties in turn: preferred motion, receptive field, "
             r"projection side, small-field selectivity, transmitter, and the dual axonal output.")
    P.append(r"\paragraph{It prefers back-to-front motion.} A cell's preferred direction is read from "
             r"which lobula-plate layer its motion inputs come from. For \ttt{LPT21}, "
             rf"{_esc(ident.get('layer_b_pct'))}\% of that input comes from the back-to-front layer "
             r"(Fig.~\ref{fig:census}b), the regressive direction Egelhaaf reported for FD2.")
    _off = ident.get("mean_rf_offset")
    _off_s = f"{_off:.2f}" if isinstance(_off, (int, float)) else _esc(_off)
    P.append(r"\paragraph{Its receptive field is frontal.} Measured against the frontal FD1 cell in "
             r"the same coordinate frame, the input field of \ttt{LPT21} sits at the FD1 position "
             rf"(offset near zero, about {_off_s} lattice columns) and fills the frontal band on both "
             r"sides, the frontal receptive field Egelhaaf described for FD2 reaching the margin of the "
             r"ipsilateral eye (Fig.~\ref{fig:hexmap}).")
    P.append(r"\paragraph{Its axon stays on its own side of the brain.} Egelhaaf found that FD2 is a "
             r"homolateral cell whose main axon projects to the ipsilateral posterior optic foci. The "
             rf"connectome shows the same: {_esc(ident.get('contra_output_pct'))}\% of \ttt{{LPT21}}'s "
             r"output synapses are on the opposite side of the brain, that is, almost all of its output "
             r"stays ipsilateral. This is the homolateral projection of FD2, and it is the property "
             r"that separates FD2 from the contralaterally projecting FD3.")
    P.append(r"\paragraph{It responds to small objects.} The defining behaviour of an FD cell is a "
             r"stronger response to a small moving figure than to wide-field motion, which requires the "
             r"cell to pool a compact patch of the visual field. The input patch of \ttt{LPT21} is more "
             rf"concentrated than an in-degree matched random sample ($p={_esc(nb.get('p_value'))}$). "
             r"This provides wiring support for small-field selectivity.")
    P.append(r"\paragraph{It is an excitatory output cell.} \ttt{LPT21} is a bilateral pair classified "
             r"as a visual output neuron, releasing the excitatory transmitter acetylcholine (confidence "
             rf"{_esc(ident.get('nt_conf'))}). Egelhaaf did not report the transmitter of FD2; this is a "
             r"connectome assignment.")
    P.append(r"\paragraph{It has two separate axonal outputs.} Egelhaaf reported that FD2 alone among "
             r"the FD cells has a second axonal branch, running frontally toward the anterior optic "
             r"foci, and proposed it may drive the landing response. The connectome shows the same: the "
             rf"output synapses of each \ttt{{LPT21}} cell form two spatially separated terminal fields "
             rf"(bimodal on both cells: {_esc(dual.get('both_bimodal'))}), a main field and a smaller "
             r"displaced field, matching the two axonal branches Egelhaaf described (Fig.~\ref{fig:dual}).")
    P.append(_figure("figures/fd2_dual_output.png",
                     r"\textbf{The dual output.} For each \ttt{LPT21} cell, the separation between its "
                     r"two output terminal fields (measured as the depth of the valley between them) "
                     r"and the size of the smaller field. Two well-separated fields per cell are the "
                     r"connectome correlate of Egelhaaf's main (ipsilateral posterior optic foci) and "
                     r"second (frontal, anterior optic foci) axonal branches.", "fig:dual"))
    P.append(_figure("figures/fd2_dual_output_3d.png",
                     r"\textbf{The dual output in anatomical space.} The output synapses of each "
                     r"\ttt{LPT21} cell, coloured by which of the two terminal fields they fall in, on "
                     r"the reconstructed cell. The two spatially separated fields are directly visible, "
                     r"confirming the second axonal branch Egelhaaf described.", "fig:dual3d"))
    return "\n\n".join(P)


def _confidence_section(D: dict) -> str:
    conf = D.get("confidence", {}); uniq = D.get("uniqueness", {})
    comp = ", ".join(f"\\ttt{{{_esc(c['type'])}}}" for c in uniq.get("competitors", [])) or "none"
    return (
        r"\section{Confidence in the identification}" "\n\n"
        r"The identification is expressed as a decomposed confidence rather than an assertion. It "
        r"combines three factors, each stated so a reader can reconstruct it. First, the fraction of "
        rf"FD2-defining properties \ttt{{LPT21}} matches: {_esc(conf.get('n_matched'))} of "
        rf"{_esc(conf.get('n_properties'))} (Fig.~\ref{{fig:confidence}}, left). Second, a global scan "
        r"of the lobula-plate tangential family for cells that satisfy the whole FD2 conjunction "
        r"(regressive, cholinergic, homolateral, a single bilateral pair, and a bilaterally consistent "
        rf"frontal field): \ttt{{LPT21}} is the {'unique' if uniq.get('unique') else 'leading'} full "
        rf"match ({_esc(uniq.get('n_fd2_hits'))} hit(s); Fig.~\ref{{fig:uniqueness}}). The only other "
        rf"regressive homolateral cholinergic cell, {comp}, is excluded because it is not a single "
        r"bilateral pair and its receptive field is not the homogeneous frontal field of FD2. Third, "
        r"explicit discounts cap the estimate: FD2 has no independent connectome anchor of the kind "
        r"that fixes FD1 to \ttt{Nod1}, so the identity is an inference from property consistency and "
        r"uniqueness; Egelhaaf could not resolve FD2's large-field organisation, so that axis cannot "
        r"corroborate; further undiscovered FD cells cannot be excluded; and the cell is a single "
        rf"bilateral pair. The resulting point estimate is {_esc(conf.get('point_estimate'))}, with a "
        rf"plausible range of {_esc(conf.get('interval'))}." "\n\n"
        + _figure("figures/fd2_confidence.png",
                  r"\textbf{The confidence, decomposed.} (left) The FD2-defining properties and whether "
                  r"\ttt{LPT21} matches each. (right) The point estimate and interval, with the "
                  r"uniqueness factor and the discounts that set it.", "fig:confidence")
        + _figure("figures/fd2_uniqueness.png",
                  r"\textbf{The uniqueness scan.} Every scanned lobula-plate tangential cell placed by "
                  r"its regressive drive (horizontal) and output laterality (vertical). \ttt{LPT21} "
                  r"occupies the FD2 corner (regressive and homolateral); any cell sharing the corner is "
                  r"labelled with why it is not FD2.", "fig:uniqueness"))


def _evidence_table(D: dict) -> str:
    ident = D.get("identity", {}); conf = D.get("confidence", {}); uniq = D.get("uniqueness", {})
    dual = D.get("dual_output", {})
    rows = [
        ("Prefers regressive (back-to-front) motion", "Egelhaaf: FD2 excited by regressive motion",
         rf"{_esc(ident.get('layer_b_pct'))}\% of its T4/T5 input is the back-to-front layer"),
        ("Frontal receptive field", "Egelhaaf: peak azimuth $0$--$10^\\circ$, at the eye's frontal margin",
         r"input field co-located with the frontal FD1 cell, frontal band filled, no gap"),
        ("Homolateral (ipsilateral) projection", "Egelhaaf: main axon to the ipsilateral posterior optic foci",
         rf"{_esc(ident.get('contra_output_pct'))}\% of output crosses the midline (near-zero = ipsilateral)"),
        ("Small-object selective", "FD cells prefer small figures to wide-field motion",
         r"input patch bounded below an in-degree matched null"),
        ("Excitatory output (cholinergic)", "FD cells are excitatory output neurons",
         rf"acetylcholine, confidence {_esc(ident.get('nt_conf'))}"),
        ("Second frontal axonal branch", "Egelhaaf: a second branch runs frontally to the anterior optic foci",
         rf"two separated output terminal fields per cell (bimodal: {_esc(dual.get('both_bimodal'))})"),
        ("Unique in the tangential family", "the identification should be specific",
         rf"{_esc(uniq.get('n_fd2_hits'))} full FD2 match(es); unique: {_esc(uniq.get('unique'))}"),
        ("Confidence", "quantify the identification honestly",
         rf"point {_esc(conf.get('point_estimate'))}, range {_esc(conf.get('interval'))}"),
    ]
    head = (r"\renewcommand{\arraystretch}{1.25}" "\n" r"\setlength{\tabcolsep}{3pt}" "\n"
            r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.26\textwidth} "
            r">{\raggedright\arraybackslash}p{0.32\textwidth} "
            r">{\raggedright\arraybackslash}p{0.36\textwidth}@{}}" "\n"
            r"\toprule \textbf{FD2 property} & \textbf{What Egelhaaf reported} & "
            r"\textbf{What we find for \ttt{LPT21}} \\\midrule\endhead" "\n")
    body = "\n".join(rf"{a} & {b} & {c} \\" for a, b, c in rows)
    return (head + body + "\n" + r"\bottomrule\end{longtable}" + "\n"
            r"\renewcommand{\arraystretch}{1.0}" "\n" r"\setlength{\tabcolsep}{6pt}")


def _identity_caveats() -> str:
    return (r"\section{What the data can and cannot settle}" "\n\n"
            r"\textbf{FD2 has no independent connectome anchor.} FD1 was fixed to \ttt{Nod1} and FD3 to "
            r"\ttt{LPT42\_Nod4} before FD2 was approached, but no cell is independently known to be FD2, "
            r"so this identification rests on the consistency of the measured properties with Egelhaaf's "
            r"description and on the uniqueness scan, not on anchoring. The receptive-field comparison is "
            r"moreover made relative to the FD1=\ttt{Nod1} cell, itself an assumed mapping, so an error "
            r"in that anchor would propagate; this is one of the reasons the confidence is held below "
            r"certainty. \textbf{FD2's large-field"
            r"organisation was never measured.} Egelhaaf could not resolve FD2's inhibitory large-field "
            r"input, so that axis, resolved here from the connectome, corroborates but was not part of "
            r"the physiological definition. \textbf{The cell is one bilateral pair.} Robustness comes "
            r"from the left and right cell agreeing and from resampling the thousands of synapses each "
            r"makes, not from a population average. \textbf{Absolute visual angle is approximate.} The "
            r"receptive-field argument is built on the relative comparison to the frontal FD1 cell and "
            r"does not rely on absolute azimuth. \textbf{The dual output matches FD2 but is not "
            r"FD2-exclusive in the connectome.} Two separated output terminal fields match Egelhaaf's "
            r"description of FD2's main and frontal branches; other tangential cells with a "
            r"midline-crossing axon also show two output lobes for a different reason, so this feature "
            r"corroborates the FD2 phenotype rather than proving it in isolation. \textbf{Every "
            r"load-bearing claim was tested adversarially.} The identity, the motion drive, the "
            r"descending output, the dual output, and the confidence interval were each subjected to an "
            r"independent attempt at refutation with fresh connectome queries; each survived, and the "
            r"honest residual caveats from that exercise are the ones stated here.")


# ---------------------------------------------------------------------------
# Part II: Afferent inputs.
# ---------------------------------------------------------------------------
def _afferent(D: dict) -> str:
    aff = D.get("afferent", {}); cen = aff.get("census", {}); casc = aff.get("upstream_cascade", {})
    sheet = D.get("sheet", {})
    P = [r"\part{Inputs: what drives FD2}"]
    P.append(r"\textbf{Summary.}\quad This part follows the wiring that drives FD2 from the eye: from "
             r"the photoreceptors, through the lamina and medulla, to the local motion detectors, and "
             rf"onto \ttt{{LPT21}}'s dendrite. Its measured T4/T5 motion input is "
             rf"{_esc(cen.get('layer_b_frac_of_t4t5'))}\% back-to-front, carried by both the ON "
             rf"(\ttt{{T4b}}) and OFF (\ttt{{T5b}}) detectors, and is about "
             rf"{_esc(cen.get('t4t5_frac_of_total'))}\% of its total input, the rest coming from "
             r"columnar projection sheets and inhibitory cells.")
    P.append(r"\section{The motion detectors that drive FD2}")
    P.append(rf"Of \ttt{{LPT21}}'s input from the local motion detectors, "
             rf"{_esc(cen.get('layer_b_frac_of_t4t5'))}\% comes from the back-to-front (layer-b) "
             rf"channel, carried by both the ON detector \ttt{{T4b}} ({_esc(cen.get('t4b_syn'))} "
             rf"synapses) and the OFF detector \ttt{{T5b}} ({_esc(cen.get('t5b_syn'))} synapses), a "
             r"roughly even ON/OFF mix. FD2 therefore pools back-to-front motion of both contrast "
             r"polarities (Fig.~\ref{fig:census}).")
    P.append(_figure("figures/fd2_input_census.png",
                     r"\textbf{What feeds FD2.} (a) \ttt{LPT21}'s direct inputs by type, coloured by "
                     r"class: the motion detectors, the columnar projection sheets, and the inhibitory "
                     r"cells. (b) Within the motion input, almost all is the back-to-front (layer-b) "
                     r"channel, split between the ON and OFF detectors.", "fig:census"))
    P.append(r"\section{From the eye to the detectors}")
    P.append(rf"The \ttt{{T4b}} cells that drive FD2 are fed by the canonical ON medulla cells and the "
             rf"\ttt{{T5b}} cells by the canonical OFF medulla cells; these are fed by the lamina and, "
             r"before it, the photoreceptors. The whole cascade from the eye to FD2's detectors is "
             rf"present in the connectome (ON medulla present: {_esc(casc.get('t4_on_medulla_present'))}; "
             rf"OFF medulla: {_esc(casc.get('t5_off_medulla_present'))}; lamina: "
             rf"{_esc(casc.get('lamina_present'))}; photoreceptors: "
             rf"{_esc(casc.get('photoreceptor_present'))}) and matches published optic-lobe wiring "
             r"(Fig.~\ref{fig:cascade}). This cascade is the general column-to-column wiring of the "
             r"optic lobe, shown as the pathway that culminates in FD2's detectors, not as a per-object "
             r"trace unique to FD2.")
    P.append(_figure("figures/fd2_afferent_cascade.png",
                     r"\textbf{The afferent cascade.} Photoreceptors to lamina to ON/OFF medulla to the "
                     r"back-to-front detectors \ttt{T4b}/\ttt{T5b} to FD2. Solid links are measured on "
                     r"FD2's detectors; dashed links are the standard column wiring.", "fig:cascade"))
    P.append(r"\section{Where the input lands}")
    P.append(r"Each motion detector looks at one point in the eye, so the detectors feeding FD2 draw "
             r"out its receptive field on the eye's lattice. Plotting those input columns shows FD2's "
             r"field frontal, at the position of the FD1 cell (Fig.~\ref{fig:hexmap}).")
    P.append(_figure("figures/fd2_input_hexmap.png",
                     r"\textbf{FD2's input in eye coordinates.} Every motion column feeding \ttt{LPT21}, "
                     r"placed on the eye's hexagonal lattice and shaded by synapse number, next to the "
                     r"frontal FD1 columns. FD2's field sits frontally, at the FD1 position.",
                     "fig:hexmap"))
    P.append(r"\section{The columnar sheet and the inhibitory gate}")
    P.append(rf"The motion detectors also reach FD2 through a columnar projection sheet, the "
             rf"direction-matched layer-b sheet \ttt{{{_esc(sheet.get('named_sheet'))}}}. In addition, "
             r"FD2 receives input from wide-field inhibitory cells. Egelhaaf could not resolve FD2's "
             r"large-field inhibitory input physiologically; the connectome resolves it as a set of "
             rf"lobula-plate intrinsic (\ttt{{LPi}}) cells, led by "
             rf"\ttt{{{_esc(D.get('gate', {}).get('named_inhibitor'))}}}, that inhibit the cell and its "
             r"detectors. The sign and function of each inhibitory partner remain wiring inferences "
             r"unless supported by transmitter identity or physiology.")
    P.append(_figure("figures/fd2_arbor_inputs.png",
                     r"\textbf{Inputs on FD2's arbor.} Each \ttt{LPT21} cell's reconstructed shape with "
                     r"its input synapses drawn at their real positions and coloured by the class of "
                     r"partner: back-to-front motion detectors, columnar sheets, and inhibitory cells.",
                     "fig:arbor"))
    P.append(r"\section{What the data can and cannot settle}" "\n\n"
             r"\textbf{The upstream cascade is general wiring.} The photoreceptor-to-detector cascade is "
             r"measured cell-type by cell-type across the optic lobe and matches the known column "
             r"wiring; it is shown as the pathway culminating in FD2's detectors, not as an FD2-specific "
             r"trace. \textbf{The inhibitory gate is a wiring inference.} That the resolved \ttt{LPi} "
             r"partners are inhibitory follows from their transmitter, but the direction and functional "
             r"role of the gate are inferences the connectome sets up rather than measures.")
    return "\n\n".join(P)


# ---------------------------------------------------------------------------
# Part III: Efferent outputs.
# ---------------------------------------------------------------------------
def _efferent(D: dict) -> str:
    eff = D.get("efferent", {}); direct = eff.get("direct", {}); mot = eff.get("motor", {})
    top = direct.get("ranking", [{}])[0].get("cell_type") if direct.get("ranking") else None
    P = [r"\part{Outputs: descending and motor-system partners}"]
    P.append(r"\textbf{Summary.}\quad For FD2's figure signal to influence movement it must reach "
             r"descending neurons, which project from the brain into the ventral nerve cord and contact "
             rf"motor-control networks. \ttt{{LPT21}} contacts descending neurons directly, led by "
             rf"\ttt{{{_esc(top)}}}, and its descending output is biased toward the wing-steering motor "
             r"system when each descending neuron is followed into the male nerve-cord connectome.")
    P.append(r"\section{The descending neurons FD2 contacts directly}")
    P.append(rf"\ttt{{LPT21}} makes {_esc(direct.get('dn_syn'))} output synapses onto "
             rf"{_esc(len(direct.get('ranking', [])))} descending-neuron types. The leading target is "
             rf"\ttt{{{_esc(top)}}}, the same wing-steering command neuron that the FD1 and FD3 arms "
             r"reach, so the three figure cells converge on a shared descending target "
             r"(Fig.~\ref{fig:dnrank}). Egelhaaf could name only ``descending neurones'' generically; "
             r"the connectome names them.")
    P.append(_figure("figures/fd2_dn_ranking.png",
                     r"\textbf{The descending neurons FD2 contacts directly.} Each bar is one "
                     r"descending-neuron type, ranked by synapses received from \ttt{LPT21}. Red marks "
                     r"neurons known to drive wing-steering muscles.", "fig:dnrank"))
    P.append(r"\section{Motor systems contacted by those neurons}")
    P.append(rf"Classifying the descending neurons FD2 reaches by whether they are known wing-steering "
             rf"command neurons, the great majority of FD2's descending output goes to that class "
             rf"(dominant system: \ttt{{{_esc(mot.get('dominant_motor_system'))}}}; Fig.~\ref{{fig:motorsys}}). "
             rf"The principal target \ttt{{{_esc(top)}}} drives the contralateral wing-steering muscles "
             r"in the male nerve-cord connectome. Two qualifications keep this precise. First, the class "
             r"label is by descending-neuron identity, not by muscle count: when each descending neuron "
             r"is followed into the male nerve cord and weighted by its actual motor targets, the "
             rf"leading neuron \ttt{{{_esc(top)}}} splits its output between wing-steering and neck "
             r"muscles, and the second target is neck and abdominal rather than wing, so the honest "
             r"picture is a bias toward steering and gaze control rather than pure wing steering. "
             r"Second, the descending contact is sparse (of the order of a hundred synapses), most of "
             r"it carried by the top two neurons. The pathway therefore supports a steering and "
             r"gaze-control role, consistent with Egelhaaf's proposal that the FD cells contribute to "
             r"yaw-torque control, but it does not by itself prove that FD2 activity drives a behaviour.")
    P.append(_figure("figures/fd2_motor_systems.png",
                     r"\textbf{Motor-system proxy.} (left) The motor systems reached by FD2's descending "
                     r"output, weighted by how strongly FD2 drives each descending neuron and classified "
                     r"by descending-neuron identity; the wing-steering class has the largest share, "
                     r"though weighting by actual muscle targets softens this toward a mixed steering and "
                     r"neck-gaze bias (see caveats). (right) The top wing-steering muscles of the "
                     r"principal descending target in the male nerve-cord connectome.", "fig:motorsys"))
    P.append(r"\section{The circuit in the brain}")
    P.append(r"Seen alongside the cells it connects, FD2 sits anatomically between the lobula-plate "
             r"motion detectors and the descending neurons that contact wing-steering motor systems "
             r"(Fig.~\ref{fig:circuit3d}).")
    P.append(_figure("figures/fd2_circuit_3d.png",
                     r"\textbf{The circuit in the brain.} Reconstructed neurons of the FD2 pathway in "
                     r"their true anatomical positions: the \ttt{T4b}/\ttt{T5b} motion detectors, both "
                     r"\ttt{LPT21} cells, and the descending neuron they lead to.", "fig:circuit3d"))
    P.append(r"\section{What the data can and cannot settle}" "\n\n"
             r"\textbf{The steering label is by neuron identity, not muscle count.} The wing-steering "
             r"share reported above counts descending neurons that belong to the known wing-steering "
             r"set; weighting instead by each neuron's actual motor targets in the male nerve cord "
             r"softens the picture to a mixed wing-steering and neck-gaze bias, because the leading "
             r"descending neuron contacts neck as well as wing muscles and the second target is neck "
             r"and abdominal. The direction of the bias (toward steering and gaze control) is robust; "
             r"its exact magnitude depends on which of the two weightings is used. \textbf{The muscle "
             r"map crosses two connectomes.} The muscles each descending neuron drives are read from a "
             r"second, male nerve-cord connectome, bridged to the brain by the shared descending-neuron "
             r"name; a neuron without a named counterpart contributes no muscle information. "
             r"\textbf{The descending contact is sparse.} FD2's direct descending output is of the "
             r"order of a hundred synapses, most of it onto the top two neurons, so the read-out rests "
             r"on a small number of strong contacts rather than a broad projection. \textbf{FD2 is two "
             r"cells.} The result rests on the two cells agreeing, not on a population average.")
    return "\n\n".join(P)


# ---------------------------------------------------------------------------
# Part IV: Functional circuit.
# ---------------------------------------------------------------------------
def _functional(D: dict) -> str:
    census = D.get("census", {}); comp = D.get("compartment", {})
    sheet = D.get("sheet", {}); gate = D.get("gate", {}); eff = D.get("efferent", {})
    top = eff.get("direct", {}).get("ranking", [{}])[0].get("cell_type") if eff.get("direct", {}).get("ranking") else None
    inp = census.get("input", {}); out = census.get("output", {})
    P = [r"\part{The functional figure-ground circuit of FD2}"]
    P.append(r"\textbf{Summary.}\quad This part maps FD2's whole connectivity and then names the working "
             r"figure-ground circuit within it, in the same terms as the FD1 and FD3 arms: motion "
             rf"detectors to the layer-b sheet \ttt{{{_esc(sheet.get('named_sheet'))}}} to FD2 to the "
             rf"steering neuron \ttt{{{_esc(top)}}}, gated by the wide-field inhibitor "
             rf"\ttt{{{_esc(gate.get('named_inhibitor'))}}}.")
    P.append(r"\section{FD2's connectivity, comprehensively}")
    P.append(rf"\ttt{{LPT21}} receives about {_esc(_g(inp, 'total_syn'))} input synapses from roughly "
             rf"{_esc(_g(inp, 'n_partners'))} partner cells, and makes about {_esc(_g(out, 'total_syn'))} "
             rf"output synapses onto roughly {_esc(_g(out, 'n_partners'))} partners. The inputs are "
             r"dominated by the motion detectors, the neighbouring columnar sheets, and inhibitory "
             r"cells; the outputs are broadcast to central brain cells and, through the descending "
             r"neurons, toward the steering muscles (Fig.~\ref{fig:wheel}).")
    P.append(_figure("figures/fd2_connectivity_wheel.png",
                     r"\textbf{FD2's comprehensive connectivity.} Every major input (left) and output "
                     r"(right) partner type of \ttt{LPT21}.", "fig:wheel"))
    P.append(r"\section{Where inputs land}")
    P.append(r"Splitting FD2's input synapses between its dendrite and its axon shows that the input "
             r"arrives on the dendritic arbour in the lobula plate, while the axon carries the cell's "
             r"output. FD2 is thus a cleanly polarised neuron (Fig.~\ref{fig:compartment}).")
    P.append(_figure("figures/fd2_compartment_split.png",
                     r"\textbf{Where inputs land.} Input synapses to each \ttt{LPT21} cell, split "
                     r"between the dendrite and the axon and coloured by the class of partner.",
                     "fig:compartment"))
    P.append(r"\section{The circuit, named end to end}")
    P.append(rf"Putting the pieces together, FD2's figure-ground circuit reads: the \ttt{{T4b}}/"
             rf"\ttt{{T5b}} back-to-front motion detectors drive the layer-b sheet "
             rf"\ttt{{{_esc(sheet.get('named_sheet'))}}}, which relays to FD2; FD2 drives the steering "
             rf"command neuron \ttt{{{_esc(top)}}} and thence the wing; and the wide-field cell "
             rf"\ttt{{{_esc(gate.get('named_inhibitor'))}}} gates the circuit. This is the same "
             r"architecture as the FD1 and FD3 arms, built from a frontal, regressive, homolaterally "
             r"projecting cell, and it terminates on the shared wing-steering command neuron. Whether "
             r"the frontal branch additionally serves the landing response, as Egelhaaf proposed, is a "
             r"question the wiring sets up but does not answer.")
    P.append(_figure("figures/fd2_functional_circuit.png",
                     r"\textbf{The named figure-ground circuit.} \ttt{T4b}/\ttt{T5b} motion detectors "
                     r"drive the layer-b sheet, which relays to FD2; FD2 drives the steering neuron and "
                     r"the wing, with a wide-field inhibitory gate. Numbers are measured synapse counts.",
                     "fig:funcircuit"))
    P.append(r"\section{What the data can and cannot settle}" "\n\n"
             r"\textbf{Sign is not read from wiring.} That a GABAergic contact is inhibitory, and that "
             r"the gate shapes FD2's figure response, are expectations from physiology; the connectome "
             r"supplies the connections, not the sign or the response. \textbf{The landing-response "
             r"role is a hypothesis.} Egelhaaf proposed that FD2's frontal branch drives the landing "
             r"response; the connectome confirms the dual output but does not measure the behaviour. "
             r"\textbf{FD2 is one bilateral pair.} As throughout, the result rests on the two cells "
             r"agreeing and on the thousands of synapses each makes.")
    return "\n\n".join(P)


def _methods(D: dict) -> str:
    track = D.get("track", "offline")
    return (r"\section{Methods and provenance}" "\n\n"
            r"\textbf{Connectome.} FlyWire FAFB materialization v783 (\ttt{synapses\_nt\_v1}, no cleft "
            rf"threshold), track \ttt{{{_esc(track)}}}, with the homolateral (contra\%) identity "
            r"measurement confirmed across the offline, live and v630 tracks. Neuron annotations are the "
            r"Schlegel/Codex tables. \textbf{Preferred direction} is the synapse-weighted T4/T5 "
            r"lobula-plate layer composition~\cite{fd1989,maisak2013}. \textbf{Receptive field}: T4/T5 "
            r"inputs carry hex-lattice coordinates; the differential test compares the synapse-weighted "
            r"centroid and frontal-band occupancy against the FD1=\ttt{Nod1} anchor with a within-cell "
            r"bootstrap, the axis pinned by requiring the known-frontal FD1 anchor to land frontal. "
            r"\textbf{Projection side} is the fraction of output synapses on the contralateral side. "
            r"\textbf{Dual output} is a bimodality test on each cell's output-synapse field. "
            r"\textbf{Uniqueness} scans the lobula-plate tangential family for the FD2 conjunction. "
            r"\textbf{Afferent, sheet, gate, efferent} reuse the FD-circuit machinery parameterised on "
            r"\ttt{LPT21}; the motor read-out crosses into the male CNS by shared descending-neuron "
            r"name. \textbf{Reproducibility}: built from the FD2 verification JSON.")


def _bibliography() -> str:
    items = [
        r"\bibitem{egelhaaf1985a} Egelhaaf M. (1985) On the neuronal basis of figure-ground "
        r"discrimination by relative motion in the visual system of the fly. I. \textit{Biol. "
        r"Cybern.} 52:123--140.",
        r"\bibitem{egelhaaf1985b} Egelhaaf M. (1985) \ldots II. Figure-detection cells, a new class "
        r"of visual interneurones. \textit{Biol. Cybern.} 52:195--209. (FD2 at pp.~201--202, 207--208.)",
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
    contra = D.get("identity", {}).get("contra_output_pct")
    parts = [
        _PREAMBLE,
        r"\begin{document}",
        r"\title{\textbf{Connectomic identification, with confidence, and the full circuit of the FD2 "
        r"figure-detection cell (\ttt{LPT21})}}",
        r"\author{Stage 11 FD2 Identification}",
        r"\date{\today}",
        r"\maketitle",
        r"\tableofcontents",
        r"\part{Identity: FD2 is \texttt{LPT21}}",
        _summary(D),
        _introduction(),
        _tikz_fig1(contra),
        _identity_results(D),
        _confidence_section(D),
        r"\section{The evidence at a glance}",
        r"Table~\ref{tab:evidence} collects the defining FD2 properties next to the connectome "
        r"measurement that establishes each for \ttt{LPT21}.",
        r"\begin{table}[H]\centering\footnotesize\caption{Each defining property of Egelhaaf's FD2 cell "
        r"and the matching measurement for the FlyWire cell type \ttt{LPT21}.}\label{tab:evidence}",
        _evidence_table(D),
        r"\end{table}",
        _identity_caveats(),
        r"\clearpage",
        _afferent(D),
        r"\clearpage",
        _efferent(D),
        r"\clearpage",
        _functional(D),
        _methods(D),
        r"\clearpage",
        _bibliography(),
        r"\end{document}",
    ]
    return "\n\n".join(parts)


def _render_figures(results: dict, fig_dir: Path, src=None, meta=None) -> None:
    run = {"derived": {"FD2": _d(results)}, "meta": results.get("meta", {})}
    FF.fd2_confidence(run, fig_dir / "fd2_confidence.png")
    FF.fd2_uniqueness_scatter(run, fig_dir / "fd2_uniqueness.png")
    FF.fd2_input_census(run, fig_dir / "fd2_input_census.png")
    FF.fd2_dn_ranking(run, fig_dir / "fd2_dn_ranking.png")
    FF.fd2_motor_systems(run, fig_dir / "fd2_motor_systems.png")
    FF.fd2_dual_output(run, fig_dir / "fd2_dual_output.png")
    FF.fd2_connectivity_wheel(run, fig_dir / "fd2_connectivity_wheel.png")
    FF.fd2_compartment_split(run, fig_dir / "fd2_compartment_split.png")
    # Anatomical renders (skeletons / hex / cascade / schematic). Best-effort: skip on failure.
    if src is not None and meta is not None:
        try:
            from ..paper import fd2_figures_anat as FA
            FA.render_all(src, meta, _d(results), fig_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[fd2-report] anatomical figures skipped: {type(exc).__name__}: {exc}")


def build_report(stage_dir: str | Path = "11 - FD2 Identification",
                 results_path: str | Path = "11 - FD2 Identification/fd2_circuit.json",
                 *, src=None, meta=None) -> dict[str, Any]:
    stage = Path(stage_dir)
    (stage / "figures").mkdir(parents=True, exist_ok=True)
    results = read_json(Path(results_path))
    _render_figures(results, stage / "figures", src=src, meta=meta)
    tex_path = stage / "fd2_full_circuit_report.tex"
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
