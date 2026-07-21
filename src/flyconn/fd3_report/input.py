"""Build + compile the Stage-7 *input-pathway* report (LaTeX -> PDF) from the Family-P JSON.

The third FD3 report, companion to ``builder.py`` (identity, Family K) and ``descending.py``
(output, Family L). It answers: WHAT DRIVES FD3 -- the afferent pathway from the photoreceptors,
through the motion detectors, to FD3's dendrite, plus FD3's central inputs and its contralateral
inhibition. Reads ONLY the ``families.P`` block of the verification JSON and writes
``7 - FD3 Identification/fd3_input_report.{tex,pdf}``.

Prose is READER-FACING: each result is a finding (what we measure + what it means), with no
verdict / claim vocabulary. A guard test enforces this. Real anatomical figures (the FD3 arbor
with coloured input synapses, the retinotopic input-column map, the 3D circuit, the tiered
afferent cascade) are rendered when a FlyWire source is available, and otherwise reused from the
``figures/`` directory if a prior run left them there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import read_json
from ..paper import figures as PF
from .builder import _PREAMBLE, _figure, _esc, _g, _compile


def _figure_fullpage(rel_png: str, caption: str, label: str) -> str:
    """A figure on its own float page ([p]), sized to fill the text block so the reconstructed
    neurons render as large as the page allows (used for the two anatomical skeleton figures)."""
    return (r"\begin{figure}[p]\centering" "\n"
            rf"\includegraphics[width=\textwidth,height=0.86\textheight,keepaspectratio]{{{rel_png}}}" "\n"
            rf"\caption{{{caption}}}\label{{{label}}}" "\n"
            r"\end{figure}" "\n")


def _summary(P: dict) -> str:
    census = _g(P, "census", default={})
    t45f = _g(census, "t4t5_frac_of_total")
    lb = _g(census, "layer_b_frac_of_t4t5")
    return (
        r"\textbf{Summary.}\quad "
        r"The figure-detection cell FD3 -- the cell type \ttt{LPT42\_Nod4} in the fly connectome -- "
        r"signals a small object moving from back to front on one side of the visual field. This "
        r"report follows the wiring that \emph{drives} it: from the photoreceptors of the eye, "
        r"through the lamina and medulla, to the local motion detectors, and onto FD3's dendrite, "
        r"together with the central cells and the inhibition from the other eye that shape its "
        r"response. We find that FD3's measured T4/T5 motion input comes almost entirely from the back-to-front "
        rf"(regressive) motion channel -- {_esc(lb)}\% of its detector input is that one channel -- "
        r"carried by both the ON (\ttt{T4b}) and the OFF (\ttt{T5b}) detectors. Those detectors are "
        r"themselves fed by the canonical columnar cascade (photoreceptors, lamina, and the ON/OFF "
        r"medulla cells), which is present in the connectome and matches the literature. Importantly, "
        rf"motion is only about {_esc(t45f)}\% of FD3's total input: most of its input comes from "
        r"neighbouring columnar projection sheets and from inhibitory cells, including contralateral "
        r"inhibition driven by the opposite eye. Unlike the previously mapped FD1 cell, FD3 has no "
        r"centrifugal gate and no front-to-back drive -- it is a parallel, oppositely-tuned arm of "
        r"the figure-ground system, not a copy of FD1.")


def _introduction() -> str:
    return (
        r"\section{From the eye to FD3}" "\n\n"
        r"A moving object is first transduced by the photoreceptors, split into ON and OFF channels "
        r"in the lamina and medulla, and turned into local direction-of-motion signals by the "
        r"\ttt{T4} (ON) and \ttt{T5} (OFF) cells of the lobula plate. Each \ttt{T4}/\ttt{T5} cell "
        r"prefers one of four directions, sorted into four layers of the lobula plate: layer~a is "
        r"front-to-back, layer~b is back-to-front, and layers c and d are up and down~\cite{maisak2013}. A "
        r"figure-detection cell reads out this motion map. The companion identity report showed that "
        r"FD3 reads the back-to-front (layer~b) map with a receptive field placed off to the side "
        r"and a gap directly in front; the descending report followed FD3's output to the "
        r"descending neurons. Here we extend the trace on the input side, naming the cells that "
        r"feed FD3 and tracing the pathway back to the eye.")


def _detectors_section(P: dict) -> str:
    census = _g(P, "census", default={})
    cascade = _g(P, "upstream_cascade", default={})
    oo = _g(census, "on_off_split", default={})
    lb = _g(census, "layer_b_frac_of_t4t5")
    on = _g(cascade, "on_limb_t4b", default={})
    off = _g(cascade, "off_limb_t5b", default={})
    parts = [r"\section{The motion detectors that drive FD3}"]
    parts.append(
        rf"Of FD3's input from the local motion detectors, {_esc(lb)}\% comes from the "
        r"back-to-front (layer~b) channel, the direction FD3 is excited by, and essentially none "
        r"from the opposite direction. That layer~b drive is carried by \emph{both} kinds of "
        rf"detector: the ON detector \ttt{{T4b}} ({_esc(oo.get('T4b_ON'))} synapses) and the OFF "
        rf"detector \ttt{{T5b}} ({_esc(oo.get('T5b_OFF'))} synapses), a roughly even ON/OFF mix. So "
        r"FD3 pools back-to-front motion of both contrast polarities, supporting an object signal "
        r"across contrast polarity.")
    parts.append(_figure("figures/fd3_input_census.png",
                         r"\textbf{What feeds FD3.} (a) FD3's direct inputs by cell type, coloured by "
                         r"class; the motion detectors (red) are only a minority of the total. "
                         r"(b) Within the motion input, almost all of it is the back-to-front "
                         r"(layer~b) channel, split between the ON and OFF detectors.", "fig:census"))
    parts.append(
        r"Following those detectors one step further back, the \ttt{T4b} cells that drive FD3 are "
        rf"themselves fed by the canonical ON medulla cells ({_esc(', '.join(on.get('expected_medulla_present', [])))}), "
        r"and the \ttt{T5b} cells by the canonical OFF medulla cells "
        rf"({_esc(', '.join(off.get('expected_medulla_present', [])))}); these in turn are fed by the "
        r"lamina and, before it, the photoreceptors. The whole cascade from the eye to FD3's "
        r"detectors is present in the connectome and matches published optic-lobe wiring. We show "
        r"it as the afferent pathway below, marking which links we measure directly onto FD3 and "
        r"which are the general column-to-column wiring of the optic lobe.")
    parts.append(_figure("figures/fd3_afferent_cascade.png",
                         r"\textbf{The afferent cascade.} Photoreceptors $\rightarrow$ lamina "
                         r"$\rightarrow$ ON/OFF medulla $\rightarrow$ the back-to-front detectors "
                         r"\ttt{T4b}/\ttt{T5b} $\rightarrow$ FD3 $\rightarrow$ the steering neuron "
                         r"\ttt{DNp26} $\rightarrow$ wing. Solid links are measured on FD3 itself; "
                         r"dashed links are the standard column wiring measured cell-type by "
                         r"cell-type; the greyed link is the histaminergic front end known from "
                         r"physiology.", "fig:cascade"))
    return "\n\n".join(parts)


def _where_section(P: dict) -> str:
    """The two real-anatomy figures (retinotopic map + arbor), described in plain language."""
    parts = [r"\section{Where the input lands}"]
    parts.append(
        r"Each motion detector looks at one point in the eye's field of view, so the set of "
        r"detectors feeding FD3 draws out FD3's receptive field directly on the eye's lattice. "
        r"Plotting those input columns shows FD3's field sitting off to the side, shifted away from "
        r"straight ahead relative to the FD1 cell -- the lateral placement with a frontal gap that "
        r"defines FD3.")
    parts.append(_figure("figures/fd3_input_hexmap.png",
                         r"\textbf{FD3's input in eye coordinates.} Every motion column feeding FD3 "
                         r"(left) and FD1 (right), placed on the eye's hexagonal lattice and shaded "
                         r"by synapse number. FD3's centre of mass sits further to the side.",
                         "fig:hexmap"))
    parts.append(
        r"On the cell itself, the different kinds of input are not scattered at random: the motion "
        r"detectors, the neighbouring projection sheets, and the inhibitory cells each contact "
        r"characteristic parts of FD3's arbor. Rendering FD3's reconstructed shape with its input "
        r"synapses marked shows where each class of input arrives.")
    parts.append(_figure_fullpage("figures/fd3_arbor_inputs.png",
                         r"\textbf{Inputs on FD3's arbor.} Each FD3 cell's reconstructed shape (thin "
                         r"cable) with its input synapses drawn at their real positions and coloured "
                         r"by the class of cell they come from: back-to-front motion detectors "
                         r"(\ttt{T4b}/\ttt{T5b}), columnar projection sheets (\ttt{LPC}/\ttt{LLPC}), "
                         r"and contralateral inhibitors. The motion input clusters on the dendritic "
                         r"tuft in the lobula plate; the crossing axon carries the contralateral "
                         r"inhibitory contacts.", "fig:arbor"))
    parts.append(
        r"Seen alongside the cells it connects, FD3 sits anatomically between lobula-plate motion "
        r"detectors and descending neurons that contact wing-steering motor systems.")
    parts.append(_figure_fullpage("figures/fd3_circuit_3d.png",
                         r"\textbf{The circuit in the brain.} Reconstructed neurons of the FD3 "
                         r"pathway in their true anatomical positions: the \ttt{T4b} (ON) and "
                         r"\ttt{T5b} (OFF) motion detectors, both FD3 cells with their axons crossing "
                         r"toward the midline, and the descending neuron \ttt{DNp26}, whose dendrite "
                         r"meets FD3's axon terminal before descending toward the nerve cord.",
                         "fig:circuit3d"))
    return "\n\n".join(parts)


def _central_section(P: dict) -> str:
    central = _g(P, "central_inputs", default={})
    census = _g(P, "census", default={})
    t45f = _g(census, "t4t5_frac_of_total")
    top = central.get("top_types", []) if central else []
    names = ", ".join(
        rf"\ttt{{{_esc(t['cell_type'])}}} ({_esc(t.get('nt', '?'))}, layer {_esc(t.get('dominant_layer', '?'))})"
        for t in top[:4]
    ) or "several columnar and inhibitory sheets"
    n_layerb = central.get("n_layerb_carriers")
    return (
        r"\section{The central cells that converge on FD3}" "\n\n"
        rf"The motion detectors are only about {_esc(t45f)}\% of FD3's input. The larger share comes "
        rf"from central and optic-lobe projection inputs, led by {names}. These inputs are not a "
        r"single layer-b excitatory copy of the T4/T5 drive: the leading set includes cholinergic, "
        r"GABAergic and glutamatergic types, and their dominant layer annotations span horizontal "
        r"and vertical channels. In the current derived table, "
        rf"{_esc(n_layerb)} leading central type is marked as a layer-b carrier. FD3 is therefore "
        r"built from a back-to-front T4/T5 motion core plus broader contextual and inhibitory "
        r"inputs. This is a wiring description; the sign and function of each non-T4/T5 partner "
        r"remain hypotheses unless supported by transmitter identity or physiology.")


def _inhibition_section(P: dict) -> str:
    contra = _g(P, "contra_inhibition", default={})
    both = contra.get("both_present") if contra else None
    dom = contra.get("dominant_direction") if contra else None
    return (
        r"\section{Inhibition from the other eye}" "\n\n"
        r"FD3 is not only excited by an object on its own side; it is also inhibited by wide-field "
        r"motion, including motion seen by the \emph{opposite} eye. In the connectome FD3 receives "
        r"input from inhibitory (GABAergic) cells on the far side of the brain, "
        + (r"with classified synapses in both the front-to-back and back-to-front channels "
           + (rf"but dominated by the {_esc(dom)} channel, " if dom else "")
           if both else r"")
        + r"consistent with Egelhaaf's description of contralateral inhibition. Measured this way, "
        r"FD3's crossed inhibition is direction-dominant (mostly front-to-back) rather than evenly "
        r"bidirectional, and it is the \emph{opposite} dominant direction from the FD1 cell's "
        r"crossed inhibition -- a genuine contrast, though the fine split of the minority channel is "
        r"below what the current annotation can resolve. The report therefore treats the crossed "
        r"inhibitory motif as present and compatible with physiology rather than as a fully "
        r"enumerated mechanism. Its computational role remains a hypothesis: crossed inhibition is "
        r"positioned to suppress wide-field background motion, but the connectome alone does not "
        r"measure the resulting membrane-potential response.")


def _implications_section(P: dict) -> str:
    return (
        r"\section{FD3 is a parallel arm, not a copy of FD1}" "\n\n"
        r"Putting the input side together with the identity and output reports, FD3 emerges as a "
        r"candidate sensory-to-descending channel that runs in parallel to the FD1 arm and is tuned to the "
        r"opposite direction. Both cells end on the same steering neuron, \ttt{DNp26}, but they are "
        r"driven differently: FD1 reads front-to-back motion through a centrifugally gated sheet, "
        r"whereas FD3 reads back-to-front motion, pooled from both ON and OFF detectors and a broad "
        r"set of columnar sheets, with no such gate and with crossed inhibition from the other eye. "
        r"The connectome thus supports a model in which oppositely tuned figure-sensitive arms "
        r"converge on overlapping descending steering targets. Whether this anatomical convergence "
        r"implements direction-invariant figure tracking requires physiology or perturbation.")


def _caveats_section(P: dict) -> str:
    return (
        r"\section{What the data can and cannot settle}" "\n\n"
        r"\textbf{Motion is a fraction of the input.} The statement that FD3's motion drive is "
        r"almost entirely back-to-front refers to its \ttt{T4}/\ttt{T5} input specifically; that "
        r"motion input is itself only about a quarter of FD3's total input, and both numbers are "
        r"reported so the first is not mistaken for the second. \textbf{The upstream cascade is "
        r"general wiring.} The photoreceptor-to-detector cascade is measured cell-type by cell-type "
        r"across the optic lobe and matches the known column wiring; it is shown as the pathway that "
        r"culminates in FD3's detectors, not as a per-object trace unique to FD3. \textbf{The ON/OFF "
        r"and transmitter labels are from physiology.} That \ttt{T4} is the ON and \ttt{T5} the OFF "
        r"detector, and that the photoreceptors are histaminergic, come from published physiology; "
        r"the connectome supplies the connections and the cell names, not the sign of each cell "
        r"\cite{maisak2013,hardie1989}. "
        r"(The automated transmitter table mislabels the photoreceptors, and that value is not "
        r"used.) \textbf{The crossed inhibition is only partly resolved.} The inhibitory partners "
        r"from the opposite eye are sparse in the current annotation, so the bidirectional inhibition "
        r"is reported as present and consistent with the physiology rather than fully enumerated. "
        r"\textbf{FD3 is two cells.} As in the other reports, FD3 is a single left/right pair, so "
        r"the result rests on the two cells agreeing and on the thousands of synapses each receives, "
        r"not on a population average.")


def _methods_section(track: str) -> str:
    return (
        r"\section{Methods and provenance}" "\n\n"
        rf"\textbf{{Connectome.}} Inputs are read from the FlyWire FAFB brain connectome "
        rf"(materialization v783) on the \ttt{{{_esc(track)}}} track. \textbf{{Direct "
        r"inputs}: presynaptic partners of \ttt{LPT42\_Nod4}, aggregated by type and split into "
        r"motion detectors (with their lobula-plate layer composition), columnar projection sheets, "
        r"and inhibitory cells, with contralateral input identified by soma side. \textbf{Upstream "
        r"cascade}: the presynaptic \ttt{T4b}/\ttt{T5b} cells that drive FD3 are followed one step "
        r"back to their medulla inputs, and the lamina and photoreceptor layers are confirmed "
        r"present. \textbf{Crossed "
        r"inhibition}: contralateral GABAergic partners stratified by the lobula-plate layer their "
        r"type reads. \textbf{Figures}: neuron shapes are reconstructed skeletons; synapse positions "
        r"are the real synaptic coordinates; the eye-coordinate map uses each detector's retinotopic "
        r"column. \textbf{Reproducibility}: built from the Family-P verification JSON by the FD3 "
        r"report CLI. "
        rf"Primary track: \ttt{{{_esc(track)}}}.")


def _bibliography() -> str:
    items = [
        r"\bibitem{egelhaaf1985b} Egelhaaf M. (1985) On the neuronal basis of figure-ground "
        r"discrimination by relative motion in the visual system of the fly. II. Figure-detection "
        r"cells, a new class of visual interneurones. \textit{Biol. Cybern.} 52:195--209.",
        r"\bibitem{egelhaaf1985c} Egelhaaf M. (1985) \ldots III. Possible input circuitries and "
        r"behavioural significance of the FD-cells. \textit{Biol. Cybern.} 52:267--280.",
        r"\bibitem{fischbach1989} Fischbach K.-F., Dittrich A.P.M. (1989) The optic lobe of "
        r"\textit{Drosophila melanogaster}. I. A Golgi analysis of wild-type structure. "
        r"\textit{Cell Tissue Res.} 258:441--475.",
        r"\bibitem{maisak2013} Maisak M.S. et al. (2013) A directional tuning map of "
        r"\textit{Drosophila} elementary motion detectors. \textit{Nature} 500:212--216.",
        r"\bibitem{hardie1989} Hardie R.C. (1989) A histamine-activated chloride channel "
        r"involved in neurotransmission at a photoreceptor synapse. \textit{Nature} 339:704--706.",
        r"\bibitem{dorkenwald2024} Dorkenwald S. et al. (2024) Neuronal wiring diagram of an adult "
        r"brain. \textit{Nature} 634:124--138.",
    ]
    return (r"\begin{thebibliography}{9}" "\n" + "\n".join(items) + "\n"
            r"\end{thebibliography}")


def build_tex(results: dict) -> str:
    fam = results["families"]["P"] if "families" in results else results
    P = fam["derived"]
    track = results.get("meta", {}).get("flywire_track", P.get("track", "live"))
    parts = [
        _PREAMBLE,
        r"\begin{document}",
        r"\title{\textbf{The input pathway of the FD3 figure-detection cell "
        r"(photoreceptor $\rightarrow$ \ttt{T4b}/\ttt{T5b} $\rightarrow$ \ttt{LPT42\_Nod4})}}",
        r"\author{Stage 7 FD3 Identification, input pathway}",
        r"\date{\today}",
        r"\maketitle",
        _summary(P),
        _introduction(),
        _detectors_section(P),
        _where_section(P),
        _central_section(P),
        _inhibition_section(P),
        _implications_section(P),
        _caveats_section(P),
        _methods_section(track),
        _bibliography(),
        r"\end{document}",
    ]
    return "\n\n".join(parts)


def _render_figures(results: dict, fig_dir: Path, *, src=None, meta=None) -> None:
    """Render the scalar census figure (always, from JSON) + the anatomical figures (best-effort,
    only when a FlyWire source + metadata are supplied; otherwise reuse any existing PNGs)."""
    fam = results["families"]["P"] if "families" in results else results
    p_derived = fam["derived"]
    run = {"derived": {"P": p_derived}, "results": {"P": []}}
    PF.fd3_input_census(run, fig_dir / "fd3_input_census.png")
    # Anatomical figures need real skeleton/synapse data. Render if a source was provided.
    if src is not None and meta is not None:
        from ..paper import figures_anat as FA
        FA.render_all(src, meta, p_derived, fig_dir)


def build_report(stage_dir: str | Path = "7 - FD3 Identification",
                 results_path: str | Path = "5 - Paper Verification/verification_results.json",
                 *, src=None, meta=None) -> dict[str, Any]:
    stage = Path(stage_dir)
    (stage / "figures").mkdir(parents=True, exist_ok=True)
    results = read_json(Path(results_path))
    _render_figures(results, stage / "figures", src=src, meta=meta)
    tex_path = stage / "fd3_input_report.tex"
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
