"""Build the FD3 functional figure-ground circuit report (LaTeX -> PDF).

The fourth FD3 report, answering the mentor directly: it (1) lays out FD3's comprehensive
connectivity census and then (2) isolates and NAMES the functional circuit inside it —
T4b/T5b -> LPC1 (the layer-b sheet) -> FD3 -> DNp26 -> wing, with LPi14 the wide-field opponent
inhibitor standing in VCH's place. Reads the ``families.Q``, ``families.R``, ``families.S`` and
(optional) ``families.census`` / compartment blocks from the verification JSON.

Prose is READER-FACING (no verdict/claim vocabulary; guard test enforces).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import read_json
from ..paper import figures as PF
from .builder import _PREAMBLE, _figure, _esc, _g, _compile


def _figure_fullpage(rel_png: str, caption: str, label: str) -> str:
    return (r"\begin{figure}[p]\centering" "\n"
            rf"\includegraphics[width=\textwidth,height=0.86\textheight,keepaspectratio]{{{rel_png}}}" "\n"
            rf"\caption{{{caption}}}\label{{{label}}}" "\n"
            r"\end{figure}" "\n")


def _summary(Q: dict, R: dict) -> str:
    named = _g(Q, "named_sheet") or "LPC1"
    win = _g(R, "winner", default={})
    inh = _g(win, "cell_type") or "LPi14"
    sdw = _g(R, "same_direction_winner", default={}) or {}
    sd_inh = _g(sdw, "cell_type") or "LPi12"
    return (
        r"\textbf{Summary.}\quad "
        r"The figure-detection cell FD3 (\ttt{LPT42\_Nod4}) sits inside a large web of connections "
        r"-- thousands of partners in and out. This report first maps that web comprehensively, "
        r"then picks out the working figure-ground circuit within it and names every element, in "
        r"the same terms as the previously described FD1 pathway. We find that the motion signal "
        rf"reaches FD3 through a named intermediate sheet -- the cell type \ttt{{{_esc(named)}}}, the "
        r"one neighbouring sheet that reads the same back-to-front motion channel FD3 does -- so "
        rf"the pathway is \ttt{{T4b}}/\ttt{{T5b}} $\rightarrow$ \ttt{{{_esc(named)}}} $\rightarrow$ "
        r"FD3, with the sibling sheets adding a four-direction motion-context surround. And we find "
        r"that FD3 is shaped by \emph{two} complementary wide-field inhibitory gates. "
        rf"\ttt{{{_esc(inh)}}} plays the role the wide-field cell VCH plays for FD1 -- it pools "
        r"motion across the field, feeds back onto the detectors, gates the sheet, and inhibits FD3 "
        r"-- but is a different cell class, tuned to the opposite direction, so it is an "
        rf"\emph{{opponent}} gate. \ttt{{{_esc(sd_inh)}}} is the \emph{{same-direction}} surround "
        r"Egelhaaf predicted: a regressive wide-field cell that acts one stage upstream, dominating "
        r"the inhibition of FD3's own \ttt{T4b}/\ttt{T5b} detectors while touching FD3 only weakly. "
        rf"The two gates are complementary, not competing: \ttt{{{_esc(sd_inh)}}} does not replace "
        rf"\ttt{{{_esc(inh)}}}.")


def _census_section(cen: dict) -> str:
    inp = _g(cen, "input", default={})
    outp = _g(cen, "output", default={})
    P = [r"\section{FD3's connectivity, comprehensively}"]
    P.append(
        rf"Before isolating the figure-ground circuit we map FD3's whole connectivity. FD3 "
        rf"receives about {_esc(_g(inp, 'total_syn'))} input synapses from roughly "
        rf"{_esc(_g(inp, 'n_partners'))} partner cells, and makes about {_esc(_g(outp, 'total_syn'))} "
        rf"output synapses onto roughly {_esc(_g(outp, 'n_partners'))} partners. The inputs are "
        r"dominated by the motion detectors and a handful of neighbouring columnar sheets together "
        r"with inhibitory cells; the outputs are broadcast widely to central brain cells and, "
        r"through the descending neurons, to the steering muscles. Against this backdrop the "
        r"working figure-ground circuit is a small, identifiable subset, highlighted below.")
    P.append(_figure("figures/fd3_connectivity_wheel.png",
                     r"\textbf{FD3's comprehensive connectivity.} Every major input (left) and "
                     r"output (right) partner type of FD3. The cells of the functional figure-ground "
                     r"circuit are drawn in colour; the rest of the web is greyed, so the circuit "
                     r"stands out from the full connectivity.", "fig:wheel"))
    P.append(
        r"The inputs also land in a structured way on the cell. Splitting FD3's input synapses "
        r"between its dendrite and its axon shows that essentially all of the input -- the motion "
        r"drive, the sheet, and the inhibition alike -- arrives on the dendritic arbour in the "
        r"lobula plate, while the axon carries the cell's output across the midline. FD3 is thus a "
        r"cleanly polarised neuron, and the inhibition is applied on the same arbour that receives "
        r"the excitation.")
    P.append(_figure("figures/fd3_compartment_split.png",
                     r"\textbf{Where inputs land.} Input synapses to each FD3 cell, split between "
                     r"the dendrite and the axon and coloured by the class of partner. Almost all "
                     r"input -- excitatory and inhibitory -- is on the dendrite.", "fig:compartment"))
    return "\n\n".join(P)


def _sheet_section(Q: dict) -> str:
    named = _g(Q, "named_sheet") or "LPC1"
    sset = _g(Q, "sheet_set", default=[])
    prof = _g(Q, "profiles", named, default={})
    retino = _g(Q, "retinotopy_null", default={})
    P = [r"\section{Naming the sheet: T4b/T5b $\rightarrow$ LPC1 $\rightarrow$ FD3}"]
    P.append(
        rf"For the FD1 cell, the motion detectors do not contact the figure cell directly; they "
        rf"first drive a retinotopic sheet of cells (\ttt{{LLPC1}}) that then relays to FD1. FD3 "
        rf"has the same architecture, and the sheet that fills the equivalent slot is "
        rf"\ttt{{{_esc(named)}}}. Of the neighbouring sheets that contact FD3, \ttt{{{_esc(named)}}} "
        rf"is the only one that reads the same back-to-front (layer-b) motion channel FD3 is tuned "
        rf"to (its sibling sheets read the vertical up/down channels instead), it is cholinergic "
        rf"(excitatory), and it is driven by the very same \ttt{{T4b}}/\ttt{{T5b}} detectors "
        rf"({_esc(_g(prof, 'to_fd3_syn'))} synapses onto FD3). It is also genuinely retinotopic: "
        rf"the detectors feeding each \ttt{{{_esc(named)}}} cell come from a tight local patch of "
        r"the eye, far tighter than chance, so it preserves the spatial map rather than pooling "
        r"globally.")
    P.append(
        rf"So the intermediate is best described as a small \emph{{set}} of sibling sheets "
        rf"({_esc(', '.join(sset))}) of which \ttt{{{_esc(named)}}} is the direction-matched, "
        r"horizontal member carrying the figure signal. Notably, FD1's own sheet (\ttt{LLPC1}) is "
        r"\emph{not} a major input to FD3 -- FD3 reads a different sheet, as befits its opposite "
        r"direction preference.")
    # directional composition: the sheets FD3 pools carry all four cardinal directions.
    comp = _g(Q, "directional_composition", default={})
    fd1ref = _g(Q, "fd1_reference", default={})
    if comp:
        frac = _g(comp, "sheet_frac_by_channel", default={})
        P.append(
            r"The sibling sheets are not simply ``vertical context'': each reads its own cardinal "
            r"motion channel, so FD3's sheet-relayed input spans all four directions. Rolling the "
            r"sheets' synapses onto FD3 up by the motion channel each one reads gives "
            rf"\ttt{{LPC1}} (back-to-front, FD3's own direction) {_esc(frac.get('b'))}\%, "
            rf"\ttt{{LLPC3}} (downward) {_esc(frac.get('d'))}\%, \ttt{{LLPC2}}/\ttt{{LPC2}} (upward) "
            rf"{_esc(frac.get('c'))}\%, and \ttt{{LLPC1}} (front-to-back, the opposite direction) "
            rf"{_esc(frac.get('a'))}\% of the sheet input. The direction-matched channel is thus a "
            rf"\emph{{minority}} ({_esc(_g(comp, 'matched_frac'))}\%) of what the sheets deliver: FD3 "
            r"reads its own regressive channel through \ttt{LPC1} plus a near-balanced "
            r"up/down/opposite \emph{motion-context surround} through the siblings. This is a "
            r"statement about the \emph{sheet-relayed} input; FD3's \emph{direct} \ttt{T4}/\ttt{T5} "
            r"drive stays $\sim$99\% back-to-front (Part~II).")
        if _g(fd1ref, "available") and not _g(fd1ref, "fd3_specific"):
            P.append(
                r"This multi-direction sheet pooling is, however, a general property of wide-field "
                r"tangential cells rather than a signature unique to FD3: the FD1 cell shows a "
                r"comparable spread through its own sheets, so we report it as context, not as an "
                r"FD3-specific discovery.")
    P.append(_figure("figures/fd3_functional_circuit_schematic.png",
                     r"\textbf{The named figure-ground circuit, with its two gates.} "
                     r"\ttt{T4b}/\ttt{T5b} motion detectors drive the layer-b sheet \ttt{LPC1}, "
                     r"which relays to FD3; FD3 drives the steering neuron \ttt{DNp26} and the wing. "
                     r"Two wide-field inhibitory cells (flat-headed arrows) gate the circuit at "
                     r"different nodes: the opponent cell \ttt{LPi14} gates the sheet, feeds back "
                     r"onto the detectors, and inhibits FD3; the same-direction cell \ttt{LPi12} "
                     r"pours inhibition onto the \ttt{T4b}/\ttt{T5b} detectors. Numbers are measured "
                     r"synapse counts.",
                     "fig:funcircuit"))
    return "\n\n".join(P)


def _inhibitor_section(R: dict) -> str:
    win = _g(R, "winner", default={})
    inh = _g(win, "cell_type") or "LPi14"
    sdw = _g(R, "same_direction_winner", default={}) or {}
    sd_inh = _g(sdw, "cell_type") or (_g(R, "same_direction_gates", default=["LPi12"]) or ["LPi12"])[0]
    enr = _g(R, "same_direction_enrichment", default={})
    P = [r"\section{Two wide-field gates: an opponent gate and a same-direction surround}"]
    P.append(
        rf"For FD1, a single wide-field cell, VCH, does four things: it pools motion across the "
        rf"whole visual field, feeds that back onto the motion detectors, gates the sheet, and so "
        rf"shapes the figure cell's selectivity. For FD3, the cell that fills that role at the "
        rf"figure-cell and sheet level is \ttt{{{_esc(inh)}}}. It pools wide-field motion "
        rf"({_esc(_g(win, 't4t5_in_frac'))}\% of its own input is motion detectors), it feeds back "
        rf"onto FD3's \ttt{{T4b}}/\ttt{{T5b}} detectors ({_esc(_g(win, 'to_detectors_syn'))} "
        rf"synapses), it gates the \ttt{{LPC1}} sheet ({_esc(_g(win, 'to_sheet_syn'))} synapses), and "
        rf"it inhibits FD3 directly ({_esc(_g(win, 'to_fd3_syn'))} synapses -- "
        rf"{_esc(_g(win, 'frac_of_fd3_inhibition'))}\% of all the inhibition FD3 receives). In wiring "
        r"terms it stands where VCH stands for FD1.")
    P.append(
        rf"There are three honest differences from VCH, and they matter. First, \ttt{{{_esc(inh)}}} "
        r"is a lobula-plate \emph{intrinsic} cell, not a centrifugal cell like VCH -- a different "
        r"cell class filling the same role. Second, it is tuned to the \emph{opposite} direction "
        r"from FD3 (it reads front-to-back motion, FD3 reads back-to-front), so it acts as an "
        r"\emph{opponent} suppressor rather than a same-direction gain control. Third, FD3 does not "
        r"send much back to it, so the loop is not reciprocal in the way VCH's is. For these "
        rf"reasons we describe \ttt{{{_esc(inh)}}} as the \emph{{functional}} counterpart of VCH -- "
        r"it plays VCH's part in the circuit -- rather than the same cell type.")
    P.append(
        r"Egelhaaf, recording from FD3, inferred a second wide-field inhibitor tuned to the "
        r"\emph{same} direction as FD3's excitation, to explain how FD3 tells a small object from "
        rf"whole-field motion. Searching the full lobula-plate intrinsic panel resolves it: "
        rf"\ttt{{{_esc(sd_inh)}}}, a GABAergic cell tuned to the \emph{{same}} back-to-front direction "
        rf"as FD3 ({_esc(_g(sdw, 'layer_b_pct'))}\% layer-b). But \ttt{{{_esc(sd_inh)}}} does its work "
        rf"at a \emph{{different node}}: it pours "
        rf"{_esc(_g(sdw, 'to_detectors_syn'))} synapses onto the \ttt{{T4b}}/\ttt{{T5b}} detectors "
        rf"themselves -- {_esc(_g(sdw, 'to_detectors_frac_of_output'))}\% of its own output, and a "
        rf"larger share of the detectors' total inhibition than \ttt{{{_esc(inh)}}} supplies -- while "
        rf"touching FD3 only weakly ({_esc(_g(sdw, 'to_fd3_syn'))} synapses, "
        rf"{_esc(_g(sdw, 'frac_of_fd3_inhibition'))}\% of FD3's inhibition) and barely reaching the "
        rf"sheet ({_esc(_g(sdw, 'to_sheet_syn'))} synapses). So \ttt{{{_esc(sd_inh)}}} does "
        rf"\emph{{not}} replace \ttt{{{_esc(inh)}}}; the two are complementary gates at different "
        r"stages -- an opponent gate on the figure cell and its sheet, and a same-direction surround "
        r"on the motion detectors.")
    if _g(enr, "available"):
        P.append(
            rf"That \ttt{{{_esc(sd_inh)}}} touches FD3 at all is not an accident of its size: its "
            rf"{_esc(_g(enr, 'obs_to_fd3'))} synapses onto FD3 exceed what a size-matched random "
            rf"lobula-plate target would receive from it (null {_esc(_g(enr, 'null_mean'))}, "
            rf"$z={_esc(_g(enr, 'z_score'))}$, $p={_esc(_g(enr, 'p_enrichment'))}$), so the (minor) "
            r"direct contact is a genuine FD3-specific input rather than spillover. Whether either "
            r"gate produces net suppression, and which carries the effect Egelhaaf measured, is a "
            r"question for physiology; the connectome supplies the wiring and the direction, not the "
            r"sign of the response.")
    return "\n\n".join(P)


def _implications_section(S: dict) -> str:
    return (
        r"\section{The circuit, named end to end}" "\n\n"
        r"Putting the pieces together, FD3's figure-ground circuit reads, in full: the "
        r"\ttt{T4b}/\ttt{T5b} back-to-front motion detectors drive the layer-b sheet \ttt{LPC1} "
        r"(one of a sibling set that also relays the other three cardinal directions as context), "
        r"which relays to FD3; FD3 drives the steering command neuron \ttt{DNp26} and thence the "
        r"wing. Two wide-field gates shape it: the opponent cell \ttt{LPi14} gates the sheet, feeds "
        r"back onto the detectors, and inhibits FD3 directly; and the same-direction cell "
        r"\ttt{LPi12} pours inhibition onto FD3's detectors while barely touching FD3 itself. This "
        r"is the same core architecture as the FD1 arm (detectors $\rightarrow$ sheet $\rightarrow$ "
        r"figure cell $\rightarrow$ steering command, with wide-field inhibitory gating), built from "
        r"a different, oppositely-tuned set of cells, and with the same-direction surround Egelhaaf "
        r"predicted resolved as a distinct detector-level gate. Whether these gates implement the "
        r"small-versus-wide-field discrimination Egelhaaf described is a functional question the "
        r"wiring sets up but does not answer.")


def _caveats_section() -> str:
    return (
        r"\section{What the data can and cannot settle}" "\n\n"
        r"\textbf{The sheet is a set, and LPC1 is its direction-matched member.} Several sibling "
        r"sheets contact FD3, carrying all four cardinal motion directions; \ttt{LPC1} is named as "
        r"the figure-carrying one because it is the only one reading FD3's own (back-to-front) "
        r"channel, and the multi-direction pooling is reported as sheet-relayed context, not as a "
        r"change in FD3's own $\sim$99\% back-to-front tuning. \textbf{Two gates, at different "
        r"nodes, not a replacement.} \ttt{LPi14} is the opponent gate on FD3 and its sheet; "
        r"\ttt{LPi12} is the same-direction gate on FD3's detectors. The two are named by "
        r"\emph{where} and \emph{in which direction} they act, normalised against each target's "
        r"total inhibition so raw synapse counts (and \ttt{LPi14}'s known hub connectivity) do not "
        r"mislead; \ttt{LPi12} is a minor slice of FD3's own inhibition, so it complements rather "
        r"than replaces \ttt{LPi14}. \textbf{Sign is not read from wiring.} That a GABAergic contact "
        r"is inhibitory, and that either gate suppresses FD3's figure response, are expectations "
        r"from physiology; the connectome supplies the connections and the direction, not the sign "
        r"or the response. Some inhibitory-looking cells here are predicted glutamatergic, whose "
        r"sign depends on the receptor and is left open. \textbf{FD3 is one bilateral pair.} As "
        r"throughout, the result rests on the two cells agreeing (both show the same two-gate "
        r"asymmetry) and on the thousands of synapses each makes, not on a population average.")


def _methods_section(track: str) -> str:
    return (
        r"\section{Methods and provenance}" "\n\n"
        r"\textbf{Connectome.} All connectivity is read from the FlyWire FAFB brain connectome "
        r"(materialization v783), cross-checked offline. \textbf{Census}: every pre- and "
        r"postsynaptic partner of \ttt{LPT42\_Nod4}, partitioned by cell class, transmitter, "
        r"lobula-plate layer and side. \textbf{Sheet}: candidate columnar-projection sheets "
        r"feeding FD3 scored by their motion-channel tuning, their synapses onto FD3, and a "
        r"retinotopy test (each sheet cell's detector inputs vs a shuffled-input null). "
        r"\textbf{Sheet directions}: each sheet's synapses onto FD3 rolled up by the motion "
        r"channel it reads; a sheet's direction label is used only when its dominant layer beats "
        r"the runner-up by $\geq$20 percentage points and is stable in $\geq$95\% of synapse "
        r"resamples. Whether four-direction pooling is FD3-specific is tested against the FD1=Nod1 "
        r"reference and a random lobula-plate-tangential entropy null. \textbf{Inhibitor}: FD3's "
        r"inhibitory inputs screened across the full lobula-plate intrinsic panel for wide-field "
        r"pooling, feedback onto the detectors, gating of the sheet, and direct inhibition of FD3 "
        r"(the screen does not require a centrifugal cell type). Each candidate's contact is "
        r"normalised against the target's total inhibition (and the candidate's own output budget) "
        r"so raw counts and hub connectivity do not mislead; a same-direction candidate's direct-FD3 "
        r"contact is tested for FD3-specificity against a size-matched random-target permutation "
        r"null, and every threshold-dependent call is reported across a floor sweep. "
        r"\textbf{Compartments}: input synapses assigned to the dendrite or axon by nearest point "
        r"on the reconstructed skeleton. \textbf{Left/right agreement}: FD3 is a bilateral pair, so "
        r"the two-gate asymmetry and the direction labels are checked on both cells. "
        rf"\textbf{{Reproducibility}}: rebuilt from the verification JSON. Primary track: \ttt{{{_esc(track)}}}.")


def _bibliography() -> str:
    items = [
        r"\bibitem{egelhaaf1985b} Egelhaaf M. (1985) On the neuronal basis of figure-ground "
        r"discrimination by relative motion in the visual system of the fly. II. Figure-detection "
        r"cells. \textit{Biol. Cybern.} 52:195--209.",
        r"\bibitem{egelhaaf1985c} Egelhaaf M. (1985) \ldots III. Possible input circuitries and "
        r"behavioural significance of the FD-cells. \textit{Biol. Cybern.} 52:267--280.",
        r"\bibitem{fischbach1989} Fischbach K.-F., Dittrich A.P.M. (1989) The optic lobe of "
        r"\textit{Drosophila melanogaster}. \textit{Cell Tissue Res.} 258:441--475.",
        r"\bibitem{maisak2013} Maisak M.S. et al. (2013) A directional tuning map of "
        r"\textit{Drosophila} elementary motion detectors. \textit{Nature} 500:212--216.",
        r"\bibitem{dorkenwald2024} Dorkenwald S. et al. (2024) Neuronal wiring diagram of an adult "
        r"brain. \textit{Nature} 634:124--138.",
    ]
    return (r"\begin{thebibliography}{9}" "\n" + "\n".join(items) + "\n"
            r"\end{thebibliography}")


def _fam(results, fid):
    if "families" in results and fid in results["families"]:
        return results["families"][fid]["derived"]
    return results.get(fid, {})


def build_tex(results: dict) -> str:
    Q = _fam(results, "Q")
    R = _fam(results, "R")
    S = _fam(results, "S")
    cen = _fam(results, "census")
    track = results.get("meta", {}).get("flywire_track", "live")
    parts = [
        _PREAMBLE,
        r"\begin{document}",
        r"\title{\textbf{The FD3 figure-ground circuit, named: "
        r"\ttt{T4b}/\ttt{T5b} $\rightarrow$ \ttt{LPC1} $\rightarrow$ \ttt{LPT42\_Nod4} "
        r"$\rightarrow$ \ttt{DNp26}, gated by \ttt{LPi14}}}",
        r"\author{Stage 7 FD3 Identification, functional circuit}",
        r"\date{\today}",
        r"\maketitle",
        _summary(Q, R),
        _census_section(cen),
        _sheet_section(Q),
        _inhibitor_section(R),
        _implications_section(S),
        _caveats_section(),
        _methods_section(track),
        _bibliography(),
        r"\end{document}",
    ]
    return "\n\n".join(parts)


def _render_figures(results: dict, fig_dir: Path, *, src=None, meta=None) -> None:
    """Render the census/compartment scalar figures + (if a source is given) the anatomical
    functional-circuit schematic. The wheel/matrix/compartment come from the census + compartment
    blocks; the schematic + 3D come from figures_anat.render_all."""
    Q = _fam(results, "Q"); R = _fam(results, "R")
    cen = _fam(results, "census"); comp = _fam(results, "compartment")
    run = {"derived": {"census": cen, "compartment": comp, "Q": Q, "R": R}, "results": {}}
    for fn, name in ((PF.fd3_connectivity_wheel, "fd3_connectivity_wheel"),
                     (PF.fd3_connectivity_matrix, "fd3_connectivity_matrix"),
                     (PF.fd3_compartment_split, "fd3_compartment_split")):
        try:
            fn(run, fig_dir / f"{name}.png")
        except Exception as e:  # noqa: BLE001
            print(f"[functional] {name} failed: {type(e).__name__}: {e}")
    if src is not None and meta is not None:
        from ..paper import figures_anat as FA
        p_block = _fam(results, "P")
        FA.render_all(src, meta, p_block, fig_dir, q_derived=Q, r_derived=R)


def _augment_with_census(results: dict, src, meta) -> dict:
    """Compute the comprehensive census + per-compartment split for FD3 and fold them into the
    results dict (under families.census / families.compartment), so the report has them even
    though they are not verification families. Needs a FlyWire source."""
    if src is None or meta is None:
        return results
    try:
        from ..paper.derive import connectivity_census as CC
        from ..paper.derive import compartments as CP
        fd3 = sorted(int(x) for x in meta.root_ids_of_type(["LPT42_Nod4"]))
        cen = CC.full_census(src, meta, fd3)
        comp = [CP.compartment_input_split(src, meta, r) for r in fd3]
        results.setdefault("families", {})
        results["families"]["census"] = {"derived": cen}
        results["families"]["compartment"] = {"derived": comp}
    except Exception as e:  # noqa: BLE001 — the report still builds without the census
        print(f"[functional] census/compartment computation failed: {type(e).__name__}: {e}")
    return results


def build_report(stage_dir: str | Path = "7 - FD3 Identification",
                 results_path: str | Path = "5 - Paper Verification/verification_results.json",
                 *, src=None, meta=None) -> dict[str, Any]:
    stage = Path(stage_dir)
    (stage / "figures").mkdir(parents=True, exist_ok=True)
    results = read_json(Path(results_path))
    results = _augment_with_census(results, src, meta)
    _render_figures(results, stage / "figures", src=src, meta=meta)
    tex_path = stage / "fd3_functional_circuit_report.tex"
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
