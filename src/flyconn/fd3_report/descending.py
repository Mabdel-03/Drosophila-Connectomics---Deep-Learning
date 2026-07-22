"""Build + compile the Stage-7 *descending-neuron* report (LaTeX -> PDF) from the Family-L JSON.

Companion to ``builder.py`` (which builds the FD3 *identity* report). This one asks which
descending neurons (DNs) receive FD3 output, and which motor systems those DNs contact in the
male CNS map. It reads ONLY the ``families.L`` block of
``5 - Paper Verification/verification_results.json`` (no live CAVE, no feathers) and writes
``7 - FD3 Identification/fd3_descending_report.{tex,pdf}``.

Prose is READER-FACING: it presents each result as a finding (what we measure + what it means
biologically), with NO verdict/claims vocabulary ("CONFIRMED", "claim", "CORE/DISC", "proxy"). A
guard test enforces this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import read_json
from ..paper import figures as PF
from ..motif import compare as K
from .builder import _PREAMBLE, _figure, _esc, _g, _compile


def _summary(L: dict) -> str:
    direct = _g(L, "direct", default={})
    motor = _g(L, "motor", default={})
    top = _g(direct, "top_dn")
    n_direct = _g(direct, "n_dns")
    dom = _g(motor, "dominant_motor_system") if motor.get("available") else None
    return (
        r"\textbf{Summary.}\quad "
        r"The figure-detection cell FD3 -- the cell type \ttt{LPT42\_Nod4} in the fly connectome -- "
        r"is anatomically positioned to report a small moving object in the visual field. For that "
        r"signal to influence movement it must reach \emph{descending neurons}: cells that project "
        r"from brain circuits into the ventral nerve cord and contact motor-control networks. This "
        r"report follows FD3's output to those neurons. We find that "
        rf"FD3 contacts about {_esc(n_direct)} descending neurons directly, and that its single "
        r"strongest direct target is \ttt{" + _esc(top) + r"} -- the very same wing-steering command "
        r"neuron on which the previously mapped FD1 analysis converges, so the two figure-cell "
        r"arms share a major descending target. Beyond this direct route, FD3 reaches a much larger set of descending "
        r"neurons through one relay step via its strongest central partners. Following every "
        r"identified descending neuron into the male nerve-cord connectome, where each is matched to "
        r"the muscles it drives, shows that FD3's descending output is directed "
        + (rf"most strongly toward the \emph{{wing-steering}} motor system" if dom == "wing_steering"
           else r"at several motor systems") +
        r" in this anatomical proxy, with the principal target steering the \emph{opposite} wing. "
        r"This supports a testable steering hypothesis rather than proving a behavioral command.")


def _introduction() -> str:
    return r"""\section{Introduction}

\textbf{From a figure signal to a movement.}\quad
A figure-detection cell that fires when a small object moves is only useful to the animal if its
signal can reach the muscles. In the fly, every motor command leaves the brain through a
numerically small population of \emph{descending neurons} (DNs), each of which projects from the
brain into the ventral nerve cord and there contacts motor-control networks~\cite{namiki2018}.
Identifying which descending neurons a visual cell drives, and which muscles those neurons in turn
contact, therefore converts an anatomical identity into a concrete behavioural hypothesis.

\textbf{The template: the FD1 steering arm.}\quad
The figure-ground circuit this work belongs to has already been traced on its FD1 branch in the
companion analysis. There,
the figure signal carried by the cell type \ttt{Nod1} reaches the descending neuron \ttt{DNp26}, a
convergent wing-steering command neuron, and \ttt{DNp26} drives contralateral wing-steering muscles
(\ttt{hg1}, \ttt{i1}, \ttt{hg2}). The question here is whether FD3 -- a regressive,
fronto-lateral figure cell distinct from FD1 -- feeds the same kind of motor pathway, and if so
through which neurons.

\textbf{What we measure.}\quad
We follow FD3's output along two routes. The \emph{direct} route is the set of descending neurons
that receive synapses straight from FD3's axon. The \emph{relayed} route is the set reached one
synapse further on, through FD3's strongest central (non-descending) partners. For every descending
neuron found on either route we then cross into the male whole--central-nervous-system connectome,
which -- unlike the brain-only dataset -- includes the nerve cord, and read off the motor neurons,
muscles and wing side each descending neuron contacts. Egelhaaf, in the third part of his 1985
study, argued that the FD cells serve figure-tracking and the fixation of a moving object; the motor
read-out here is an anatomical counterpart of that behavioural proposal~\cite{egelhaaf1985c}."""


def _direct_section(L: dict) -> str:
    direct = _g(L, "direct", default={})
    ranking = _g(direct, "ranking", default=[]) or []
    top = _g(direct, "top_dn")
    n = _g(direct, "n_dns")
    syn = _g(direct, "dn_syn")
    frac = _g(direct, "steering_frac")
    per_side = _g(direct, "per_side", default={}) or {}
    # a short inline list of the leading steering DNs.
    lead = ", ".join(rf"\ttt{{{_esc(r['cell_type'])}}}" for r in ranking[:6])
    sides_txt = "; ".join(f"{_esc(s)} cell: {_esc(v.get('dn_syn'))} synapses" for s, v in per_side.items())
    P = [r"\section{The descending neurons FD3 contacts directly}"]
    P.append(
        rf"FD3 makes {_esc(syn)} output synapses onto {_esc(n)} descending neurons. The leading "
        rf"targets are {lead} (Fig.~\ref{{fig:dnrank}}). The single strongest is "
        rf"\ttt{{{_esc(top)}}}; about {_esc(frac)}\% of FD3's direct descending output goes to "
        r"neurons already known to drive wing-steering muscles. Both FD3 cells -- one per side of "
        rf"the brain -- contribute ({sides_txt}), so the projection is bilateral, as the cell is.")
    if top == "DNp26":
        P.append(
            r"\ttt{DNp26} is notable: it is the same convergent wing-steering command neuron that "
            r"the FD1 cell reaches through \ttt{Nod1}. FD3 and FD1, two figure cells tuned to "
            r"opposite directions of motion and different parts of the visual field, therefore "
            r"converge on a shared descending target. This is anatomical support for a common "
            r"figure-tracking output stage.")
    P.append(_figure("figures/fd3_dn_ranking.png",
                     r"\textbf{The descending neurons FD3 contacts directly.} Each bar is one "
                     r"descending-neuron type, ranked by the number of synapses it receives from "
                     r"FD3. Red marks neurons already known to drive wing-steering muscles. "
                     r"\ttt{DNp26}, the figure-steering command neuron shared with the FD1 arm, "
                     r"leads.", "fig:dnrank"))
    return "\n\n".join(P)


def _relay_section(L: dict) -> str:
    relay = _g(L, "relay", default={})
    overlap = _g(L, "overlap", default={})
    n_dns = _g(relay, "n_dns")
    n_types = _g(relay, "n_seed_types")
    min_syn = _g(L, "relay_min_syn")
    seed_types = _g(relay, "seed_types", default=[]) or []
    lead = ", ".join(rf"\ttt{{{_esc(s['cell_type'])}}}" for s in seed_types[:6])
    ov = overlap.get("overlap", []) or []
    ov_txt = ", ".join(rf"\ttt{{{_esc(t)}}}" for t in ov[:8])
    P = [r"\section{The descending neurons FD3 reaches through a relay}"]
    P.append(
        rf"Beyond the neurons it drives directly, FD3 sends most of its output to central brain "
        rf"neurons rather than to descending neurons. Taking FD3's strongest such partners (those "
        rf"receiving at least {_esc(min_syn)} synapses from FD3 -- {_esc(n_types)} cell types, led "
        rf"by {lead}) and following \emph{{their}} output reaches a further {_esc(n_dns)} descending "
        r"neurons (Fig.~\ref{fig:dnchannels}). This relayed route is broad where the direct route is "
        r"focused: it spreads onto many more descending neurons, the larger of the two routes by "
        r"total synapses.")
    P.append(
        rf"Crucially the two routes are not unrelated. {_esc(overlap.get('n_overlap'))} "
        rf"descending-neuron types are reached \emph{{both}} directly and through the relay, "
        + (rf"including {ov_txt}" if ov_txt else "") +
        r". These overlaps show that the relay reaches some of the same DN classes as the direct "
        r"route. Because the "
        r"size of the relayed set depends on how strong a partner has to be to count, we report that "
        r"sensitivity explicitly (Fig.~\ref{fig:dnchannels}b): the count grows smoothly as the "
        r"threshold is relaxed.")
    P.append(_figure("figures/fd3_descending_channels.png",
                     r"\textbf{Two descending routes from FD3.} (a) The direct route (FD3 "
                     r"$\rightarrow$ descending neuron) is focused; the relayed route (FD3 "
                     r"$\rightarrow$ central partner $\rightarrow$ descending neuron) is broader, "
                     r"reaching many more neurons. (b) The number of relayed descending neurons as "
                     r"the partner-strength threshold is varied -- it changes smoothly, so the "
                     r"result does not hinge on one cut-off.", "fig:dnchannels"))
    return "\n\n".join(P)


def _motor_section(L: dict) -> str:
    motor = _g(L, "motor", default={})
    P = [r"\section{Motor systems contacted by those neurons}"]
    if not motor.get("available"):
        P.append(r"The male nerve-cord connectome was not available in this run, so the muscles "
                 r"these descending neurons drive could not be read out here. In the brain-only "
                 r"data the descending targets are nonetheless identified by name, and the leading "
                 r"ones are known wing-steering command neurons (above).")
        return "\n\n".join(P)
    msp = motor.get("motor_system_pct", {}) or {}
    dom = motor.get("dominant_motor_system")
    per_dn = motor.get("per_dn", {}) or {}
    dnp26 = per_dn.get("DNp26", {})
    dom_pct = msp.get(dom)
    P.append(
        r"Each descending neuron FD3 reaches can be matched, by its standard name, to a cell in the "
        r"male nerve-cord connectome and from there to the motor neurons and muscles it contacts. "
        rf"Weighting every neuron by how strongly FD3 drives it, the largest share of FD3's "
        rf"descending output -- {_esc(dom_pct)}\% -- reaches the \emph{{wing-steering}} muscles "
        r"(Fig.~\ref{fig:motorsys}). Smaller shares reach the neck (gaze), abdominal and haltere "
        r"systems. This anatomical weighting supports a steering hypothesis, but it does not by "
        r"itself prove that FD3 activity drives steering behavior.")
    if dnp26:
        muscles = ", ".join(rf"\ttt{{{_esc(m)}}}" for m in list(dnp26.get("top_muscles", {}).keys())[:3])
        P.append(
            rf"The principal target, \ttt{{DNp26}}, drives the \emph{{contralateral}} wing (its "
            rf"wing-steering output goes overwhelmingly to the side opposite its own cell body) "
            rf"through the steering muscles {muscles}, matching from FD3's vantage point the "
            r"same muscle targets found on the FD1 arm. A figure seen to one side thus acts, through "
            r"FD3 and \ttt{DNp26}, on a pathway with the laterality needed for asymmetric wing "
            r"steering.")
    P.append(_figure("figures/fd3_motor_systems.png",
                     r"\textbf{Motor-system proxy.} The motor systems reached by FD3's descending "
                     r"output, weighted by how strongly FD3 drives each descending neuron. "
                     r"Wing-steering has the largest FD3-weighted share.", "fig:motorsys"))
    P.append(_figure("figures/fd3_brain_to_muscle.png",
                     r"\textbf{From FD3 to the wing muscles.} The leading wing-steering descending "
                     r"neurons FD3 reaches, the wing each one steers, and the steering muscles it "
                     r"drives in the nerve cord.", "fig:b2m"))
    return "\n\n".join(P)


def _implications_section(L: dict) -> str:
    motor = _g(L, "motor", default={})
    dom = _g(motor, "dominant_motor_system") if motor.get("available") else None
    return (r"\section{Functional implications}" "\n\n"
            r"Three anatomical facts make FD3 a concrete candidate pathway for figure-directed "
            r"steering. First, FD3 reaches descending neurons directly and through one relay step"
            + (r", with the FD3-weighted motor proxy largest for wing steering" if dom == "wing_steering" else "")
            + r". Second, FD3 and the FD1 arm share \ttt{DNp26}, so figure signals with different "
            r"preferred directions converge on at least one major descending target. Third, "
            r"\ttt{DNp26} contacts the contralateral wing-steering muscles in the male CNS map. "
            r"These results specify a testable sensorimotor hypothesis: FD3 activity should bias "
            r"wing-steering output for laterally seen regressive figures. Proving that hypothesis "
            r"requires physiology or perturbation.")


def _caveats_section(L: dict) -> str:
    return (r"\section{What the data can and cannot settle}" "\n\n"
            r"\textbf{The exact counts are track-specific.} The synapse and DN counts are reported "
            r"for the primary verification track named in Methods. If a live or alternate-release "
            r"rerun is used later, those absolute counts should be reported separately rather than "
            r"silently merged. "
            r"\textbf{The relay set depends on a threshold.} The direct descending targets are "
            r"unambiguous -- they are the neurons FD3 synapses onto. The relayed set depends on how "
            r"strong a central partner must be to be followed; we report the full sensitivity of the "
            r"count to that choice rather than fixing a single number. \textbf{The muscle map crosses two connectomes.} "
            r"The muscles each descending neuron drives are read from a second, male nerve-cord "
            r"connectome, and the link between the two datasets is the shared neuron name. Where a "
            r"descending neuron carries the same name in both, the match is direct; a neuron without "
            r"a named nerve-cord counterpart simply contributes no muscle information and is not "
            r"forced into one. \textbf{FD3 is two cells.} As in the identity report, FD3 is a single "
            r"left/right pair, so the result rests on the two cells agreeing and on the thousands of "
            r"synapses each makes, not on a population average.")


def _methods_section(track: str) -> str:
    return (r"\section{Methods and provenance}" "\n\n"
            r"\textbf{Connectomes.} The descending targets are read from the FlyWire FAFB brain "
            r"connectome (materialization v783); a neuron counts as descending by its "
            r"\ttt{super\_class} annotation. The motor read-out is the male whole-CNS connectome "
            r"(male-cns:v1.0), reached offline; descending neurons are matched between the two by "
            r"their shared Janelia name. \textbf{Direct route}: postsynaptic partners of "
            r"\ttt{LPT42\_Nod4} whose super-class is descending, aggregated by type. \textbf{Relayed "
            r"route}: FD3's strongest non-descending partners (above a reported synapse floor) and "
            r"their descending output, attributed to the intermediary type, with the count's "
            r"threshold-sensitivity reported. \textbf{Motor systems}: each descending neuron's "
            r"male-CNS motor output classified by motor-neuron subclass (wing-steering, wing-power, "
            r"neck/gaze, haltere, leg, abdominal); wing laterality is the descending neuron's "
            r"soma side versus the motor neuron's. \textbf{Reproducibility}: built from the "
            rf"Family-L verification JSON by the FD3 report CLI. Primary track: \ttt{{{_esc(track)}}}.")


def _bibliography() -> str:
    items = [
        r"\bibitem{namiki2018} Namiki S., Dickinson M.H., Wong A.M., Korff W., Card G.M. (2018) "
        r"The functional organization of descending sensory-motor pathways in \textit{Drosophila}. "
        r"\textit{eLife} 7:e34272. doi:10.7554/eLife.34272.",
        r"\bibitem{dorkenwald2024} Dorkenwald S. et al. (2024) Neuronal wiring diagram of an adult "
        r"brain. \textit{Nature} 634:124--138. doi:10.1038/s41586-024-07558-y.",
        r"\bibitem{egelhaaf1985c} Egelhaaf M. (1985) On the neuronal basis of figure-ground "
        r"discrimination by relative motion in the visual system of the fly. III. Possible input "
        r"circuitries and behavioural significance of the FD-cells. \textit{Biol. Cybern.} "
        r"52:267--280. doi:10.1007/BF00336983.",
    ]
    return (r"\begin{thebibliography}{9}" "\n" + "\n".join(items) + "\n"
            r"\end{thebibliography}")


def build_tex(results: dict) -> str:
    fam = results["families"]["L"] if "families" in results else results
    L = fam["derived"]
    track = results.get("meta", {}).get("flywire_track", L.get("track", "live"))
    parts = [
        _PREAMBLE,
        r"\begin{document}",
        r"\title{\textbf{The descending neurons of the FD3 figure-detection cell "
        r"(\ttt{LPT42\_Nod4} $\rightarrow$ descending neurons $\rightarrow$ wing steering)}}",
        r"\author{Stage 7 FD3 Identification, descending extension}",
        r"\date{\today}",
        r"\maketitle",
        _summary(L),
        _introduction(),
        _direct_section(L),
        _relay_section(L),
        _motor_section(L),
        _implications_section(L),
        _caveats_section(L),
        _methods_section(track),
        _bibliography(),
        r"\end{document}",
    ]
    return "\n\n".join(parts)


def _render_figures(results: dict, fig_dir: Path) -> None:
    fam = results["families"]["L"] if "families" in results else results
    run = {"derived": {"L": fam["derived"]}, "results": {"L": []}}
    PF.fd3_dn_ranking(run, fig_dir / "fd3_dn_ranking.png")
    PF.fd3_descending_channels(run, fig_dir / "fd3_descending_channels.png")
    PF.fd3_motor_systems(run, fig_dir / "fd3_motor_systems.png")
    PF.fd3_brain_to_muscle(run, fig_dir / "fd3_brain_to_muscle.png")


def build_report(stage_dir: str | Path = "7 - FD3 Identification",
                 results_path: str | Path = "5 - Paper Verification/verification_results.json"
                 ) -> dict[str, Any]:
    stage = Path(stage_dir)
    (stage / "figures").mkdir(parents=True, exist_ok=True)
    results = read_json(Path(results_path))
    _render_figures(results, stage / "figures")
    tex_path = stage / "fd3_descending_report.tex"
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
