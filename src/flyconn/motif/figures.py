"""Diagnostic figures for the VCH-T4/T5 verification (headless / Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import vch_config as C  # noqa: E402


def _grouped_bar(ax, labels, report_vals, computed_vals, title, ylabel):
    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w / 2, report_vals, w, label="report", color="#c44e52")
    ax.bar(x + w / 2, computed_vals, w, label="computed (raw)", color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()


def subtype_figure(report_subtypes: dict, computed_table, kind: str, out: Path) -> Path:
    """report vs computed neuron counts per subtype. report_subtypes: {sub: (neurons, syn)}."""
    labels = list(report_subtypes.keys())
    rep = [report_subtypes[s][0] for s in labels]
    comp = [int(computed_table["neurons"].get(s, 0)) for s in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _grouped_bar(ax, labels, rep, comp, f"T4/T5 {kind} subtypes (neurons)", "neurons")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def hemisphere_figure(ins_x, outs_x, out: Path) -> Path:
    """Histogram of VCH input vs output synapse medio-lateral (x) positions."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = 60
    ax.hist(ins_x, bins=bins, alpha=0.6, label="input synapses (on VCH dendrite)",
            color="#4c72b0", density=True)
    ax.hist(outs_x, bins=bins, alpha=0.6, label="output synapses (on VCH axon)",
            color="#dd8452", density=True)
    for v, lab, c in [(np.median(ins_x), "in median", "#4c72b0"),
                      (np.median(outs_x), "out median", "#dd8452")]:
        ax.axvline(v, color=c, ls="--", lw=1)
    ax.set_xlabel("synapse x position (voxels)")
    ax.set_ylabel("density")
    ax.set_title("VCH input vs output synapse positions (laterality test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def top20_figure(top20_df, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 7))
    labels = [f"{r.cell_type} ({int(r.root_id) % 100000})" for r in top20_df.itertuples()]
    ax.barh(range(len(top20_df)), top20_df["syn"], color="#55a868")
    ax.set_yticks(range(len(top20_df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("synapses from 918 reciprocal T4/T5")
    ax.set_title("Top-20 downstream targets (computed, raw track)")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def tracks_figure(report: dict, primary: dict, secondary: dict, out: Path) -> Path:
    """Paired bars: report vs raw vs proofread for the four headline counts."""
    keys = ["total_input_syn", "total_output_syn", "upstream_partners", "downstream_partners"]
    labels = ["input syn", "output syn", "upstream", "downstream"]
    x = np.arange(len(keys))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w, [report[k] for k in keys], w, label="report (live API)", color="#c44e52")
    ax.bar(x, [primary[k] for k in keys], w, label="raw synapse table", color="#4c72b0")
    ax.bar(x + w, [secondary[k] for k in keys], w, label="proofread edges_full", color="#8172b3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("count")
    ax.set_title("VCH headline counts across data tracks")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out
