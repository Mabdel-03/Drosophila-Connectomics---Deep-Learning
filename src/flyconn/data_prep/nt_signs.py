"""Neurotransmitter -> synaptic sign policies.

Sign is a per-PRESYNAPTIC-NEURON property (Dale's principle: a neuron releases one
transmitter), looked up from the neuron table's ``nt_canonical`` column. Policies are
declared in configs/data_v783.yaml so the choice is explicit and reproducible.

Key fly-specific fact baked in: **glutamate is frequently INHIBITORY in Drosophila**
(via GluClalpha channels), so the default ``flyvis_standard`` policy sets Glut = -1,
unlike the mammalian convention.

A sign of 0 means "exclude from the signed matrix" (e.g. modulatory monoamines);
those synapses still appear in the unsigned raw-count matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def sign_vector_for_nodes(
    cfg: Config, neurons: pd.DataFrame, policy: str
) -> np.ndarray:
    """Return an int8 array sign[idx] for every node, under the named policy.

    neurons must be the canonical table (row i == node idx i) with an
    ``nt_canonical`` column. Neurons with unknown/None NT get sign 0.
    """
    if policy not in cfg.nt_policies:
        raise KeyError(
            f"Unknown nt policy {policy!r}; available: {sorted(cfg.nt_policies)}"
        )
    mapping = cfg.nt_policies[policy]  # e.g. {'acetylcholine':1,'gaba':-1,...}

    if "nt_canonical" not in neurons.columns:
        raise KeyError("neurons table missing 'nt_canonical' column")
    # Defensive: ensure rows are in idx order.
    nt = neurons.sort_values("idx")["nt_canonical"].to_numpy()
    signs = np.zeros(len(nt), dtype=np.int8)
    for canonical, s in mapping.items():
        signs[np.array([str(x).lower() == canonical for x in nt])] = np.int8(s)
    return signs


def available_policies(cfg: Config) -> list[str]:
    return list(cfg.nt_policies.keys())
