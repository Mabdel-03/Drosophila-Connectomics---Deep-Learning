"""Two weight-initialization modes for ConnectomeNet edge magnitudes.

Both keep the fixed edge set + sign + connectivity mask from the data; they differ only
in the per-edge MAGNITUDE they assign before the spectral-radius rescale.

  A "from_data" : magnitudes proportional to the real synapse counts -> the network at
                  init IS the (scaled) real wiring. Tests whether the raw connectome
                  already computes something useful.
  B "random"    : magnitudes drawn from a LOG-NORMAL fit to the empirical counts (counts
                  are heavy-tailed, so log-normal not Normal). Same edge set/sign/mask,
                  random amplitudes with matched distribution. The essential control:
                  shows the task isn't solved just by reading off synapse counts.

Magnitude target ``t[e] >= 0`` is converted to theta via softplus_inv inside the model.
``alpha0`` (~0.01) is the flyvis-style unit scaling so total drive is ~O(1) at init; the
subsequent spectral-radius rescale (model.rescale_to_radius) sets the final scale, so
alpha0 mainly sets the relative spread.
"""

from __future__ import annotations

import numpy as np
import torch

from .connectome_net import ConnectomeNet
from .subgraphs import Subgraph

ALPHA0 = 0.01


def _from_data_magnitude(count: np.ndarray, alpha0: float = ALPHA0) -> np.ndarray:
    c = np.asarray(count, dtype=np.float64)
    mean_c = c.mean() if c.size and c.mean() > 0 else 1.0
    return (alpha0 * c / mean_c).astype(np.float32)


def _lognormal_magnitude(count: np.ndarray, alpha0: float = ALPHA0,
                         seed: int = 0) -> np.ndarray:
    c = np.asarray(count, dtype=np.float64)
    pos = c[c > 0]
    if pos.size == 0:
        return np.full(len(c), alpha0, dtype=np.float32)
    logc = np.log(pos)
    mu, sigma = logc.mean(), logc.std() + 1e-8
    rng = np.random.default_rng(seed)
    t = np.exp(mu + sigma * rng.standard_normal(len(c)))   # log-normal samples
    t = alpha0 * t / t.mean()
    return t.astype(np.float32)


def init_magnitudes(sub: Subgraph, mode: str, *, alpha0: float = ALPHA0,
                    seed: int = 0) -> np.ndarray:
    """Return per-edge magnitude target t[e] >= 0 for the given init mode."""
    if mode in ("A", "from_data"):
        return _from_data_magnitude(sub.count, alpha0=alpha0)
    if mode in ("B", "random"):
        return _lognormal_magnitude(sub.count, alpha0=alpha0, seed=seed)
    raise ValueError(f"unknown init mode {mode!r}; use 'from_data'/'A' or 'random'/'B'")


def apply_init(model: ConnectomeNet, sub: Subgraph, mode: str, *,
               alpha0: float = ALPHA0, seed: int = 0, rescale: bool = True) -> dict:
    """Set model.theta from the chosen init mode, then spectral-radius rescale.

    Returns a small report dict (achieved spectral radius, magnitude stats).
    """
    mag = init_magnitudes(sub, mode, alpha0=alpha0, seed=seed)
    model.set_theta_from_magnitude(torch.from_numpy(mag))
    report = {
        "init_mode": mode,
        "alpha0": alpha0,
        "mag_mean": float(mag.mean()),
        "mag_std": float(mag.std()),
        "rho_before": model.estimate_spectral_radius(),
    }
    if rescale:
        report["rho_after"] = model.rescale_to_radius()
    return report
