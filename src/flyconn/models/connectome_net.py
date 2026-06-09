"""ConnectomeNet — the connectome-constrained trainable core.

Connectivity (which edges exist) and synaptic SIGN are FIXED from the data; only the
edge MAGNITUDES are learned (the user's "purest" choice — no per-node bias/leak/gain).

Weight parametrization (Dale's law hard-enforced):
    W_value[e] = sign[e] * softplus(theta[e])          # sign fixed buffer; magnitude >= 0
Only ``theta`` (length E) is a Parameter. ``sign``, ``row_idx``, ``col_idx`` are buffers,
so absent edges have structurally zero gradient (the connectivity mask is implicit) and
gradient descent can only rescale a synapse's magnitude, never flip its sign.

Dynamics (leaky-integrator unroll; FF and RNN are the SAME forward, different config):
    h_0 = 0
    h_{t+1} = (1 - alpha) * h_t + alpha * phi( scatter(W, h_t) + x_t )
where scatter does the sparse W @ h via gather-presyn + index_add-to-postsyn (this avoids
the torch.sparse_csr_tensor autograd-detach trap, pytorch #98929).
    * ff_unroll : alpha ~= 1 (no state carry), x injected only at t=0, read at step T
    * rnn       : alpha < 1 (leaky memory), x injected every step, full BPTT

Stability (no learnable node params, so this matters): edge magnitudes are rescaled at
init so the assembled sparse W has spectral radius ~= ``target_radius`` (estimated by
sparse power iteration — never densified). phi defaults to tanh (bounded).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .subgraphs import Subgraph


def softplus_inv(t: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus: theta such that softplus(theta) == t (t > 0)."""
    # log(expm1(t)); numerically stable for large t via the identity t + log1p(-exp(-t)).
    return torch.where(t > 20, t, torch.log(torch.expm1(t.clamp(min=1e-12))))


@dataclass
class NetConfig:
    arch: str = "ff_unroll"        # 'ff_unroll' | 'rnn'
    T: int = 10                    # unroll steps
    alpha: float = 1.0             # leak (ff_unroll ~1.0; rnn ~0.2)
    inject: str = "t0"             # 't0' (ff) | 'persistent' (rnn)
    nonlinearity: str = "tanh"     # 'tanh' | 'relu'
    target_radius: float = 0.9     # spectral radius after init rescale
    checkpoint: bool = False       # gradient-checkpoint the time loop (rnn, large T)

    @classmethod
    def for_arch(cls, arch: str, **over) -> "NetConfig":
        if arch == "ff_unroll":
            base = dict(arch="ff_unroll", alpha=1.0, inject="t0", T=10, checkpoint=False)
        elif arch == "rnn":
            base = dict(arch="rnn", alpha=0.2, inject="persistent", T=15, checkpoint=True)
        else:
            raise ValueError(f"unknown arch {arch!r}")
        base.update(over)
        return cls(**base)


class ConnectomeNet(nn.Module):
    """Recurrent connectome core with fixed mask+sign and learnable edge magnitudes."""

    def __init__(self, sub: Subgraph, cfg: NetConfig):
        super().__init__()
        self.cfg = cfg
        self.N = sub.N
        self.subgraph_id = sub.subgraph_id
        # Fixed structural buffers (NOT parameters).
        self.register_buffer("row_idx", torch.as_tensor(sub.row_idx, dtype=torch.long))
        self.register_buffer("col_idx", torch.as_tensor(sub.col_idx, dtype=torch.long))
        self.register_buffer("sign", torch.as_tensor(sub.sign, dtype=torch.float32))
        # The ONLY core parameter: per-edge magnitude pre-activation.
        self.theta = nn.Parameter(torch.zeros(len(sub.row_idx), dtype=torch.float32))
        self._phi = torch.tanh if cfg.nonlinearity == "tanh" else F.relu

    # ---- weight ----
    def edge_weight(self) -> torch.Tensor:
        """Signed effective weight per edge: sign * softplus(theta). Dale-constrained."""
        return self.sign * F.softplus(self.theta)

    def set_theta_from_magnitude(self, mag: torch.Tensor) -> None:
        """Set theta so softplus(theta) == mag (mag >= 0). Used by init modes."""
        with torch.no_grad():
            self.theta.copy_(softplus_inv(mag.to(self.theta)))

    # ---- forward ----
    def _scatter(self, w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Sparse W @ h via gather-presyn + index_add to postsyn. h:[B,N] -> [B,N]."""
        msg = w.unsqueeze(0) * h[:, self.col_idx]                  # [B, E]
        out = torch.zeros_like(h)
        out.index_add_(1, self.row_idx, msg)                      # scatter to postsyn
        return out

    def _step(self, h: torch.Tensor, x_t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        pre = self._scatter(w, h) + x_t
        a = self.cfg.alpha
        return (1.0 - a) * h + a * self._phi(pre)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Unroll T steps. x:[B,N] external current. Returns final activations [B,N].

        ff_unroll: inject x at t=0 only. rnn: inject x every step.
        """
        B = x.shape[0]
        h = x.new_zeros(B, self.N)
        w = self.edge_weight()
        persistent = self.cfg.inject == "persistent"
        zero = torch.zeros_like(x)
        for t in range(self.cfg.T):
            x_t = x if (persistent or t == 0) else zero
            if self.cfg.checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    self._step, h, x_t, w, use_reentrant=False
                )
            else:
                h = self._step(h, x_t, w)
        return h

    # ---- spectral-radius init rescale (stability; no densification) ----
    @torch.no_grad()
    def estimate_spectral_radius(self, n_iter: int = 30, seed: int = 0) -> float:
        """Power-iterate the assembled sparse W using only sparse matvecs."""
        g = torch.Generator(device=self.theta.device).manual_seed(seed)
        v = torch.randn(1, self.N, generator=g, device=self.theta.device)
        v = v / (v.norm() + 1e-12)
        w = self.edge_weight()
        lam = 0.0
        for _ in range(n_iter):
            u = self._scatter(w, v)
            nrm = u.norm()
            if nrm < 1e-20:
                return 0.0
            lam = float(nrm)
            v = u / nrm
        return lam

    @torch.no_grad()
    def rescale_to_radius(self, target: float | None = None, n_iter: int = 30) -> float:
        """Scale all magnitudes so spectral radius ~= target. Returns the achieved radius."""
        target = self.cfg.target_radius if target is None else target
        rho = self.estimate_spectral_radius(n_iter=n_iter)
        if rho <= 1e-12:
            return rho
        scale = target / rho
        # softplus(theta) -> scale * softplus(theta); reset theta accordingly.
        new_mag = (F.softplus(self.theta) * scale).clamp(min=1e-12)
        self.theta.copy_(softplus_inv(new_mag))
        return self.estimate_spectral_radius(n_iter=n_iter)


class ConnectomeClassifier(nn.Module):
    """Encoder (784 -> input nodes) + frozen-mask ConnectomeNet core + linear readout.

    Only the encoder, the core's edge magnitudes (theta), and the readout are trained.
    """

    def __init__(self, core: ConnectomeNet, input_local: np.ndarray,
                 readout_local: np.ndarray, n_pixels: int = 784, n_classes: int = 10):
        super().__init__()
        self.core = core
        self.register_buffer("input_local", torch.as_tensor(input_local, dtype=torch.long))
        self.register_buffer("readout_local", torch.as_tensor(readout_local, dtype=torch.long))
        self.encoder = nn.Linear(n_pixels, len(input_local))
        self.readout = nn.Linear(len(readout_local), n_classes)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        B = pixels.shape[0]
        cur = self.encoder(pixels)                              # [B, n_input]
        x = pixels.new_zeros(B, self.core.N)
        x[:, self.input_local] = cur                           # inject into input nodes
        h = self.core(x)                                       # [B, N]
        feats = h[:, self.readout_local]                      # [B, n_readout]
        return self.readout(feats)                            # [B, 10] logits
