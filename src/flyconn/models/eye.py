"""The rigid (non-learned) "eye": render an image onto photoreceptors retinotopically.

This REPLACES the learned ``Linear(784 -> n_photoreceptors)`` encoder of the original
ConnectomeClassifier. The image enters the brain the way the fly's eye delivers it: each
photoreceptor sits at a retinal-lattice position (from flyconn.data_prep.retinotopy) and
receives a fixed luminance read from the image at that position (flyvis BoxEye-style area
sampling). Zero learned parameters; fully deterministic.

Mechanics, precomputed once per subgraph into a single sparse matrix ``B[n_photo, 784]``
so the forward is one matmul:

    drive[B, n_photo] = drive_gain * standardize( pixels[B, 784] @ B.T )
    x[B, N] = scatter drive into the photoreceptor local ids; everything else 0.

``B[r, :]`` is the (normalized) box-filter footprint of the image region under receptor r's
retinal column, with eye routing (``eyes``), contralateral mirroring (``mirror_lr``) and the
unassigned-receptor fill policy (``fill``) folded in. Receptors of all photoreceptor types
(R1-6/R7/R8) get the same column luminance (grayscale MNIST has one channel; ``channels``
can restrict to R1-6 as an ablation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data_prep.retinotopy import PHOTORECEPTOR_TYPES, load_retinotopy
from .io_inject import input_local_ids
from .subgraphs import Subgraph

IMG_H = IMG_W = 28


@dataclass
class EyeMap:
    """Frozen, deterministic image->photoreceptor map for one subgraph."""

    subgraph_id: str
    N: int                       # subgraph node count
    photo_local: np.ndarray      # int64[n_photo] local ids of photoreceptor input nodes
    B: np.ndarray                # float32[n_photo, 784] box-filter footprints (rows ~sum 1)
    eye_of_photo: np.ndarray     # int8[n_photo] 0=left 1=right
    n_assigned: int              # receptors with a retinal column (rest have all-zero B rows)
    drive_gain: float
    standardize: bool
    source: str

    @property
    def n_photo(self) -> int:
        return len(self.photo_local)


def _column_centers(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Map integer hex (p, q) per eye to normalized image coords in [0, 1]^2.

    Uses the lattice's own min/max so the eye's field fills the image. Returns [n, 2]
    (u = horizontal, v = vertical). NaN rows (unassigned) stay NaN.
    """
    uv = np.stack([p.astype(np.float64), q.astype(np.float64)], axis=1)
    out = np.full_like(uv, np.nan)
    finite = np.isfinite(uv).all(1)
    if finite.sum() == 0:
        return out
    sub = uv[finite]
    lo, hi = sub.min(0), sub.max(0)
    span = hi - lo
    span[span == 0] = 1.0
    out[finite] = (sub - lo) / span
    return out


def _box_filter_row(u: float, v: float, radius_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Pixel indices + area-overlap weights for a box of half-width radius_px at (u,v).

    (u, v) are in [0, 1] image coords. Returns (flat_pixel_idx, weights) with weights
    normalized to sum 1. A simple separable box (area sampling) — anti-aliases when the
    lattice is coarser than the image and reduces to nearest-pixel when it is finer.
    """
    cx = u * (IMG_W - 1)
    cy = v * (IMG_H - 1)
    x0 = int(np.floor(cx - radius_px))
    x1 = int(np.ceil(cx + radius_px))
    y0 = int(np.floor(cy - radius_px))
    y1 = int(np.ceil(cy + radius_px))
    xs = np.clip(np.arange(x0, x1 + 1), 0, IMG_W - 1)
    ys = np.clip(np.arange(y0, y1 + 1), 0, IMG_H - 1)
    xs = np.unique(xs)
    ys = np.unique(ys)
    gx, gy = np.meshgrid(xs, ys)
    # Triangular (tent) weight by distance from center -> smooth footprint, sums normalized.
    wx = np.clip(1.0 - np.abs(gx - cx) / max(radius_px, 1.0), 0.0, None)
    wy = np.clip(1.0 - np.abs(gy - cy) / max(radius_px, 1.0), 0.0, None)
    w = (wx * wy).ravel()
    idx = (gy.ravel() * IMG_W + gx.ravel())
    s = w.sum()
    if s <= 0:
        # Degenerate: fall back to the single nearest pixel.
        nx, ny = int(round(cx)), int(round(cy))
        return np.array([ny * IMG_W + nx]), np.array([1.0])
    return idx, w / s


def build_eye_map(
    sub: Subgraph,
    *,
    source: str = "auto",
    drive_gain: float = 0.5,
    eyes: str = "both",
    channels: str = "all",
    fill: str = "zero",
    mirror_lr: bool = True,
    standardize: bool = True,
) -> EyeMap:
    """Construct the deterministic image->photoreceptor map for ``sub``.

    eyes: 'both' (image to both eyes), 'left'/'right' (drive one eye, zero the other).
    channels: 'all' (R1-6/R7/R8 all driven) or 'R1-6' (only R1-6; ablation).
    fill: 'zero' (unassigned receptors get no drive) or 'nearest' (assign to nearest
          assigned column by image coords).
    mirror_lr: LR-flip the image for the RIGHT eye (eyes are mirror-symmetric retinotopically).
    """
    photo_local = input_local_ids(sub)               # local ids of R1-6/R7/R8
    n_photo = photo_local.size
    neurons = sub.neurons
    ret = load_retinotopy(neurons, source=source)    # aligned to neurons index

    # Per-photoreceptor retinal coords + eye.
    p = ret["p"].to_numpy()[photo_local]
    q = ret["q"].to_numpy()[photo_local]
    eye = ret["eye"].to_numpy().astype("float64")[photo_local]   # 0 left, 1 right, NaN none
    ctype = neurons["cell_type"].to_numpy()[photo_local]

    # Channel ablation: zero out non-R1-6 receptors' drive if requested.
    channel_ok = np.ones(n_photo, dtype=bool)
    if channels == "R1-6":
        channel_ok = ctype == "R1-6"

    # Normalize each eye's lattice to image coords SEPARATELY so each eye fills the frame.
    uv = np.full((n_photo, 2), np.nan)
    for eye_code in (0, 1):
        sel = eye == eye_code
        if sel.sum() == 0:
            continue
        uv[sel] = _column_centers(p[sel], q[sel])
    # Mirror the right eye horizontally so a frontal stimulus is retinotopically aligned.
    if mirror_lr:
        right = eye == 1
        uv[right, 0] = 1.0 - uv[right, 0]

    # Eye routing: which receptors actually see the image.
    sees = channel_ok & np.isfinite(uv).all(1)
    if eyes == "left":
        sees &= eye == 0
    elif eyes == "right":
        sees &= eye == 1
    # 'both': all assigned receptors see it.

    # Fill policy for receptors with no assigned column.
    if fill == "nearest":
        sees = _fill_nearest(uv, eye, sees, channel_ok)

    # Receptive radius ~ half the inter-column pitch so neighboring boxes tile.
    n_assigned = int(sees.sum())
    radius_px = _receptive_radius(uv[sees]) if n_assigned else 1.0

    # Build the sparse B as a dense [n_photo, 784] (n_photo ~<=11k, 784 cols -> <=8.7M
    # floats per subgraph; trivial). Rows for non-seeing receptors stay all-zero.
    B = np.zeros((n_photo, IMG_H * IMG_W), dtype=np.float32)
    for r in np.flatnonzero(sees):
        idx, w = _box_filter_row(uv[r, 0], uv[r, 1], radius_px)
        B[r, idx] = w.astype(np.float32)

    eye_of_photo = np.where(np.isfinite(eye), eye, -1).astype(np.int8)
    return EyeMap(
        subgraph_id=sub.subgraph_id, N=sub.N, photo_local=photo_local.astype(np.int64),
        B=B, eye_of_photo=eye_of_photo, n_assigned=n_assigned,
        drive_gain=float(drive_gain), standardize=bool(standardize), source=source,
    )


def _receptive_radius(uv_seen: np.ndarray) -> float:
    """Half the median nearest-neighbor spacing (in pixels) of the seeing receptors."""
    if len(uv_seen) < 2:
        return 1.0
    px = uv_seen * np.array([IMG_W - 1, IMG_H - 1])
    # Approximate NN spacing via a coarse grid density (avoid O(n^2) on ~5k points).
    area = (np.ptp(px[:, 0]) + 1e-6) * (np.ptp(px[:, 1]) + 1e-6)
    spacing = np.sqrt(area / len(px))
    return float(max(0.75, spacing / 2.0))


def _fill_nearest(uv, eye, sees, channel_ok):
    """Assign each unseen-but-channel-ok receptor the coords of its nearest seen one."""
    sees = sees.copy()
    for eye_code in (0, 1):
        donor = sees & (eye == eye_code)
        need = channel_ok & ~sees & (eye == eye_code) & np.isfinite(uv).all(1)
        if donor.sum() == 0 or need.sum() == 0:
            continue
        d_uv = uv[donor]
        for r in np.flatnonzero(need):
            j = np.argmin(((d_uv - uv[r]) ** 2).sum(1))
            uv[r] = d_uv[j]
        sees[need] = True
    return sees


class RigidEye(torch.nn.Module):
    """Parameter-free module: pixels[B,784] -> external current x[B, N]."""

    def __init__(self, eye_map: EyeMap):
        super().__init__()
        self.N = eye_map.N
        self.drive_gain = eye_map.drive_gain
        self.standardize = eye_map.standardize
        self.register_buffer("photo_local", torch.as_tensor(eye_map.photo_local, dtype=torch.long))
        self.register_buffer("B", torch.as_tensor(eye_map.B, dtype=torch.float32))
        # Standardization stats over the seeing receptors are computed lazily on first
        # batch's column luminance? No — standardize per-batch per-eye is data-dependent
        # and would leak. We standardize each receptor's drive by the FIXED row energy of
        # B (||B[r]||_1 == 1 for seers), so the only normalization is per-eye mean removal
        # done at inject time over the receptor population (image-content dependent but
        # symmetric across classes; documented choice).

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        # pixels: [B, 784] (already MNIST-normalized upstream).
        drive = pixels @ self.B.t()                       # [B, n_photo]
        if self.standardize:
            # Remove per-image mean over receptors (kills the global-brightness DC that
            # would otherwise dominate a fixed-drive tanh core), keep per-receptor scale.
            drive = drive - drive.mean(dim=1, keepdim=True)
        drive = self.drive_gain * drive
        x = pixels.new_zeros(pixels.shape[0], self.N)
        x[:, self.photo_local] = drive
        return x
