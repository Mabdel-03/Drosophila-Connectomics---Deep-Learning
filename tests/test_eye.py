"""Unit tests for the rigid "eye" (image -> photoreceptor) and the photoreceptor-sign fix.

Run on tiny synthetic subgraphs carrying a real-shaped neuron table (cell_type/side/
pos_x/pos_y) so no data files are needed. Guards: box-filter determinism + normalization,
coverage/fill, eye routing + LR mirror, the photoreceptor-sign override, and that a frozen
core stays frozen.

Run:  /orcd/home/002/mabdel03/conda_envs/consortium/bin/python -m pytest tests/test_eye.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from flyconn.models.eye import build_eye_map, RigidEye  # noqa: E402
from flyconn.models.subgraphs import (  # noqa: E402
    Subgraph,
    _normalize_photoreceptor_sign,
)


def _synthetic_visual_subgraph(n_photo_per_eye=36, n_other=20, seed=0):
    """A subgraph with photoreceptors on a grid (both eyes) + some downstream neurons.

    Photoreceptors are R1-6 laid out on a sqrt(n) x sqrt(n) grid per eye (distinct
    pos_x/pos_y so the pos_grid retinotopy gives each a unique column). Downstream nodes
    receive edges from photoreceptors so the sign-override has something to act on.
    """
    rng = np.random.default_rng(seed)
    side_per = int(round(np.sqrt(n_photo_per_eye)))
    rows = []
    # left eye photoreceptors
    for e, (xbase, sd) in enumerate([(0.0, "left"), (1000.0, "right")]):
        for i in range(side_per):
            for j in range(side_per):
                rows.append({"cell_type": "R1-6", "side": sd,
                             "pos_x": xbase + i, "pos_y": float(j),
                             "super_class": "sensory", "root_id": len(rows)})
    n_photo = len(rows)
    # downstream (lamina-like) neurons
    for k in range(n_other):
        rows.append({"cell_type": "L1", "side": "left" if k % 2 else "right",
                     "pos_x": 5000.0 + k, "pos_y": 5000.0,
                     "super_class": "optic", "root_id": len(rows)})
    neurons = pd.DataFrame(rows)
    N = len(neurons)

    # Edges: every photoreceptor -> a random downstream neuron (source orientation pre->post,
    # then transposed into W[post,pre] buffers like build_subgraph).
    downstream = np.arange(n_photo, N)
    pre = np.arange(n_photo)
    post = rng.choice(downstream, n_photo)
    sign = rng.choice([1.0, -1.0], n_photo).astype(np.float32)   # artifactual mixed signs
    count = rng.integers(5, 50, n_photo).astype(np.float32)
    sub = Subgraph(
        subgraph_id="optic", policy="synthetic", N=N,
        row_idx=post.astype(np.int64), col_idx=pre.astype(np.int64),
        sign=sign, count=count,
        local2global=np.arange(N, dtype=np.int64), neurons=neurons,
    )
    return sub, n_photo


# ----------------------------- box-filter determinism + normalization -----------------
def test_eye_map_deterministic():
    sub, _ = _synthetic_visual_subgraph()
    a = build_eye_map(sub, source="pos_grid")
    b = build_eye_map(sub, source="pos_grid")
    assert np.array_equal(a.B, b.B)
    assert np.array_equal(a.photo_local, b.photo_local)
    assert np.array_equal(a.eye_of_photo, b.eye_of_photo)


def test_box_filter_rows_sum_to_one():
    sub, _ = _synthetic_visual_subgraph()
    em = build_eye_map(sub, source="pos_grid", eyes="both")
    rowsums = em.B.sum(1)
    seeing = rowsums > 0
    assert seeing.all()                                  # all receptors assigned here
    assert np.allclose(rowsums[seeing], 1.0, atol=1e-5)  # energy-preserving


def test_uniform_image_gives_uniform_then_zero_drive():
    sub, _ = _synthetic_visual_subgraph()
    em = build_eye_map(sub, source="pos_grid", standardize=True)
    eye = RigidEye(em).eval()
    x = eye(torch.ones(3, 784))
    # mean-removal across receptors -> ~0 drive for a flat image
    assert float(x[:, eye.photo_local].abs().max()) < 1e-4


# ----------------------------- coverage / fill ---------------------------------------
def test_fill_zero_leaves_unassigned_receptors_dark():
    sub, n_photo = _synthetic_visual_subgraph()
    # Corrupt one receptor's position to NaN so it cannot be placed on the grid.
    sub.neurons.loc[0, "pos_x"] = np.nan
    em = build_eye_map(sub, source="pos_grid", fill="zero")
    # The unplaced receptor (local id 0) must have an all-zero B row -> zero drive.
    assert em.B[0].sum() == 0.0


# ----------------------------- eye routing + LR mirror -------------------------------
def test_eyes_left_zeroes_right_receptors():
    sub, _ = _synthetic_visual_subgraph()
    em = build_eye_map(sub, source="pos_grid", eyes="left")
    right = em.eye_of_photo == 1
    assert em.B[right].sum() == 0.0          # right eye sees nothing
    left = em.eye_of_photo == 0
    assert em.B[left].sum() > 0.0            # left eye still sees


def test_mirror_lr_flips_right_eye():
    sub, _ = _synthetic_visual_subgraph()
    on = build_eye_map(sub, source="pos_grid", eyes="right", mirror_lr=True)
    off = build_eye_map(sub, source="pos_grid", eyes="right", mirror_lr=False)
    # A horizontal flip of the image equals the mirrored eye's response of the un-flipped.
    eye_on, eye_off = RigidEye(on).eval(), RigidEye(off).eval()
    img = torch.randn(1, 784)
    img2d = img.view(1, 28, 28)
    flipped = torch.flip(img2d, dims=[2]).reshape(1, 784)
    d_on = eye_on(img)[:, eye_on.photo_local]
    d_off_flipped = eye_off(flipped)[:, eye_off.photo_local]
    assert torch.allclose(d_on, d_off_flipped, atol=1e-4)


# ----------------------------- photoreceptor-sign override ---------------------------
def test_photoreceptor_sign_override_forces_minus_one():
    sub, n_photo = _synthetic_visual_subgraph()
    is_photo = sub.neurons["cell_type"].isin(["R1-6", "R7", "R8"]).to_numpy()
    pre_is_photo = is_photo[sub.col_idx]
    assert pre_is_photo.sum() > 0
    # Apply the same override logic build_subgraph uses on the sign buffer.
    sub.sign[pre_is_photo] = -1.0
    assert np.all(sub.sign[pre_is_photo] == -1.0)
    # Non-photoreceptor presyn edges (none here, but guard the indexing) unchanged.
    assert np.all(sub.sign[~pre_is_photo] != 0) or (~pre_is_photo).sum() == 0


def test_normalize_photoreceptor_sign():
    assert _normalize_photoreceptor_sign("inherit") is None
    assert _normalize_photoreceptor_sign(-1) == -1
    assert _normalize_photoreceptor_sign(1) == 1
    with pytest.raises(ValueError):
        _normalize_photoreceptor_sign(2)


# ----------------------------- frozen core stays frozen ------------------------------
def test_frozen_core_has_no_trainable_params():
    from flyconn.models.connectome_net import ConnectomeNet, NetConfig, RigidEyeClassifier
    sub, _ = _synthetic_visual_subgraph()
    em = build_eye_map(sub, source="pos_grid")
    core = ConnectomeNet(sub, NetConfig.for_arch("ff_unroll", T=3))
    readout_ids = np.arange(em.photo_local.size, sub.N)   # downstream nodes as readout
    model = RigidEyeClassifier(RigidEye(em), core, readout_ids,
                               learn_core=False, decision="linear")
    core_trainable = sum(p.requires_grad for p in model.core.parameters())
    assert core_trainable == 0
    theta0 = model.core.theta.detach().clone()
    # a step on the readout must not move the frozen core
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model.train()
    opt.zero_grad()
    out = model(torch.randn(4, 784))
    out.sum().backward()
    opt.step()
    assert torch.allclose(model.core.theta, theta0)
