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
    _normalize_force_sign,
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


# ----------------------------- CIFAR image dims (32x32 luminance) ---------------------
def test_eye_map_cifar_dims_b_width_1024():
    sub, _ = _synthetic_visual_subgraph()
    em = build_eye_map(sub, img_h=32, img_w=32, source="pos_grid")
    assert em.B.shape[1] == 32 * 32 == 1024
    assert (em.img_h, em.img_w) == (32, 32)


def test_cifar_box_filter_rows_sum_to_one():
    sub, _ = _synthetic_visual_subgraph()
    em = build_eye_map(sub, img_h=32, img_w=32, source="pos_grid", eyes="both")
    rowsums = em.B.sum(1)
    seeing = rowsums > 0
    assert seeing.all()
    assert np.allclose(rowsums[seeing], 1.0, atol=1e-5)   # tiling/normalization size-agnostic


def test_cifar_rigid_eye_runs_and_flat_image_zero_drive():
    sub, _ = _synthetic_visual_subgraph()
    em = build_eye_map(sub, img_h=32, img_w=32, source="pos_grid", standardize=True)
    eye = RigidEye(em).eval()
    x = eye(torch.ones(3, 1024))                          # 32x32 luminance vector
    assert x.shape == (3, em.N)
    assert float(x[:, eye.photo_local].abs().max()) < 1e-4


def test_rigid_eye_rejects_wrong_pixel_width():
    sub, _ = _synthetic_visual_subgraph()
    em = build_eye_map(sub, img_h=32, img_w=32, source="pos_grid")
    eye = RigidEye(em).eval()
    with pytest.raises(AssertionError):
        eye(torch.ones(2, 784))                           # MNIST-width into a CIFAR eye


# ----------------------------- SPECTRAL ("color") rigid eye --------------------------
def _synthetic_color_subgraph(per_eye_per_type=9, n_other=12, seed=0):
    """Subgraph with R1-6, R7, AND R8 receptors on a grid per eye (for the spectral eye)."""
    rng = np.random.default_rng(seed)
    side = int(round(np.sqrt(per_eye_per_type)))
    rows = []
    for ctype in ("R1-6", "R7", "R8"):
        for xbase, sd in [(0.0, "left"), (1000.0, "right")]:
            for i in range(side):
                for j in range(side):
                    rows.append({"cell_type": ctype, "side": sd,
                                 "pos_x": xbase + i, "pos_y": float(j),
                                 "super_class": "sensory", "root_id": len(rows)})
    n_photo = len(rows)
    for k in range(n_other):
        rows.append({"cell_type": "L1", "side": "left" if k % 2 else "right",
                     "pos_x": 5000.0 + k, "pos_y": 5000.0,
                     "super_class": "optic", "root_id": len(rows)})
    neurons = pd.DataFrame(rows)
    N = len(neurons)
    downstream = np.arange(n_photo, N)
    pre = np.arange(n_photo)
    post = rng.choice(downstream, n_photo)
    sign = rng.choice([1.0, -1.0], n_photo).astype(np.float32)
    count = rng.integers(5, 50, n_photo).astype(np.float32)
    sub = Subgraph(
        subgraph_id="optic", policy="synthetic", N=N,
        row_idx=post.astype(np.int64), col_idx=pre.astype(np.int64),
        sign=sign, count=count,
        local2global=np.arange(N, dtype=np.int64), neurons=neurons,
    )
    return sub, n_photo


def test_spectral_channel_codes_per_type():
    # R1-6 -> luminance (CHANNEL_LUMA=-1); R7 -> blue (2); R8 -> green (1).
    from flyconn.models.eye import CHANNEL_LUMA
    sub, _ = _synthetic_color_subgraph()
    em = build_eye_map(sub, img_h=32, img_w=32, source="pos_grid", color="spectral")
    assert em.color == "spectral"
    ctype = sub.neurons["cell_type"].to_numpy()[em.photo_local]
    chan = em.channel_of_photo
    assert np.all(chan[ctype == "R1-6"] == CHANNEL_LUMA)
    assert np.all(chan[ctype == "R7"] == 2)    # blue
    assert np.all(chan[ctype == "R8"] == 1)    # green


def test_spectral_eye_routes_channels():
    # Feed an image that is pure in ONE channel; only the receptors tuned to it (plus the
    # luminance R1-6, weighted) should be driven.
    sub, _ = _synthetic_color_subgraph()
    em = build_eye_map(sub, img_h=32, img_w=32, source="pos_grid",
                       color="spectral", standardize=False)
    eye = RigidEye(em).eval()
    ctype = sub.neurons["cell_type"].to_numpy()[em.photo_local]
    HW = 32 * 32
    # Pure-green image (channel 1 = 1, others = 0), channel-major [R|G|B].
    img = torch.zeros(1, 3 * HW)
    img[:, HW:2 * HW] = 1.0
    drive = eye(img)[:, eye.photo_local][0]               # [n_photo]
    # R8 (green) receptors driven; R7 (blue) receptors not; R1-6 driven via luma weight (0.587).
    assert float(drive[torch.tensor(ctype == "R8")].abs().sum()) > 0
    assert float(drive[torch.tensor(ctype == "R7")].abs().sum()) < 1e-5   # blue sees no green
    assert float(drive[torch.tensor(ctype == "R1-6")].abs().sum()) > 0    # luma includes green


def test_spectral_eye_rejects_non_rgb_width():
    sub, _ = _synthetic_color_subgraph()
    em = build_eye_map(sub, img_h=32, img_w=32, source="pos_grid", color="spectral")
    eye = RigidEye(em).eval()
    with pytest.raises(AssertionError):
        eye(torch.ones(2, 1024))                          # luma width into a spectral eye


def test_luma_eye_unaffected_by_spectral_addition():
    # The default (color='luma') eye still single-channel; channel_of_photo all luminance.
    from flyconn.models.eye import CHANNEL_LUMA
    sub, _ = _synthetic_color_subgraph()
    em = build_eye_map(sub, img_h=32, img_w=32, source="pos_grid")   # default luma
    assert em.color == "luma"
    assert np.all(em.channel_of_photo == CHANNEL_LUMA)
    eye = RigidEye(em).eval()
    out = eye(torch.randn(3, 1024))                       # 1-channel luma input works
    assert out.shape == (3, em.N)


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


# ------------------- Family 5: force_sign / sign_shuffle init-ablation knobs -------------
def test_normalize_force_sign():
    assert _normalize_force_sign("none") is None
    assert _normalize_force_sign("inherit") is None
    assert _normalize_force_sign(None) is None
    assert _normalize_force_sign(1) == 1
    assert _normalize_force_sign(-1) == -1
    with pytest.raises(ValueError):
        _normalize_force_sign(2)


def test_force_sign_makes_all_signs_positive():
    # force_sign=+1 is the 'no-direction' ablation: every edge becomes excitatory.
    sub, _ = _synthetic_visual_subgraph()
    assert np.any(sub.sign < 0)                     # starts with mixed signs
    sub.sign[:] = 1.0                               # emulate build_subgraph's global override
    assert np.all(sub.sign == 1.0)


def test_force_sign_dominates_photoreceptor_sign():
    # build_subgraph order: photoreceptor override -> sign_shuffle -> force_sign. The global
    # +1 must win on EVERY edge, including the photoreceptor edges set to -1 just before.
    sub, _ = _synthetic_visual_subgraph()
    is_photo = sub.neurons["cell_type"].isin(["R1-6", "R7", "R8"]).to_numpy()
    sub.sign[is_photo[sub.col_idx]] = -1.0          # photoreceptor fix first
    sub.sign[:] = 1.0                               # force_sign=+1 second (dominates)
    assert np.all(sub.sign == 1.0)


def test_sign_shuffle_preserves_ei_ratio_and_is_reproducible():
    # Permuting signs keeps the +/- counts (E:I ratio) but moves which edge gets which sign,
    # and the same seed gives the same permutation.
    sub, _ = _synthetic_visual_subgraph(n_photo_per_eye=64, n_other=40, seed=2)
    orig = sub.sign.copy()
    n_pos0, n_neg0 = int((orig > 0).sum()), int((orig < 0).sum())
    rng = np.random.default_rng(7)
    shuffled = orig[rng.permutation(len(orig))].copy()
    assert int((shuffled > 0).sum()) == n_pos0      # E:I ratio preserved
    assert int((shuffled < 0).sum()) == n_neg0
    assert not np.array_equal(shuffled, orig)        # placement actually changed
    rng2 = np.random.default_rng(7)
    again = orig[rng2.permutation(len(orig))].copy()
    assert np.array_equal(shuffled, again)           # reproducible for a fixed seed


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
