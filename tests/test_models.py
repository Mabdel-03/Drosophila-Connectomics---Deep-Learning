"""Unit tests for the stage-3 connectome-constrained model core.

These run on tiny synthetic graphs (no GPU, no data files) and guard the four traps
called out in the design: autograd-detach, sign-flip, mask-fixed, and orientation.

Run:  /orcd/home/002/mabdel03/conda_envs/consortium/bin/python -m pytest tests/test_models.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flyconn.models.connectome_net import (  # noqa: E402
    ConnectomeNet, NetConfig, softplus_inv,
)
from flyconn.models.init_modes import init_magnitudes  # noqa: E402
from flyconn.models.subgraphs import subgraph_from_arrays  # noqa: E402


def _toy_chain():
    # 0 -> 1 -> 2 (excitatory chain), source orientation (pre, post)
    pre = np.array([0, 1]); post = np.array([1, 2])
    sign = np.array([1.0, 1.0]); count = np.array([10.0, 20.0])
    return subgraph_from_arrays(3, pre, post, sign, count)


def _toy_random(N=50, E=200, seed=0):
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, N, E); post = rng.integers(0, N, E)
    sign = rng.choice([1.0, -1.0], E); count = rng.integers(5, 100, E).astype(float)
    return subgraph_from_arrays(N, pre, post, sign, count, subgraph_id="rand")


# ---------------- autograd: gradient must flow to theta ----------------
def test_grad_flows_to_theta():
    sub = _toy_random()
    net = ConnectomeNet(sub, NetConfig.for_arch("ff_unroll", T=5))
    x = torch.randn(4, sub.N)
    out = net(x)
    out.sum().backward()
    assert net.theta.grad is not None
    assert float(net.theta.grad.abs().sum()) > 0  # catches CSR-detach trap (#98929)


# ---------------- Dale: sign never flips under training ----------------
def test_sign_never_flips():
    sub = _toy_random()
    net = ConnectomeNet(sub, NetConfig.for_arch("rnn", T=4))
    sign0 = net.sign.clone()
    opt = torch.optim.Adam(net.parameters(), lr=0.5)
    for _ in range(20):
        opt.zero_grad()
        net(torch.randn(8, sub.N)).pow(2).sum().backward()
        opt.step()
    w = net.edge_weight()
    # effective weight sign equals the fixed sign buffer for every edge
    assert torch.all(torch.sign(w) == sign0)
    # softplus magnitude stays strictly nonnegative
    assert torch.all(torch.nn.functional.softplus(net.theta) >= 0)


# ---------------- mask: absent edges have no parameter / zero structural grad ----------------
def test_mask_is_fixed():
    sub = _toy_random(N=50, E=200)
    net = ConnectomeNet(sub, NetConfig.for_arch("ff_unroll", T=3))
    # Only E parameters exist — there is no dense N*N weight to leak into absent edges.
    assert net.theta.numel() == sub.E == 200
    # The number of edges is fixed; forward never creates new ones.
    net(torch.randn(2, sub.N)).sum().backward()
    assert net.theta.grad.numel() == sub.E


# ---------------- orientation: signal flows pre -> post (A.T used) ----------------
def test_orientation_pre_to_post():
    sub = _toy_chain()
    net = ConnectomeNet(sub, NetConfig.for_arch("ff_unroll", T=3, alpha=1.0))
    net.set_theta_from_magnitude(torch.ones(sub.E))     # unit magnitudes
    # inject only at node 0; with T>=2 it should reach node 2 (0->1->2), not backward.
    x = torch.zeros(1, 3); x[0, 0] = 5.0
    h = net(x)
    assert abs(float(h[0, 2])) > 1e-4        # node 2 (downstream) activated
    # a reversed injection at node 2 should NOT drive node 0 (no 2->0 edge)
    x2 = torch.zeros(1, 3); x2[0, 2] = 5.0
    h2 = net(x2)
    assert abs(float(h2[0, 0])) < 1e-4


# ---------------- init fidelity ----------------
def test_init_from_data_roundtrip():
    sub = _toy_random()
    net = ConnectomeNet(sub, NetConfig.for_arch("ff_unroll"))
    mag = init_magnitudes(sub, "from_data", alpha0=0.01)
    net.set_theta_from_magnitude(torch.from_numpy(mag))
    got = torch.nn.functional.softplus(net.theta).detach().numpy()
    assert np.allclose(got, mag, atol=1e-4)   # softplus(softplus_inv(t)) == t


def test_init_random_matches_lognormal_stats():
    sub = _toy_random(N=80, E=2000, seed=1)
    mag = init_magnitudes(sub, "random", alpha0=0.01, seed=3)
    # log of sampled magnitudes should be roughly normal (finite, no NaN, positive)
    assert np.all(mag > 0) and np.all(np.isfinite(mag))
    # heavy tail preserved: max/median ratio is well above 1
    assert mag.max() / np.median(mag) > 2.0


# ---------------- Family 5: constant-magnitude ('no-magnitude') init ----------------
def test_init_constant_is_uniform():
    # constant init -> every edge magnitude identical (counts discarded), positive, finite.
    sub = _toy_random(N=60, E=300, seed=5)
    mag = init_magnitudes(sub, "constant", alpha0=0.01)
    assert mag.shape == (sub.E,)
    assert np.all(mag > 0) and np.all(np.isfinite(mag))
    assert np.allclose(mag, mag[0]) and np.allclose(mag, 0.01)   # uniform == alpha0
    # ...and unlike from_data it does NOT track the (heterogeneous) synapse counts.
    fd = init_magnitudes(sub, "from_data", alpha0=0.01)
    assert not np.allclose(fd, fd[0])


def test_constant_softplus_uniform_before_rescale():
    # softplus(theta) is uniform after a constant init (the 'no-magnitude' guarantee).
    sub = _toy_random(N=50, E=200)
    net = ConnectomeNet(sub, NetConfig.for_arch("ff_unroll"))
    net.set_theta_from_magnitude(torch.from_numpy(init_magnitudes(sub, "constant")))
    got = torch.nn.functional.softplus(net.theta)
    assert torch.allclose(got, got[0].expand_as(got), atol=1e-5)


def test_nomag_effective_weights_uniform_magnitude():
    # After constant init + spectral rescale: |W| is uniform and signs come from the data.
    sub = _toy_random(N=80, E=400, seed=6)      # mixed signs from _toy_random
    net = ConnectomeNet(sub, NetConfig.for_arch("ff_unroll", target_radius=0.9))
    net.set_theta_from_magnitude(torch.from_numpy(init_magnitudes(sub, "constant")))
    net.rescale_to_radius()
    w = net.edge_weight()
    assert torch.allclose(w.abs(), w.abs()[0].expand_as(w), atol=1e-5)   # |w| uniform
    assert torch.all(torch.sign(w) == net.sign)                          # signs from data


def test_unknown_init_mode_raises():
    sub = _toy_random()
    with pytest.raises(ValueError):
        init_magnitudes(sub, "bogus")


def test_softplus_inv_roundtrip():
    t = torch.tensor([1e-3, 0.1, 1.0, 5.0, 50.0])
    assert torch.allclose(torch.nn.functional.softplus(softplus_inv(t)), t, atol=1e-4)


# ---------------- spectral radius rescale keeps things bounded ----------------
def test_spectral_rescale_and_stability():
    sub = _toy_random(N=100, E=800, seed=2)
    net = ConnectomeNet(sub, NetConfig.for_arch("rnn", T=50, target_radius=0.9))
    net.set_theta_from_magnitude(torch.from_numpy(init_magnitudes(sub, "from_data")))
    rho = net.rescale_to_radius()
    assert rho == pytest.approx(0.9, abs=0.15)
    # 50-step unroll stays finite/bounded
    h = net(torch.randn(4, sub.N))
    assert torch.isfinite(h).all()
    assert float(h.abs().max()) < 100.0


# ================= Family 3: recmul (recurrent-multiply) dynamics =================
# These guard the new `dynamics='linear'` step rule (pure signed matmul h <- W h, the
# literal "multiply by the connectivity matrix over and over" model) and the learned-eye
# core-freeze that enables the frozen-W cells.
from flyconn.models.connectome_net import ConnectomeClassifier  # noqa: E402


def test_linear_dynamics_is_pure_matmul():
    # dynamics='linear' must apply exactly scatter(W,h)+x_t — no leak, no tanh.
    sub = _toy_random()
    net = ConnectomeNet(sub, NetConfig.for_arch("ff_unroll", T=2, dynamics="linear"))
    net.set_theta_from_magnitude(torch.full((sub.E,), 0.5))
    h = torch.randn(3, sub.N)
    x_t = torch.randn(3, sub.N)
    w = net.edge_weight()
    got = net._step(h, x_t, w)
    expected = net._scatter(w, h) + x_t            # pure matmul, no nonlinearity
    assert torch.allclose(got, expected, atol=1e-6)
    # ... and it is NOT the saturating tanh step for a large input.
    tanh_net = ConnectomeNet(sub, NetConfig.for_arch("ff_unroll", T=2, dynamics="tanh"))
    tanh_net.set_theta_from_magnitude(torch.full((sub.E,), 0.5))
    big = torch.full((1, sub.N), 5.0)
    lin = net._step(big.clone(), torch.zeros(1, sub.N), w)
    tnh = tanh_net._step(big.clone(), torch.zeros(1, sub.N), tanh_net.edge_weight())
    assert float(tnh.abs().max()) <= 1.0 + 1e-5    # tanh is bounded
    assert float(lin.abs().max()) > 1.0 + 1e-5     # linear is not


def test_linear_rms_unit_rms():
    # dynamics='linear' + state_norm='rms' renormalizes each step to unit per-row RMS.
    sub = _toy_random(N=60, E=300)
    net = ConnectomeNet(sub, NetConfig.for_arch(
        "ff_unroll", T=3, dynamics="linear", state_norm="rms"))
    net.set_theta_from_magnitude(torch.from_numpy(init_magnitudes(sub, "from_data")))
    w = net.edge_weight()
    h = net._step(torch.randn(5, sub.N), torch.randn(5, sub.N), w)
    rms = h.pow(2).mean(dim=1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def _toy_classifier(dynamics="linear", learn_core=True, n_pixels=16, T=10):
    """A tiny ConnectomeClassifier on a synthetic graph (no data files)."""
    sub = _toy_random(N=40, E=160, seed=7)
    net = ConnectomeNet(sub, NetConfig.for_arch("ff_unroll", T=T, dynamics=dynamics))
    net.set_theta_from_magnitude(torch.from_numpy(init_magnitudes(sub, "from_data")))
    input_local = np.array([0, 1, 2, 3])
    readout_local = np.array([sub.N - 1, sub.N - 2, sub.N - 3])
    model = ConnectomeClassifier(net, input_local, readout_local,
                                 n_pixels=n_pixels, n_classes=10)
    if not learn_core:
        for p in model.core.parameters():
            p.requires_grad_(False)
    return model, sub


def test_frozen_core_no_theta_grad():
    # learn_core=False: only encoder+readout learn; the connectome theta gets no gradient.
    model, _ = _toy_classifier(learn_core=False, n_pixels=16)
    logits = model(torch.randn(8, 16))
    logits.sum().backward()
    assert model.core.theta.grad is None
    assert model.core.theta.requires_grad is False
    assert model.encoder.weight.grad is not None
    assert model.readout.weight.grad is not None


def test_recmul_classifier_shape_and_grad():
    # Full depth-10 recmul classifier: [B, n_pixels] -> [B, 10]; grad to encoder/readout/theta.
    model, _ = _toy_classifier(dynamics="linear", learn_core=True, n_pixels=16, T=10)
    logits = model(torch.randn(8, 16))
    assert logits.shape == (8, 10)
    logits.sum().backward()
    assert model.core.theta.grad is not None         # trainable-W gets a gradient
    assert float(model.core.theta.grad.abs().sum()) > 0
    assert model.encoder.weight.grad is not None
    assert model.readout.weight.grad is not None


def test_linear_rms_bounded_depth10():
    # linear_rms stays finite/bounded over depth 10 regardless of spectral radius.
    sub = _toy_random(N=120, E=1000, seed=4)
    net = ConnectomeNet(sub, NetConfig.for_arch(
        "ff_unroll", T=10, dynamics="linear", state_norm="rms"))
    # Deliberately do NOT rescale to a small radius — rms renorm should bound it anyway.
    net.set_theta_from_magnitude(torch.from_numpy(init_magnitudes(sub, "from_data")))
    x = torch.zeros(4, sub.N)
    x[:, :8] = torch.randn(4, 8)                      # inject at a few input nodes (t=0)
    h = net(x)
    assert torch.isfinite(h).all()
    assert float(h.abs().max()) < 100.0


# ================= CIFAR-10 input dims (learned encoder n_pixels) =================
# The learned-eye ConnectomeClassifier must accept the CIFAR pixel widths (1024 luminance /
# 3072 rgb), not just MNIST's 784. build_classifier passes n_pixels from the dataset spec;
# these guard the encoder shape directly so a regression there fails fast without GPU/data.
@pytest.mark.parametrize("n_pixels", [784, 1024, 3072])
def test_classifier_accepts_cifar_pixel_widths(n_pixels):
    model, _ = _toy_classifier(dynamics="linear", learn_core=True, n_pixels=n_pixels, T=4)
    assert model.encoder.in_features == n_pixels
    logits = model(torch.randn(6, n_pixels))
    assert logits.shape == (6, 10)
    logits.sum().backward()
    assert model.encoder.weight.grad is not None
    assert model.readout.weight.grad is not None
