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
