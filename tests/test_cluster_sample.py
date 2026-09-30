"""Tests for the cluster-sampling estimator (head of clusters + importance-sampled tail)."""

from __future__ import annotations

import math

import pytest
import torch

from ssa.attn.cluster_sample import build_clusters, certified_head, cluster_sample, log_mass_bounds
from _fixtures import Geom, make_qkv

N_CL, WINDOW, B = 80, 16, 8     # 80 clustered prefill keys (10 clusters) + 16 recent-window keys


def _setup(seed=0, qscale=3.0):
    q, K, V = make_qkv(Geom(H=2, H_kv=1, d=16, n_k=N_CL + WINDOW), seed=seed)
    q, K, V = q * qscale, K[:, 0], V[:, 0]          # q [G=2, d], K/V [n, d]
    state = build_clusters(K[:N_CL], B=B, seed=seed)
    return q, K, V, state


def _dense(q, K, V):
    A = torch.softmax(q @ K.t() / math.sqrt(q.shape[1]), dim=-1)
    return A @ V


def test_build_clusters_valid():
    _, K, _, st = _setup()
    assert st.labels.shape == (N_CL,) and st.labels.min() >= 0 and st.labels.max() < st.n_clusters
    torch.testing.assert_close(st.cdir.norm(dim=1), torch.ones(st.n_clusters, dtype=torch.float64))
    kn = K[:N_CL] / K[:N_CL].norm(dim=1, keepdim=True)
    ang = (kn - st.cdir[st.labels]).norm(dim=1)
    assert torch.all(ang <= st.rho[st.labels] + 1e-12)
    torch.testing.assert_close(st.kmag, K[:N_CL].norm(dim=1))


def test_bounds_bracket_true_mass():
    q, K, _, st = _setup(seed=1)
    scale = 1 / math.sqrt(q.shape[1])
    log_mhat, log_L, log_U = log_mass_bounds(q, st, scale)
    s = q @ K[:N_CL].t() * scale                                   # [G, n]
    log_m = torch.stack([torch.logsumexp(s[:, st.labels == b], dim=1)
                         for b in range(st.n_clusters)], dim=1)    # [G, k]
    assert torch.all(log_L <= log_m + 1e-9) and torch.all(log_m <= log_U + 1e-9)
    assert torch.all(log_L <= log_mhat + 1e-9) and torch.all(log_mhat <= log_U + 1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_certified_head_contains_top_token(seed):
    q, K, _, st = _setup(seed=seed, qscale=5.0)
    scale = 1 / math.sqrt(q.shape[1])
    cert = certified_head(q, st, scale)                            # [G, k] bool
    top = (q @ K[:N_CL].t()).argmax(dim=1)                         # [G]
    for g in range(q.shape[0]):
        assert cert[g, st.labels[top[g]]]


def test_all_clusters_in_head_equals_dense():
    q, K, V, st = _setup(seed=2)
    out, _ = cluster_sample(q, K, V, st, h=st.n_clusters, S=0, R=3)
    for r in range(3):
        torch.testing.assert_close(out[r], _dense(q, K, V), rtol=1e-10, atol=1e-12)


def test_drop_equals_softmax_over_head_and_window():
    q, K, V, st = _setup(seed=3)
    out, kread = cluster_sample(q, K, V, st, h=2, S=0, R=1)
    for g in range(q.shape[0]):
        keep = kread[0, g]
        ref = _dense(q[g:g + 1], K[keep], V[keep])[0]
        torch.testing.assert_close(out[0, g], ref, rtol=1e-10, atol=1e-12)
        assert keep[N_CL:].all()                                    # window always read
        assert keep[:N_CL].sum() < N_CL


@pytest.mark.parametrize("proposal", ["mag", "oracle"])
def test_numerator_and_denominator_unbiased(proposal):
    q, K, V, st = _setup(seed=4)
    g = torch.Generator().manual_seed(0)
    R = 20000
    _, _, parts = cluster_sample(q, K, V, st, h=2, S=3, R=R, alpha=0.1, generator=g,
                                 proposal=proposal, return_parts=True)
    for est, exact in ((parts["Z_hat"], parts["Z"]), (parts["N_hat"], parts["N"])):
        mean = est.mean(0)
        se = est.std(0) / math.sqrt(R)
        z = ((mean - exact) / se.clamp_min(1e-300)).abs()
        assert z.max() < 5.0, float(z.max())


def test_ratio_estimate_converges_with_S():
    q, K, V, st = _setup(seed=5)
    ref = _dense(q, K, V)
    errs = []
    for S in (2, 8, 32):
        g = torch.Generator().manual_seed(1)
        out, _ = cluster_sample(q, K, V, st, h=1, S=S, R=400, generator=g)
        errs.append(float(((out - ref) ** 2).sum(-1).mean()))
    assert errs[0] > errs[1] > errs[2]


def test_reads_cover_head_window_and_sampled():
    q, K, V, st = _setup(seed=6)
    g = torch.Generator().manual_seed(2)
    _, kread = cluster_sample(q, K, V, st, h=1, S=2, R=50, generator=g)
    assert kread.shape == (50, 2, N_CL + WINDOW)
    assert kread[..., N_CL:].all()
    n_read = kread[..., :N_CL].sum(-1)
    max_read = st.sizes.sort(descending=True).values[:3].sum()      # at most h + S = 3 clusters
    assert torch.all(n_read >= 1) and torch.all(n_read <= max_read)


def test_oracle_proposal_lowers_variance():
    q, K, V, st = _setup(seed=7, qscale=5.0)
    ref = _dense(q, K, V)
    err = {}
    for prop in ("mag", "oracle"):
        g = torch.Generator().manual_seed(3)
        out, _ = cluster_sample(q, K, V, st, h=0, S=4, R=2000, alpha=0.1, generator=g, proposal=prop)
        err[prop] = float(((out - ref) ** 2).sum(-1).mean())
    assert err["oracle"] < err["mag"]
