"""Tests for SphereState: incremental hypersphere bins with δ-triggered recentering (v1)."""

from __future__ import annotations

import math

import torch

from ssa.attn import attn
from ssa.attn.sphere_skip import fixed_directions, region_stats, sphere_skip
from ssa.attn.sphere_state import SphereState
from _fixtures import Geom, make_qkv

W = 6


def _stream(seed=0, n=260, H=8, H_kv=2, d=16):
    q, K, V = make_qkv(Geom(H=H, H_kv=H_kv, d=d, n_k=n), seed=seed)
    g = torch.Generator().manual_seed(seed + 100)
    Q = torch.randn(n, H, d, generator=g, dtype=torch.float64) * 3
    return Q, K + 1.5, V                       # shared key offset, like real keys


def test_delta_zero_matches_stateless_every_step():
    Q, K, V = _stream()
    st = SphereState(C=16, window=W, delta=0.0)
    for n in range(150, 170):
        out, info = st.attend(Q[n - 1], K[:n], V[:n], budget=0.2, group="sum_share")
        ref, rinfo = sphere_skip(Q[n - 1], K[:n], V[:n], budget=0.2, C=16, window=W, center=True,
                                 group="sum_share", return_info=True)
        assert torch.equal(info.selected, rinfo.selected), n
        torch.testing.assert_close(out, ref)


def test_delta_inf_never_rebuilds_and_stats_match_batch():
    Q, K, V = _stream(seed=1)
    st = SphereState(C=16, window=W, delta=math.inf)
    for n in range(100, 180):
        st.attend(Q[n - 1], K[:n], V[:n], budget=0.1, group="sum_share")
    assert st.rebuilds == 0
    n = 179
    for h in range(2):
        binned = K[1:n - W, h]
        mu0 = K[1:100 - W, h].mean(0)                    # mean at the initial build
        dirs = fixed_directions(16, 16, seed=0, dtype=torch.float64)
        ref = region_stats(binned - mu0, dirs)
        got = st.heads[h]
        torch.testing.assert_close(got["count"], ref["count"])
        torch.testing.assert_close(got["mmax"], ref["mmax"])
        torch.testing.assert_close(got["mmin"], ref["mmin"])
        torch.testing.assert_close(got["labels"], ref["labels"])


def test_smaller_delta_rebuilds_more():
    Q, K, V = _stream(seed=2, n=400)
    counts = []
    for delta in (0.0, 0.02, 0.1, math.inf):
        st = SphereState(C=16, window=W, delta=delta)
        for n in range(60, 400):
            st.attend(Q[n - 1], K[:n], V[:n], budget=0.1, group="sum_share")
        counts.append(st.rebuilds)
    assert counts[0] >= counts[1] >= counts[2] >= counts[3] == 0
    assert counts[0] > counts[2]


def test_full_budget_is_dense_for_any_delta():
    Q, K, V = _stream(seed=3)
    for delta in (0.0, 0.05, math.inf):
        st = SphereState(C=16, window=W, delta=delta)
        for n in range(80, 120):
            out, _ = st.attend(Q[n - 1], K[:n], V[:n], budget=1.0, group="sum_share")
        torch.testing.assert_close(out, attn(Q[118], K[:119], V[:119], impl="dense"))


def test_new_sequence_resets():
    Q, K, V = _stream(seed=4)
    st = SphereState(C=16, window=W, delta=math.inf)
    for n in range(100, 110):
        st.attend(Q[n - 1], K[:n], V[:n], budget=0.1, group="sum_share")
    out, _ = st.attend(Q[49], K[:50], V[:50], budget=0.1, group="sum_share")   # shorter: new sequence
    ref = sphere_skip(Q[49], K[:50], V[:50], budget=0.1, C=16, window=W, center=True, group="sum_share")
    torch.testing.assert_close(out, ref)
