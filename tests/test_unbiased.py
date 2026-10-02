"""Unbiasedness test: the santa* estimators satisfy E[estimate] == dense.

We assert the Monte-Carlo mean lands within a few standard errors of the float64
dense reference, coordinate-wise. topk is the biased baseline and is explicitly
NOT subject to this test.
"""

from __future__ import annotations

import pytest
import torch

from ssa.attn.dense import dense
from ssa.harness.variance import collect_estimates, mc_mean_and_stderr
from _fixtures import Geom, make_qkv

IMPLS = ["santa", "santa_strat", "santa_sys"]


@pytest.mark.parametrize("impl", IMPLS)
def test_mc_mean_matches_dense(impl):
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=128), seed=0, dtype=torch.float64)
    ref = dense(q, K, V).to(torch.float64)

    runs, S = 2000, 32
    est = collect_estimates(impl, q, K, V, S=S, runs=runs, base_seed=1000)
    mean, stderr = mc_mean_and_stderr(est)

    z = (mean - ref).abs() / stderr.clamp_min(1e-300)
    # ~64 coordinates: a 5-sigma band makes false failures astronomically unlikely
    assert z.max().item() < 5.0, f"{impl}: max |z| = {z.max().item():.2f} (mean biased?)"


@pytest.mark.parametrize("tail", ["sys", "strat", "iid"])
@pytest.mark.parametrize("k_h", [2, 8])
def test_hybrid_mc_mean_matches_dense(k_h, tail):
    # The hybrid must be exactly unbiased for any distribution.
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=128), seed=0, dtype=torch.float64)
    ref = dense(q, K, V).to(torch.float64)

    runs, S_tail = 2000, 24
    est = collect_estimates(
        "santa_hybrid", q, K, V, S=S_tail, runs=runs, base_seed=2000,
        k_h=k_h, tail=tail,
    )
    mean, stderr = mc_mean_and_stderr(est)
    z = (mean - ref).abs() / stderr.clamp_min(1e-300)
    assert z.max().item() < 5.0, f"hybrid k_h={k_h} tail={tail}: max |z| = {z.max().item():.2f}"


@pytest.mark.parametrize("impl", IMPLS)
def test_estimator_mean_is_seed_independent_in_expectation(impl):
    # Two disjoint seed blocks should agree within combined MC error.
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=64), seed=7, dtype=torch.float64)
    a = collect_estimates(impl, q, K, V, S=16, runs=1500, base_seed=0)
    b = collect_estimates(impl, q, K, V, S=16, runs=1500, base_seed=5000)
    ma, sa = mc_mean_and_stderr(a)
    mb, sb = mc_mean_and_stderr(b)
    z = (ma - mb).abs() / (sa**2 + sb**2).clamp_min(1e-300).sqrt()
    assert z.max().item() < 5.0
