"""Tests for the bound-ordered cluster-coverage diagnostic."""

from __future__ import annotations

import torch

from ssa.harness.bound_order import cluster_summary, coverage_curves, spherical_kmeans_best


def _keys(n=120, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float64) * torch.rand(n, 1, generator=g,
                                                                             dtype=torch.float64) * 3


def test_restarts_never_worse():
    K = _keys()
    Kn = K / K.norm(dim=1, keepdim=True)
    _, obj1 = spherical_kmeans_best(Kn, k=15, restarts=1, seed=0)
    _, obj10 = spherical_kmeans_best(Kn, k=15, restarts=10, seed=0)
    assert obj10 >= obj1 - 1e-9


def test_upper_bound_holds_for_every_cluster():
    K = _keys(seed=1)
    labels, _ = spherical_kmeans_best(K / K.norm(dim=1, keepdim=True), k=15, restarts=3, seed=1)
    st = cluster_summary(K, labels)
    for seed in range(5):
        q = torch.randn(3, 16, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) * 2
        s = q @ K.t()
        true_max = torch.full((3, st["k"]), -torch.inf, dtype=torch.float64).scatter_reduce(
            1, labels.expand(3, -1), s, reduce="amax")
        x = q @ st["cdir"].t() + q.norm(dim=1, keepdim=True) * st["rho"]
        ub = torch.where(x >= 0, st["mmax"] * x, st["mmin"] * x)
        assert torch.all(ub >= true_max - 1e-9)


def test_coverage_curves_monotone_and_complete():
    K = _keys(seed=2)
    labels, _ = spherical_kmeans_best(K / K.norm(dim=1, keepdim=True), k=15, restarts=2, seed=2)
    st = cluster_summary(K, labels)
    q = torch.randn(4, 16, generator=torch.Generator().manual_seed(9), dtype=torch.float64) * 3
    grid = torch.linspace(0, 1, 51, dtype=torch.float64)
    cur = coverage_curves(q, K, labels, st, scale=0.25, grid=grid, seed=0)
    for name in ("oracle", "bound", "meandir_maxmag", "random"):
        c = cur[name]                                   # [G, len(grid)]
        assert torch.all(c[:, 1:] >= c[:, :-1] - 1e-12)
        torch.testing.assert_close(c[:, -1], torch.ones(4, dtype=torch.float64))
        assert torch.all(c[:, 0] == 0)
        assert torch.all(cur["oracle"] >= c - 1e-12)    # nothing beats ordering by true mass
