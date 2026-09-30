"""Tests for the partition-coverage comparison harness."""

from __future__ import annotations

import torch

from ssa.harness.sphere_coverage import captured_fraction


def _case(seed=0, n=150, d=16):
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(n, d, generator=g, dtype=torch.float64) * (0.5 + torch.rand(n, 1, generator=g,
                                                                                  dtype=torch.float64))
    q = torch.randn(4, d, generator=g, dtype=torch.float64) * 3
    return q, K


def test_captured_fraction_bounds_and_oracle():
    q, K = _case()
    budgets = (0.1, 0.3, 1.0)
    parts = {"random64": ("fixed", {"kind": "random", "C": 16}),
             "cross": ("fixed", {"kind": "cross", "C": 32}),
             "kmeans8": ("kmeans", {"B": 8, "restarts": 2}),
             "oracle": ("oracle", {})}
    res = {nm: captured_fraction(q, K, scale=0.25, window=8, budgets=budgets, method=m, seed=0, **kw)
           for nm, (m, kw) in parts.items()}
    for nm, r in res.items():
        assert r["captured"].shape == (4, 3)
        assert torch.all(r["captured"] >= 0) and torch.all(r["captured"] <= 1 + 1e-12)
        assert torch.all(r["captured"][:, 1:] >= r["captured"][:, :-1] - 1e-12)
        torch.testing.assert_close(r["captured"][:, -1], torch.ones(4, dtype=torch.float64))
        assert r["read"].shape == (4, 3)
        assert torch.all(r["read"] >= torch.tensor(budgets, dtype=torch.float64) - 1e-12)  # budget met
