"""Tests for the multi-context sink / coverage analysis."""

from __future__ import annotations

import torch

from ssa.harness.sink_coverage import exact_mask, query_coverage


def test_exact_mask():
    m = exact_mask(10, window=3)
    assert m.tolist() == [True] + [False] * 6 + [True] * 3
    assert exact_mask(3, window=5).all()


def _case(seed=0, n=90, d=16):
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(n, d, generator=g, dtype=torch.float64) * (0.5 + torch.rand(n, 1, generator=g,
                                                                                  dtype=torch.float64))
    q = torch.randn(3, d, generator=g, dtype=torch.float64) * 3
    return q, K


def test_query_coverage_basic():
    q, K = _case()
    r = query_coverage(q, K, scale=0.25, window=8, B=4, restarts=2, seed=0, targets=(0.9, 0.99))
    A = torch.softmax(q @ K.t() * 0.25, -1)
    torch.testing.assert_close(r["sink"], A[:, 0])
    ex = exact_mask(K.shape[0], 8)
    torch.testing.assert_close(r["exact_mass"], A[:, ex].sum(1))
    for t in ("0.9", "0.99"):
        for rank in ("est", "oracle"):
            f = r[f"frac_{rank}_{t}"]
            assert f.shape == (3,) and torch.all(f > 0) and torch.all(f <= 1)
    assert torch.all(r["frac_est_0.99"] >= r["frac_est_0.9"])


def test_oracle_needs_no_more_clusters_than_estimate():
    for seed in range(4):
        q, K = _case(seed)
        r = query_coverage(q, K, scale=0.25, window=8, B=4, restarts=2, seed=seed, targets=(0.9,))
        assert torch.all(r["nclu_oracle_0.9"] <= r["nclu_est_0.9"])
