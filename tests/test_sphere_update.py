"""Tests for the mean-drift measurement and δ sweep harness."""

from __future__ import annotations

import math

import torch

from ssa.harness.sphere_update import delta_sweep, mean_moves


def test_mean_moves_scale_as_one_over_n():
    g = torch.Generator().manual_seed(0)
    K = torch.randn(2000, 1, 16, generator=g, dtype=torch.float64) + 3.0
    m = mean_moves(K, window=4, start=100)
    n = m["n"]
    scaled = m["move_rel"] * n                              # ≈ ‖k − μ‖ / r̄ ≈ 1 for iid keys
    assert 0.6 < float(scaled.median()) < 1.4
    assert m["move_rel"][:50].mean() > m["move_rel"][-50:].mean()


def test_delta_sweep_monotone_rebuilds_and_zero_error_at_full_budget():
    g = torch.Generator().manual_seed(1)
    n_end, H, H_kv, d = 300, 4, 2, 16
    K = torch.randn(n_end, H_kv, d, generator=g, dtype=torch.float64) + 1.0
    V = torch.randn(n_end, H_kv, d, generator=g, dtype=torch.float64)
    Q = torch.randn(n_end, H, d, generator=g, dtype=torch.float64) * 3
    eval_steps = list(range(n_end - 10, n_end + 1))
    res = delta_sweep(Q, K, V, start=60, eval_ns=eval_steps, deltas=(0.0, 0.05, math.inf),
                      budgets=(0.2, 1.0), C=8, window=4)
    rates = [res[f"delta={x}"]["rebuild_rate"] for x in (0.0, 0.05, math.inf)]
    assert rates[0] >= rates[1] >= rates[2] == 0.0
    for x in (0.0, 0.05, math.inf):
        assert res[f"delta={x}"]["b=1.0"]["err_num"] < 1e-20
        assert res[f"delta={x}"]["b=0.2"]["err_den"] > 0
