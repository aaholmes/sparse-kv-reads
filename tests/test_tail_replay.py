"""Smoke test for the offline tail-sampling replay."""

from __future__ import annotations

import torch

from ssa.harness.tail_replay import eval_step


def test_eval_step_full_budget_zero_error_and_sampling_reads_more():
    g = torch.Generator().manual_seed(0)
    H, H_kv, d, n = 4, 2, 16, 400
    q = torch.randn(H, d, generator=g, dtype=torch.float64) * 3
    K = torch.randn(H_kv, n, d, generator=g, dtype=torch.float64) + 1.0
    V = torch.randn(H_kv, n, d, generator=g, dtype=torch.float64)
    cfg = {"all": (1.0, 0, 0.0), "top": (0.1, 0, 0.0), "samp": (0.1, 8, 0.2)}
    r = eval_step(q, K, V, n, configs=cfg, R=8, seed=0)
    assert r["all"]["bias2"] + r["all"]["var"] < 1e-20
    assert r["samp"]["reads"] > r["top"]["reads"]
    assert r["samp"]["var"] > 0
