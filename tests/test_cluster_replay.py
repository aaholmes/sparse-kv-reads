"""Tests for the offline cluster-sampling replay harness."""

from __future__ import annotations

import math

import torch

from ssa.harness.cluster_replay import reads_to_reach, replay_layer, summarize_layer


def _fake_layer(T=3, H=4, Hkv=2, d=16, P=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(T, H, d, generator=g, dtype=torch.float64) * 3
    K = torch.randn(P + T, Hkv, d, generator=g, dtype=torch.float64)
    V = torch.randn(P + T, Hkv, d, generator=g, dtype=torch.float64)
    return q, K, V, P


CONFIGS = [("santa_sys", {"S": 8}), ("cs_drop", {"h": 2}), ("cs_drop", {"h": 10_000}),
           ("cluster_sample", {"h": 1, "S": 4})]


def test_replay_layer_accounting():
    q, K, V, P = _fake_layer()
    res = replay_layer(q, K, V, prefill=P, scale=1 / math.sqrt(16), configs=CONFIGS, R=16, B=8)
    T = q.shape[0]
    for name in ("santa_sys|S=8", "cs_drop|h=2", "cs_drop|h=10000", "cluster_sample|h=1,S=4"):
        r = res["configs"][name]
        assert len(r["mse"]) == T and len(r["reads"]) == T
    assert all(x >= 0.5 for x in res["configs"]["santa_sys|S=8"]["reads"])   # all keys read
    exact = res["configs"]["cs_drop|h=10000"]
    assert max(exact["mse"]) < 1e-20                                            # every cluster read
    assert all(x > 1.0 for x in exact["reads"])                                 # plus summary overhead
    assert all(0 < x < 1 for x in res["configs"]["cs_drop|h=2"]["reads"])
    assert len(res["ref2"]) == T


def test_summarize_layer_bootstrap():
    q, K, V, P = _fake_layer(T=6)
    res = replay_layer(q, K, V, prefill=P, scale=0.25, configs=CONFIGS, R=8, B=8)
    s = summarize_layer(res, n_boot=50, seed=0, ref_cfg="santa_sys|S=8",
                        family="cluster_sample")
    row = s["configs"]["santa_sys|S=8"]
    assert row["rel_mse_ci"][0] <= row["rel_mse"] <= row["rel_mse_ci"][1]
    assert "reads_to_match_ref" in s


def test_reads_to_reach_interpolates_frontier():
    pts = [(0.1, 1e-2), (0.2, 1e-3), (0.4, 1e-4), (0.3, 5e-2)]   # (reads, err); (0.3,..) dominated
    assert math.isclose(reads_to_reach(pts, 1e-3), 0.2)
    r = reads_to_reach(pts, math.sqrt(1e-2 * 1e-3))                # log-midpoint
    assert math.isclose(r, math.exp((math.log(0.1) + math.log(0.2)) / 2), rel_tol=1e-9)
    assert reads_to_reach(pts, 1e-6) == math.inf
    assert reads_to_reach(pts, 1.0) == 0.1
