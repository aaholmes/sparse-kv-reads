"""run_conditions: TVD / KL / top-1 / NLL vs dense for arbitrary attention conditions."""

from __future__ import annotations

import torch

from ssa.harness.accept_sweep import paired_summary, run_conditions
from _tiny_model import TinyCfg, tiny_model


def _chunks(c, n=3, T=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, c.vocab_size, (1, T), generator=g) for _ in range(n)]


def test_dense_and_full_budget_have_zero_tvd():
    c = TinyCfg(max_position_embeddings=64)
    model = tiny_model(c).eval()
    conds = [("dense", {}), ("voronoi_skip", {"budget": 1.0, "C": 8, "window": 4, "center": True}),
             ("voronoi_skip", {"budget": 0.05, "C": 8, "window": 2, "center": True})]
    res = run_conditions(model, _chunks(c), conditions=conds, prefill_len=24)
    assert [r["impl"] for r in res] == ["dense", "voronoi_skip", "voronoi_skip"]
    for r in res[:2]:
        assert max(r["chunk_tvd"]) < 1e-5 and max(r["chunk_kl"]) < 1e-5
        assert r["top1_agree"] == 1.0
    assert res[2]["tvd"] > 1e-4
    assert 0 < res[2]["kv_read_fraction"] < 1.2
    assert len(res[2]["chunk_tvd"]) == 3 and len(res[2]["chunk_n"]) == 3


def test_paired_summary_intervals():
    c = TinyCfg(max_position_embeddings=64)
    model = tiny_model(c).eval()
    conds = [("dense", {}), ("voronoi_skip", {"budget": 0.05, "C": 8, "window": 2, "center": True})]
    res = run_conditions(model, _chunks(c, n=4), conditions=conds, prefill_len=24)
    s = paired_summary(res, n_boot=200, seed=0)
    row = s[1]
    assert row["tvd_ci"][0] <= row["tvd"] <= row["tvd_ci"][1]
    assert row["dppl_pct_ci"][0] <= row["dppl_pct"] <= row["dppl_pct_ci"][1]
    assert s[0]["dppl_pct"] == 0.0


def test_v1_delta_zero_matches_stateless_and_reports_rebuilds():
    c = TinyCfg(max_position_embeddings=64)
    model = tiny_model(c).eval()
    base = {"budget": 0.2, "C": 8, "window": 2, "group": "sum_share"}
    conds = [("dense", {}), ("voronoi_skip", {**base, "center": True}),
             ("voronoi_skip_v1", {**base, "delta": 0.0}), ("voronoi_skip_v1", {**base, "delta": float("inf")})]
    res = run_conditions(model, _chunks(c), conditions=conds, prefill_len=24)
    assert abs(res[1]["tvd"] - res[2]["tvd"]) < 1e-6
    assert res[2]["rebuild_rate"] > 0 and res[3]["rebuild_rate"] == 0


def test_chunked_prefill_gives_identical_dense_logits():
    from ssa.harness.accept_sweep import _decode_logits
    c = TinyCfg(max_position_embeddings=64)
    model = tiny_model(c).eval()
    ids = _chunks(c, n=1)[0]
    with torch.inference_mode():
        a = _decode_logits(model, ids, prefill_len=24)
        b = _decode_logits(model, ids, prefill_len=24, prefill_chunk=7)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


def test_quant_condition_runs_last_and_changes_logits():
    c = TinyCfg(max_position_embeddings=64, hidden_size=64, intermediate_size=128)
    model = tiny_model(c).eval()
    res = run_conditions(model, _chunks(c), conditions=[("dense", {}), ("quant", {"n_bits": 4, "group_size": 32})],
                         prefill_len=24)
    assert res[1]["tvd"] > 0 and res[1]["kv_read_fraction"] == 1.0
