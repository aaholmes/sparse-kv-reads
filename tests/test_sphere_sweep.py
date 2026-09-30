"""Tests for the offline sphere_skip sweep (flat and two-level partitions)."""

from __future__ import annotations

import math

import torch

from ssa.attn import attn
from ssa.attn.sphere_skip import sphere_skip
from ssa.harness.sphere_sweep import step_eval, tree_partition
from _fixtures import Geom, make_qkv


def _case(seed=0, n=300):
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=n), seed=seed)
    return q * 3, K + 2.0, V            # shared offset in keys, like real ones


def test_flat_matches_sphere_skip():
    q, K, V = _case()
    res = step_eval(q, K, V, scale=0.25, window=6, Cs=(16,), groups=("per_head", "max_rel"),
                    budgets=(0.2,), trees=(), santa_S=())
    for group in ("per_head", "max_rel"):
        out, info = sphere_skip(q, K, V, budget=0.2, C=16, window=6, center=True, group=group,
                                return_info=True)
        ref = attn(q, K, V, impl="dense")
        r = res[f"flat C=16 {group} b=0.2"]
        torch.testing.assert_close(torch.tensor(r["err_num"], dtype=torch.float64), ((out - ref) ** 2).sum(), rtol=1e-8, atol=1e-12)
        overhead = 16 + 2 * 16 / 16
        rows = 2 * info.kv_union.double().mean() + overhead
        assert math.isclose(r["kv_frac"], float(rows) / (2 * 300), rel_tol=1e-9)


def test_full_budget_zero_error_flat_and_tree():
    q, K, V = _case(seed=1)
    res = step_eval(q, K, V, scale=0.25, window=6, Cs=(8,), groups=("per_head",), budgets=(1.0,),
                    trees=((4, 4, 2.0),), santa_S=())
    for k, r in res.items():
        assert r["err_num"] < 1e-20, k


def test_tree_partition_consistency():
    g = torch.Generator().manual_seed(0)
    Kr = torch.randn(200, 16, generator=g, dtype=torch.float64)
    t = tree_partition(Kr, C1=4, C2=5, seed=0)
    assert t["fine_labels"].max() < 20 and t["coarse_labels"].max() < 4
    torch.testing.assert_close(t["fine_labels"] // 5, t["coarse_labels"])
    assert int(t["fine"]["count"].sum()) == 200 and int(t["coarse"]["count"].sum()) == 200


def test_tree_reads_fewer_summaries_than_flat_leaves():
    q, K, V = _case(seed=2, n=600)
    res = step_eval(q, K, V, scale=0.25, window=6, Cs=(64,), groups=("max_rel",), budgets=(0.1,),
                    trees=((8, 8, 2.0),), santa_S=())
    tr = res["tree 8x8 g=2.0 max_rel b=0.1"]
    assert tr["overhead_rows"] < 64 + 1e-9           # coarse 8 + a few expanded cells × 8
    assert tr["kv_frac"] > 0 and tr["err_num"] >= 0


def test_santa_reference_reads_all_keys():
    q, K, V = _case(seed=3)
    res = step_eval(q, K, V, scale=0.25, window=6, Cs=(), groups=(), budgets=(), trees=(), santa_S=(16,),
                    seed=0)
    assert res["santa_sys S=16"]["kv_frac"] >= 0.5
