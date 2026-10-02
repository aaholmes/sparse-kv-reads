"""Contiguous-block sampling: aggregation, limits, and the unbiasedness test."""

from __future__ import annotations

import pytest
import torch

from ssa.attn import attn, available
from ssa.attn.block import block_aggregate, santa_block
from ssa.attn.dense import dense
from ssa.attn.geometry import attn_weights, gqa_expand
from ssa.harness.variance import collect_estimates, mc_mean_and_stderr
from _fixtures import Geom, make_qkv

METHODS = ["sys", "strat", "iid"]


# --- block_aggregate ---------------------------------------------------------

def test_block_mass_normalised_and_b1_identity():
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=32), seed=0, dtype=torch.float64)
    A = attn_weights(q, K)
    V_exp = gqa_expand(V, 4)

    m1, vbar1 = block_aggregate(A, V_exp, 1)
    torch.testing.assert_close(m1, A, rtol=1e-12, atol=1e-12)          # B=1: m == A
    torch.testing.assert_close(vbar1, V_exp.permute(1, 0, 2), rtol=1e-12, atol=1e-12)

    m4, _ = block_aggregate(A, V_exp, 4)
    torch.testing.assert_close(m4.sum(-1), torch.ones(4, dtype=torch.float64))  # sums to 1


def test_block_aggregate_handles_ragged_last_block():
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=30), seed=1, dtype=torch.float64)  # 30 % 7 != 0
    A = attn_weights(q, K)
    m, vbar = block_aggregate(A, gqa_expand(V, 4), 7)
    assert m.shape[1] == (30 + 6) // 7  # ceil(30/7) = 5 blocks
    torch.testing.assert_close(m.sum(-1), torch.ones(4, dtype=torch.float64))


# --- contract ----------------------------------------------------------------

def test_registered_shape_and_info():
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=64), seed=2, dtype=torch.float64)
    assert "santa_block" in available()
    g = torch.Generator().manual_seed(0)
    out, info = attn(q, K, V, impl="santa_block", S=8, B=8, generator=g, return_info=True)
    assert out.shape == (4, 16)
    assert info.block_idx.shape == (4, 8) and info.B == 8
    assert torch.all(info.reads <= 64) and torch.all(info.reads <= 8 * 8)


# --- limit gates -------------------------------------------------------------

def test_b1_equals_iid_santa():
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=64), seed=3, dtype=torch.float64)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    block_b1 = santa_block(q, K, V, S=32, B=1, method="iid", generator=g1)
    pure_iid = attn(q, K, V, impl="santa", S=32, generator=g2)
    torch.testing.assert_close(block_b1, pure_iid, rtol=1e-12, atol=1e-12)


def test_full_block_equals_dense_and_deterministic():
    g = Geom(H=4, H_kv=2, d=16, n_k=32)
    q, K, V = make_qkv(g, seed=4, dtype=torch.float64)
    ref = dense(q, K, V)
    outs = [santa_block(q, K, V, S=4, B=g.n_k, generator=torch.Generator().manual_seed(s))
            for s in (0, 1, 2)]
    for o in outs:
        torch.testing.assert_close(o, ref, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=0)  # zero variance


# --- unbiasedness test -------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("B", [2, 4, 8])
def test_block_unbiased(B, method):
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=128), seed=0, dtype=torch.float64)
    ref = dense(q, K, V).to(torch.float64)
    est = collect_estimates("santa_block", q, K, V, S=32, runs=2000, base_seed=500,
                            B=B, method=method)
    mean, stderr = mc_mean_and_stderr(est)
    z = (mean - ref).abs() / stderr.clamp_min(1e-300)
    assert z.max().item() < 5.0, f"B={B} method={method}: max|z|={z.max().item():.2f}"
