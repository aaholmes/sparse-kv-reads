"""Triton label-weighted flash-decoding kernel vs the reference (GPU only)."""

from __future__ import annotations

import pytest
import torch

from ssa.attn.labeled import label_weighted_attention

pytestmark = pytest.mark.requires_cuda

H, H_KV, D, L = 32, 8, 128, 257          # Qwen3-4B geometry; 256 bins + exact label


def _inputs(n, *, seed=0, dtype=torch.bfloat16, capacity=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    cap = capacity or n
    q = torch.randn(H, D, device="cuda", generator=g).to(dtype) * 2
    Kbuf = torch.randn(1, H_KV, cap, D, device="cuda", generator=g).to(dtype)
    Vbuf = torch.randn(1, H_KV, cap, D, device="cuda", generator=g).to(dtype)
    labels = torch.randint(0, L, (H_KV, n), device="cuda", generator=g).to(torch.int16)
    return q, Kbuf[0, :, :n], Vbuf[0, :, :n], labels


IMPLS = ["masked_scan", "compact"]


def _fn(impl):
    from ssa.kernels import labeled_attn as la
    return {"masked_scan": la.label_weighted_attention_triton,
            "compact": la.label_weighted_attention_compact}[impl]


def _check(q, K, V, labels, w, impl="masked_scan", **kw):
    out = _fn(impl)(q, K, V, labels, w, **kw)
    ref = label_weighted_attention(q.double(), K.double(), V.double(), labels, w.double())
    assert out.shape == (H, D) and out.dtype == q.dtype
    torch.testing.assert_close(out.double(), ref, rtol=2e-2, atol=2e-2)
    return out


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("n", [1, 63, 64, 65, 1000, 4097])
def test_all_ones_is_dense(n, impl):
    q, K, V, labels = _inputs(n, seed=n)
    w = torch.ones(H_KV, L, device="cuda")
    out = _check(q, K, V, labels, w, impl=impl)
    sdpa = torch.nn.functional.scaled_dot_product_attention(
        q.view(1, H, 1, D), K.unsqueeze(0).repeat_interleave(H // H_KV, 1),
        V.unsqueeze(0).repeat_interleave(H // H_KV, 1)).view(H, D)
    torch.testing.assert_close(out.float(), sdpa.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("per_head", [False, True])
def test_random_selection(per_head, impl):
    q, K, V, labels = _inputs(3000, seed=1)
    g = torch.Generator(device="cuda").manual_seed(2)
    w = (torch.rand(H if per_head else H_KV, L, device="cuda", generator=g) < 0.2).float()
    w[:, L - 1] = 1.0                                            # exact set always read
    _check(q, K, V, labels, w, impl=impl)


@pytest.mark.parametrize("impl", IMPLS)
def test_non_unit_weights(impl):
    q, K, V, labels = _inputs(2500, seed=3)
    g = torch.Generator(device="cuda").manual_seed(4)
    w = torch.rand(H_KV, L, device="cuda", generator=g) * 5
    w[w < 1.0] = 0.0
    w[:, L - 1] = 1.0
    _check(q, K, V, labels, w, impl=impl)


@pytest.mark.parametrize("impl", IMPLS)
def test_unread_rows_are_never_used(impl):
    q, K, V, labels = _inputs(2000, seed=5)
    w = torch.zeros(H_KV, L, device="cuda")
    w[:, :64] = 1.0
    w[:, L - 1] = 1.0
    f = _fn(impl)
    a = f(q, K, V, labels, w)
    skip = torch.gather(w, 1, labels.long()) == 0                # [H_kv, n]
    K2, V2 = K.clone(), V.clone()
    K2[skip] = float("nan")
    V2[skip] = float("nan")
    b = f(q, K2, V2, labels, w)
    assert torch.isfinite(b).all()
    exact = impl == "masked_scan"          # compaction order is nondeterministic (atomics)
    torch.testing.assert_close(a, b, rtol=0 if exact else 1e-2, atol=0 if exact else 1e-2)


@pytest.mark.parametrize("impl", IMPLS)
def test_strided_cache_view_with_spare_capacity(impl):
    q, K, V, labels = _inputs(1500, seed=6, capacity=4096)       # K, V are views into a larger buffer
    assert not K.is_contiguous()
    w = torch.ones(H_KV, L, device="cuda")
    w[:, 5:100] = 0.0
    _check(q, K, V, labels, w, impl=impl)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("block_n,num_splits", [(32, 1), (64, 7), (128, 40)])
def test_tiling_parameters(block_n, num_splits, impl):
    q, K, V, labels = _inputs(5000, seed=7)
    w = (torch.rand(H_KV, L, device="cuda", generator=torch.Generator(device="cuda").manual_seed(8)) < 0.3).float()
    w[:, L - 1] = 1.0
    _check(q, K, V, labels, w, impl=impl, block_n=block_n, num_splits=num_splits)


@pytest.mark.parametrize("impl", IMPLS)
def test_long_context(impl):
    q, K, V, labels = _inputs(65536, seed=9)
    w = (torch.rand(H_KV, L, device="cuda", generator=torch.Generator(device="cuda").manual_seed(10)) < 0.1).float()
    w[:, L - 1] = 1.0
    _check(q, K, V, labels, w, impl=impl)


def test_compaction_list_is_exactly_the_needed_rows():
    from ssa.kernels.labeled_attn import compact_positions
    _, _, _, labels = _inputs(5000, seed=11)
    g = torch.Generator(device="cuda").manual_seed(12)
    for rows in (H_KV, H):
        w = (torch.rand(rows, L, device="cuda", generator=g) < 0.25).float()
        idx, cnt = compact_positions(labels, w)
        wk = w if rows == H_KV else w.view(H_KV, H // H_KV, L).amax(1)
        need = torch.gather(wk, 1, labels.long()) > 0
        for kv in range(H_KV):
            got = torch.sort(idx[kv, :int(cnt[kv])]).values
            torch.testing.assert_close(got.long(), need[kv].nonzero().squeeze(1))
