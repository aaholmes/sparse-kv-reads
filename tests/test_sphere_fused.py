"""Fused Triton bin maintenance and selection vs the batched PyTorch index (GPU only)."""

from __future__ import annotations

import math

import pytest
import torch

from ssa.attn.sphere_gpu import SphereIndexGPU

pytestmark = pytest.mark.requires_cuda

H, H_KV, D, C, W = 32, 8, 128, 256, 64


def _stream(n, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    K = (torch.randn(H_KV, n, D, device="cuda", generator=g) + 0.7).to(torch.bfloat16)
    V = torch.randn(H_KV, n, D, device="cuda", generator=g).to(torch.bfloat16)
    Q = (torch.randn(n, H, D, device="cuda", generator=g) * 2).to(torch.bfloat16)
    return Q, K, V


def _pair(delta, check_every=1, async_check=False):
    from ssa.kernels.sphere_fused import SphereIndexFused
    kw = dict(C=C, window=W, delta=delta, capacity=4096, check_every=check_every)
    return SphereIndexGPU(**kw), SphereIndexFused(**kw, async_check=async_check)


def test_binning_matches_torch_index():
    Q, K, V = _stream(1400)
    ref, fus = _pair(math.inf)
    for n in range(1000, 1400):
        ref.observe(K, n)
        fus.observe(K, n)
    same = (ref.labels[:, :1400] == fus.labels[:, :1400]).float().mean().item()
    assert same > 0.999, same
    if same == 1.0:
        torch.testing.assert_close(fus.count, ref.count)
        torch.testing.assert_close(fus.mmax, ref.mmax)
        torch.testing.assert_close(fus.mmin, ref.mmin)
        torch.testing.assert_close(fus.sum_dir, ref.sum_dir, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(fus.ksum, ref.ksum, rtol=1e-5, atol=1e-3)


def test_drift_flags_and_rebuilds_match():
    Q, K, V = _stream(1200, seed=1)
    ref, fus = _pair(0.003)
    for n in range(300, 1200):
        ref.observe(K, n)
        fus.observe(K, n)
    assert fus.rebuilds > 0
    assert abs(fus.rebuilds - ref.rebuilds) <= max(2, 0.02 * ref.rebuilds)


@pytest.mark.parametrize("group", ["sum_share", "per_head"])
@pytest.mark.parametrize("budget", [0.05, 0.2, 0.5, 1.0])
def test_selection_matches_torch_index(group, budget):
    Q, K, V = _stream(1500, seed=2)
    ref, fus = _pair(math.inf)
    ref.observe(K, 1500)
    fus.observe(K, 1500)
    for attr in ("labels", "sum_dir", "mmax", "mmin", "count", "ksum", "mu_ref", "rbar"):
        getattr(fus, attr).copy_(getattr(ref, attr))                 # identical state
    for t in (1400, 1450, 1499):
        _, w_r = ref.labels_and_weights(Q[t], n=1500, budget=budget, group=group)
        _, w_f = fus.labels_and_weights(Q[t], n=1500, budget=budget, group=group)
        assert w_f.shape == w_r.shape
        assert torch.equal(w_f[:, C], w_r[:, C])
        diff = (w_f[:, :C] != w_r[:, :C]).sum(1)
        assert int(diff.max()) <= 1, diff                             # at most one boundary bin per row
        cnt = fus.count.repeat_interleave(H // H_KV, 0) if group == "per_head" else fus.count
        need = math.ceil(budget * (fus.end - 1))
        assert torch.all((w_f[:, :C] * cnt).sum(1) >= min(need, int(cnt.sum(1).min())))


def test_end_to_end_step_matches_reference_attention():
    from ssa.attn.labeled import label_weighted_attention
    Q, K, V = _stream(2100, seed=3)
    _, fus = _pair(0.03, check_every=16)
    for n in range(2000, 2100):
        out, lab, w = fus.attend(Q[n - 1], K[:, :n], V[:, :n], n=n, budget=0.2, group="sum_share")
    ref = label_weighted_attention(Q[2098].double(), K[:, :2099].double(), V[:, :2099].double(), lab, w.double())
    torch.testing.assert_close(out.double(), ref, rtol=2e-2, atol=2e-2)


def test_async_drift_check_rebuilds_without_host_sync():
    Q, K, V = _stream(1200, seed=5)
    ref, fus = _pair(0.003, check_every=4, async_check=True)
    for n in range(300, 1200):
        ref.observe(K, n)
        fus.observe(K, n)
    assert fus.rebuilds > 0
    assert abs(fus.rebuilds - ref.rebuilds) <= max(4, 0.15 * ref.rebuilds)


def test_triton_merge_matches_torch_merge():
    from ssa.kernels.labeled_attn import _merge, merge_splits
    g = torch.Generator(device="cuda").manual_seed(6)
    for S, G in ((1, 4), (18, 4), (72, 2)):
        m = torch.randn(H_KV, S, G, device="cuda", generator=g) * 3
        m[:, 0, 0] = -1.0e30                                          # an empty split
        l = torch.rand(H_KV, S, G, device="cuda", generator=g)
        a = torch.randn(H_KV, S, G, D, device="cuda", generator=g)
        ref = _merge(m, l, a, H_KV * G, D, torch.float32)
        got = merge_splits(m, l, a, torch.float32)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
