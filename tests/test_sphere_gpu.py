"""Batched GPU bin index must reproduce SphereState's labels and weights."""

from __future__ import annotations

import math

import pytest
import torch

from ssa.attn.sphere_gpu import SphereIndexGPU
from ssa.attn.sphere_skip import fixed_directions, region_stats
from ssa.attn.sphere_state import SphereState
from _fixtures import Geom, make_qkv

W, C = 6, 16
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _stream(seed=0, n=300, H=8, H_kv=2, d=16):
    q, K, V = make_qkv(Geom(H=H, H_kv=H_kv, d=d, n_k=n), seed=seed)
    g = torch.Generator().manual_seed(seed + 100)
    Q = torch.randn(n, H, d, generator=g, dtype=torch.float64) * 3
    return Q, K + 1.5, V


def _cache(K, device):
    return K.permute(1, 0, 2).to(device)            # [H_kv, n, d] view-like engine layout


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("group", ["sum_share", "per_head"])
def test_matches_sphere_state_every_step(device, group):
    Q, K, V = _stream()
    ref = SphereState(C=C, window=W, delta=0.0)
    gpu = SphereIndexGPU(C=C, window=W, delta=0.0, capacity=400, check_every=1)
    for n in range(150, 175):
        ref.observe(K[:n])
        lab_r, w_r = ref.labels_and_weights(Q[n - 1], n=n, budget=0.2, group=group)
        gpu.observe(_cache(K[:n], device), n)
        lab_g, w_g = gpu.labels_and_weights(Q[n - 1].to(device), n=n, budget=0.2, group=group)
        assert torch.equal(lab_g.cpu(), lab_r), n
        torch.testing.assert_close(w_g.cpu(), w_r)
    assert gpu.rebuilds == ref.rebuilds


@pytest.mark.parametrize("device", DEVICES)
def test_delta_inf_never_rebuilds_and_stats_match_batch(device):
    Q, K, V = _stream(seed=1)
    gpu = SphereIndexGPU(C=C, window=W, delta=math.inf, capacity=400, check_every=1)
    for n in range(100, 180):
        gpu.observe(_cache(K[:n], device), n)
    assert gpu.rebuilds == 0
    n = 179
    dirs = fixed_directions(C, 16, seed=0, dtype=torch.float64)
    for h in range(2):
        mu0 = K[1:100 - W, h].mean(0)
        ref = region_stats(K[1:n - W, h] - mu0, dirs)
        torch.testing.assert_close(gpu.count[h].cpu(), ref["count"])
        torch.testing.assert_close(gpu.mmax[h].cpu(), ref["mmax"])
        torch.testing.assert_close(gpu.mmin[h].cpu(), ref["mmin"])
        assert torch.equal(gpu.labels[h, 1:n - W].long().cpu(), ref["labels"])


@pytest.mark.parametrize("device", DEVICES)
def test_rebuild_count_monotone_in_delta(device):
    Q, K, V = _stream(seed=2, n=400)
    counts = []
    for delta in (0.0, 0.02, 0.1, math.inf):
        gpu = SphereIndexGPU(C=C, window=W, delta=delta, capacity=400, check_every=1)
        for n in range(60, 400):
            gpu.observe(_cache(K[:n], device), n)
        counts.append(gpu.rebuilds)
    assert counts[0] >= counts[1] >= counts[2] >= counts[3] == 0 and counts[0] > counts[2]


@pytest.mark.parametrize("device", DEVICES)
def test_new_sequence_resets(device):
    Q, K, V = _stream(seed=3)
    gpu = SphereIndexGPU(C=C, window=W, delta=math.inf, capacity=400, check_every=1)
    for n in range(100, 110):
        gpu.observe(_cache(K[:n], device), n)
    gpu.observe(_cache(K[:50], device), 50)
    ref = SphereState(C=C, window=W, delta=math.inf)
    ref.observe(K[:50])
    lab_r, w_r = ref.labels_and_weights(Q[49], n=50, budget=0.1, group="sum_share")
    lab_g, w_g = gpu.labels_and_weights(Q[49].to(device), n=50, budget=0.1, group="sum_share")
    assert torch.equal(lab_g.cpu(), lab_r)
    torch.testing.assert_close(w_g.cpu(), w_r)


@pytest.mark.requires_cuda
def test_bf16_end_to_end_with_kernel():
    from ssa.attn.labeled import label_weighted_attention
    from ssa.kernels.labeled_attn import label_weighted_attention_compact
    Q, K, V = _stream(seed=4, n=600, H=32, H_kv=8, d=128)
    Kc = K.permute(1, 0, 2).to("cuda", torch.bfloat16)
    Vc = V.permute(1, 0, 2).to("cuda", torch.bfloat16)
    gpu = SphereIndexGPU(C=64, window=64, delta=0.03, capacity=1024, check_every=16)
    for n in range(500, 600):
        gpu.observe(Kc[:, :n], n)
    q = Q[599].to("cuda", torch.bfloat16)
    lab, w = gpu.labels_and_weights(q, n=600, budget=0.2, group="sum_share")
    out = label_weighted_attention_compact(q, Kc[:, :600], Vc[:, :600], lab, w)
    ref = label_weighted_attention(q.double(), Kc[:, :600].double(), Vc[:, :600].double(), lab, w.double())
    torch.testing.assert_close(out.double(), ref, rtol=2e-2, atol=2e-2)
