"""8-bit copies of the per-cluster vectors read at every step (GPU only)."""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.requires_cuda

H, H_KV, D, W = 32, 8, 128, 64
KW = dict(C=256, C_init=64, split_factor=2.0, window=W, delta=math.inf, capacity=4096, partition="kmeans")


def _stream(n, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    K = (torch.randn(H_KV, n, D, device="cuda", generator=g) + 0.7).to(torch.bfloat16)
    Q = (torch.randn(n, H, D, device="cuda", generator=g) * 2).to(torch.bfloat16)
    return Q, K


def _q8(x):
    return torch.floor(torch.nn.functional.normalize(x, dim=-1) * 127 + 0.5).to(torch.int8)


def test_the_8_bit_copies_track_the_float_vectors_through_inserts_and_splits():
    from ssa.kernels.sphere_fused import SphereIndexFused
    Q, K = _stream(2600)
    idx = SphereIndexFused(**KW)
    assert idx.summary_bits == 8
    idx.observe(K, 1000)
    for n in range(1001, 2600):
        idx.observe(K, n)
    assert idx.splits > 0
    live = idx.count > 0
    assert (idx.dir8[live].int() - _q8(idx.sum_dir)[live].int()).abs().max() <= 1
    assert (idx.cent8[live].int() - _q8(idx.cent)[live].int()).abs().max() <= 1
    assert torch.all(idx.count <= idx.cap.unsqueeze(1))


@pytest.mark.parametrize("budget", [0.1, 0.3])
def test_selection_with_8_bit_directions_agrees_with_float32_on_identical_state(budget):
    from ssa.kernels.sphere_fused import SphereIndexFused
    Q, K = _stream(2600, seed=1)
    a = SphereIndexFused(**KW)
    a.observe(K, 2600)
    b = SphereIndexFused(**KW, summary_bits=32)
    b.observe(K, 2600)
    assert torch.equal(a.labels[:, :a.end], b.labels[:, :b.end])          # the host fit does not depend on the copies
    agree = []
    for t in range(0, 200, 10):
        _, wa = a.labels_and_weights(Q[t].float(), n=2600, budget=budget)
        _, wb = b.labels_and_weights(Q[t].float(), n=2600, budget=budget)
        agree.append(float(((wa > 0) == (wb > 0)).float().mean()))
    assert min(agree) > 0.97, min(agree)
    _, w = a.labels_and_weights(Q[0].float(), n=2600, budget=1.0)
    assert torch.equal(w[:, :256] > 0, a.count > 0)


def test_float32_mode_keeps_no_copies():
    from ssa.kernels.sphere_fused import SphereIndexFused
    Q, K = _stream(1500, seed=2)
    idx = SphereIndexFused(**KW, summary_bits=32)
    idx.observe(K, 1000)
    for n in range(1001, 1500):
        idx.observe(K, n)
    assert idx.dir8 is None and idx.cent8 is None and idx.splits > 0
    counts = torch.stack([torch.bincount(idx.labels[h, 1:idx.end].long(), minlength=256) for h in range(H_KV)])
    torch.testing.assert_close(idx.count, counts.float())
