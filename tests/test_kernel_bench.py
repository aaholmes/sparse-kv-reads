"""Smoke tests for the kernel microbenchmark harness (GPU only)."""

from __future__ import annotations

import pytest
import torch

from ssa.harness.kernel_bench import bytes_needed, make_labels, time_call

pytestmark = pytest.mark.requires_cuda


@pytest.mark.parametrize("layout", ["scattered", "contiguous"])
def test_make_labels_hits_target_fraction(layout):
    labels, w = make_labels(8, 10_000, C=256, frac=0.2, layout=layout, device="cuda", seed=0)
    assert labels.shape == (8, 10_000) and labels.dtype == torch.int16
    sel = (torch.gather(w, 1, labels.long()) > 0).float().mean().item()
    assert 0.17 < sel < 0.25
    assert (labels[:, 0] == 256).all() and (labels[:, -64:] == 256).all()     # exact set


def test_bytes_and_timer():
    labels, w = make_labels(8, 4096, C=256, frac=1.0, layout="scattered", device="cuda", seed=1)
    b = bytes_needed(labels, w, d=128, elem=2)
    assert b == 8 * 4096 * 2 * 128 * 2 + labels.numel() * 2 + w.numel() * 4
    x = torch.randn(1000, device="cuda")
    t = time_call(lambda: x * 2, reps=5, flush=True)
    assert t > 0
