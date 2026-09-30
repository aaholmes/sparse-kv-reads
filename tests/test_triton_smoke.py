"""Triton compiles and runs on this GPU (skipped without CUDA)."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.requires_cuda
def test_triton_toy_kernel_runs():
    import triton
    import triton.language as tl

    @triton.jit
    def add_k(x, y, o, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)

    x = torch.randn(10_000, device="cuda")
    y = torch.randn_like(x)
    o = torch.empty_like(x)
    add_k[(triton.cdiv(x.numel(), 1024),)](x, y, o, x.numel(), BLOCK=1024)
    torch.testing.assert_close(o, x + y)
