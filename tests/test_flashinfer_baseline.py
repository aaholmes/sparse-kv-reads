"""FlashInfer's exact decode, used as a timing baseline, must match exact attention (GPU, optional)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("flashinfer")
pytestmark = pytest.mark.requires_cuda


def test_flashinfer_decode_matches_sdpa_on_the_engine_layout():
    from ssa.harness.kernel_bench import flashinfer_decode
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    K = torch.randn(8, 3000, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    V = torch.randn_like(K)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.view(1, 32, 1, 128).float(), K.unsqueeze(0).float(), V.unsqueeze(0).float(), enable_gqa=True).view(32, 128)
    for tc in (False, True):
        torch.testing.assert_close(flashinfer_decode(q, K, V, tensor_cores=tc).float(), ref, rtol=2e-2, atol=2e-2)
