"""CUDA-graph decoding with FlashInfer's paged decode as the exact attention (GPU, optional)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("flashinfer")
pytestmark = pytest.mark.requires_cuda

from _tiny_model import TinyCfg, tiny_model  # noqa: E402

CFG = TinyCfg(head_dim=128, max_position_embeddings=512, num_attention_heads=4, num_key_value_heads=2)
P, T = 200, 30


def _logits(mode, capture, dtype=torch.bfloat16, seed=0):
    from ssa.models.graph_decode import GraphDecoder
    model = tiny_model(CFG, seed=seed).to("cuda").to(dtype).eval()
    ids = torch.randint(0, CFG.vocab_size, (1, P + T + 1), device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(seed))
    cache = model.alloc_cache(-(-(P + T + 2) // 16) * 16, dtype=dtype)
    out = []
    with torch.inference_mode():
        model(ids[:, :P], cache, start_pos=0)
        if mode == "eager":                                            # the engine's own decode
            for t in range(P, P + T):
                out.append(model(ids[:, t:t + 1], cache)[0, -1].float().clone())
            return torch.stack(out)
        dec = GraphDecoder(model, cache, mode=mode)
        dec.prepare(P)
        if capture:
            dec.capture()
        for t in range(P, P + T):
            out.append(dec.step(ids[:, t:t + 1], t)[0, -1].float().clone())
    return torch.stack(out)


@pytest.mark.parametrize("capture", [False, True])
def test_flashinfer_is_as_close_to_float32_as_the_engine_exact_kernel(capture):
    """FlashInfer needs bf16, which alone moves logits by ~0.1 here; compare both bf16 paths with
    a float32 run of the same model instead of with each other."""
    ref = _logits("eager", False, dtype=torch.float32)
    err_dense = (_logits("dense", capture) - ref).abs()
    err_fi = (_logits("flashinfer", capture) - ref).abs()
    assert err_fi.mean() <= 1.2 * err_dense.mean() and err_fi.max() <= 1.5 * err_dense.max()
    assert err_fi.max() < 0.01 * ref.abs().max()


def test_paged_view_reads_each_sequence_in_place():
    from ssa.kernels.flashinfer_graph import FlashInferDecode
    B, H, H_kv, d, L = 3, 8, 2, 128, 160
    g = torch.Generator(device="cuda").manual_seed(0)
    K = torch.randn(B, H_kv, L, d, device="cuda", dtype=torch.bfloat16, generator=g)
    V = torch.randn_like(K)
    q = torch.randn(B, H, d, device="cuda", dtype=torch.bfloat16, generator=g)
    fi = FlashInferDecode(H=H, H_kv=H_kv, d=d, max_len=L, dtype=torch.bfloat16, device="cuda", batch=B)
    for n in (1, 17, 100, 160):
        fi.plan(n)
        out = fi(q, K, V)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(2).float(), K[:, :, :n].float(), V[:, :, :n].float(), enable_gqa=True).squeeze(2)
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
