"""Batched CUDA-graph decode: a batch of B equal-length sequences must match B separate
batch-1 runs, for every attention mode (GPU only)."""

from __future__ import annotations

import pytest
import torch

from _tiny_model import TinyCfg, tiny_model

pytestmark = pytest.mark.requires_cuda

CFG = TinyCfg(head_dim=16, max_position_embeddings=512, num_attention_heads=4, num_key_value_heads=2)
P, T, B = 200, 30, 3
CLUSTER = dict(budget=0.3, C=16, window=4, delta=float("inf"), check_every=4)


def _ids(seed=0):
    return torch.randint(0, CFG.vocab_size, (B, P + T + 1), device="cuda",
                         generator=torch.Generator(device="cuda").manual_seed(seed))


def _run(model, ids, mode, capture, cfg, dtype):
    """Logits ``[b, T, vocab]`` from decoding the rows of ``ids`` together."""
    from ssa.models.graph_decode import GraphDecoder
    b = ids.shape[0]
    cache = model.alloc_cache(-(-(P + T + 2) // 16) * 16, max_batch=b, dtype=dtype)
    out = []
    with torch.inference_mode():
        model(ids[:, :P], cache, start_pos=0)
        dec = GraphDecoder(model, cache, mode=mode, **cfg)
        dec.prepare(P)
        if capture:
            dec.capture()
        for t in range(P, P + T):
            out.append(dec.step(ids[:, t:t + 1], t)[:, -1].float().clone())
    return torch.stack(out, 1)


def _batched_vs_separate(mode, capture, cfg, dtype, model_seed=0):
    model = tiny_model(CFG, seed=model_seed).to("cuda").to(dtype).eval()
    ids = _ids(model_seed)
    together = _run(model, ids, mode, capture, cfg, dtype)
    apart = torch.cat([_run(model, ids[i:i + 1], mode, capture, cfg, dtype) for i in range(B)])
    return together, apart


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("mode,cfg", [("dense", {}), ("cluster", CLUSTER)])
def test_batch_matches_separate_runs(mode, cfg, capture):
    together, apart = _batched_vs_separate(mode, capture, cfg, torch.float32)
    assert together.shape == (B, T, CFG.vocab_size)
    torch.testing.assert_close(together, apart, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("capture", [False, True])
def test_flashinfer_batch_matches_separate_runs(capture):
    pytest.importorskip("flashinfer")
    cfg = TinyCfg(head_dim=128, max_position_embeddings=512, num_attention_heads=4, num_key_value_heads=2)
    model = tiny_model(cfg, seed=1).to("cuda").to(torch.bfloat16).eval()
    ids = torch.randint(0, cfg.vocab_size, (B, P + T + 1), device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(1))
    together = _run(model, ids, "flashinfer", capture, {}, torch.bfloat16)
    apart = torch.cat([_run(model, ids[i:i + 1], "flashinfer", capture, {}, torch.bfloat16) for i in range(B)])
    assert (together - apart).abs().max() < 0.01 * apart.abs().max()       # bf16: batch shapes change matmul rounding
