"""Capture harness: records decode queries and the final K/V cache without changing outputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ssa.harness.capture_qkv import capture
from _tiny_model import TinyCfg, tiny_model

P, T = 10, 5


def _logits(model, ids):
    cache = model.alloc_cache(ids.shape[1] + 1)
    outs = []
    with torch.inference_mode():
        model(ids[:, :P], cache, start_pos=0)
        for t in range(P, P + T):
            outs.append(model(ids[:, t:t + 1], cache))
    return torch.cat(outs, 1)


def test_capture_shapes_and_no_perturbation():
    c = TinyCfg()
    model = tiny_model(c).eval()
    ids = torch.randint(0, c.vocab_size, (1, P + T))
    ref = _logits(model, ids)
    cap = capture(model, ids, prefill=P, steps=T)
    L, H, Hkv, d = c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.head_dim
    assert cap["q"].shape == (L, T, H, d)
    assert cap["k"].shape == (L, P + T, Hkv, d) and cap["v"].shape == (L, P + T, Hkv, d)
    torch.testing.assert_close(cap["logits"], ref, rtol=1e-5, atol=1e-5)
    # hooks removed afterwards
    torch.testing.assert_close(_logits(model, ids), ref, rtol=0, atol=0)


def test_captured_step_reproduces_attention():
    c = TinyCfg()
    model = tiny_model(c).eval()
    ids = torch.randint(0, c.vocab_size, (1, P + T))
    seen = {}

    from engine.attention import Attention
    mod = [m for m in model.modules() if isinstance(m, Attention)][1]

    def spy(q, fk, fv, *, scale, layer_idx):
        G = q.shape[1] // fk.shape[1]
        out = F.scaled_dot_product_attention(q, fk.repeat_interleave(G, 1), fv.repeat_interleave(G, 1))
        seen.setdefault("outs", []).append(out[0, :, 0].clone())
        return out

    mod.decode_attn_op = spy
    _logits(model, ids)
    mod.decode_attn_op = None
    cap = capture(model, ids, prefill=P, steps=T)
    G = c.num_attention_heads // c.num_key_value_heads
    for i in range(T):
        q = cap["q"][1, i]                                   # [H, d]
        K = cap["k"][1, :P + i + 1].repeat_interleave(G, 1)  # [n, H, d]
        V = cap["v"][1, :P + i + 1].repeat_interleave(G, 1)
        A = torch.softmax(torch.einsum("hd,nhd->hn", q, K) * cap["scale"], -1)
        torch.testing.assert_close(torch.einsum("hn,nhd->hd", A, V), seen["outs"][i],
                                   rtol=1e-5, atol=1e-5)
