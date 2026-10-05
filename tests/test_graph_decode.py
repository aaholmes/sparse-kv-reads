"""CUDA-graph decode: must match the engine's eager decode (GPU only)."""

from __future__ import annotations

import pytest
import torch

from _tiny_model import TinyCfg, tiny_model

pytestmark = pytest.mark.requires_cuda

CFG = TinyCfg(head_dim=16, max_position_embeddings=512, num_attention_heads=4, num_key_value_heads=2)
P, T = 200, 30
SPHERE = dict(budget=0.3, C=16, window=4, delta=float("inf"), check_every=4, partition="random")


def _setup(seed=0):
    model = tiny_model(CFG, seed=seed).to("cuda").eval()
    ids = torch.randint(0, CFG.vocab_size, (1, P + T + 1), device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(seed))
    return model, ids


def _engine_logits(model, ids):
    cache = model.alloc_cache(P + T + 2)
    out = []
    with torch.inference_mode():
        model(ids[:, :P], cache, start_pos=0)
        for t in range(P, P + T):
            out.append(model(ids[:, t:t + 1], cache)[0, -1].float().clone())
    return torch.stack(out)


def _graph_logits(model, ids, mode, capture, **kw):
    from ssa.models.graph_decode import GraphDecoder
    cache = model.alloc_cache(P + T + 2)
    out = []
    with torch.inference_mode():
        model(ids[:, :P], cache, start_pos=0)
        dec = GraphDecoder(model, cache, mode=mode, **kw)
        dec.prepare(P)
        if capture:
            dec.capture()
        for t in range(P, P + T):
            out.append(dec.step(ids[:, t:t + 1], t)[0, -1].float().clone())
    return torch.stack(out), dec


@pytest.mark.parametrize("capture", [False, True])
def test_dense_matches_engine(capture):
    model, ids = _setup()
    ref = _engine_logits(model, ids)
    got, _ = _graph_logits(model, ids, "dense", capture)
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


def test_dense_replay_matches_eager_exactly():
    model, ids = _setup(1)
    a, _ = _graph_logits(model, ids, "dense", False)
    b, _ = _graph_logits(model, ids, "dense", True)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def _engine_sphere_logits(model, ids):
    """Engine eager decode with a SphereIndexFused per layer, built at the end of the prompt
    (as GraphDecoder.prepare does), so both paths bin the same keys with the same mean."""
    from ssa.kernels.sphere_fused import SphereIndexFused
    from engine.attention import Attention
    cache = model.alloc_cache(P + T + 2)
    mods = [m for m in model.modules() if isinstance(m, Attention)]
    out = []
    with torch.inference_mode():
        model(ids[:, :P], cache, start_pos=0)
        for i, mod in enumerate(mods):
            idx = SphereIndexFused(C=SPHERE["C"], window=SPHERE["window"], delta=SPHERE["delta"],
                                   capacity=cache.max_seq_len, check_every=SPHERE["check_every"])
            idx.observe(cache.k[i][0], P)

            def op(q, fk, fv, *, scale, layer_idx, idx=idx):
                o, _, _ = idx.attend(q[0, :, 0, :].contiguous(), fk[0], fv[0], n=fk.shape[2],
                                     budget=SPHERE["budget"], group="sum_share")
                return o.view(1, -1, 1, q.shape[-1])
            mod.decode_attn_op = op
        for t in range(P, P + T):
            out.append(model(ids[:, t:t + 1], cache)[0, -1].float().clone())
    for mod in mods:
        mod.decode_attn_op = None
    return torch.stack(out)


@pytest.mark.parametrize("capture", [False, True])
def test_sphere_matches_engine_with_same_index(capture):
    model, ids = _setup(2)
    ref = _engine_sphere_logits(model, ids)
    got, dec = _graph_logits(model, ids, "voronoi", capture, **SPHERE)
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
    assert dec.attn[0].head_steps > 0


def test_sphere_recenters_between_replays():
    model, ids = _setup(4)
    _, dec = _graph_logits(model, ids, "voronoi", True, **{**SPHERE, "delta": 0.0, "check_every": 1})
    assert sum(i.rebuilds for i in dec.attn) > 0


def test_sphere_full_budget_equals_dense():
    model, ids = _setup(3)
    a, _ = _graph_logits(model, ids, "dense", True)
    b, _ = _graph_logits(model, ids, "voronoi", True, **{**SPHERE, "budget": 1.0})
    torch.testing.assert_close(a, b, rtol=2e-2, atol=2e-2)
