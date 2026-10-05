"""Fitted per-head centroids in the Triton region kernels and the CUDA-graph decoder (GPU only)."""

from __future__ import annotations

import math

import pytest
import torch

from ssa.attn.sphere_gpu import SphereIndexGPU
from _tiny_model import TinyCfg, tiny_model

pytestmark = pytest.mark.requires_cuda

H, H_KV, D, C, W = 32, 8, 128, 256, 64


def _stream(n, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    K = (torch.randn(H_KV, n, D, device="cuda", generator=g) + 0.7).to(torch.bfloat16)
    Q = (torch.randn(n, H, D, device="cuda", generator=g) * 2).to(torch.bfloat16)
    return Q, K


def test_fused_binning_and_selection_match_the_torch_index_with_fitted_centroids():
    from ssa.kernels.sphere_fused import SphereIndexFused
    Q, K = _stream(1400)
    kw = dict(C=C, window=W, delta=math.inf, capacity=4096, check_every=1, partition="kmeans")
    ref, fus = SphereIndexGPU(**kw), SphereIndexFused(**kw, async_check=False)
    for n in range(1000, 1400, 3):
        ref.observe(K, n)
        fus.observe(K, n)
    torch.testing.assert_close(fus.cent, ref.cent)                         # one fit, at the first build
    same = (ref.labels[:, :1397] == fus.labels[:, :1397]).float().mean().item()
    assert same > 0.999, same
    n = 1397
    lab_r, w_r = ref.labels_and_weights(Q[n - 1], n=n, budget=0.2)
    lab_f, w_f = fus.labels_and_weights(Q[n - 1], n=n, budget=0.2)
    assert (w_r == w_f).float().mean() > 0.99


def test_fitted_centroids_differ_from_random_directions():
    from ssa.kernels.sphere_fused import SphereIndexFused
    Q, K = _stream(1200, seed=1)
    a = SphereIndexFused(C=C, window=W, delta=math.inf, capacity=4096, partition="kmeans")
    b = SphereIndexFused(C=C, window=W, delta=math.inf, capacity=4096)
    a.observe(K, 1200)
    b.observe(K, 1200)
    assert (a.labels[:, :1100] != b.labels[:, :1100]).float().mean() > 0.5


@pytest.mark.parametrize("capture", [False, True])
def test_graph_decoder_with_fitted_centroids_matches_the_engine_op(capture):
    """The CUDA-graph path and the engine's eager fused op, both with fitted centroids, must agree."""
    from ssa.models.graph_decode import GraphDecoder
    cfg = TinyCfg(head_dim=16, max_position_embeddings=512, num_attention_heads=4, num_key_value_heads=2)
    kw = dict(budget=0.3, C=16, window=4, delta=float("inf"), check_every=4, partition="kmeans")
    P, T = 200, 30
    model = tiny_model(cfg, seed=2).to("cuda").eval()
    ids = torch.randint(0, cfg.vocab_size, (1, P + T + 1), device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(2))

    def run(graph):
        cache = model.alloc_cache(-(-(P + T + 2) // 16) * 16)
        out = []
        with torch.inference_mode():
            model(ids[:, :P], cache, start_pos=0)
            if graph:
                dec = GraphDecoder(model, cache, mode="cluster", **kw)
                dec.prepare(P)
                if capture:
                    dec.capture()
                for t in range(P, P + T):
                    out.append(dec.step(ids[:, t:t + 1], t)[0, -1].float().clone())
            else:                                    # engine decode with an index built at the end of the
                from engine.attention import Attention          # prompt, as GraphDecoder.prepare does
                from ssa.kernels.sphere_fused import SphereIndexFused
                mods = [m for m in model.modules() if isinstance(m, Attention)]
                for i, mod in enumerate(mods):
                    idx = SphereIndexFused(C=kw["C"], window=kw["window"], delta=kw["delta"], partition="kmeans",
                                           capacity=cache.max_seq_len, check_every=kw["check_every"])
                    idx.observe(cache.k[i][0], P)

                    def op(q, fk, fv, *, scale, layer_idx, idx=idx):
                        o, _, _ = idx.attend(q[0, :, 0, :].contiguous(), fk[0], fv[0], n=fk.shape[2],
                                             budget=kw["budget"], group="sum_share")
                        return o.view(1, -1, 1, q.shape[-1])
                    mod.decode_attn_op = op
                for t in range(P, P + T):
                    out.append(model(ids[:, t:t + 1], cache)[0, -1].float().clone())
                for mod in mods:
                    mod.decode_attn_op = None
        return torch.stack(out)

    step_diff = (run(True) - run(False)).abs().amax(1)                    # [T]
    # The two pick kernels can order a region on the selection boundary differently (rounding), which
    # changes one step's reads; allow that on a few steps, and require the rest to agree closely.
    assert (step_diff < 2e-2).float().mean() >= 0.9, step_diff
    assert step_diff.max() < 0.3, step_diff
