"""Split-on-overflow with the Triton kernels and the CUDA-graph decoder (GPU only)."""

from __future__ import annotations

import math

import pytest
import torch

from ssa.attn.sphere_gpu import SphereIndexGPU
from _tiny_model import TinyCfg, tiny_model

pytestmark = pytest.mark.requires_cuda

H, H_KV, D, W = 32, 8, 128, 64
KW = dict(C=256, C_init=64, split_factor=2.0, window=W, delta=math.inf, capacity=4096, check_every=1,
          partition="kmeans")


def _stream(n, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    K = (torch.randn(H_KV, n, D, device="cuda", generator=g) + 0.7).to(torch.bfloat16)
    Q = (torch.randn(n, H, D, device="cuda", generator=g) * 2).to(torch.bfloat16)
    return Q, K


def test_fused_index_splits_like_the_torch_index_and_never_fills_spare_slots():
    from ssa.kernels.sphere_fused import SphereIndexFused
    Q, K = _stream(2600)
    ref, fus = SphereIndexGPU(**KW), SphereIndexFused(**KW, async_check=False)
    for n in range(1000, 2600, 4):
        ref.observe(K, n)
        fus.observe(K, n)
    assert fus.splits > 0 and torch.equal(fus.n_c, ref.n_c)
    end = fus.end
    assert torch.all(fus.labels[:, 1:end].long() < fus.n_c.unsqueeze(1))
    slots = torch.arange(256, device="cuda").unsqueeze(0)
    assert torch.all(fus.count[slots >= fus.n_c.unsqueeze(1)] == 0)
    same = (ref.labels[:, :end] == fus.labels[:, :end]).float().mean().item()
    assert same > 0.99, same
    counts = torch.stack([torch.bincount(fus.labels[h, 1:end].long(), minlength=256) for h in range(H_KV)])
    torch.testing.assert_close(fus.count, counts.float())


@pytest.mark.parametrize("capture", [False, True])
def test_graph_decoder_with_splitting_is_exact_at_full_budget_and_splits(capture):
    from ssa.models.graph_decode import GraphDecoder
    cfg = TinyCfg(head_dim=16, max_position_embeddings=1024, num_attention_heads=4, num_key_value_heads=2)
    P, T = 200, 400
    model = tiny_model(cfg, seed=3).to("cuda").eval()
    ids = torch.randint(0, cfg.vocab_size, (1, P + T + 1), device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(3))

    def run(mode, **kw):
        cache = model.alloc_cache(-(-(P + T + 2) // 16) * 16)
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

    ref, _ = run("dense")
    got, dec = run("cluster", budget=1.0, C=64, C_init=16, split_factor=2.0, window=4, delta=float("inf"),
                   check_every=4)
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
    assert sum(i.splits for i in dec.attn) > 0
    assert all(int(i.n_c.max()) <= 64 for i in dec.attn)


def test_flagged_splits_without_a_host_sync_keep_clusters_near_the_cap_and_spare_slots_unselected():
    from ssa.kernels.sphere_fused import SphereIndexFused
    Q, K = _stream(2600, seed=1)
    fus = SphereIndexFused(**{**KW, "check_every": 4})                    # flags read asynchronously
    for n in range(1000, 2600):
        fus.observe(K, n)
    torch.cuda.synchronize()
    assert fus.splits > 0
    end = fus.end
    counts = torch.stack([torch.bincount(fus.labels[h, 1:end].long(), minlength=256) for h in range(H_KV)])
    torch.testing.assert_close(fus.count, counts.float())
    assert torch.all(fus.count <= fus.cap.unsqueeze(1) + 40)              # a split lags its flag by a few checks
    labels, w = fus.labels_and_weights(Q[0].float(), n=2600, budget=1.0)
    slots = torch.arange(256, device="cuda").unsqueeze(0)
    spare = slots >= fus.n_c.unsqueeze(1)
    assert spare.any() and torch.all(w[:, :256][spare] == 0)
    assert torch.equal(w[:, :256] > 0, fus.count > 0)
