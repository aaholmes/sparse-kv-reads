"""`SphereIndexGPU(partition="kmeans")`: centroids fitted once on the prompt, then kept."""

from __future__ import annotations

import math

import pytest
import torch

from ssa.attn.sphere_gpu import SphereIndexGPU
from ssa.harness.partition_ablation import select
from ssa.models.patch import install, uninstall
from _fixtures import Geom, make_qkv
from _tiny_model import TinyCfg, tiny_model

C, W = 16, 8


def _qkv(n=500, seed=0):
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=n), seed=seed)
    return q, K.permute(1, 0, 2).contiguous() + 1.5, V.permute(1, 0, 2).contiguous()


def test_first_build_matches_the_ablation_reference():
    q, K, _ = _qkv()
    n = 400
    idx = SphereIndexGPU(C=C, window=W, delta=math.inf, capacity=500, partition="kmeans")
    idx.observe(K, n)
    lab, w = idx.labels_and_weights(q, n=n, budget=0.2)
    lab_ref, w_ref, _ = select(q, K, n=n, budget=0.2, partition="kmeans", center=True, score="length", C=C, window=W)
    assert torch.equal(lab.long(), lab_ref)
    torch.testing.assert_close(w, w_ref)


def test_new_keys_join_the_nearest_fitted_centroid_and_centroids_stay_fixed():
    q, K, _ = _qkv(seed=1)
    idx = SphereIndexGPU(C=C, window=W, delta=math.inf, capacity=500, partition="kmeans")
    idx.observe(K, 400)
    cent = idx.cent.clone()
    for n in range(401, 480, 7):
        idx.observe(K, n)
    assert torch.equal(idx.cent, cent)
    Kr = K[:, 392:idx.end] - idx.mu_ref.unsqueeze(1)                      # keys binned after the fit
    nearest = torch.einsum("hmd,hcd->hmc", torch.nn.functional.normalize(Kr, dim=-1), cent).argmax(-1)
    assert torch.equal(idx.labels[:, 392:idx.end].long(), nearest)


def test_random_partition_is_unchanged_by_default():
    q, K, _ = _qkv(seed=2)
    a = SphereIndexGPU(C=C, window=W, delta=math.inf, capacity=500)
    b = SphereIndexGPU(C=C, window=W, delta=math.inf, capacity=500, partition="random")
    a.observe(K, 400)
    b.observe(K, 400)
    assert torch.equal(a.labels, b.labels)


def test_engine_op_with_fitted_centroids_is_exact_at_full_budget():
    cfg = TinyCfg(head_dim=16, max_position_embeddings=256, num_attention_heads=4, num_key_value_heads=2)
    model = tiny_model(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 170), generator=torch.Generator().manual_seed(0))

    def decode():
        cache = model.alloc_cache(171)
        with torch.inference_mode():
            model(ids[:, :150], cache, start_pos=0)
            return torch.stack([model(ids[:, t:t + 1], cache)[0, -1].float() for t in range(150, 170)])

    ref = decode()
    install(model, "voronoi_tail", budget=1.0, order="drop", partition="kmeans", C=16, window=4, capacity=256)
    got = decode()
    uninstall(model)
    torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)


def test_fitted_centroids_are_the_default_for_the_engine_ops_and_graph_decoder():
    import inspect
    from ssa.models.graph_decode import GraphDecoder
    from ssa.models import patch
    assert inspect.signature(GraphDecoder.__init__).parameters["partition"].default == "kmeans"
    src = inspect.getsource(patch.make_decode_op)
    assert src.count('cfg.get("partition", "kmeans")') == 2 and 'cfg.get("partition", "random")' not in src


def test_older_sweep_presets_pin_random_directions():
    from ssa.harness.ppl_sweep import CONDITION_PRESETS
    for name in ("fused_hi", "sphere_fused_refs", "sample_grid", "tail", "tail_conv"):
        for impl, cfg in CONDITION_PRESETS[name]:
            if impl in ("voronoi_fused", "voronoi_tail", "voronoi_sample"):
                assert cfg.get("partition") == "random", (name, impl)
    assert all(c.get("partition") == "kmeans" for i, c in CONDITION_PRESETS["fused_kmeans"] if i != "dense")
