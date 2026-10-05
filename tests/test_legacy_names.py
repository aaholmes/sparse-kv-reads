"""The method was renamed from ``sphere_*`` to ``voronoi_*`` to ``cluster_*``; result files keep the old names."""

from __future__ import annotations

import json

import torch

from ssa.attn import attn, canonical
from ssa.harness.plot_tvd import build_series
from ssa.models.patch import install, uninstall
from _fixtures import Geom, make_qkv
from _tiny_model import TinyCfg, tiny_model


def test_canonical_maps_every_old_name():
    new = {"skip": "cluster_skip", "skip_v1": "cluster_skip_v1", "fused": "cluster_fused",
           "sample": "cluster_tail_sample", "tail": "cluster_tail"}
    for suffix, name in new.items():
        assert canonical(f"sphere_{suffix}") == name
        assert canonical(f"voronoi_{suffix}") == name
        assert canonical(name) == name
    assert canonical("santa_sys") == "santa_sys"


def test_attn_accepts_the_old_name():
    q, K, V = make_qkv(Geom(n_k=64), seed=0)
    a = attn(q, K, V, impl="cluster_skip", budget=0.3, C=8, window=4)
    for old in ("sphere_skip", "voronoi_skip"):
        torch.testing.assert_close(attn(q, K, V, impl=old, budget=0.3, C=8, window=4), a)


def test_engine_op_accepts_the_old_name():
    cfg = TinyCfg(head_dim=16, max_position_embeddings=128, num_attention_heads=4, num_key_value_heads=2)
    model = tiny_model(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 90), generator=torch.Generator().manual_seed(0))
    outs = []
    for name in ("cluster_tail", "voronoi_tail", "sphere_tail"):
        install(model, name, budget=0.2, order=1, C=8, window=4, capacity=128)
        cache = model.alloc_cache(91)
        with torch.inference_mode():
            model(ids[:, :80], cache, start_pos=0)
            outs.append(torch.stack([model(ids[:, t:t + 1], cache)[0, -1] for t in range(80, 90)]))
        uninstall(model)
    torch.testing.assert_close(outs[0], outs[1])
    torch.testing.assert_close(outs[0], outs[2])


def test_plot_reads_old_result_files_into_the_new_series(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"summary": [
        {"impl": "sphere_skip", "cfg": {"budget": 0.1, "C": 256}, "kv_read_fraction": 0.13, "tvd": 0.063,
         "tvd_ci": [0.053, 0.074]},
        {"impl": "sphere_tail", "cfg": {"budget": 0.1, "C": 256, "order": 1}, "kv_read_fraction": 0.147,
         "tvd": 0.061, "tvd_ci": [0.052, 0.070]},
    ]}))
    s = build_series([p], regions=256)
    assert s["cluster_skip"]["x"] == [13.0] and s["cluster_tail"]["x"] == [14.7]
