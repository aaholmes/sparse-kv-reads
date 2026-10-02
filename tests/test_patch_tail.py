"""The tail estimate plugged into the engine's decode seam (pure PyTorch, runs on CPU)."""

from __future__ import annotations

import torch

from ssa.models.patch import install, uninstall
from _tiny_model import TinyCfg, tiny_model

CFG = TinyCfg(head_dim=16, max_position_embeddings=256, num_attention_heads=4, num_key_value_heads=2)
P, T = 150, 20
KW = dict(C=16, window=4, delta=0.03, check_every=4, capacity=256)


def _decode(model, ids):
    cache = model.alloc_cache(ids.shape[1] + 1)
    out = []
    with torch.inference_mode():
        model(ids[:, :P], cache, start_pos=0)
        for t in range(P, P + T):
            out.append(model(ids[:, t:t + 1], cache)[0, -1].float())
    return torch.stack(out)


def _setup():
    model = tiny_model(CFG).eval()
    ids = torch.randint(0, CFG.vocab_size, (1, P + T), generator=torch.Generator().manual_seed(0))
    return model, ids


def test_full_budget_matches_dense():
    model, ids = _setup()
    ref = _decode(model, ids)
    install(model, "voronoi_tail", budget=1.0, order=1, **KW)
    got = _decode(model, ids)
    uninstall(model)
    torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)


def test_drop_mode_reads_fewer_rows_and_counts_only_the_existing_summaries():
    model, ids = _setup()
    drop = install(model, "voronoi_tail", budget=0.2, order="drop", **KW)
    a = _decode(model, ids)
    uninstall(model)
    est = install(model, "voronoi_tail", budget=0.2, order=1, **KW)
    b = _decode(model, ids)
    uninstall(model)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert not torch.allclose(a, b)                             # the estimate changes the output
    assert 0.0 < drop.kv_read_fraction < 1.0
    assert est.kv_read_fraction > drop.kv_read_fraction         # plus the value sums per region


def test_drop_layers_fall_back_to_dropping_on_those_layers_only():
    model, ids = _setup()
    runs = {}
    for name, extra in (("est", {}), ("drop", {"order": "drop"}), ("mixed", {"drop_layers": [0]}),
                        ("all_dropped", {"drop_layers": [0, 1]})):
        cfg = {"order": 1, **extra}
        install(model, "voronoi_tail", budget=0.2, **cfg, **KW)
        runs[name] = _decode(model, ids)
        uninstall(model)
    torch.testing.assert_close(runs["all_dropped"], runs["drop"], rtol=1e-5, atol=1e-5)
    assert not torch.allclose(runs["mixed"], runs["est"]) and not torch.allclose(runs["mixed"], runs["drop"])
