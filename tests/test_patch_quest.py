"""Quest-style page selection plugged into the engine's decode seam (pure PyTorch, CPU)."""

from __future__ import annotations

import torch

from ssa.models.patch import install, uninstall
from _tiny_model import TinyCfg, tiny_model

CFG = TinyCfg(head_dim=16, max_position_embeddings=256, num_attention_heads=4, num_key_value_heads=2)
P, T = 150, 20


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
    install(model, "quest_matched", budget=1.0, page=16, window=4)
    got = _decode(model, ids)
    uninstall(model)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_partial_budget_reads_fewer_rows_and_counts_page_bounds():
    model, ids = _setup()
    stats = install(model, "quest_matched", budget=0.2, page=16, window=4)
    got = _decode(model, ids)
    uninstall(model)
    assert torch.isfinite(got).all()
    assert 0.0 < stats.kv_read_fraction < 0.8
    assert stats.kv_read_fraction > 0.2 * stats.read_fraction       # page bounds add reads
