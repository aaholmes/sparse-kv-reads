"""The fused sphere_skip kernels plugged into the engine's decode seam (GPU only)."""

from __future__ import annotations

import pytest
import torch

from ssa.models.patch import install, uninstall
from _tiny_model import TinyCfg, tiny_model

pytestmark = pytest.mark.requires_cuda

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
    model = tiny_model(CFG).to("cuda").eval()
    ids = torch.randint(0, CFG.vocab_size, (1, P + T), device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(0))
    return model, ids


def test_full_budget_matches_dense():
    model, ids = _setup()
    ref = _decode(model, ids)
    stats = install(model, "cluster_fused", budget=1.0, C=16, window=4, delta=0.03, group="sum_share")
    got = _decode(model, ids)
    uninstall(model)
    torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)
    assert stats.kv_read_fraction > 1.0            # every row plus the region summaries


def test_partial_budget_reads_fewer_rows_and_stays_finite():
    model, ids = _setup()
    stats = install(model, "cluster_fused", budget=0.2, C=16, window=4, delta=0.03, group="sum_share")
    got = _decode(model, ids)
    uninstall(model)
    assert torch.isfinite(got).all()
    assert 0.0 < stats.read_fraction < 0.8
    assert 0.0 < stats.kv_read_fraction < 1.0


def test_untracked_mode_records_nothing():
    model, ids = _setup()
    stats = install(model, "cluster_fused", budget=0.2, C=16, window=4, track_reads=False)
    _decode(model, ids)
    uninstall(model)
    assert stats.steps == 0 and stats.kv_read_fraction is None


def test_sample_with_zero_draws_equals_fused():
    model, ids = _setup()
    install(model, "cluster_fused", budget=0.2, C=16, window=4, delta=0.03)
    a = _decode(model, ids)
    uninstall(model)
    install(model, "cluster_tail_sample", budget=0.2, S=0, C=16, window=4, delta=0.03)
    b = _decode(model, ids)
    uninstall(model)
    torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)


def test_sample_reads_more_rows_than_head_alone():
    model, ids = _setup()
    s1 = install(model, "cluster_fused", budget=0.1, C=16, window=4, delta=0.03)
    _decode(model, ids)
    uninstall(model)
    s2 = install(model, "cluster_tail_sample", budget=0.1, S=3, alpha=0.2, C=16, window=4, delta=0.03)
    got = _decode(model, ids)
    uninstall(model)
    assert torch.isfinite(got).all()
    assert s2.read_fraction > s1.read_fraction
