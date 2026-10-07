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


def test_fused_index_splits_about_as_often_as_the_torch_index_and_never_fills_spare_slots():
    from ssa.kernels.sphere_fused import SphereIndexFused
    Q, K = _stream(2600)
    ref, fus = SphereIndexGPU(**KW), SphereIndexFused(**KW, async_check=False, summary_bits=32)
    for n in range(1000, 2600, 4):
        ref.observe(K, n)
        fus.observe(K, n)
    assert fus.splits > 0 and (fus.n_c - ref.n_c).abs().max() <= 4        # the kernel splits per key, in float32
    assert torch.all(fus.count <= fus.cap.unsqueeze(1))
    end = fus.end
    assert torch.all(fus.labels[:, 1:end].long() < fus.n_c.unsqueeze(1))
    slots = torch.arange(256, device="cuda").unsqueeze(0)
    assert torch.all(fus.count[slots >= fus.n_c.unsqueeze(1)] == 0)
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


def test_splits_inside_the_binning_kernel_keep_every_cluster_under_the_cap_with_no_host_work():
    from ssa.kernels.sphere_fused import SphereIndexFused
    Q, K = _stream(2600, seed=1)
    fus = SphereIndexFused(**{**KW, "check_every": 4})
    fus.observe(K, 1000)
    at_fit = fus.splits
    fus._split_overflow = None                                            # any host split during decoding would raise
    for n in range(1001, 2600):
        fus.observe(K, n)
    assert fus.splits > at_fit
    end = fus.end
    counts = torch.stack([torch.bincount(fus.labels[h, 1:end].long(), minlength=256) for h in range(H_KV)])
    torch.testing.assert_close(fus.count, counts.float())
    assert torch.all(fus.count <= fus.cap.unsqueeze(1))                   # split in the step that passes the cap
    live = fus.count > 0
    Kr = K.float() - fus.mu_ref.unsqueeze(1)
    mag = Kr.norm(dim=-1)
    Kn = Kr / mag.unsqueeze(-1)
    lab = fus.labels[:, 1:end].long()
    sd = torch.zeros_like(fus.sum_dir).scatter_add_(1, lab.unsqueeze(-1).expand(-1, -1, D), Kn[:, 1:end])
    torch.testing.assert_close(fus.sum_dir, sd, rtol=1e-3, atol=1e-3)
    mx = torch.zeros_like(fus.mmax).scatter_reduce_(1, lab, mag[:, 1:end], reduce="amax")
    torch.testing.assert_close(fus.mmax[live], mx[live], rtol=1e-4, atol=1e-4)
    labels, w = fus.labels_and_weights(Q[0].float(), n=2600, budget=1.0)
    slots = torch.arange(256, device="cuda").unsqueeze(0)
    spare = slots >= fus.n_c.unsqueeze(1)
    assert spare.any() and torch.all(w[:, :256][spare] == 0)
    assert torch.equal(w[:, :256] > 0, live)


def test_a_lopsided_cluster_is_cut_at_the_median_inside_the_kernel():
    from ssa.kernels.sphere_fused import SphereIndexFused
    g = torch.Generator(device="cuda").manual_seed(2)
    K = torch.randn(1, 1400, D, device="cuda", generator=g)
    main = torch.randn(1, 1, D, device="cuda", generator=g)
    K[:, 700:] = 4 * main + 0.2 * torch.randn(1, 700, D, device="cuda", generator=g)     # later keys: one tight lump
    K[:, 900::97] = -4 * main                                                           # with a few far outliers
    fus = SphereIndexFused(C=64, C_init=16, split_factor=2.0, window=8, delta=math.inf, capacity=2048,
                           partition="kmeans")
    fus.observe(K, 600)
    for n in range(601, 1400):
        fus.observe(K, n)
    assert torch.all(fus.count <= fus.cap.unsqueeze(1))
    assert int(fus.n_c[0]) < 64                                           # balanced cuts do not exhaust the slots


@pytest.mark.parametrize("lumpy", [False, True])
def test_each_kernel_split_matches_the_reference_rule_on_the_same_members(lumpy):
    from ssa.attn.sphere_gpu import _bisect_padded
    from ssa.kernels.sphere_fused import SphereIndexFused
    Q, K = _stream(1800, seed=3)
    if lumpy:                                                             # forces lopsided 2-means, so median cuts
        g = torch.Generator(device="cuda").manual_seed(4)
        main = torch.randn(H_KV, 1, D, device="cuda", generator=g)
        K[:, 1100:] = (3 * main + 0.3 * torch.randn(H_KV, 700, D, device="cuda", generator=g)).to(K.dtype)
        K[:, 1150::41] = (-3 * main).to(K.dtype)
    fus = SphereIndexFused(**KW)
    fus.observe(K, 1000)
    checked = median_cuts = 0
    for n in range(1001, 1800):
        lab0, nc0, end0 = fus.labels.clone(), fus.n_c.clone(), fus.end
        fus.observe(K, n)
        for h in (fus.n_c > nc0).nonzero().flatten().tolist():
            new = int(nc0[h])
            assert int(fus.n_c[h]) == new + 1 and fus.end == end0 + 1
            moved = (fus.labels[h, 1:fus.end] == new).nonzero().flatten() + 1
            c = int(lab0[h, moved[0]]) if int(moved[0]) < end0 else int(lab0[h, moved[1]])
            was = lab0[h, 1:fus.end].clone()
            was[end0 - 1] = c if int(fus.labels[h, end0]) in (c, new) else -1
            pos = (was == c).nonzero().flatten() + 1
            assert set(fus.labels[h, pos].tolist()) == {c, new}           # one cluster became two
            Kn = torch.nn.functional.normalize(K[h, pos].float() - fus.mu_ref[h], dim=-1).unsqueeze(0)
            valid = torch.ones(1, len(pos), dtype=torch.bool, device="cuda")
            side = _bisect_padded(Kn, valid)[0]
            got = (fus.labels[h, pos] == new).long()
            assert int((side != got).sum()) <= max(1, 0.02 * len(pos)), (h, n, int((side != got).sum()), len(pos))
            share = got.float().mean()
            assert 0.25 <= share <= 0.75 or abs(int(got.sum()) - (len(pos) - len(pos) // 2)) == 0
            median_cuts += int(got.sum()) == len(pos) - len(pos) // 2
            checked += 1
    assert checked >= 20
    if lumpy:
        assert median_cuts > 0


def test_periodic_refit_in_the_graph_decoder_is_exact_at_full_budget_and_matches_a_fresh_fit():
    from ssa.kernels.graph_kernels import SphereIndexGraph
    from ssa.models.graph_decode import GraphDecoder
    cfg = TinyCfg(head_dim=16, max_position_embeddings=1024, num_attention_heads=4, num_key_value_heads=2)
    P, T = 200, 300
    model = tiny_model(cfg, seed=5).to("cuda").eval()
    ids = torch.randint(0, cfg.vocab_size, (1, P + T + 1), device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(5))

    def run(mode, **kw):
        cache = model.alloc_cache(-(-(P + T + 2) // 16) * 16)
        out = []
        with torch.inference_mode():
            model(ids[:, :P], cache, start_pos=0)
            dec = GraphDecoder(model, cache, mode=mode, **kw)
            dec.prepare(P)
            dec.capture()
            for t in range(P, P + T):
                out.append(dec.step(ids[:, t:t + 1], t)[0, -1].float().clone())
        return torch.stack(out), dec, cache

    ref, _, _ = run("dense")
    got, dec, cache = run("cluster", budget=1.0, C=32, window=4, delta=float("inf"), refit_every=100)
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
    idx = dec.attn[1]
    assert idx.refits == 3                                                 # the last one at the final step
    fresh = SphereIndexGraph(budget=1.0, C=32, window=4, delta=float("inf"), capacity=idx.capacity, partition="kmeans")
    fresh.prepare(dec._heads(cache.k[1]), P + T, 4)
    assert torch.equal(fresh.labels[:, :fresh.end], idx.labels[:, :fresh.end])
    torch.testing.assert_close(fresh.count, idx.count)


def test_growing_cap_and_reset_in_the_kernels():
    from ssa.kernels.sphere_fused import SphereIndexFused
    Q, K = _stream(4000, seed=5)
    kw = {**KW, "check_every": 4}
    fixed, grow = SphereIndexFused(**kw), SphereIndexFused(**kw, grow_cap=True)
    reset = SphereIndexFused(**{**kw, "C": 128}, reset_at=128)
    for idx in (fixed, grow, reset):
        idx.observe(K, 1000)
        for n in range(1001, 4000):
            idx.observe(K, n)
            torch.cuda.synchronize()
    end = grow.end
    assert torch.all(grow.count <= grow._cap_now().unsqueeze(1))
    assert grow.n_c.max() < 0.7 * fixed.n_c.min()
    assert reset.resets >= 1 and torch.all(reset.n_c <= 128)
    for idx in (grow, reset):
        counts = torch.stack([torch.bincount(idx.labels[h, 1:end].long(), minlength=idx.C) for h in range(H_KV)])
        torch.testing.assert_close(idx.count, counts.float())
        assert torch.all(idx.labels[:, 1:end].long() < idx.n_c.unsqueeze(1))


def test_graph_decoder_with_reset_is_exact_at_full_budget():
    from ssa.models.graph_decode import GraphDecoder
    cfg = TinyCfg(head_dim=16, max_position_embeddings=1024, num_attention_heads=4, num_key_value_heads=2)
    P, T = 200, 600
    model = tiny_model(cfg, seed=6).to("cuda").eval()
    ids = torch.randint(0, cfg.vocab_size, (1, P + T + 1), device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(6))

    def run(mode, **kw):
        cache = model.alloc_cache(-(-(P + T + 2) // 16) * 16)
        out = []
        with torch.inference_mode():
            model(ids[:, :P], cache, start_pos=0)
            dec = GraphDecoder(model, cache, mode=mode, **kw)
            dec.prepare(P)
            dec.capture()
            for t in range(P, P + T):
                out.append(dec.step(ids[:, t:t + 1], t)[0, -1].float().clone())
                torch.cuda.synchronize()
        return torch.stack(out), dec

    ref, _ = run("dense")
    got, dec = run("cluster", budget=1.0, C=32, C_init=16, split_factor=2.0, reset_at=32, window=4,
                   delta=float("inf"), check_every=4)
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
    assert sum(i.resets for i in dec.attn) > 0
    got, dec = run("cluster", budget=1.0, C=64, C_init=16, split_factor=2.0, grow_cap=True, window=4,
                   delta=float("inf"), check_every=4)
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
    assert all(int(i.n_c.max()) < 48 for i in dec.attn)


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("cfg", [dict(C=64, C_init=16, split_factor=2.0), dict(C=32), dict(C=32, partition="random"),
                                 dict(C=64, C_init=16, split_factor=2.0, summary_bits=32)])
def test_one_insertion_launch_for_all_layers_gives_identical_decoding(capture, cfg):
    from ssa.models.graph_decode import GraphDecoder
    mc = TinyCfg(head_dim=16, max_position_embeddings=1024, num_attention_heads=4, num_key_value_heads=2)
    P, T, B = 200, 300, 2
    model = tiny_model(mc, seed=7).to("cuda").eval()
    ids = torch.randint(0, mc.vocab_size, (B, P + T + 1), device="cuda", generator=torch.Generator(device="cuda").manual_seed(7))

    def run(**kw):
        cache = model.alloc_cache(-(-(P + T + 2) // 16) * 16, max_batch=B)
        out = []
        with torch.inference_mode():
            model(ids[:, :P], cache, start_pos=0)
            dec = GraphDecoder(model, cache, mode="cluster", budget=0.3, window=4, delta=float("inf"), check_every=4,
                               **cfg, **kw)
            dec.prepare(P)
            if capture:
                dec.capture()
            for t in range(P, P + T):
                out.append(dec.step(ids[:, t:t + 1], t)[:, -1].float().clone())
                torch.cuda.synchronize()
        return torch.stack(out), dec

    a, da = run()
    b, db = run(fused_insert=True)
    assert db.insert_all is not None and len(db.attn) > 1
    assert torch.equal(a, b)
    for x, y in zip(da.attn, db.attn):
        end = int(x.end_dev[0])
        assert torch.equal(x.labels[:, :end], y.labels[:, :end]) and torch.equal(x.count, y.count)
        assert torch.equal(x.n_c, y.n_c) and x.splits == y.splits
    if "split_factor" in cfg:
        assert sum(i.splits for i in db.attn) > 0


def test_variance_trigger_in_the_kernel():
    from ssa.kernels.sphere_fused import SphereIndexFused
    Q, K = _stream(2600, seed=8)
    kw = {**KW, "split_factor": 0.0, "var_min": 6}
    fus, off = SphereIndexFused(**kw, var_factor=1.0), SphereIndexFused(**kw, var_factor=50.0)
    ref = SphereIndexGPU(**kw, var_factor=1.0)
    for idx in (fus, off, ref):
        idx.observe(K, 1000)
    assert fus.splits == 0 and torch.all(torch.isinf(fus.cap)) and torch.all(fus.vthr > 0)
    v0 = fus._spread()[fus.count > 0].mean()
    for n in range(1001, 2600):
        for idx in (fus, off, ref):
            idx.observe(K, n)
    assert off.splits == 0                                                # a threshold no cluster reaches: no splits
    assert fus.splits > 0 and torch.all(fus.n_c <= 256)
    assert (fus.n_c - ref.n_c.to(fus.n_c.device)).abs().float().mean() <= 0.15 * (ref.n_c.float().mean() - 64)   # about as many splits as the reference
    end = fus.end
    counts = torch.stack([torch.bincount(fus.labels[h, 1:end].long(), minlength=256) for h in range(H_KV)])
    torch.testing.assert_close(fus.count, counts.float())
    assert fus._spread()[fus.count >= 2].mean() < v0


@pytest.mark.parametrize("capture", [False, True])
def test_graph_decoder_with_variance_trigger_is_exact_at_full_budget(capture):
    from ssa.models.graph_decode import GraphDecoder
    cfg = TinyCfg(head_dim=16, max_position_embeddings=1024, num_attention_heads=4, num_key_value_heads=2)
    P, T = 200, 400
    model = tiny_model(cfg, seed=9).to("cuda").eval()
    ids = torch.randint(0, cfg.vocab_size, (1, P + T + 1), device="cuda", generator=torch.Generator(device="cuda").manual_seed(9))

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
    for fused in (False, True):
        got, dec = run("cluster", budget=1.0, C=64, C_init=16, var_factor=1.0, var_min=4, window=4, delta=float("inf"),
                       check_every=4, fused_insert=fused)
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
        assert sum(i.splits for i in dec.attn) > 0
