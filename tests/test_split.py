"""Split-on-overflow (v1): a fitted cluster that grows past its cap is split in two."""

from __future__ import annotations

import math

import torch

from ssa.attn.labeled import label_weighted_attention
from ssa.attn.sphere_gpu import SphereIndexGPU
from ssa.attn.tail_estimate import SphereIndexTail
from ssa.models.patch import install, uninstall
from _fixtures import Geom, make_qkv
from _tiny_model import TinyCfg, tiny_model

W = 8
KW = dict(C=64, C_init=16, split_factor=2.0, window=W, delta=math.inf, capacity=1200, partition="kmeans")


def _qkv(n=1100, seed=0):
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=n), seed=seed)
    return q, K.permute(1, 0, 2).contiguous() + 1.0, V.permute(1, 0, 2).contiguous()


def _recount(idx, K):
    """Cluster statistics recomputed from the labels."""
    H_kv, C, d = idx.H_kv, idx.C, idx.d
    cnt = torch.zeros(H_kv, C, dtype=torch.float64)
    sd = torch.zeros(H_kv, C, d, dtype=torch.float64)
    mx = torch.zeros(H_kv, C, dtype=torch.float64)
    mn = torch.full((H_kv, C), math.inf, dtype=torch.float64)
    for h in range(H_kv):
        for j in range(idx.start, idx.end):
            c = int(idx.labels[h, j])
            kr = K[h, j] - idx.mu_ref[h]
            m = kr.norm()
            cnt[h, c] += 1
            sd[h, c] += kr / m
            mx[h, c] = max(mx[h, c], m)
            mn[h, c] = min(mn[h, c], m)
    return cnt, sd, mx, mn


def test_clusters_stay_under_the_cap_and_statistics_match_a_recount():
    q, K, V = _qkv()
    idx = SphereIndexGPU(**KW)
    idx.observe(K, 500)
    n0 = idx.n_c.clone()
    for n in range(520, 1100, 20):
        idx.observe(K, n)
    assert idx.splits > 0 and torch.all(idx.n_c > n0) and torch.all(idx.n_c <= 64)
    slots = torch.arange(64).unsqueeze(0)
    assert torch.all(idx.count[slots >= idx.n_c.unsqueeze(1)] == 0)                 # spare slots are empty
    assert torch.all(idx.labels[:, 1:idx.end] < idx.n_c.unsqueeze(1))
    assert torch.all(idx.count <= idx.cap.unsqueeze(1) + 20)                        # checked after each batch of 20 keys
    cnt, sd, mx, mn = _recount(idx, K)
    torch.testing.assert_close(idx.count, cnt)
    torch.testing.assert_close(idx.sum_dir, sd, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(idx.mmax, mx)
    torch.testing.assert_close(idx.mmin, mn)


def test_selection_at_full_budget_is_exact_with_spare_slots():
    q, K, V = _qkv(seed=1)
    idx = SphereIndexGPU(**KW)
    for n in (500, 800, 1100):
        idx.observe(K, n)
    labels, w = idx.labels_and_weights(q, n=1100, budget=1.0)
    G = q.shape[0] // K.shape[0]
    s = torch.einsum("hd,hnd->hn", q, K.repeat_interleave(G, 0)) / math.sqrt(q.shape[1])
    dense = torch.einsum("hn,hnd->hd", torch.softmax(s, -1), V.repeat_interleave(G, 0))
    torch.testing.assert_close(label_weighted_attention(q, K, V, labels, w), dense, rtol=1e-10, atol=1e-10)


def test_without_a_split_factor_nothing_splits():
    q, K, V = _qkv(seed=2)
    idx = SphereIndexGPU(**{**KW, "split_factor": 0.0})
    for n in range(500, 1100, 50):
        idx.observe(K, n)
    assert idx.splits == 0 and torch.all(idx.n_c == 16)


def test_tail_sums_follow_the_splits():
    q, K, V = _qkv(seed=3)
    idx = SphereIndexTail(**KW)
    for n in range(500, 1100, 30):
        idx.observe(K, V, n)
    assert idx.splits > 0
    tz = torch.zeros_like(idx.tz)
    tn = torch.zeros_like(idx.tn)
    for h in range(idx.H_kv):
        for j in range(idx.start, idx.end):
            c = int(idx.labels[h, j])
            e = torch.exp(idx.t0[h] * (K[h, j] - idx.mu_ref[h]).norm())
            tz[h, c] += e
            tn[h, c] += e * V[h, j]
    torch.testing.assert_close(idx.tz, tz)
    torch.testing.assert_close(idx.tn, tn, rtol=1e-9, atol=1e-9)


def test_engine_op_with_splitting_is_exact_at_full_budget_and_counts_active_clusters():
    cfg = TinyCfg(head_dim=16, max_position_embeddings=256, num_attention_heads=4, num_key_value_heads=2)
    model = tiny_model(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 230), generator=torch.Generator().manual_seed(0))

    def decode():
        cache = model.alloc_cache(231)
        with torch.inference_mode():
            model(ids[:, :100], cache, start_pos=0)
            return torch.stack([model(ids[:, t:t + 1], cache)[0, -1].float() for t in range(100, 230)])

    ref = decode()
    stats = install(model, "cluster_tail", budget=1.0, order="drop", C=32, C_init=8, split_factor=2.0, window=4,
                    capacity=256)
    got = decode()
    uninstall(model)
    torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)
    full = (2 * stats.reads_sum + stats.steps * 32 * (1 + 2 / 16)) / (2 * stats.n_k_sum)   # if every slot counted
    assert stats.kv_read_fraction < full


def test_nothing_is_over_the_cap_once_the_prompt_is_clustered():
    q, K, V = _qkv(seed=4)
    g = torch.Generator().manual_seed(4)
    lump = torch.randn(2, 1, 16, generator=g, dtype=K.dtype)
    K[:, 100:500] = 1.0 + 3 * lump + 0.3 * torch.randn(2, 400, 16, generator=g, dtype=K.dtype)   # 400 near-parallel keys
    idx = SphereIndexGPU(**{**KW, "C": 128, "split_factor": 1.2})
    idx.observe(K, 900)
    assert torch.all(idx.count <= idx.cap.unsqueeze(1))                 # no backlog left for decoding
    cnt, sd, mx, mn = _recount(idx, K)
    torch.testing.assert_close(idx.count, cnt)
    torch.testing.assert_close(idx.sum_dir, sd, rtol=1e-9, atol=1e-9)
    assert idx.splits > 0


def test_a_split_leaves_two_substantial_halves():
    from ssa.attn.sphere_gpu import bisect_directions
    g = torch.Generator().manual_seed(0)
    main = torch.nn.functional.normalize(torch.randn(1, 16, generator=g, dtype=torch.float64), dim=-1)
    pts = torch.nn.functional.normalize(main + 0.05 * torch.randn(99, 16, generator=g, dtype=torch.float64), dim=-1)
    outlier = torch.nn.functional.normalize(-main + 0.01 * torch.randn(1, 16, generator=g, dtype=torch.float64), dim=-1)
    side, cents = bisect_directions(torch.cat([pts, outlier]))          # plain 2-means would peel off the outlier
    share = side.double().mean()
    assert 0.25 <= share <= 0.75
    assert torch.allclose(cents.norm(dim=-1), torch.ones(2, dtype=torch.float64))


def test_later_splits_are_few_and_only_follow_overflow():
    q, K, V = _qkv(seed=5)
    idx = SphereIndexGPU(**{**KW, "C": 128})
    idx.observe(K, 900)
    s0 = idx.splits
    for n in range(901, 1100):
        idx.observe(K, n)
    assert idx.splits - s0 <= 2 * 2 * 199 / float(idx.cap.min())       # at most ~one split per cap/2 new keys per head
    assert torch.all(idx.count <= idx.cap.unsqueeze(1))


def test_a_cap_that_grows_with_the_context_keeps_the_cluster_count_bounded():
    q, K, V = _qkv(n=3000, seed=6)
    kw = {**KW, "C": 256, "capacity": 3100}
    fixed, grow = SphereIndexGPU(**kw), SphereIndexGPU(**kw, grow_cap=True)
    for idx in (fixed, grow):
        idx.observe(K, 500)
        for n in range(510, 3000, 10):
            idx.observe(K, n)
    assert torch.all(grow._cap_now() > 5 * grow.cap)                         # six times the keys of the fit
    assert torch.all(grow.count <= grow._cap_now().unsqueeze(1) + 10)
    assert torch.all(grow.n_c <= 3 * 16) and torch.all(fixed.n_c > 4 * 16)
    cnt, sd, mx, mn = _recount(grow, K)
    torch.testing.assert_close(grow.count, cnt)


def test_reset_refits_a_head_once_its_cluster_count_has_doubled():
    q, K, V = _qkv(n=3000, seed=7)
    idx = SphereIndexGPU(**{**KW, "C": 32, "capacity": 3100}, reset_at=32)
    idx.observe(K, 500)
    cap0 = idx.cap.clone()
    seen = 0
    for n in range(510, 3000, 10):
        idx.observe(K, n)
        seen = max(seen, int(idx.n_c.max()))
        assert torch.all(idx.n_c < 32)                                       # a head at the limit is refitted in that step
    assert idx.resets >= 2 and seen >= 28
    assert torch.all(idx.cap > cap0)                                         # the cap is set again from the keys at the reset
    cnt, sd, mx, mn = _recount(idx, K)
    torch.testing.assert_close(idx.count, cnt)
    torch.testing.assert_close(idx.sum_dir, sd, rtol=1e-9, atol=1e-9)
    labels, w = idx.labels_and_weights(q, n=2990, budget=1.0)
    G = q.shape[0] // K.shape[0]
    s = torch.einsum("hd,hnd->hn", q, K[:, :2990].repeat_interleave(G, 0)) / math.sqrt(q.shape[1])
    dense = torch.einsum("hn,hnd->hd", torch.softmax(s, -1), V[:, :2990].repeat_interleave(G, 0))
    torch.testing.assert_close(label_weighted_attention(q, K[:, :2990], V[:, :2990], labels, w), dense, rtol=1e-10, atol=1e-10)


def test_an_absolute_cap_does_not_depend_on_the_prompt_length():
    q, K, V = _qkv(n=2000, seed=8)
    kw = {**KW, "C": 256, "capacity": 2100, "split_factor": 0.0, "cap_keys": 24}
    short, long = SphereIndexGPU(**kw), SphereIndexGPU(**kw)
    short.observe(K, 200)
    long.observe(K, 1500)
    assert torch.all(short.cap == 24) and torch.all(long.cap == 24)
    assert torch.all(long.count <= 24) and long.splits > 0               # 1,492 keys in 16 clusters: split at the fit
    for n in range(210, 1500, 10):
        short.observe(K, n)
    assert torch.all(short.count <= 24 + 10)
    assert (short.n_c.float().mean() / long.n_c.float().mean() - 1).abs() < 0.35   # similar counts at equal context
    cnt, sd, mx, mn = _recount(short, K)
    torch.testing.assert_close(short.count, cnt)
