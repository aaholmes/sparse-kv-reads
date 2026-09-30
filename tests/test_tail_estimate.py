"""Estimating the dropped bins from per-bin sums."""

from __future__ import annotations

import math

import torch

from ssa.attn.tail_estimate import SphereIndexTail, attend_with_tail

D, W = 8, 4
C = 2 * D                                     # "cross" directions: ±e_i, one bin per signed axis


def _cross_cache(seed=0, n=200, H_kv=2, equal_lengths=True):
    """Keys μ + L·(±e_i): every bin holds one exact direction, and the ± pairs make the
    binned keys' mean exactly μ, so centering recovers L·(±e_i)."""
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(H_kv, D, generator=g, dtype=torch.float64)
    K = torch.empty(H_kv, n, D, dtype=torch.float64)
    K[:, 0] = torch.randn(H_kv, D, generator=g, dtype=torch.float64)
    axis_len = 1.0 + 2.0 * torch.rand(H_kv, D, generator=g, dtype=torch.float64)
    for j in range(1, n, 2):
        i = torch.randint(0, D, (H_kv,), generator=g)
        L = axis_len[torch.arange(H_kv), i]
        if not equal_lengths:
            L = L * (0.5 + torch.rand(H_kv, generator=g, dtype=torch.float64))
        e = torch.nn.functional.one_hot(i, D).double()
        K[:, j] = mu + L.unsqueeze(1) * e
        if j + 1 < n:
            K[:, j + 1] = mu - L.unsqueeze(1) * e
    V = torch.randn(H_kv, n, D, generator=g, dtype=torch.float64)
    return K, V


def _dense(q, K, V, alive=None):
    H, d = q.shape
    G = H // K.shape[0]
    s = torch.einsum("hd,hnd->hn", q, K.repeat_interleave(G, 0)) / math.sqrt(d)
    if alive is not None:
        s = s.masked_fill(~alive, -math.inf)
    return torch.einsum("hn,hnd->hd", torch.softmax(s, -1), V.repeat_interleave(G, 0))


def _index(K, V, n, **kw):
    idx = SphereIndexTail(C=C, window=W, delta=math.inf, capacity=K.shape[1], kind="cross", **kw)
    idx.observe(K, V, n)
    return idx


def _query(seed, H=4):
    return torch.randn(H, D, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) * 2


def test_first_order_estimate_is_exact_when_each_bin_has_one_direction_and_length():
    K, V = _cross_cache()
    n = K.shape[1]
    idx = _index(K, V, n, t0=0.3)
    q = _query(1)
    labels, w = idx.labels_and_weights(q, n=n, budget=0.1)
    assert (w[:, :C] == 0).any() and (w[:, :C] == 1).any()      # some bins read, some estimated
    got = attend_with_tail(idx, q, K[:, :n], V[:, :n], labels, w, order=1)
    torch.testing.assert_close(got, _dense(q, K[:, :n], V[:, :n]), rtol=1e-10, atol=1e-10)


def test_second_order_estimate_reduces_to_first_order_with_one_direction_per_bin():
    K, V = _cross_cache()
    n = K.shape[1]
    idx = _index(K, V, n, t0=0.3)
    q = _query(1)
    labels, w = idx.labels_and_weights(q, n=n, budget=0.1)
    got = attend_with_tail(idx, q, K[:, :n], V[:, :n], labels, w, order=2)
    torch.testing.assert_close(got, _dense(q, K[:, :n], V[:, :n]), rtol=1e-8, atol=1e-8)


def test_second_order_raises_the_tail_mass_when_directions_spread():
    K, V = _cross_cache(seed=9)
    K[:, 1:] = K[:, 1:] + 0.3 * torch.randn(K[:, 1:].shape, generator=torch.Generator().manual_seed(9),
                                            dtype=torch.float64)
    n = K.shape[1]
    idx = _index(K, V, n)
    q = _query(2)
    _, w = idx.labels_and_weights(q, n=n, budget=0.1)
    from ssa.attn.tail_estimate import tail_log_estimates
    z1 = torch.logsumexp(tail_log_estimates(idx, q, w, order=2, shrink=False)[0], -1)
    z0 = torch.logsumexp(tail_log_estimates(idx, q, w, order=1)[0], -1)
    assert torch.all(z1 >= z0)


def test_zero_order_estimate_is_not_exact_away_from_the_reference():
    K, V = _cross_cache()
    n = K.shape[1]
    idx = _index(K, V, n, t0=0.3)
    q = _query(1)
    labels, w = idx.labels_and_weights(q, n=n, budget=0.1)
    got = attend_with_tail(idx, q, K[:, :n], V[:, :n], labels, w, order=0)
    assert (got - _dense(q, K[:, :n], V[:, :n])).abs().max() > 1e-3


def test_full_budget_reads_everything_and_estimates_nothing():
    K, V = _cross_cache(seed=2, equal_lengths=False)
    n = K.shape[1]
    idx = _index(K, V, n)
    q = _query(3)
    labels, w = idx.labels_and_weights(q, n=n, budget=1.0)
    got = attend_with_tail(idx, q, K[:, :n], V[:, :n], labels, w, order=1)
    torch.testing.assert_close(got, _dense(q, K[:, :n], V[:, :n]), rtol=1e-10, atol=1e-10)


def _sums_from_scratch(idx, K, V):
    """Per-bin sums over the binned, non-evicted positions, using the index's labels and mean."""
    H_kv = K.shape[0]
    Z = torch.zeros(H_kv, C, dtype=torch.float64)
    L = torch.zeros_like(Z)
    M = torch.zeros_like(Z)
    N = torch.zeros(H_kv, C, D, dtype=torch.float64)
    cnt = torch.zeros_like(Z)
    for h in range(H_kv):
        for j in range(idx.start, idx.end):
            c = int(idx.labels[h, j])
            m = (K[h, j] - idx.mu_ref[h]).norm()
            e = torch.exp(idx.t0[h] * m)
            Z[h, c] += e
            L[h, c] += m * e
            M[h, c] += m * m * e
            N[h, c] += e * V[h, j]
            cnt[h, c] += 1
    return Z, L, N, cnt, M


def test_incremental_sums_match_a_recount():
    K, V = _cross_cache(seed=4, equal_lengths=False)
    idx = SphereIndexTail(C=C, window=W, delta=math.inf, capacity=K.shape[1], kind="cross", t0=0.2)
    for n in range(60, 200, 7):
        idx.observe(K, V, n)
    Z, L, N, cnt, M = _sums_from_scratch(idx, K, V)
    torch.testing.assert_close(idx.tz, Z)
    torch.testing.assert_close(idx.tl, L)
    torch.testing.assert_close(idx.tq, M)
    torch.testing.assert_close(idx.tn, N)
    torch.testing.assert_close(idx.count, cnt)


def test_recentering_rebuilds_the_sums():
    K, V = _cross_cache(seed=5, equal_lengths=False)
    K = K + torch.linspace(0, 3, K.shape[1], dtype=torch.float64).view(1, -1, 1)   # drifting mean
    idx = SphereIndexTail(C=C, window=W, delta=0.0, capacity=K.shape[1], kind="cross", t0=0.2,
                          check_every=1)
    for n in range(60, 200, 5):
        idx.observe(K, V, n)
    assert idx.rebuilds > 0
    Z, L, N, _, M = _sums_from_scratch(idx, K, V)
    torch.testing.assert_close(idx.tz, Z)
    torch.testing.assert_close(idx.tn, N)


def test_eviction_subtracts_old_keys_from_every_bin_sum():
    K, V = _cross_cache(seed=6, equal_lengths=False)
    n = K.shape[1]
    idx = _index(K, V, n, t0=0.2)
    sum_dir_before = idx.sum_dir.clone()
    idx.evict(K, V, 81)
    assert idx.start == 81
    Z, L, N, cnt, M = _sums_from_scratch(idx, K, V)
    torch.testing.assert_close(idx.tz, Z)
    torch.testing.assert_close(idx.tl, L)
    torch.testing.assert_close(idx.tq, M)
    torch.testing.assert_close(idx.tn, N)
    torch.testing.assert_close(idx.count, cnt)
    assert not torch.equal(idx.sum_dir, sum_dir_before)
    torch.testing.assert_close(idx.ksum, K[:, 81:idx.end].sum(1))


def test_attention_after_eviction_matches_dense_over_the_remaining_tokens():
    K, V = _cross_cache(seed=7)
    n = K.shape[1]
    idx = _index(K, V, n, t0=0.3)
    idx.evict(K, V, 61)
    q = _query(8)
    labels, w = idx.labels_and_weights(q, n=n, budget=0.1)
    got = attend_with_tail(idx, q, K[:, :n], V[:, :n], labels, w, order=1)
    alive = torch.ones(q.shape[0], n, dtype=torch.bool)
    alive[:, 1:61] = False
    torch.testing.assert_close(got, _dense(q, K[:, :n], V[:, :n], alive), rtol=1e-10, atol=1e-10)
