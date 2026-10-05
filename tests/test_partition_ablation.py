"""Partition / centering / score ablation: its two corners must reproduce the known methods."""

from __future__ import annotations

import math

import torch

from ssa.attn.clusterkv import clusterkv_labels_and_weights
from ssa.attn.labeled import label_weighted_attention
from ssa.attn.sphere_gpu import SphereIndexGPU
from ssa.harness.partition_ablation import select
from _fixtures import Geom, make_qkv

C, W = 16, 8


def _qkv(n=400, seed=0):
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=n), seed=seed)
    return q, K.permute(1, 0, 2).contiguous() + 1.5, V.permute(1, 0, 2).contiguous()


def test_random_centered_length_score_is_the_fixed_direction_method():
    q, K, V = _qkv()
    n = K.shape[1]
    idx = SphereIndexGPU(C=C, window=W, delta=math.inf, capacity=n)
    idx.observe(K, n)
    lab_ref, w_ref = idx.labels_and_weights(q, n=n, budget=0.2)
    lab, w, _ = select(q, K, n=n, budget=0.2, partition="random", center=True, score="length", C=C, window=W)
    assert torch.equal(lab, lab_ref.long())
    torch.testing.assert_close(w, w_ref)


def test_kmeans_raw_qcentroid_is_clusterkv_matched():
    q, K, V = _qkv(seed=1)
    n = K.shape[1]
    lab_ref, w_ref = clusterkv_labels_and_weights(q, K, n=n, budget=0.2, variant="matched", C=C, window=W)
    lab, w, _ = select(q, K, n=n, budget=0.2, partition="kmeans", center=False, score="qcentroid", C=C, window=W)
    assert torch.equal(lab, lab_ref)
    torch.testing.assert_close(w, w_ref)


def test_every_combination_is_exact_at_full_budget_and_counts_its_summaries():
    q, K, V = _qkv(seed=2)
    n = K.shape[1]
    G = q.shape[0] // K.shape[0]
    s = torch.einsum("hd,hnd->hn", q, K.repeat_interleave(G, 0)) / math.sqrt(q.shape[1])
    dense = torch.einsum("hn,hnd->hd", torch.softmax(s, -1), V.repeat_interleave(G, 0))
    for partition in ("random", "kmeans"):
        for center in (True, False):
            for score in ("length", "qcentroid"):
                lab, w, summary = select(q, K, n=n, budget=1.0, partition=partition, center=center, score=score,
                                         C=C, window=W)
                torch.testing.assert_close(label_weighted_attention(q, K, V, lab, w), dense, rtol=1e-10, atol=1e-10)
                assert summary == (C * (1 + 2 / 16) if score == "length" else C)


def test_stale_fit_uses_only_early_keys_and_assigns_the_rest_to_the_nearest_centroid():
    from ssa.attn.clusterkv import spherical_kmeans
    from ssa.harness.partition_ablation import group
    q, K, V = _qkv(n=600, seed=3)
    n = 600
    fresh = group(K, n, partition="kmeans", center=True, C=C, window=W)
    same = group(K, n, partition="kmeans", center=True, C=C, window=W, fit_tokens=n - W - 1)
    assert torch.equal(fresh["assign"], same["assign"])                  # fitting on every binned key = fresh
    stale = group(K, n, partition="kmeans", center=True, C=C, window=W, fit_tokens=200)
    X = K[:, 1:n - W] - K[:, 1:n - W].mean(1, keepdim=True)
    _, cent = spherical_kmeans(X[:, :200], C=C)
    nearest = torch.einsum("hmd,hcd->hmc", torch.nn.functional.normalize(X, dim=-1), cent).argmax(-1)
    assert torch.equal(stale["assign"], nearest)
    assert not torch.equal(stale["assign"], fresh["assign"])
