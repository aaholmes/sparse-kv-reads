"""ClusterKV-style selection: cosine k-means clusters of keys, ranked by q·centroid."""

from __future__ import annotations

import math

import torch

from ssa.attn.clusterkv import clusterkv_labels_and_weights, spherical_kmeans
from ssa.attn.labeled import label_weighted_attention
from _fixtures import Geom, make_qkv


def _qkv(n=400, seed=0):
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=n), seed=seed)
    return q, K.permute(1, 0, 2).contiguous(), V.permute(1, 0, 2).contiguous()


def _dense(q, K, V):
    G = q.shape[0] // K.shape[0]
    s = torch.einsum("hd,hnd->hn", q, K.repeat_interleave(G, 0)) / math.sqrt(q.shape[1])
    return torch.einsum("hn,hnd->hd", torch.softmax(s, -1), V.repeat_interleave(G, 0))


def test_kmeans_recovers_well_separated_directions():
    g = torch.Generator().manual_seed(0)
    centers = torch.nn.functional.normalize(torch.randn(4, 16, generator=g, dtype=torch.float64), dim=-1)
    truth = torch.arange(200) % 4
    X = (centers[truth] * 5 + 0.05 * torch.randn(200, 16, generator=g, dtype=torch.float64)).unsqueeze(0)
    assign, cent = spherical_kmeans(X, C=4, iters=10, seed=0)
    for c in range(4):                                       # every true group lands in one cluster
        assert assign[0, truth == c].unique().numel() == 1
    assert torch.allclose(cent.norm(dim=-1), torch.ones(1, 4, dtype=torch.float64))


def test_full_budget_equals_exact_attention():
    q, K, V = _qkv(seed=1)
    n = K.shape[1]
    for kw in (dict(variant="plain", clustered_end=n - 10), dict(variant="matched", C=8)):
        labels, w = clusterkv_labels_and_weights(q, K, n=n, budget=1.0, **kw)
        torch.testing.assert_close(label_weighted_attention(q, K, V, labels, w), _dense(q, K, V),
                                   rtol=1e-10, atol=1e-10)


def test_variants_have_their_own_always_read_tokens_and_selection_rows():
    q, K, V = _qkv(seed=2)
    n = K.shape[1]
    lab_p, w_p = clusterkv_labels_and_weights(q, K, n=n, budget=0.1, variant="plain", clustered_end=n - 10)
    exact_p = w_p.shape[1] - 1
    assert w_p.shape[0] == q.shape[0]                                     # one selection per query head
    assert torch.all(lab_p[:, :16] == exact_p) and torch.all(lab_p[:, n - 10:] == exact_p)
    lab_m, w_m = clusterkv_labels_and_weights(q, K, n=n, budget=0.1, variant="matched", C=8, window=64)
    exact_m = w_m.shape[1] - 1
    assert w_m.shape[0] == K.shape[0]                                     # shared per KV head
    assert torch.all(lab_m[:, 0] == exact_m) and torch.all(lab_m[:, n - 64:] == exact_m)
    assert w_p.shape[1] - 1 == round((n - 10 - 16) / 80)                  # ~one cluster per 80 tokens


def test_budget_controls_tokens_selected():
    q, K, V = _qkv(n=800, seed=3)
    n = K.shape[1]
    labels, w = clusterkv_labels_and_weights(q, K, n=n, budget=0.25, variant="matched", C=16, window=64)
    m = n - 64 - 1
    sel = (torch.gather(w, 1, labels) > 0).sum(1) - 65                   # minus the always-read tokens
    assert torch.all(sel >= 0.25 * m)
