"""Quest-style page selection: per-page key bounds, page ranking, and the label/weight interface."""

from __future__ import annotations

import math

import torch

from ssa.attn.labeled import label_weighted_attention
from ssa.attn.quest import page_bounds, page_scores, quest_labels_and_weights
from _fixtures import Geom, make_qkv


def _qkv(n=300, seed=0):
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=n), seed=seed)
    return q, K.permute(1, 0, 2).contiguous(), V.permute(1, 0, 2).contiguous()     # [H_kv, n, d]


def _dense(q, K, V):
    G = q.shape[0] // K.shape[0]
    s = torch.einsum("hd,hnd->hn", q, K.repeat_interleave(G, 0)) / math.sqrt(q.shape[1])
    return torch.einsum("hn,hnd->hd", torch.softmax(s, -1), V.repeat_interleave(G, 0))


def test_page_score_bounds_every_key_in_the_page():
    q, K, _ = _qkv()
    kmin, kmax = page_bounds(K, start=0, end=K.shape[1], page=16)
    s = page_scores(q, kmin, kmax)                                  # [H, pages]
    G = q.shape[0] // K.shape[0]
    qk = torch.einsum("hd,hnd->hn", q, K.repeat_interleave(G, 0))
    for p in range(s.shape[1]):
        best = qk[:, p * 16:(p + 1) * 16].max(1).values
        assert torch.all(s[:, p] >= best - 1e-9)


def test_full_budget_reads_everything_and_equals_exact_attention():
    q, K, V = _qkv(seed=1)
    for variant in ("quest_plain", "quest_matched"):
        labels, w = quest_labels_and_weights(q, K, n=K.shape[1], budget=1.0, variant=variant)
        got = label_weighted_attention(q, K, V, labels, w)
        torch.testing.assert_close(got, _dense(q, K, V), rtol=1e-10, atol=1e-10)


def test_variants_differ_in_selection_rows_and_exact_set():
    q, K, V = _qkv(seed=2)
    n = K.shape[1]
    lab_q, w_q = quest_labels_and_weights(q, K, n=n, budget=0.1, variant="quest_plain")
    lab_m, w_m = quest_labels_and_weights(q, K, n=n, budget=0.1, variant="quest_matched", window=64)
    assert w_q.shape[0] == q.shape[0]                               # one selection per query head
    assert w_m.shape[0] == K.shape[0]                               # one shared selection per KV head
    exact = w_m.shape[1] - 1
    assert torch.all(lab_m[:, 0] == exact) and torch.all(lab_m[:, n - 64:] == exact)
    assert torch.all(w_m[:, exact] == 1)


def test_budget_controls_how_many_tokens_each_head_selects():
    q, K, V = _qkv(n=640, seed=3)
    n = K.shape[1]
    labels, w = quest_labels_and_weights(q, K, n=n, budget=0.25, variant="quest_plain")
    G = q.shape[0] // K.shape[0]
    lab = labels.long().repeat_interleave(G, 0)
    per_head = (torch.gather(w, 1, lab) > 0).sum(1)
    assert torch.all(per_head >= 0.25 * n) and torch.all(per_head <= 0.25 * n + 16)
