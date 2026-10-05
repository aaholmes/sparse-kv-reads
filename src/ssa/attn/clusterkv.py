"""ClusterKV-style selection (arXiv:2412.03213), as a reference baseline.

Keys are grouped by direction with k-means on cosine similarity (no mean subtraction); a query
scores each cluster by ``q·centroid`` (unit centroids) and the top clusters are read exactly up
to the budget. Two variants, both producing the ``labels``/``w`` inputs of
``label_weighted_attention``:

- ``plain``: ClusterKV's reported settings: about one cluster per 80 tokens, the first 16 tokens
  always read, and each query head selecting its own clusters (the KV head reads the union).
  ClusterKV clusters new tokens every 320 decode steps; tokens after ``clustered_end`` (those not
  yet clustered) are treated as always read, which is this implementation's assumption.
- ``matched``: ``voronoi_skip``'s always-read set (token 0 and the last ``window`` tokens) and one
  shared selection per KV head (ranked by the sum over its query heads of each head's softmax over
  cluster scores), with ``C`` clusters; the remaining differences are k-means clusters compared
  with fixed directions (and our mean-centering), and ``q·centroid`` compared with our score.

Pure PyTorch; the clustering is recomputed on each call (reference, not a kernel).
"""

from __future__ import annotations

import math

import torch


def spherical_kmeans(X: torch.Tensor, *, C: int, iters: int = 10, seed: int = 0):
    """k-means on cosine similarity, batched over heads: ``X [h, m, d]`` -> ``assign [h, m]``,
    unit ``centroids [h, C, d]``. Starts from the points at ``C`` random positions, the same positions for every head (so a head's
    result does not depend on where it sits in a batch);
    an empty cluster keeps its previous centroid."""
    h, m, d = X.shape
    Xn = torch.nn.functional.normalize(X, dim=-1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    init = torch.randperm(m, generator=g)[torch.arange(C) % m].expand(h, C).to(X.device)   # same for every head
    cent = torch.gather(Xn, 1, init.unsqueeze(-1).expand(h, C, d)).clone()
    assign = None
    for _ in range(iters):
        assign = torch.einsum("hmd,hcd->hmc", Xn, cent).argmax(-1)
        sums = torch.zeros(h, C, d, dtype=X.dtype, device=X.device).scatter_add_(
            1, assign.unsqueeze(-1).expand(h, m, d), Xn)
        cnt = torch.zeros(h, C, dtype=X.dtype, device=X.device).scatter_add_(1, assign, torch.ones_like(assign, dtype=X.dtype))
        new = torch.nn.functional.normalize(sums, dim=-1)
        cent = torch.where((cnt > 0).unsqueeze(-1), new, cent)
    assign = torch.einsum("hmd,hcd->hmc", Xn, cent).argmax(-1)
    return assign, cent


def _take_until(rank_scores: torch.Tensor, sizes: torch.Tensor, need: int) -> torch.Tensor:
    """Per row: clusters in decreasing score until their sizes (per row) reach ``need``."""
    R, C = rank_scores.shape
    order = rank_scores.argsort(1, descending=True)
    cum = torch.gather(sizes, 1, order).cumsum(1)
    n_sel = (cum < need).sum(1) + 1
    rank = torch.empty_like(order).scatter_(1, order, torch.arange(C, device=order.device).expand(R, C))
    return rank < n_sel.unsqueeze(1)


def clusterkv_labels_and_weights(q: torch.Tensor, K: torch.Tensor, *, n: int, budget: float, variant: str = "plain",
                                 C: int | None = None, window: int = 64, sink: int = 16,
                                 clustered_end: int | None = None, tokens_per_cluster: int = 80,
                                 iters: int = 10, seed: int = 0, clusters=None):
    """``labels [H_kv, n]`` (cluster id; label ``C`` = always read) and ``w`` (``[H, C+1]`` for
    ``plain``, ``[H_kv, C+1]`` for ``matched``) for ``q [H, d]``, ``K [H_kv, ≥n, d]``. ``clusters``
    reuses an ``(assign, centroids)`` pair from ``spherical_kmeans`` on the same keys and range."""
    if variant not in ("plain", "matched"):
        raise ValueError(f"unknown variant {variant!r}")
    H, d = q.shape
    H_kv = K.shape[0]
    G = H // H_kv
    if variant == "plain":
        start, end = sink, (n if clustered_end is None else min(clustered_end, n))
    else:
        start, end = 1, max(1, n - window)
    m = max(0, end - start)
    if C is None:
        C = max(1, round(m / tokens_per_cluster))
    C = min(C, max(1, m))
    labels = torch.full((H_kv, n), C, dtype=torch.long, device=K.device)
    rows = H if variant == "plain" else H_kv
    w = torch.zeros(rows, C + 1, dtype=q.dtype, device=q.device)
    w[:, C] = 1.0
    if m == 0:
        return labels, w
    assign, cent = clusters if clusters is not None else spherical_kmeans(K[:, start:end], C=C, iters=iters, seed=seed)
    labels[:, start:end] = assign
    sizes = torch.zeros(H_kv, C, dtype=torch.long, device=K.device).scatter_add_(1, assign, torch.ones_like(assign))
    s = torch.einsum("hgd,hcd->hgc", q.view(H_kv, G, d), cent.to(q.dtype))       # q·centroid
    need = math.ceil(budget * m)
    if variant == "plain":
        w[:, :C] = _take_until(s.reshape(H, C), sizes.repeat_interleave(G, 0), need).to(w.dtype)
    else:
        z = torch.softmax(s / math.sqrt(d), dim=2).sum(1)
        w[:, :C] = _take_until(z, sizes, need).to(w.dtype)
    return labels, w
