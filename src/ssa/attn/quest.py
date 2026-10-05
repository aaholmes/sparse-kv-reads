"""Quest-style page selection (Tang et al., ICML 2024, arXiv:2406.10774), as a reference baseline.

The cache is split into pages of ``page`` consecutive tokens. Each page stores the element-wise
minimum and maximum of its keys; a query scores a page by ``Σ_i max(q_i·min_i, q_i·max_i)``, an
upper bound on ``q·k`` over the page's keys, and the top pages are read exactly up to the budget.

Two variants, both producing the ``labels``/``w`` inputs of ``label_weighted_attention``:

- ``quest_plain``: every position is paged and every query head selects its own pages (the KV head
  then reads the union). The paper does not state that recent tokens are always read, nor whether
  selection is per head; per-head selection is this implementation's choice. The paper keeps the
  first two layers dense ("we only apply Quest and all baselines on later layers"), which a caller
  can do by not applying it there.
- ``quest_matched``: token 0 and the last ``window`` tokens are always read, and the query heads of
  each KV head share one selection, ranked by the sum over heads of each head's softmax over page
  scores; these match ``cluster_skip``, so the remaining differences are pages by position compared
  with regions by direction, and Quest's bound compared with our score.

Pure PyTorch; pages are rebuilt from the cache on each call (reference, not a kernel).
"""

from __future__ import annotations

import math

import torch


def page_bounds(K: torch.Tensor, *, start: int, end: int, page: int):
    """Element-wise min and max of keys ``K[:, start:end]`` per page: ``[H_kv, pages, d]`` each."""
    H_kv, _, d = K.shape
    m = end - start
    P = -(-m // page)
    pad = P * page - m
    x = K[:, start:end]
    lo = torch.cat([x, x[:, -1:].expand(H_kv, pad, d)], 1) if pad else x       # pad with a real key
    return lo.view(H_kv, P, page, d).amin(2), lo.view(H_kv, P, page, d).amax(2)


def page_scores(q: torch.Tensor, kmin: torch.Tensor, kmax: torch.Tensor) -> torch.Tensor:
    """Upper bound on ``q·k`` per query head and page: ``[H, pages]``."""
    H, d = q.shape
    G = H // kmin.shape[0]
    qx = q.view(kmin.shape[0], G, 1, d)
    return torch.maximum(qx * kmin.unsqueeze(1), qx * kmax.unsqueeze(1)).sum(-1).view(H, -1)


def _take_until(rank_scores: torch.Tensor, sizes: torch.Tensor, need: int) -> torch.Tensor:
    """Per row of ``rank_scores [R, P]``: pages in decreasing score until their sizes reach ``need``."""
    R, P = rank_scores.shape
    order = rank_scores.argsort(1, descending=True)
    cum = sizes[order].cumsum(1)
    n_sel = (cum < need).sum(1) + 1
    rank = torch.empty_like(order).scatter_(1, order, torch.arange(P, device=order.device).expand(R, P))
    return rank < n_sel.unsqueeze(1)


def quest_labels_and_weights(q: torch.Tensor, K: torch.Tensor, *, n: int, budget: float, variant: str = "quest_plain",
                             page: int = 16, window: int = 64):
    """``labels [H_kv, n]`` (page index; label ``P`` = always read) and ``w`` (``[H, P+1]`` for
    ``quest_plain``, ``[H_kv, P+1]`` for ``quest_matched``) for ``q [H, d]``, ``K [H_kv, ≥n, d]``."""
    if variant not in ("quest_plain", "quest_matched"):
        raise ValueError(f"unknown variant {variant!r}")
    H, d = q.shape
    H_kv = K.shape[0]
    G = H // H_kv
    start, end = (0, n) if variant == "quest_plain" else (1, max(1, n - window))
    m = end - start
    P = -(-m // page) if m > 0 else 0
    labels = torch.full((H_kv, n), P, dtype=torch.long, device=K.device)
    if m > 0:
        labels[:, start:end] = (torch.arange(m, device=K.device) // page).expand(H_kv, m)
    rows = H if variant == "quest_plain" else H_kv
    w = torch.zeros(rows, P + 1, dtype=q.dtype, device=q.device)
    w[:, P] = 1.0
    if m <= 0:
        return labels, w
    need = math.ceil(budget * m)
    sizes = torch.full((P,), page, dtype=torch.long, device=K.device)
    sizes[-1] = m - (P - 1) * page
    kmin, kmax = page_bounds(K, start=start, end=end, page=page)
    s = page_scores(q, kmin, kmax)                                           # [H, P]
    if variant == "quest_plain":
        w[:, :P] = _take_until(s, sizes, need).to(w.dtype)
    else:
        z = torch.softmax(s / math.sqrt(d), dim=1).view(H_kv, G, P).sum(1)
        w[:, :P] = _take_until(z, sizes, need).to(w.dtype)
    return labels, w
