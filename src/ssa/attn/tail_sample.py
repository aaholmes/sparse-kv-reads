"""Sampling the unselected bins.

Top-k-style selection reads the chosen ("head") bins exactly and drops the rest. Here S of
the remaining ("tail") bins are drawn by systematic sampling from a proposal ``π``, and a
drawn bin gets weight ``c_b / (S π_b)``, where ``c_b`` is how many of the S draws landed on
it. Every tail bin with ``π_b > 0`` then has expected weight exactly 1, so both sums of the
label-weighted estimator,

    N̂ = Σ_j w[label_j] e^{s_j} v_j     and     Ẑ = Σ_j w[label_j] e^{s_j},

are unbiased for the full-cache sums; their ratio (the attention output) has bias O(1/S).
Reading the drawn bins costs their keys and values, and never any key outside them.

Proposal: estimated bin mass ``count_b · exp(scale · e_b)`` per query head (``e_b`` is the
max-score estimate used for selection), normalized per head and averaged over the query
heads of the KV head, then mixed with a uniform floor ``α`` over the tail bins.
"""

from __future__ import annotations

import torch


def proposal_from_scores(e: torch.Tensor, count: torch.Tensor, *, scale: float) -> torch.Tensor:
    """``e [H_kv, G, C]`` max-score estimates (−∞ for empty bins), ``count [H_kv, C]`` →
    proposal ``[H_kv, C]`` (rows sum to 1): mean over heads of normalized ``count·e^{scale·e}``."""
    z = e * scale
    z = z - z.max(dim=2, keepdim=True).values
    m = count.unsqueeze(1) * torch.exp(z)
    m = torch.nan_to_num(m, nan=0.0)
    m = m / m.sum(dim=2, keepdim=True).clamp_min(1e-300)
    return m.mean(dim=1)


def tail_sample_weights(prop: torch.Tensor, count: torch.Tensor, head: torch.Tensor, *, S: int,
                        alpha: float, generator: torch.Generator | None = None, return_pi: bool = False):
    """Per-bin weights ``[H_kv, C]``: 1 for non-empty head bins, ``c_b/(S π_b)`` for drawn tail
    bins, 0 otherwise. ``prop`` need not be normalized; empty bins are never drawn."""
    live = count > 0
    headw = (head & live).to(prop.dtype)
    tail = (~head) & live
    if S <= 0:
        return (headw, torch.zeros_like(prop)) if return_pi else headw
    n_tail = tail.sum(1, keepdim=True).clamp_min(1)
    p = torch.where(tail, prop, torch.zeros_like(prop))
    p = p / p.sum(1, keepdim=True).clamp_min(1e-300)
    pi = torch.where(tail, (1 - alpha) * p + alpha / n_tail, torch.zeros_like(prop))
    has_tail = tail.any(1, keepdim=True)
    F = pi.cumsum(1)
    R, C = pi.shape
    u = torch.rand(R, 1, generator=generator, dtype=prop.dtype, device="cpu").to(prop.device) / S
    T = (u + torch.arange(S, dtype=prop.dtype, device=prop.device) / S) * F[:, -1:]      # < total
    idx = torch.searchsorted(F.contiguous(), T.contiguous()).clamp_(max=C - 1)
    c = torch.zeros_like(prop).scatter_add_(1, idx, torch.ones_like(T))
    c = torch.where(has_tail, c, torch.zeros_like(c))
    wt = torch.where(tail & (c > 0), c / (S * pi.clamp_min(1e-300)), torch.zeros_like(prop))
    w = headw + wt
    return (w, pi) if return_pi else w
