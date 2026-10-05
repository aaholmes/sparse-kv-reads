"""Label-weighted attention: the reference the Triton kernel must match.

Every cached position ``j`` of KV head ``g`` carries an integer label ``labels[g, j]`` (its
hypersphere bin, or a reserved label for the always-read exact set). Selection produces a
weight per label, ``w[g, label]`` (shared by the query heads of KV head ``g``) or
``w[h, label]`` (one row per query head). Then for query head ``h`` on KV head ``g``:

    out[h] = Σ_j w[label_j] e^{s_j} v_j / Σ_j w[label_j] e^{s_j},     s_j = q_h·k_j / √d

Rows with weight 0 contribute nothing and a kernel need not read them. Weight 1 on chosen
bins gives `cluster_skip`; weights ``1/(S π_b)`` on sampled bins give the tail-sampling
estimator. Inputs use the engine's cache layout ``K, V [H_kv, n, d]``, read in place.
"""

from __future__ import annotations

import math

import torch

from .geometry import _accum_dtype


def label_weighted_attention(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                             labels: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``q [H, d]``, ``K, V [H_kv, n, d]``, ``labels [H_kv, n]`` (int), ``w [H_kv or H, L]`` -> ``[H, d]``."""
    H, d = q.shape
    H_kv, n, _ = K.shape
    G = H // H_kv
    acc = _accum_dtype(q.dtype)
    lab = labels.long()
    if w.shape[0] == H_kv:
        wr = torch.gather(w.to(acc), 1, lab).repeat_interleave(G, 0)            # [H, n]
    elif w.shape[0] == H:
        wr = torch.gather(w.to(acc), 1, lab.repeat_interleave(G, 0))           # [H, n]
    else:
        raise ValueError(f"w has {w.shape[0]} rows; expected H_kv={H_kv} or H={H}")
    Kx = K.to(acc).repeat_interleave(G, 0)                                       # [H, n, d]
    s = torch.einsum("hd,hnd->hn", q.to(acc), Kx) / math.sqrt(d)
    logw = torch.where(wr > 0, wr.clamp_min(1e-300).log(), torch.full_like(wr, -math.inf))
    A = torch.softmax(s + logw, dim=-1)                                          # w e^s / Σ w e^s
    Vx = V.to(acc).repeat_interleave(G, 0)
    Vx = torch.where((wr > 0).unsqueeze(-1), Vx, torch.zeros_like(Vx))          # unread rows: no NaN leak
    return torch.einsum("hn,hnd->hd", A, Vx).to(V.dtype)
