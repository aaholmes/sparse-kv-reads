"""Exact attention implementations: ``dense`` (ground truth) and ``topk`` (biased).

``dense`` is the quantity every estimator must match in expectation. ``topk`` is
the deterministic biased baseline (keep the k highest-weight keys, renormalise).
"""

from __future__ import annotations

import torch

from . import register
from .geometry import attn_weights, gqa_expand, weighted_sum


@register("dense")
def dense(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, **cfg) -> torch.Tensor:
    """Exact ``softmax(qKᵀ/√d) V`` -> ``[H, d]``."""
    A = attn_weights(q, K)
    V_exp = gqa_expand(V, q.shape[0])
    return weighted_sum(A, V_exp)


@register("topk")
def topk(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, *, k: int, **cfg) -> torch.Tensor:
    """Keep the ``k`` highest-weight keys per head, renormalise, exact sum.

    Biased baseline — its expectation is **not** ``dense`` (so it is not tested for unbiasedness).
    With ``k >= n_k`` it reduces to ``dense``.
    """
    H = q.shape[0]
    A = attn_weights(q, K)
    n_k = A.shape[-1]
    k = min(k, n_k)
    top_vals, top_idx = torch.topk(A, k, dim=-1)
    top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
    V_exp = gqa_expand(V, H)
    # gather the top-k value rows per head: [H, k, d]
    d = V_exp.shape[-1]
    V_perm = V_exp.permute(1, 0, 2)
    V_top = torch.gather(V_perm, 1, top_idx.unsqueeze(-1).expand(H, k, d))
    return torch.einsum("hk,hkd->hd", top_vals.to(V_top.dtype), V_top)
