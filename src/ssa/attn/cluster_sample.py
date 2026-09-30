"""Cluster sampling — skip key reads with a read-free head plus an importance-sampled tail.

For one KV head and its group of query heads:

  - Keys ``[0, n_cl)`` are clustered by direction (spherical k-means). Each cluster
    stores a unit mean direction ``ĉ_b`` and angular radius ``ρ_b``; each key stores its
    length ``|k_j|``. Keys ``[n_cl, n)`` are the recent window, always read exactly.
  - Read-free mass estimate and bracket (``s_j = scale·q·k_j``):
        m̂_b = Σ_{j∈b} exp(scale·|k_j|(q·ĉ_b))
        L_b, U_b = Σ_{j∈b} exp(scale·|k_j|(q·ĉ_b ∓ ‖q‖ρ_b))      L_b ≤ m_b ≤ U_b
  - Head: the top-``h`` clusters by ``m̂_b``, read exactly.
  - Tail: ``S`` systematic draws over the remaining clusters from
    ``π_b = (1-α)·m̂_b/Σ_T m̂ + α/|T|``; each drawn cluster's keys and values are read
    and weighted by ``1/(S π_b)``.
  - Output ``(N_W + N_H + N̂_T) / (Z_W + Z_H + Ẑ_T)``. The numerator and denominator
    are each unbiased; their ratio has O(1/S) bias. ``S=0`` drops the tail (biased).

This is the offline estimator for the replay harness: it computes every score (so the
exact per-cluster masses are available for weighting and testing) but reports which
keys a real kernel would have to read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class ClusterState:
    """Maintained per-(layer, KV head) summary of the clustered keys."""

    labels: torch.Tensor    # [n_cl] cluster id per clustered key
    cdir: torch.Tensor      # [k, d] unit mean direction per cluster
    kmag: torch.Tensor      # [n_cl] per-key length
    rho: torch.Tensor       # [k] angular radius max_j ‖k̂_j − ĉ_b‖
    sizes: torch.Tensor     # [k] keys per cluster (float)

    @property
    def n_clusters(self) -> int:
        return int(self.cdir.shape[0])


def build_clusters(K: torch.Tensor, *, B: int, seed: int = 0, iters: int = 20) -> ClusterState:
    """Spherical k-means with ``ceil(n/B)`` clusters on ``K=[n, d]``; empty clusters dropped."""
    K = K.to(torch.float64)
    n, d = K.shape
    k = max(1, (n + B - 1) // B)
    kmag = K.norm(dim=1)
    Kn = K / kmag.clamp_min(1e-12).unsqueeze(1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = Kn[torch.randperm(n, generator=g)[:k].to(K.device)].clone()
    labels = torch.full((n,), -1, dtype=torch.long, device=K.device)
    for _ in range(iters):
        sim = Kn @ c.t()
        new = sim.argmax(1)
        if torch.equal(new, labels):
            break
        labels = new
        csum = torch.zeros(k, d, dtype=K.dtype, device=K.device).index_add_(0, labels, Kn)
        norms = csum.norm(dim=1)
        empty = norms < 1e-12
        c = csum / norms.clamp_min(1e-12).unsqueeze(1)
        if empty.any():  # reseed empty clusters on the worst-fitting keys
            worst = sim.max(1).values.argsort()[: int(empty.sum())]
            c[empty] = Kn[worst]
    # final centers consistent with final labels, empty clusters removed
    used, labels = torch.unique(labels, return_inverse=True)
    k = used.numel()
    csum = torch.zeros(k, d, dtype=K.dtype, device=K.device).index_add_(0, labels, Kn)
    cdir = csum / csum.norm(dim=1, keepdim=True).clamp_min(1e-12)
    ang = (Kn - cdir[labels]).norm(dim=1)
    rho = torch.zeros(k, dtype=K.dtype, device=K.device).scatter_reduce_(
        0, labels, ang, reduce="amax", include_self=True)
    sizes = torch.zeros(k, dtype=K.dtype, device=K.device).index_add_(
        0, labels, torch.ones_like(kmag))
    return ClusterState(labels=labels, cdir=cdir, kmag=kmag, rho=rho, sizes=sizes)


def _segment_logsumexp(x: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    """Per-row log Σ_{j: labels_j = b} exp(x_j). ``x=[G, n]`` -> ``[G, k]`` (−inf if empty)."""
    shift = x.max(dim=1, keepdim=True).values
    acc = torch.zeros(x.shape[0], k, dtype=x.dtype, device=x.device).index_add_(
        1, labels, torch.exp(x - shift))
    return acc.log() + shift


def log_mass_bounds(q: torch.Tensor, st: ClusterState, scale: float):
    """Read-free ``(log m̂_b, log L_b, log U_b)``, each ``[G, k]``."""
    q = q.to(torch.float64)
    d_b = (q @ st.cdir.t()) * scale                          # [G, k]
    slack = (q.norm(dim=1, keepdim=True) * scale) * st.rho   # [G, k]
    k = st.n_clusters
    out = []
    for proj in (d_b, d_b - slack, d_b + slack):
        out.append(_segment_logsumexp(st.kmag * proj[:, st.labels], st.labels, k))
    return tuple(out)


def certified_head(q: torch.Tensor, st: ClusterState, scale: float) -> torch.Tensor:
    """``[G, k]`` mask of clusters that could hold the top-scoring clustered key.

    Any cluster has ``m_b ≤ |b| e^{s_max}`` and the top token's cluster has
    ``m_{b*} ≥ e^{s_max}``, so every ``b`` with ``U_b < max_b' L_b'/|b'|`` is excluded.
    """
    _, log_L, log_U = log_mass_bounds(q, st, scale)
    tau = (log_L - st.sizes.log()).max(dim=1, keepdim=True).values
    return log_U >= tau - 1e-12


def cluster_sample(q, K, V, st: ClusterState, *, h: int, S: int, R: int = 1,
                   alpha: float = 0.1, generator: torch.Generator | None = None,
                   scale: float | None = None, proposal: str = "mag",
                   return_parts: bool = False):
    """Estimate attention for query heads ``q=[G, d]`` over ``K, V=[n, d]``.

    ``proposal="oracle"`` ranks and samples by the exact cluster masses instead of the
    magnitude estimate: not realizable without reading every key, but it bounds what a
    perfect read-free estimate could achieve at this cluster granularity.

    Returns ``out [R, G, d]`` and ``kread [R, G, n]`` (keys whose K and V rows are read),
    plus exact/estimated numerator and denominator if ``return_parts``.
    """
    q = q.to(torch.float64)
    K = K.to(torch.float64)
    V = V.to(torch.float64)
    G, d = q.shape
    n = K.shape[0]
    n_cl = st.labels.numel()
    k = st.n_clusters
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    dev = q.device

    s = (q @ K.t()) * scale                                   # [G, n]
    w = torch.exp(s - s.max(dim=1, keepdim=True).values)      # shift cancels in the ratio
    w_cl = w[:, :n_cl]
    m = torch.zeros(G, k, dtype=w.dtype, device=dev).index_add_(1, st.labels, w_cl)
    Nb = torch.zeros(G, k, d, dtype=w.dtype, device=dev).index_add_(
        1, st.labels, w_cl.unsqueeze(-1) * V[:n_cl].unsqueeze(0))
    Z_W = w[:, n_cl:].sum(1)
    N_W = w[:, n_cl:] @ V[n_cl:]

    if proposal == "oracle":
        log_mhat = m.log()
    elif proposal == "mag":
        log_mhat, _, _ = log_mass_bounds(q, st, scale)
    else:
        raise ValueError(f"unknown proposal {proposal!r}")
    head = torch.zeros(G, k, dtype=torch.bool, device=dev)
    hh = min(h, k)
    if hh > 0:
        head.scatter_(1, log_mhat.topk(hh, dim=1).indices, True)
    Z_H = (m * head).sum(1)
    N_H = (Nb * head.unsqueeze(-1)).sum(1)

    tail = ~head
    counts = torch.zeros(R, G, k, dtype=w.dtype, device=dev)
    wt = torch.zeros(R, G, k, dtype=w.dtype, device=dev)
    if S > 0 and bool(tail.any()):
        mhat = torch.exp(log_mhat - log_mhat.max(dim=1, keepdim=True).values) * tail
        n_tail = tail.sum(1, keepdim=True).clamp_min(1)
        p = mhat / mhat.sum(1, keepdim=True).clamp_min(1e-300)
        pi = ((1 - alpha) * p + alpha * tail / n_tail) * tail           # [G, k], 0 on head
        F = pi.cumsum(1)
        steps = torch.arange(S, dtype=w.dtype, device=dev) / S
        U = torch.rand(R, G, 1, generator=generator, dtype=w.dtype, device="cpu").to(dev) / S
        T = (U + steps) * F[:, -1:].unsqueeze(0)                        # strictly < total
        idx = torch.searchsorted(F.unsqueeze(0).expand(R, G, k).contiguous(), T).clamp_(max=k - 1)
        counts.scatter_add_(2, idx, torch.ones_like(T))
        wt = counts / (S * pi.clamp_min(1e-300)).unsqueeze(0)
    Z_T = (wt * m.unsqueeze(0)).sum(-1)                                  # [R, G]
    N_T = torch.einsum("rgk,gkd->rgd", wt, Nb)

    Z_hat = Z_W + Z_H + Z_T
    N_hat = N_W.unsqueeze(0) + N_H.unsqueeze(0) + N_T
    out = N_hat / Z_hat.unsqueeze(-1)

    read_cl = head.unsqueeze(0) | (counts > 0)                           # [R, G, k]
    kread = torch.ones(R, G, n, dtype=torch.bool, device=dev)
    kread[..., :n_cl] = read_cl[..., st.labels]
    if return_parts:
        parts = {"Z_hat": Z_hat, "N_hat": N_hat, "Z": w.sum(1), "N": w @ V}
        return out, kread, parts
    return out, kread
