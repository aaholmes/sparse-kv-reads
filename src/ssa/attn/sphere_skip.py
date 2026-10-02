"""`voronoi_skip` — skip key reads using a fixed partition of the unit hypersphere.

Per KV head:

  - **Exact set**, always read: token 0 (the attention sink) and the ``window`` most
    recent positions.
  - **Regions**: every other key goes to the nearest of ``C`` fixed random unit
    directions (a data-independent Voronoi partition of the sphere). Each region keeps
    a running sum of unit key directions (→ mean direction ``ĉ_r``), the min and max key
    length ``μ_r``, ``M_r``, and a count. Assignment and updates are O(C·d) per key.
  - **Ranking** per query head: ``e_r = M_r·(q·ĉ_r)`` if ``q·ĉ_r ≥ 0`` else ``μ_r·(q·ĉ_r)``,
    an estimate of the region's largest score (not a bound).
  - **Selection**: take regions in decreasing ``e_r`` until their keys reach
    ``budget × (number of non-exact keys)``; read those keys and values, then compute
    exact softmax over exact set ∪ selected keys. The unselected tail is dropped, so
    the output is biased. ``budget=1`` reproduces dense attention.

``rank="random"`` orders regions randomly: the matched-budget control.

``center=True`` subtracts the mean of the non-exact keys before partitioning. Keys share
a large common component (cosine ≈ 0.99 with the mean key at layer 0), so without
centering one region holds nearly every key. Softmax is unchanged by a shift common
to all scores, and ``q·(k − μ) = q·k − q·μ``, so ranking on centered keys loses nothing;
the output is invariant to any constant offset added to every key. (This prototype
recomputes ``μ`` each step; a serving cache would freeze it after prefill.)

``group`` sets how the G query heads sharing a KV head choose regions. ``per_head``: each
head ranks and selects on its own, and a kernel must read the union. The shared modes
rank once per KV head and select one set for the whole group (reads = the budget),
from ``z = e/√d`` per head: ``max`` = max over heads of ``z``; ``max_rel`` = max over heads
of ``z − max_r z`` (each head's gap to its own best region, so heads with larger query
norms don't dominate); ``sum_share`` = Σ over heads of ``softmax_r(z)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import register
from .geometry import _accum_dtype


def fixed_directions(C: int, d: int, *, seed: int = 0, kind: str = "random", dtype=torch.float32,
                     device="cpu"):
    """``C`` fixed unit vectors in ``R^d``.

    ``random``: i.i.d. uniform on the sphere. ``sobol``: scrambled Sobol points mapped
    through the inverse normal CDF, then normalized. ``cross``: the first ``C`` of the
    ``2d`` signed coordinate axes ``±e_i`` (``C = 2d`` is the full cross-polytope).
    """
    if kind == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(C, d, generator=g, dtype=torch.float64)
    elif kind == "sobol":
        u = torch.quasirandom.SobolEngine(d, scramble=True, seed=seed).draw(C).to(torch.float64)
        x = math.sqrt(2) * torch.erfinv(2 * u.clamp(1e-12, 1 - 1e-12) - 1)
    elif kind == "cross":
        if C > 2 * d:
            raise ValueError("cross partition has at most 2d directions")
        eye = torch.eye(d, dtype=torch.float64)
        x = torch.cat([eye, -eye])[:C]
    else:
        raise ValueError(f"unknown direction kind {kind!r}")
    return (x / x.norm(dim=1, keepdim=True)).to(device=device, dtype=dtype)


class SphereIndex:
    """Running per-region summaries; ``add`` is the per-token update a serving cache would do."""

    def __init__(self, dirs: torch.Tensor):
        self.dirs = dirs
        C, d = dirs.shape
        kw = dict(dtype=dirs.dtype, device=dirs.device)
        self.sum_dir = torch.zeros(C, d, **kw)
        self.mmax = torch.zeros(C, **kw)
        self.mmin = torch.full((C,), math.inf, **kw)
        self.count = torch.zeros(C, **kw)

    def assign(self, K: torch.Tensor) -> torch.Tensor:
        Kn = K / K.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return (Kn.to(self.dirs.dtype) @ self.dirs.t()).argmax(1)

    def add(self, K: torch.Tensor) -> torch.Tensor:
        K = K.to(self.dirs.dtype)
        mag = K.norm(dim=1)
        labels = self.assign(K)
        self.sum_dir.index_add_(0, labels, K / mag.clamp_min(1e-12).unsqueeze(1))
        self.mmax.scatter_reduce_(0, labels, mag, reduce="amax")
        self.mmin.scatter_reduce_(0, labels, mag, reduce="amin")
        self.count.index_add_(0, labels, torch.ones_like(mag))
        return labels

    def stats(self) -> dict:
        cdir = self.sum_dir / self.sum_dir.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return {"cdir": cdir, "mmax": self.mmax, "mmin": self.mmin, "count": self.count}


def region_stats(K: torch.Tensor, dirs: torch.Tensor) -> dict:
    """Batch summaries for keys ``K [n, d]``: identical to adding them one by one."""
    idx = SphereIndex(dirs)
    labels = idx.add(K)
    return {**idx.stats(), "labels": labels}


@dataclass
class SphereInfo:
    selected: torch.Tensor   # [H, n_k] keys (K and V rows) read per query head
    unique: torch.Tensor     # [H] rows read per query head
    kv_union: torch.Tensor   # [H_kv] rows read per KV head (union over its query-head group)
    overhead_rows: float     # summary reads per KV head, in key-row equivalents


def _region_rank_mask(e: torch.Tensor, count: torch.Tensor, need: int) -> torch.Tensor:
    """Select regions in decreasing ``e`` until their counts reach ``need``. ``e [G, C]``."""
    G, C = e.shape
    order = e.argsort(dim=1, descending=True)
    cum = count[order].cumsum(1)
    n_sel = (cum < need).sum(1) + 1                                   # regions to take
    rank = torch.empty_like(order).scatter_(1, order, torch.arange(C, device=e.device).expand(G, C))
    return rank < n_sel.unsqueeze(1)


GROUP_MODES = ("per_head", "max", "max_rel", "sum_share")


def select_regions(e: torch.Tensor, count: torch.Tensor, need: int, *, group: str, scale: float):
    """Region mask ``[G, C]`` from per-head estimates ``e`` (``-inf`` for empty regions)."""
    if group == "per_head":
        return _region_rank_mask(e, count, need)
    z = e * scale
    if group == "max":
        shared = z.max(0).values
    elif group == "max_rel":
        shared = (z - z.max(1, keepdim=True).values).max(0).values
    elif group == "sum_share":
        shared = torch.softmax(z, dim=1).sum(0).masked_fill(count == 0, -math.inf)
    else:
        raise ValueError(f"unknown group mode {group!r}")
    return _region_rank_mask(shared.unsqueeze(0), count, need).expand(e.shape[0], -1)


def estimate_max_score(q: torch.Tensor, st: dict) -> torch.Tensor:
    """``e [G, C]``: mean direction × max (or min) length; ``-inf`` for empty regions."""
    p = q @ st["cdir"].t()
    e = torch.where(p >= 0, st["mmax"] * p, st["mmin"] * p)
    return e.masked_fill(st["count"] == 0, -math.inf)


@register("voronoi_skip")
def sphere_skip(q, K, V, *, budget: float, C: int = 256, window: int = 64, rank: str = "est",
                seed: int = 0, kind: str = "random", center: bool = False, group: str = "per_head",
                return_info: bool = False, **cfg):
    """Attention over the exact set plus the top regions by estimated max score."""
    H, d = q.shape
    n, H_kv, _ = K.shape
    G = H // H_kv
    acc = _accum_dtype(q.dtype)
    scale = 1.0 / math.sqrt(d)
    dirs = fixed_directions(C, d, seed=seed, kind=kind, dtype=acc, device=q.device)

    ex = torch.zeros(n, dtype=torch.bool, device=q.device)
    ex[0] = True
    ex[max(0, n - window):] = True
    cl = (~ex).nonzero(as_tuple=True)[0]
    need = math.ceil(budget * cl.numel())
    if group not in GROUP_MODES:
        raise ValueError(f"unknown group mode {group!r}")

    out = torch.empty(H, d, dtype=V.dtype, device=V.device)
    selected = torch.zeros(H, n, dtype=torch.bool, device=q.device)
    for hkv in range(H_kv):
        Kh = K[:, hkv].to(acc)
        qg = q[hkv * G:(hkv + 1) * G].to(acc)
        sel = ex.expand(G, n).clone()
        if cl.numel() > 0 and need > 0:
            Kr = Kh[cl] - Kh[cl].mean(0) if center else Kh[cl]
            st = region_stats(Kr, dirs)
            if rank == "est":
                e = estimate_max_score(qg, st)
            elif rank == "random":
                g = torch.Generator(device="cpu").manual_seed(seed * 7919 + hkv * 31 + n)
                e = torch.rand(G, C, generator=g).to(q.device, acc).masked_fill(st["count"] == 0, -math.inf)
            else:
                raise ValueError(f"unknown rank {rank!r}")
            reg = select_regions(e, st["count"], need, group=group, scale=scale)
            sel[:, cl] = reg[:, st["labels"]]
        s = (qg @ Kh.t()) * scale
        A = torch.softmax(s.masked_fill(~sel, -math.inf), dim=-1)
        out[hkv * G:(hkv + 1) * G] = (A @ V[:, hkv].to(acc)).to(V.dtype)
        selected[hkv * G:(hkv + 1) * G] = sel

    if return_info:
        kv_union = selected.view(H_kv, G, n).any(1).sum(1)
        overhead = C + (C * 2) / d                                   # directions + (min, max) scalars
        return out, SphereInfo(selected=selected, unique=selected.sum(1), kv_union=kv_union,
                               overhead_rows=overhead)
    return out
