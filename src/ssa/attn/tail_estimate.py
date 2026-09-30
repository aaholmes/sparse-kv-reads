"""Estimating the dropped bins from per-bin sums.

``SphereIndexTail`` extends ``SphereIndexGPU`` with three sums per bin, kept at a reference
value ``t0`` (one per KV head) of the scalar ``t``:

    Z_b(t0) = Σ_{i∈b} e^{t0 |k_i|},   L_b = Σ_{i∈b} |k_i| e^{t0 |k_i|},   M_b = Σ_{i∈b} |k_i|² e^{t0 |k_i|},
    N_b(t0) = Σ_{i∈b} e^{t0 |k_i|} v_i

(``k_i`` centered). A key is added when it is binned, subtracted when it is evicted, and a
recentered head's sums are rebuilt. For a query ``q`` a dropped bin contributes, with
``t_b = q·ĉ_b / √d`` and the shift ``q·μ / √d`` common to every centered score,

    log Ẑ_b = log Z_b(t0) + (t_b − t0) L_b / Z_b(t0) + q·μ/√d,    N̂_b = Ẑ_b · N_b(t0) / Z_b(t0)

(``order=1``; ``order=0`` drops the slope term). ``order=2`` treats each key's cosine with
the query as varying across the bin, with mean ``a_b = ρ_b (q̂·ĉ_b)`` (``ρ_b`` = length of the
bin's mean unit direction) and variance ``v_b = (1−ρ_b²)(1−(q̂·ĉ_b)²)/(d−1)``, and adds
``(s a_b − t0) L_b/Z_b + s² v_b M_b / (2 Z_b)`` with ``s = |q|/√d`` (``shrink=False`` uses
``a_b = q̂·ĉ_b``). ``attend_with_tail`` combines the exactly read
rows with these estimates in one softmax. Reference implementation, pure PyTorch.

Eviction (``evict``) updates every bin statistic that is a sum; the per-bin maximum and minimum
key lengths cannot be decremented, so they stay as bounds over a superset until the head is
rebuilt. The Triton kernels and the engine do not evict.
"""

from __future__ import annotations

import math

import torch

from .sphere_gpu import SphereIndexGPU


class SphereIndexTail(SphereIndexGPU):
    def __init__(self, *, t0: float = 0.0, **kw):
        super().__init__(**kw)
        self.t0_init = float(t0)
        self._V = None
        self._rebuilt: set[int] = set()
        self._fresh = False

    def _alloc(self, H_kv: int, d: int, dtype, device) -> None:
        super()._alloc(H_kv, d, dtype, device)
        self.tz = torch.zeros(H_kv, self.C, dtype=dtype, device=device)
        self.tl = torch.zeros_like(self.tz)
        self.tq = torch.zeros_like(self.tz)
        self.tn = torch.zeros(H_kv, self.C, d, dtype=dtype, device=device)
        self.t0 = torch.full((H_kv,), self.t0_init, dtype=dtype, device=device)

    def _fold(self, K, V, heads: torch.Tensor, a: int, b: int, sign: float) -> None:
        """Add (``sign=+1``) or subtract (``-1``) positions ``[a, b)`` of ``heads`` to their bins' sums."""
        if b <= a:
            return
        d = self.d
        Kr = K[heads, a:b].to(self.dtype) - self.mu_ref[heads].unsqueeze(1)
        m = Kr.norm(dim=-1)                                                # [h, m]
        e = torch.exp(self.t0[heads].unsqueeze(1) * m)
        flat = (self.labels[heads, a:b].long() + heads.unsqueeze(1) * self.C).reshape(-1)
        self.tz.view(-1).index_add_(0, flat, (sign * e).reshape(-1))
        self.tl.view(-1).index_add_(0, flat, (sign * m * e).reshape(-1))
        self.tq.view(-1).index_add_(0, flat, (sign * m * m * e).reshape(-1))
        Vb = V[heads, a:b].to(self.dtype)
        self.tn.view(-1, d).index_add_(0, flat, (sign * e.unsqueeze(-1) * Vb).reshape(-1, d))

    def _build(self, K, heads, mu) -> None:
        super()._build(K, heads, mu)
        self.tz[heads] = 0
        self.tl[heads] = 0
        self.tq[heads] = 0
        self.tn[heads] = 0
        self._fold(K, self._V, heads, self.start, self.end, 1.0)
        self._rebuilt.update(heads.tolist())

    def _init(self, K, n) -> None:
        super()._init(K, n)
        self._fresh = True

    def observe(self, K: torch.Tensor, V: torch.Tensor, n: int) -> None:
        """Update from the cache ``K, V [H_kv, ≥n, d]`` holding ``n`` valid positions."""
        self._V, self._rebuilt, self._fresh = V, set(), False
        old_end = self.end
        super().observe(K, n)
        if not self._fresh:
            keep = [h for h in range(self.H_kv) if h not in self._rebuilt]
            if keep:
                self._fold(K, V, torch.tensor(keep, device=self.device), old_end, self.end, 1.0)

    def set_t0(self, K: torch.Tensor, V: torch.Tensor, t0: torch.Tensor) -> None:
        """Change the reference ``t0 [H_kv]`` and recompute every head's sums."""
        self.t0 = t0.to(self.dtype).to(self.device).clone()
        heads = torch.arange(self.H_kv, device=self.device)
        self.tz.zero_()
        self.tl.zero_()
        self.tq.zero_()
        self.tn.zero_()
        self._fold(K, V, heads, self.start, self.end, 1.0)

    def evict(self, K: torch.Tensor, V: torch.Tensor, new_start: int) -> None:
        """Remove binned positions ``[start, new_start)`` (the oldest after the first token)."""
        if new_start > self.end:
            raise ValueError(f"can only evict binned positions (< {self.end}), got {new_start}")
        a, b = self.start, new_start
        if b <= a:
            return
        heads = torch.arange(self.H_kv, device=self.device)
        d = self.d
        Kb = K[:, a:b].to(self.dtype)
        Kr = Kb - self.mu_ref.unsqueeze(1)
        Kn = Kr / Kr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        flat = (self.labels[:, a:b].long() + heads.unsqueeze(1) * self.C).reshape(-1)
        self.sum_dir.view(-1, d).index_add_(0, flat, -Kn.reshape(-1, d))
        self.count.view(-1).index_add_(0, flat, -torch.ones(flat.numel(), dtype=self.dtype, device=self.device))
        self.ksum -= Kb.sum(1)
        self._fold(K, V, heads, a, b, -1.0)
        self.start = b


def tail_log_estimates(idx: SphereIndexTail, q: torch.Tensor, w: torch.Tensor, *, order: int = 1,
                       shrink: bool = True):
    """Per query head and bin: ``log Ẑ_b [H, C]`` (−∞ unless the bin is dropped and non-empty),
    the estimated mean value ``N̂_b/Ẑ_b [H, C, d]`` (0 where not dropped), and the dropped mask."""
    H, d = q.shape
    H_kv, C, dt = idx.H_kv, idx.C, idx.dtype
    G = H // H_kv
    scale = 1.0 / math.sqrt(d)
    qx = q.to(dt)
    rows = w.to(dt)
    if rows.shape[0] == H_kv:
        rows = rows.repeat_interleave(G, 0)
    count = idx.count.repeat_interleave(G, 0)                                # [H, C]
    dropped = (rows[:, :C] == 0) & (count > 0)
    cdir = idx.sum_dir / idx.sum_dir.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    t = torch.einsum("hgd,hcd->hgc", qx.view(H_kv, G, d), cdir).reshape(H, C) * scale
    shift = torch.einsum("hgd,hd->hg", qx.view(H_kv, G, d), idx.mu_ref).reshape(H, 1) * scale
    tz = idx.tz.repeat_interleave(G, 0)
    safe = torch.where(dropped, tz, torch.ones_like(tz))
    logz = safe.log() + shift
    if order == 1:
        t0 = idx.t0.repeat_interleave(G, 0).unsqueeze(1)
        logz = logz + (t - t0) * idx.tl.repeat_interleave(G, 0) / safe
    elif order == 2:
        t0 = idx.t0.repeat_interleave(G, 0).unsqueeze(1)
        qn = qx.norm(dim=-1, keepdim=True)                                   # [H, 1]
        cos = t / (qn * scale).clamp_min(1e-300)                              # q̂·ĉ_b
        cnt = torch.where(dropped, count, torch.ones_like(count))
        rho = (idx.sum_dir.norm(dim=-1).repeat_interleave(G, 0) / cnt).clamp(max=1.0)
        a = rho * cos if shrink else cos
        v = (1 - rho ** 2) * (1 - cos ** 2).clamp_min(0) / (d - 1)
        sq = qn * scale
        logz = (logz + (sq * a - t0) * idx.tl.repeat_interleave(G, 0) / safe
                + 0.5 * sq ** 2 * v * idx.tq.repeat_interleave(G, 0) / safe)
    elif order != 0:
        raise ValueError(f"order must be 0, 1 or 2, got {order}")
    logz = torch.where(dropped, logz, torch.full_like(logz, -math.inf))
    vbar = torch.where(dropped.unsqueeze(-1), idx.tn.repeat_interleave(G, 0) / safe.unsqueeze(-1),
                       torch.zeros(1, dtype=dt, device=idx.device))
    return logz, vbar, dropped


def attend_with_tail(idx: SphereIndexTail, q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                     labels: torch.Tensor, w: torch.Tensor, *, order: int = 1, shrink: bool = True) -> torch.Tensor:
    """Exact attention over the rows with ``w > 0`` plus the estimated contribution of every
    non-empty bin with ``w == 0``. ``q [H, d]``, ``K, V [H_kv, n, d]``, ``w [H_kv or H, C+1]``."""
    H, d = q.shape
    H_kv, n, _ = K.shape
    G = H // H_kv
    dt = idx.dtype
    rows = w.to(dt)
    if rows.shape[0] == H_kv:
        rows = rows.repeat_interleave(G, 0)                                  # [H, C+1]
    wr = torch.gather(rows, 1, labels.long().repeat_interleave(G, 0))        # [H, n]
    pos = torch.arange(n, device=K.device)
    read = (wr > 0) & ((pos == 0) | (pos >= idx.start)).unsqueeze(0)
    s = torch.einsum("hd,hnd->hn", q.to(dt), K.to(dt).repeat_interleave(G, 0)) / math.sqrt(d)
    s = torch.where(read, s + wr.clamp_min(1e-300).log(), torch.full_like(s, -math.inf))
    Vx = torch.where(read.unsqueeze(-1), V.to(dt).repeat_interleave(G, 0), torch.zeros(1, dtype=dt, device=K.device))
    logz, vbar, _ = tail_log_estimates(idx, q, w, order=order, shrink=shrink)
    A = torch.softmax(torch.cat([s, logz], dim=1), dim=1)
    return torch.einsum("hm,hmd->hd", A, torch.cat([Vx, vbar], dim=1)).to(V.dtype)
