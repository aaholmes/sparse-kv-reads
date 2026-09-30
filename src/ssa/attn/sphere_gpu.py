"""Batched GPU bin index for `sphere_skip`.

One instance per layer. Holds every KV head's hypersphere bins as batched tensors and
produces the Triton kernel's inputs (``labels [H_kv, n]`` int16 and per-bin weights ``w``)
with no Python loop over heads and no host synchronization in the per-step path.

Semantics match ``SphereState`` (the reference): keys are binned once, when they leave the
recent window, centered with ``μ_ref``; a KV head is recentered and rebinned when its running
mean has moved more than ``δ · r̄``. The drift flags are computed on the GPU every step and
read on the host every ``check_every`` steps (``check_every=1`` reproduces ``SphereState``).
Pure PyTorch, so it runs on CPU as well.
"""

from __future__ import annotations

import math

import torch

from .geometry import _accum_dtype
from .sphere_skip import fixed_directions

GROUPS = ("per_head", "max", "max_rel", "sum_share")


class SphereIndexGPU:
    def __init__(self, *, C: int = 256, window: int = 64, delta: float = 0.03, capacity: int = 65536,
                 check_every: int = 16, seed: int = 0, kind: str = "random"):
        self.C, self.window, self.delta, self.capacity = C, window, delta, capacity
        self.check_every, self.seed, self.kind = check_every, seed, kind
        self.initialized = False
        self.n = 0
        self.end = 0
        self.rebuilds = 0
        self.head_steps = 0
        self._steps = 0

    # -- state -------------------------------------------------------------
    def _alloc(self, H_kv: int, d: int, dtype, device) -> None:
        C = self.C
        kw = dict(dtype=dtype, device=device)
        self.dirs = fixed_directions(C, d, seed=self.seed, kind=self.kind, dtype=dtype, device=device)
        self.sum_dir = torch.zeros(H_kv, C, d, **kw)
        self.mmax = torch.zeros(H_kv, C, **kw)
        self.mmin = torch.full((H_kv, C), math.inf, **kw)
        self.count = torch.zeros(H_kv, C, **kw)
        self.ksum = torch.zeros(H_kv, d, **kw)
        self.mu_ref = torch.zeros(H_kv, d, **kw)
        self.rbar = torch.zeros(H_kv, **kw)
        self.flags = torch.zeros(H_kv, dtype=torch.bool, device=device)
        self.labels = torch.full((H_kv, self.capacity), C, dtype=torch.int16, device=device)
        self.H_kv, self.d, self.dtype, self.device = H_kv, d, dtype, device

    def _bin(self, Kr: torch.Tensor, heads: torch.Tensor):
        """Assign centered keys ``Kr [h, m, d]`` of ``heads`` and fold them into the bin stats."""
        h, m, d = Kr.shape
        mag = Kr.norm(dim=-1)                                              # [h, m]
        Kn = Kr / mag.clamp_min(1e-12).unsqueeze(-1)
        lab = (Kn @ self.dirs.t()).argmax(-1)                              # [h, m]
        flat = (lab + heads.unsqueeze(1) * self.C).reshape(-1)
        self.sum_dir.view(-1, d).index_add_(0, flat, Kn.reshape(-1, d))
        self.mmax.view(-1).scatter_reduce_(0, flat, mag.reshape(-1), reduce="amax")
        self.mmin.view(-1).scatter_reduce_(0, flat, mag.reshape(-1), reduce="amin")
        self.count.view(-1).index_add_(0, flat, torch.ones_like(mag).reshape(-1))
        return lab

    def _build(self, K: torch.Tensor, heads: torch.Tensor, mu: torch.Tensor | None) -> None:
        """(Re)build the bins of ``heads`` from binned keys ``K[heads, 1:end]``."""
        Kb = K[heads, 1:self.end].to(self.dtype)                           # [h, nb, d]
        nb = Kb.shape[1]
        if mu is None:
            mu = Kb.mean(1) if nb else torch.zeros(len(heads), self.d, dtype=self.dtype, device=self.device)
        self.sum_dir[heads] = 0
        self.mmax[heads] = 0
        self.mmin[heads] = math.inf
        self.count[heads] = 0
        self.mu_ref[heads] = mu
        self.ksum[heads] = Kb.sum(1)
        if nb:
            Kr = Kb - mu.unsqueeze(1)
            self.rbar[heads] = Kr.pow(2).sum(-1).mean(1).sqrt()
            self.labels[heads, 1:self.end] = self._bin(Kr, heads).to(torch.int16)
        else:
            self.rbar[heads] = 0

    def _init(self, K: torch.Tensor, n: int) -> None:
        H_kv, _, d = K.shape
        if n > self.capacity:
            raise ValueError(f"n={n} exceeds capacity {self.capacity}")
        self._alloc(H_kv, d, _accum_dtype(K.dtype), K.device)
        self.end = max(1, n - self.window)
        self._build(K, torch.arange(H_kv, device=K.device), None)
        self.initialized = True

    def _advance(self, K: torch.Tensor, n: int) -> None:
        new_end = max(1, n - self.window)
        heads = torch.arange(self.H_kv, device=self.device)
        if new_end > self.end:
            Kn = K[:, self.end:new_end].to(self.dtype)                     # [H_kv, m, d]
            self.ksum += Kn.sum(1)
            lab = self._bin(Kn - self.mu_ref.unsqueeze(1), heads)
            self.labels[:, self.end:new_end] = lab.to(torch.int16)
            self.end = new_end
        cnt = self.end - 1
        self.head_steps += self.H_kv
        if cnt <= 0:
            return
        mu_t = self.ksum / cnt
        dist = (mu_t - self.mu_ref).norm(dim=1)
        self.flags |= dist > self.delta * self.rbar                         # GPU-side; no sync
        self._steps += 1
        if self._steps % self.check_every == 0:
            flagged = self.flags.nonzero().squeeze(1)                       # host sync, every check_every steps
            if flagged.numel():
                self._build(K, flagged, mu_t[flagged])
                self.rebuilds += int(flagged.numel())
            self.flags.zero_()

    def observe(self, K: torch.Tensor, n: int) -> None:
        """Update from the cache ``K [H_kv, ≥n, d]`` (engine layout) holding ``n`` valid positions."""
        if not self.initialized or n < self.n or K.shape[0] != getattr(self, "H_kv", -1):
            self._init(K, n)
        else:
            self._advance(K, n)
        self.n = n

    # -- selection ---------------------------------------------------------
    @staticmethod
    def _rank_mask(e: torch.Tensor, count: torch.Tensor, need: int) -> torch.Tensor:
        """Rows of ``e [R, C]``: take bins in decreasing ``e`` until their counts reach ``need``."""
        R, C = e.shape
        order = e.argsort(dim=1, descending=True)
        cum = count.gather(1, order).cumsum(1)
        n_sel = (cum < need).sum(1) + 1
        rank = torch.empty_like(order).scatter_(1, order, torch.arange(C, device=e.device).expand(R, C))
        return rank < n_sel.unsqueeze(1)

    def labels_and_weights(self, q: torch.Tensor, *, n: int, budget: float, group: str = "sum_share"):
        """Kernel inputs: ``labels [H_kv, n]`` int16 (label ``C`` = exact set) and ``w``
        (``[H_kv, C+1]`` shared, ``[H, C+1]`` per query head)."""
        if group not in GROUPS:
            raise ValueError(f"unknown group mode {group!r}")
        H, d = q.shape
        H_kv, C = self.H_kv, self.C
        G = H // H_kv
        scale = 1.0 / math.sqrt(d)
        rows = H if group == "per_head" else H_kv
        w = torch.zeros(rows, C + 1, dtype=self.dtype, device=self.device)
        w[:, C] = 1.0
        need = math.ceil(budget * (self.end - 1))
        if self.end > 1 and need > 0:
            cdir = self.sum_dir / self.sum_dir.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            p = torch.einsum("hgd,hcd->hgc", q.view(H_kv, G, d).to(self.dtype), cdir)
            e = torch.where(p >= 0, self.mmax.unsqueeze(1) * p, self.mmin.unsqueeze(1) * p)
            empty = (self.count == 0).unsqueeze(1)
            e = e.masked_fill(empty, -math.inf)                             # [H_kv, G, C]
            if group == "per_head":
                w[:, :C] = self._rank_mask(e.reshape(H, C), self.count.repeat_interleave(G, 0), need).to(w.dtype)
            else:
                z = e * scale
                if group == "max":
                    shared = z.max(1).values
                elif group == "max_rel":
                    shared = (z - z.max(2, keepdim=True).values).max(1).values
                else:
                    shared = torch.softmax(z, dim=2).sum(1).masked_fill(self.count == 0, -math.inf)
                w[:, :C] = self._rank_mask(shared, self.count, need).to(w.dtype)
        return self.labels[:, :n], w
