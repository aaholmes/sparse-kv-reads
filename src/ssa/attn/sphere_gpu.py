"""Batched GPU bin index for `cluster_skip`.

One instance per layer. Holds every KV head's hypersphere bins as batched tensors and
produces the Triton kernel's inputs (``labels [H_kv, n]`` int16 and per-bin weights ``w``)
with no Python loop over heads and no host synchronization in the per-step path.

Semantics match ``SphereState`` (the reference): keys are binned once, when they leave the
recent window, centered with ``μ_ref``; a KV head is recentered and rebinned when its running
mean has moved more than ``δ · r̄``. The drift flags are computed on the GPU every step and
read on the host every ``check_every`` steps (``check_every=1`` reproduces ``SphereState``).
Pure PyTorch, so it runs on CPU as well.

``partition="random"`` (default) assigns each key to the nearest of ``C`` fixed random directions
shared by all heads. ``partition="kmeans"`` fits ``C`` centroids per KV head by cosine k-means on
the centered keys at the first build (the end of the prompt) and keeps them: later keys join the
nearest centroid, and recentering rebins with the centroids unchanged.

Split on overflow (v1; ``C_init < C`` and ``split_factor > 0``, with ``partition="kmeans"``): only
``C_init`` clusters are fitted, leaving ``C − C_init`` spare slots, and a cluster may hold at most
``cap = split_factor ×`` the mean cluster size at the fit. Right after the fit, each cluster over
the cap is divided by k-means over its own keys into about ``size ÷ mean size`` clusters, and any
piece still over the cap is split in two until none is. Afterwards only a cluster that has just
received a key is tested, and one over the cap is split in two by 2-means on its keys' directions;
when 2-means would leave a half with under a quarter of the keys, the cluster is cut at the median
of the keys' projections on the line between the two centroids instead. New clusters take spare
slots, and splitting stops when none are left. Clusters are never merged (a v2 question).

``cap_keys`` sets the cap as an absolute number of keys instead, so that the cluster size, and
with it the cluster count at a given context length, does not depend on how long the prompt was
(``split_factor``, 2 by default then, remains the ratio of the cap to the size the fit aims for).

``var_factor`` gives the trigger DynaKV (arXiv:2511.07427) describes, in place of or alongside the
size cap: a cluster that has just received a key is split once if the variance of its keys'
directions exceeds ``var_factor ×`` the mean variance of its head's clusters at the fit (and it
holds at least ``var_min`` keys). Nothing is split at the fit under this rule. The threshold's
definition and ``var_min`` are my choices; the paper says only that the threshold is set per head.

Two ways to bound the cluster count, which otherwise grows in proportion to the context:
``grow_cap`` makes the cap ``split_factor ×`` the current mean size of ``C_init`` clusters, so a
cluster splits only when it holds more than that multiple of its share of the keys; ``reset_at``
refits a head from all its keys, back to ``C_init`` clusters, when its count reaches that number.
"""

from __future__ import annotations

import math

import torch

from .geometry import _accum_dtype
from .sphere_skip import fixed_directions

GROUPS = ("per_head", "max", "max_rel", "sum_share")


def _kmeans_padded(Kn: torch.Tensor, valid: torch.Tensor, k: torch.Tensor, iters: int) -> torch.Tensor:
    """Cosine k-means on each row of ``Kn [S, M, d]`` (unit vectors; ``valid [S, M]`` marks real
    entries) into ``k [S]`` parts, starting from evenly spaced members. Returns ``part [S, M]``."""
    S, M, d = Kn.shape
    N = int(k.max())
    j = torch.arange(N, device=Kn.device).unsqueeze(0)
    init = (j * valid.sum(1, keepdim=True) // k.unsqueeze(1)).clamp(max=M - 1)
    cent = Kn.gather(1, init.unsqueeze(-1).expand(S, N, d)).clone()
    spare = (j >= k.unsqueeze(1)).unsqueeze(1)
    wv = valid.to(Kn.dtype)
    for it in range(iters + 1):
        part = torch.einsum("smd,snd->smn", Kn, cent).masked_fill(spare, -math.inf).argmax(-1)
        if it == iters:
            break
        sums = torch.zeros_like(cent).scatter_add_(1, part.unsqueeze(-1).expand(S, M, d), Kn * wv.unsqueeze(-1))
        cnt = torch.zeros(S, N, dtype=Kn.dtype, device=Kn.device).scatter_add_(1, part, wv)
        cent = torch.where((cnt > 0).unsqueeze(-1), torch.nn.functional.normalize(sums, dim=-1), cent)
    return part


def _bisect_padded(Kn: torch.Tensor, valid: torch.Tensor, min_share: float = 0.25, iters: int = 4) -> torch.Tensor:
    """Split each row of ``Kn [S, M, d]`` in two: 2-means on cosine similarity started from the
    first member and the member farthest from it; a row whose smaller half would hold under
    ``min_share`` of its members is cut at the median projection on the line between the two
    centroids instead. Returns ``side [S, M]`` (0 or 1)."""
    S, M, d = Kn.shape
    cnt = valid.sum(1)
    far = torch.einsum("smd,sd->sm", Kn, Kn[:, 0]).masked_fill(~valid, math.inf).argmin(1)
    two = torch.stack([Kn[:, 0], Kn[torch.arange(S, device=Kn.device), far]], 1).clone()
    Kv = Kn * valid.to(Kn.dtype).unsqueeze(-1)
    for it in range(iters + 1):
        sim = torch.einsum("smd,sjd->smj", Kn, two)
        side = (sim[..., 1] > sim[..., 0]) & valid
        if it == iters:
            break
        s1 = (Kv * side.unsqueeze(-1)).sum(1)
        s0 = Kv.sum(1) - s1
        n1 = side.sum(1)
        two = torch.stack([torch.where((cnt - n1 > 0).unsqueeze(-1), torch.nn.functional.normalize(s0, dim=-1), two[:, 0]),
                           torch.where((n1 > 0).unsqueeze(-1), torch.nn.functional.normalize(s1, dim=-1), two[:, 1])], 1)
    n1 = side.sum(1)
    lopsided = (n1 < min_share * cnt) | (n1 > (1 - min_share) * cnt)
    proj = torch.einsum("smd,sd->sm", Kn, two[:, 1] - two[:, 0]).masked_fill(~valid, math.inf)
    order = proj.argsort(dim=1, stable=True)
    rank = torch.empty_like(order).scatter_(1, order, torch.arange(M, device=Kn.device).expand(S, M))
    upper = (rank >= (cnt // 2).unsqueeze(1)) & valid
    return torch.where(lopsided.unsqueeze(1), upper, side).long()


def bisect_directions(Xn: torch.Tensor):
    """Split unit vectors ``Xn [m, d]`` in two (see ``_bisect_padded``): ``side [m]`` and the two
    halves' unit mean directions ``[2, d]``."""
    side = _bisect_padded(Xn.unsqueeze(0), torch.ones(1, Xn.shape[0], dtype=torch.bool, device=Xn.device))[0]
    cents = torch.stack([Xn[side == j].sum(0) for j in (0, 1)])
    return side, torch.nn.functional.normalize(cents, dim=-1)


class SphereIndexGPU:
    def __init__(self, *, C: int = 256, window: int = 64, delta: float = 0.03, capacity: int = 65536,
                 check_every: int = 16, seed: int = 0, kind: str = "random", partition: str = "random",
                 kmeans_iters: int = 10, C_init: int | None = None, split_factor: float = 0.0,
                 grow_cap: bool = False, reset_at: int = 0, cap_keys: float = 0.0, var_factor: float = 0.0,
                 var_min: int = 8):
        if partition not in ("random", "kmeans"):
            raise ValueError(f"unknown partition {partition!r}")
        self.partition, self.kmeans_iters = partition, kmeans_iters
        self.C_init = C if C_init is None else C_init
        self.cap_keys = float(cap_keys)         # absolute cap in keys (0: split_factor × mean size at the fit)
        self.split_factor = float(split_factor) if split_factor or not cap_keys else 2.0
        self.var_factor, self.var_min = float(var_factor), int(var_min)
        self.splitting = self.split_factor > 0 or self.var_factor > 0
        if self.C_init > C or (self.splitting and partition != "kmeans"):
            raise ValueError("C_init must be <= C, and splitting needs partition='kmeans'")
        self.splits = 0
        self.capk = self.split_factor / self.C_init if grow_cap else 0.0   # cap per binned key, when it grows
        self.reset_at = reset_at                # refit a head whose cluster count reaches this (0 = never)
        self.resets = 0
        self._fresh_fit = False
        self.C, self.window, self.delta, self.capacity = C, window, delta, capacity
        self.check_every, self.seed, self.kind = check_every, seed, kind
        self.initialized = False
        self.n = 0
        self.start = 1                  # binned positions are [start, end); [1, start) evicted
        self.end = 0
        self.rebuilds = 0
        self.head_steps = 0
        self._steps = 0

    # -- state -------------------------------------------------------------
    def _alloc(self, H_kv: int, d: int, dtype, device) -> None:
        C = self.C
        kw = dict(dtype=dtype, device=device)
        self.dirs = fixed_directions(C, d, seed=self.seed, kind=self.kind, dtype=dtype, device=device)
        self.cent = torch.zeros(H_kv, C, d, **kw) if self.partition == "kmeans" else None
        self.fitted = torch.zeros(H_kv, dtype=torch.bool, device=device)
        self.n_c = torch.full((H_kv,), self.C_init if self.partition == "kmeans" else C, dtype=torch.long,
                              device=device)                          # active clusters per head
        self.cap = torch.full((H_kv,), math.inf, **kw)                 # split a cluster above this many keys
        self.vthr = torch.full((H_kv,), math.inf, **kw)                # or above this spread of its directions
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
        if self.partition == "kmeans":
            used = int(self.n_c[heads].max())                              # score only slots in use, in chunks
            spare = torch.arange(used, device=self.device).unsqueeze(0) >= self.n_c[heads].unsqueeze(1)
            lab = torch.cat([torch.einsum("hmd,hcd->hmc", Kn[:, i:i + 4096], self.cent[heads, :used])
                             .masked_fill(spare.unsqueeze(1), -math.inf).argmax(-1)
                             for i in range(0, m, 4096)], dim=1) if m else torch.zeros(h, 0, dtype=torch.long,
                                                                                       device=self.device)
        else:
            lab = (Kn @ self.dirs.t()).argmax(-1)                          # [h, m]
        flat = (lab + heads.unsqueeze(1) * self.C).reshape(-1)
        self.sum_dir.view(-1, d).index_add_(0, flat, Kn.reshape(-1, d))
        self.mmax.view(-1).scatter_reduce_(0, flat, mag.reshape(-1), reduce="amax")
        self.mmin.view(-1).scatter_reduce_(0, flat, mag.reshape(-1), reduce="amin")
        self.count.view(-1).index_add_(0, flat, torch.ones_like(mag).reshape(-1))
        return lab

    def _build(self, K: torch.Tensor, heads: torch.Tensor, mu: torch.Tensor | None) -> None:
        """(Re)build the bins of ``heads`` from binned keys ``K[heads, start:end]``."""
        Kb = K[heads, self.start:self.end].to(self.dtype)                           # [h, nb, d]
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
            if self.partition == "kmeans":                                 # fit once per head, then keep
                todo = ~self.fitted[heads]
                if todo.any():
                    from .clusterkv import spherical_kmeans
                    _, cent = spherical_kmeans(Kr[todo], C=self.C_init, iters=self.kmeans_iters, seed=self.seed)
                    self.cent[heads[todo], :self.C_init] = cent
                    self.n_c[heads[todo]] = self.C_init
                    self.fitted[heads[todo]] = True
                    if self.split_factor > 0:
                        self.cap[heads[todo]] = self.cap_keys or self.split_factor * nb / self.C_init
                        self._fresh_fit = True
            self.rbar[heads] = Kr.pow(2).sum(-1).mean(1).sqrt()
            self.labels[heads, self.start:self.end] = self._bin(Kr, heads).to(torch.int16)
            if self.var_factor > 0 and self.partition == "kmeans" and bool(todo.any()):
                fit = heads[todo]                                          # threshold from the fit's own clusters
                used = (self.count[fit] > 0).to(self.dtype)
                self.vthr[fit] = self.var_factor * (self._spread()[fit] * used).sum(1) / used.sum(1).clamp_min(1)
        else:
            self.rbar[heads] = 0

    def _init(self, K: torch.Tensor, n: int) -> None:
        H_kv, _, d = K.shape
        if n > self.capacity:
            raise ValueError(f"n={n} exceeds capacity {self.capacity}")
        self._alloc(H_kv, d, _accum_dtype(K.dtype), K.device)
        self.start = 1
        self.end = max(1, n - self.window)
        self._build(K, torch.arange(H_kv, device=K.device), None)
        self._split_after_fit(K)
        self.initialized = True

    def _advance(self, K: torch.Tensor, n: int) -> None:
        new_end = max(1, n - self.window)
        heads = torch.arange(self.H_kv, device=self.device)
        if self.partition == "kmeans" and new_end > self.end and not bool(self.fitted.all()):
            self.end = new_end                                             # nothing was binned at the first build
            self._build(K, heads, None)
            self._split_after_fit(K)
        if new_end > self.end:
            Kn = K[:, self.end:new_end].to(self.dtype)                     # [H_kv, m, d]
            self.ksum += Kn.sum(1)
            lab = self._bin(Kn - self.mu_ref.unsqueeze(1), heads)
            self.labels[:, self.end:new_end] = lab.to(torch.int16)
            self.end = new_end
            if self.splitting:                                             # only clusters that just received a key
                got = torch.zeros(self.H_kv, self.C, dtype=torch.bool, device=self.device).scatter_(1, lab, True)
                self._split_overflow(K, got)
                if self.reset_at:
                    self._reset_heads(K, (self.n_c >= self.reset_at).nonzero().squeeze(1))
        cnt = self.end - self.start
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

    @property
    def active_clusters(self) -> float:
        """Mean number of clusters in use per KV head (all ``C`` unless slots are spare)."""
        return float(self.n_c.float().mean()) if self.initialized else float(self.C)

    def _cap_now(self) -> torch.Tensor:
        """Each head's cap in keys: the one set at the fit, or ``split_factor ×`` the current mean
        size of ``C_init`` clusters when the cap grows with the context."""
        return torch.clamp(self.cap, min=self.capk * (self.end - self.start))

    def _spread(self) -> torch.Tensor:
        """Per cluster, the variance of its keys' unit directions about their mean:
        ``1 − ‖Σ directions‖² / count²`` (0 for an empty or single-key cluster)."""
        return (1 - self.sum_dir.pow(2).sum(-1) / self.count.clamp_min(1).pow(2)).clamp_min(0) * (self.count > 0)

    def _host_end(self) -> None:
        """Bring ``self.end`` up to date before host-side work (kept on the GPU by subclasses)."""

    def _reset_heads(self, K, heads: torch.Tensor) -> None:
        """Fit ``heads`` again from all their binned keys, back to ``C_init`` clusters with a fresh
        mean and cap, then divide any cluster over the cap as after the prompt fit."""
        if heads.numel() == 0:
            return
        self._host_end()
        self.fitted[heads] = False
        self.n_c[heads] = self.C_init
        self._build(K, heads, None)
        self._split_after_fit(K)
        self.resets += int(heads.numel())

    def _before_split(self, K) -> None:
        """Hook for subclasses that keep extra per-cluster sums: called before any split."""

    def _on_split(self, K, heads: torch.Tensor, pos: torch.Tensor, slot: torch.Tensor, rows: torch.Tensor) -> None:
        """Hook: clusters ``rows`` (flat ``head · C + slot``) were rewritten; their members are the
        keys at ``pos`` of ``heads``, now in ``slot``."""

    def _split_after_fit(self, K: torch.Tensor) -> None:
        if self._fresh_fit:
            self._fresh_fit = False
            self._split_overflow(K, None, nmeans=True)

    def _split_overflow(self, K: torch.Tensor, cand: torch.Tensor | None = None, nmeans: bool = False) -> None:
        """Split clusters over their head's cap until none is, or no spare slot is left. ``cand
        [H_kv, C]`` restricts the test to those clusters (and the pieces they are split into).
        ``nmeans`` divides each cluster in one step into about size ÷ mean size parts first."""
        if not self.splitting:
            return
        slots = torch.arange(self.C, device=self.device).unsqueeze(0)
        first = True
        by_spread = cand is not None                  # a spread over the threshold splits once, on receiving a key
        while True:
            over = self.count > self._cap_now().unsqueeze(1)
            if by_spread:
                over |= (self._spread() > self.vthr.unsqueeze(1)) & (self.count >= self.var_min)
                by_spread = False
            over &= (slots < self.n_c.unsqueeze(1)) & (self.count >= 2)
            if cand is not None:
                over &= cand
            hc = over.nonzero()                                               # host sync
            if hc.shape[0] == 0:
                return
            if first:
                self._before_split(K)
                first = False
            touched = torch.zeros(self.H_kv * self.C, dtype=torch.bool, device=self.device)
            done = 0
            for i in range(0, hc.shape[0], 64):
                done += self._split_batch(K, hc[i:i + 64], nmeans, touched)
            if not done and not nmeans:
                return
            nmeans = False
            if cand is not None:
                cand = cand | touched.view(self.H_kv, self.C)

    def _split_batch(self, K, hc: torch.Tensor, nmeans: bool, touched: torch.Tensor) -> int:
        """Split the clusters ``hc [S, 2]`` (head, slot; sorted by head) together. Returns how many were split."""
        C, d, dev = self.C, self.d, self.device
        h, c = hc[:, 0], hc[:, 1]
        S = h.shape[0]
        member = self.labels[h, self.start:self.end] == c.unsqueeze(1)        # [S, binned]
        cnt = member.sum(1)
        M = int(cnt.max())
        rows, cols = member.nonzero(as_tuple=True)
        rank = torch.arange(rows.shape[0], device=dev) - (cnt.cumsum(0) - cnt)[rows]
        pos = torch.zeros(S, M, dtype=torch.long, device=dev)
        pos[rows, rank] = cols + self.start
        valid = torch.arange(M, device=dev).unsqueeze(0) < cnt.unsqueeze(1)
        Kr = K[h.unsqueeze(1), pos].to(self.dtype) - self.mu_ref[h].unsqueeze(1)
        mag = Kr.norm(dim=-1)
        Kn = Kr / mag.clamp_min(1e-12).unsqueeze(-1)
        if nmeans:
            k = torch.ceil(cnt * self.split_factor / self._cap_now()[h]).long().clamp(min=2)
            part = _kmeans_padded(Kn, valid, k, self.kmeans_iters)
        else:
            part = _bisect_padded(Kn, valid)
        N = int(part.max()) + 1
        j = torch.arange(N, device=dev).unsqueeze(0)
        sizes = torch.zeros(S, N, dtype=torch.long, device=dev).scatter_add_(1, part, valid.long())
        part = ((sizes > 0).cumsum(1) - 1).gather(1, part)                    # number the non-empty parts 0, 1, ...
        extra = (sizes > 0).sum(1) - 1                                        # new clusters each split creates
        per_head = torch.zeros(self.H_kv, dtype=torch.long, device=dev).index_add_(0, h, extra)
        base = self.n_c[h] + (extra.cumsum(0) - extra) - (per_head.cumsum(0) - per_head)[h]
        keep = (extra > 0) & (base + extra <= C)                              # spare slots are given out in order
        if not bool(keep.any()):
            return 0
        slot = torch.where(j == 0, c.unsqueeze(1), base.unsqueeze(1) + j - 1)                 # [S, N]
        tgt = (h.unsqueeze(1) * C + slot)[keep.unsqueeze(1) & (j <= extra.unsqueeze(1))]
        sel = valid & keep.unsqueeze(1)
        hh = h.unsqueeze(1).expand(S, M)[sel]
        new = slot.gather(1, part)[sel]
        flat = hh * C + new
        self.sum_dir.view(-1, d)[tgt] = 0
        self.mmax.view(-1)[tgt] = 0
        self.mmin.view(-1)[tgt] = math.inf
        self.count.view(-1)[tgt] = 0
        self.sum_dir.view(-1, d).index_add_(0, flat, Kn[sel])
        self.mmax.view(-1).scatter_reduce_(0, flat, mag[sel], reduce="amax")
        self.mmin.view(-1).scatter_reduce_(0, flat, mag[sel], reduce="amin")
        self.count.view(-1).index_add_(0, flat, torch.ones_like(mag[sel]))
        self.cent.view(-1, d)[tgt] = torch.nn.functional.normalize(self.sum_dir.view(-1, d)[tgt], dim=-1)
        self.labels[hh, pos[sel]] = new.to(torch.int16)
        self.n_c.index_add_(0, h[keep], extra[keep])
        self.splits += int(extra[keep].sum())
        touched[tgt] = True
        self._on_split(K, hh, pos[sel], new, tgt)
        return int(keep.sum())

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
        need = math.ceil(budget * (self.end - self.start))
        if self.end > self.start and need > 0:
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
