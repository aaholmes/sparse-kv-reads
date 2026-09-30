"""v1 incremental hypersphere bins for `sphere_skip`.

State for one layer, kept across decode steps. Keys are binned once, when they leave the
exact recent window: centered with the reference mean ``μ_ref``, assigned to the nearest of
``C`` fixed directions, and folded into that bin's running direction sum, max and min
length, and count. A running mean ``μ_t`` of the binned keys is kept per KV head (O(d) per
key). When ``‖μ_t − μ_ref‖ > δ · r̄`` (``r̄`` = RMS centered-key length at the last
recentering), that KV head is recentered with ``μ_t`` and all its keys reassigned.

δ = 0 recenters whenever the mean moves (identical to recomputing every step); δ = ∞
builds the bins once and only appends. Attention over the keys read is exact for any δ,
since softmax ignores the common shift ``q·μ``; δ only changes which keys are read.
"""

from __future__ import annotations

import math

import torch

from .geometry import _accum_dtype
from .sphere_skip import SphereIndex, SphereInfo, estimate_max_score, fixed_directions, select_regions


class SphereState:
    def __init__(self, *, C: int = 256, window: int = 64, delta: float = 0.0, seed: int = 0,
                 kind: str = "random"):
        self.C, self.window, self.delta, self.seed, self.kind = C, window, delta, seed, kind
        self.heads: list[dict] | None = None
        self.end = 0              # keys [1, end) are binned
        self.n = 0
        self.rebuilds = 0         # recenterings after the initial build (all KV heads)
        self.head_steps = 0       # KV-head decode steps seen (for the rebuild rate)

    # -- construction -----------------------------------------------------
    def _build_head(self, Kb: torch.Tensor, mu: torch.Tensor, dirs: torch.Tensor) -> dict:
        idx = SphereIndex(dirs)
        Kr = Kb - mu
        labels = idx.add(Kr) if Kb.shape[0] else torch.empty(0, dtype=torch.long, device=Kb.device)
        rbar = float(Kr.pow(2).sum(1).mean().sqrt()) if Kb.shape[0] else 0.0
        return {"idx": idx, "labels": labels, "mu_ref": mu.clone(), "sum": Kb.sum(0),
                "cnt": Kb.shape[0], "rbar": rbar, **idx.stats()}

    def _refresh_stats(self, h: dict) -> None:
        h.update(h["idx"].stats())

    def _init(self, K: torch.Tensor, acc) -> None:
        n, H_kv, d = K.shape
        self.dirs = fixed_directions(self.C, d, seed=self.seed, kind=self.kind, dtype=acc, device=K.device)
        self.end = max(1, n - self.window)
        self.heads = []
        for hkv in range(H_kv):
            Kb = K[1:self.end, hkv].to(acc)
            mu = Kb.mean(0) if Kb.shape[0] else torch.zeros(d, dtype=acc, device=K.device)
            self.heads.append(self._build_head(Kb, mu, self.dirs))

    def _advance(self, K: torch.Tensor, acc) -> None:
        n, H_kv, d = K.shape
        new_end = max(1, n - self.window)
        for hkv, h in enumerate(self.heads):
            if new_end > self.end:
                Kn = K[self.end:new_end, hkv].to(acc)
                h["sum"] = h["sum"] + Kn.sum(0)
                h["cnt"] += Kn.shape[0]
                lab = h["idx"].add(Kn - h["mu_ref"])
                h["labels"] = torch.cat([h["labels"], lab])
                self._refresh_stats(h)
            self.head_steps += 1
            if h["cnt"] == 0:
                continue
            mu_t = h["sum"] / h["cnt"]
            if float((mu_t - h["mu_ref"]).norm()) > self.delta * h["rbar"]:
                self.heads[hkv] = self._build_head(K[1:new_end, hkv].to(acc), mu_t, self.dirs)
                self.rebuilds += 1
        self.end = new_end

    # -- decode step -------------------------------------------------------
    def observe(self, K: torch.Tensor, acc=None) -> None:
        """Bring the bins up to date with cache ``K [n, H_kv, d]`` (build, append, maybe recenter)."""
        n, H_kv, _ = K.shape
        acc = acc or _accum_dtype(K.dtype)
        if self.heads is None or n < self.n or len(self.heads) != H_kv:
            self._init(K, acc)                     # prefill / new sequence: full build
        else:
            self._advance(K, acc)
        self.n = n

    def attend(self, q, K, V, *, budget: float, group: str = "sum_share"):
        """Attention for one decode step; returns ``(out [H, d], SphereInfo)``."""
        self.observe(K, _accum_dtype(q.dtype))
        return self.read(q, K, V, budget=budget, group=group)

    def labels_and_weights(self, q, *, n: int, budget: float, group: str = "sum_share"):
        """Kernel inputs for the current bins: ``labels [H_kv, n]`` (int16; label ``C`` = exact set)
        and ``w`` (``[H_kv, C+1]`` for shared selection, ``[H, C+1]`` per query head)."""
        H, d = q.shape
        H_kv = len(self.heads)
        G = H // H_kv
        acc = _accum_dtype(q.dtype)
        scale = 1.0 / math.sqrt(d)
        C = self.C
        labels = torch.full((H_kv, n), C, dtype=torch.int16, device=q.device)
        need = math.ceil(budget * (self.end - 1))
        rows = H if group == "per_head" else H_kv
        w = torch.zeros(rows, C + 1, dtype=acc, device=q.device)
        w[:, C] = 1.0
        for hkv, h in enumerate(self.heads):
            labels[hkv, 1:self.end] = h["labels"].to(torch.int16)
            if self.end > 1 and need > 0:
                e = estimate_max_score(q[hkv * G:(hkv + 1) * G].to(acc), h)
                reg = select_regions(e, h["count"], need, group=group, scale=scale).to(acc)   # [G, C]
                if group == "per_head":
                    w[hkv * G:(hkv + 1) * G, :C] = reg
                else:
                    w[hkv, :C] = reg[0]
        return labels, w

    def read(self, q, K, V, *, budget: float, group: str = "sum_share"):
        """Select and attend using the current bins (no update)."""
        H, d = q.shape
        n, H_kv, _ = K.shape
        G = H // H_kv
        acc = _accum_dtype(q.dtype)
        scale = 1.0 / math.sqrt(d)
        ex = torch.zeros(n, dtype=torch.bool, device=q.device)
        ex[0] = True
        ex[self.end:] = True
        need = math.ceil(budget * (self.end - 1))
        out = torch.empty(H, d, dtype=V.dtype, device=V.device)
        selected = torch.zeros(H, n, dtype=torch.bool, device=q.device)
        for hkv, h in enumerate(self.heads):
            qg = q[hkv * G:(hkv + 1) * G].to(acc)
            sel = ex.expand(G, n).clone()
            if self.end > 1 and need > 0:
                e = estimate_max_score(qg, h)
                reg = select_regions(e, h["count"], need, group=group, scale=scale)
                sel[:, 1:self.end] = reg[:, h["labels"]]
            s = (qg @ K[:, hkv].to(acc).t()) * scale
            A = torch.softmax(s.masked_fill(~sel, -math.inf), dim=-1)
            out[hkv * G:(hkv + 1) * G] = (A @ V[:, hkv].to(acc)).to(V.dtype)
            selected[hkv * G:(hkv + 1) * G] = sel
        kv_union = selected.view(H_kv, G, n).any(1).sum(1)
        return out, SphereInfo(selected=selected, unique=selected.sum(1), kv_union=kv_union,
                               overhead_rows=self.C + 2 * self.C / d)
