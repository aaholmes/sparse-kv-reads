"""Adapter + installer: plug ssa estimators into the engine's decode seam.

The engine calls ``decode_attn_op(q, full_k, full_v, *, scale, layer_idx)`` for
single-query decode steps with:
  - ``q``        : [1, H, 1, d]
  - ``full_k/v`` : [1, H_kv, n_k, d]   (pre-GQA-expansion)
and expects a return of shape ``[1, H, 1, d]``.

ssa's ``attn`` uses ``q=[H,d]``, ``K,V=[n_k,H_kv,d]`` — this module converts
between the two, drives reproducible per-(layer, position) RNG, and accumulates
the read statistics used for the bytes-avoided metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from engine.attention import Attention

from ..attn import attn
from ..sampling.draws import unique_counts

_SAMPLING_IMPLS = {"santa", "santa_strat", "santa_sys", "santa_hybrid", "skip_k"}


@dataclass
class ReadStats:
    """Accumulates value-row reads across decode steps (per head, averaged).

    ``read_fraction`` is the avoided-bytes metric: fraction of the n_k value rows
    actually read. Dense reads all of them (1.0); samplers read far fewer.
    """

    steps: int = 0
    reads_sum: float = 0.0          # Σ over steps of mean-over-heads reads
    n_k_sum: float = 0.0            # Σ over steps of n_k
    kv_rows_sum: float = 0.0        # Σ over steps of K+V rows read per KV head (union over its group)
    kv_steps: int = 0
    rebuilds: int = 0               # sphere_skip_v1: recenterings (all layers and KV heads)
    head_steps: int = 0             # sphere_skip_v1: KV-head decode steps
    gpu_reads: torch.Tensor | None = None   # sphere_fused: running sums kept on the GPU (no per-step sync)
    gpu_kv_rows: torch.Tensor | None = None

    def record_gpu(self, *, n_k: int, reads_mean: torch.Tensor, kv_rows: torch.Tensor) -> None:
        self.steps += 1
        self.kv_steps += 1
        self.n_k_sum += float(n_k)
        self.gpu_reads = reads_mean if self.gpu_reads is None else self.gpu_reads + reads_mean
        self.gpu_kv_rows = kv_rows if self.gpu_kv_rows is None else self.gpu_kv_rows + kv_rows

    def _flush_gpu(self) -> None:
        if self.gpu_reads is not None:
            self.reads_sum += float(self.gpu_reads)
            self.kv_rows_sum += float(self.gpu_kv_rows)
            self.gpu_reads = self.gpu_kv_rows = None

    @property
    def rebuild_rate(self) -> float | None:
        """Recenterings per KV head per decode step (sphere_skip_v1); None if not tracked."""
        return self.rebuilds / self.head_steps if self.head_steps else None

    def record(self, *, n_k: int, reads_per_head: torch.Tensor, kv_rows: float | None = None) -> None:
        self.steps += 1
        self.reads_sum += float(reads_per_head.float().mean().item())
        self.n_k_sum += float(n_k)
        if kv_rows is not None:
            self.kv_rows_sum += kv_rows
            self.kv_steps += 1

    @property
    def kv_read_fraction(self) -> float | None:
        """Fraction of all K and V rows read (incl. summary overhead); None if not tracked."""
        self._flush_gpu()
        if self.kv_steps != self.steps or not self.n_k_sum:
            return None
        return self.kv_rows_sum / (2 * self.n_k_sum)

    @property
    def avg_reads(self) -> float:
        return self.reads_sum / self.steps if self.steps else 0.0

    @property
    def avg_n_k(self) -> float:
        return self.n_k_sum / self.steps if self.steps else 0.0

    @property
    def read_fraction(self) -> float:
        self._flush_gpu()
        return self.reads_sum / self.n_k_sum if self.n_k_sum else 1.0


def _seed(base_seed: int, layer_idx: int, step: int) -> int:
    return (base_seed * 100_003 + layer_idx * 1_009 + step) % (2**31 - 1)


def make_decode_op(impl: str, *, base_seed: int, stats: ReadStats, cfg: dict):
    """Build a decode-seam callback for ``impl`` (closure owns its step counter)."""
    is_sampling = impl in _SAMPLING_IMPLS
    k_h = int(cfg.get("k_h", 0))
    counter = {"step": 0}
    state = None
    if impl == "sphere_skip_v1":                        # one incremental state per layer
        from ..attn.sphere_state import SphereState
        state = SphereState(C=int(cfg.get("C", 256)), window=int(cfg.get("window", 64)),
                            delta=float(cfg.get("delta", 0.0)), seed=int(cfg.get("seed", 0)),
                            kind=cfg.get("kind", "random"))

    fused = None
    if impl in ("sphere_fused", "sphere_sample"):       # fused Triton kernels, one index per layer
        from ..kernels.sphere_fused import SphereIndexFused
        fused = SphereIndexFused(C=int(cfg.get("C", 256)), window=int(cfg.get("window", 64)),
                                 delta=float(cfg.get("delta", 0.03)), capacity=int(cfg.get("capacity", 65536)),
                                 check_every=int(cfg.get("check_every", 16)), seed=int(cfg.get("seed", 0)))
        track = bool(cfg.get("track_reads", True))
        fused_prev = {"rebuilds": 0, "head_steps": 0}

    tail = None
    if impl == "sphere_tail":                          # estimated dropped bins, pure PyTorch, one index per layer
        from ..attn.tail_estimate import SphereIndexTail
        tail = SphereIndexTail(C=int(cfg.get("C", 256)), window=int(cfg.get("window", 64)),
                               delta=float(cfg.get("delta", 0.03)), capacity=int(cfg.get("capacity", 65536)),
                               check_every=int(cfg.get("check_every", 16)), seed=int(cfg.get("seed", 0)))

    def op(q, full_k, full_v, *, scale, layer_idx):
        qd = q[0, :, 0, :]                          # [H, d]
        if tail is not None:                        # order=1: estimate dropped bins; order="drop" (or a layer
                                                    # in drop_layers): drop them
            from ..attn.labeled import label_weighted_attention
            from ..attn.tail_estimate import attend_with_tail
            Kc, Vc = full_k[0], full_v[0]
            n_k = Kc.shape[1]
            tail.observe(Kc, Vc, n_k)
            labels, w = tail.labels_and_weights(qd, n=n_k, budget=float(cfg["budget"]),
                                                group=cfg.get("group", "sum_share"))
            order = "drop" if layer_idx in cfg.get("drop_layers", ()) else cfg.get("order", 1)
            d = qd.shape[1]
            if order == "drop":
                out = label_weighted_attention(qd, Kc, Vc, labels, w)
                extra = 0.0
            else:
                out = attend_with_tail(tail, qd, Kc, Vc, labels, w, order=int(order)).to(qd.dtype)
                extra = tail.C * (1 + 3 / d)                 # value sums + three scalars per bin
            rows = (torch.gather(w, 1, labels.long()) > 0).sum(1).float()         # [H_kv]
            stats.record(n_k=n_k, reads_per_head=rows,
                         kv_rows=float(2 * rows.mean()) + tail.C * (1 + 2 / d) + extra)
            counter["step"] += 1
            return out.view(1, -1, 1, d)
        if fused is not None:                       # reads the cache in place: [H_kv, n, d] views, no copy
            Kc, Vc = full_k[0], full_v[0]
            n_k = Kc.shape[1]
            if impl == "sphere_sample":                 # sample S of the unselected bins
                gen = torch.Generator().manual_seed(_seed(base_seed, layer_idx, counter["step"]))
                out, labels, w = fused.attend_sampled(qd, Kc, Vc, n=n_k, budget=float(cfg["budget"]),
                                                      S=int(cfg["S"]), alpha=float(cfg.get("alpha", 0.1)),
                                                      generator=gen)
            else:
                out, labels, w = fused.attend(qd, Kc, Vc, n=n_k, budget=float(cfg["budget"]),
                                              group=cfg.get("group", "sum_share"))
            if track:
                wk = w if w.shape[0] == Kc.shape[0] else w.view(Kc.shape[0], -1, w.shape[1]).amax(1)
                rows = (torch.gather(wk, 1, labels.long()) > 0).sum(1).float()        # [H_kv]
                stats.record_gpu(n_k=n_k, reads_mean=rows.mean(),
                                 kv_rows=2 * rows.mean() + fused.C * (1 + 2 / qd.shape[1]))
                stats.rebuilds += fused.rebuilds - fused_prev["rebuilds"]      # summed over layers
                stats.head_steps += fused.head_steps - fused_prev["head_steps"]
                fused_prev["rebuilds"], fused_prev["head_steps"] = fused.rebuilds, fused.head_steps
            counter["step"] += 1
            return out.view(1, -1, 1, qd.shape[1])
        K = full_k[0].permute(1, 0, 2).contiguous() # [n_k, H_kv, d]
        V = full_v[0].permute(1, 0, 2).contiguous()
        n_k = K.shape[0]

        call_cfg = dict(cfg)
        if impl == "skip_k":
            call_cfg["layer_idx"] = layer_idx          # for per-(layer,head) cluster seeding
        if is_sampling:
            gen = torch.Generator(device=qd.device).manual_seed(
                _seed(base_seed, layer_idx, counter["step"])
            )
            out, info = attn(qd, K, V, impl=impl, generator=gen, return_info=True, **call_cfg)
            # Distinct rows read = effective head keys + unique tail draws, capped
            # at n_k (head ∪ tail can't exceed the cache; when the head absorbs
            # ~all mass the tail is discarded, so reads saturate at n_k).
            reads = (min(k_h, n_k) + info.unique.to(qd.device).float()).clamp_(max=float(n_k))
            kv_rows = None
            if impl in ("santa", "santa_strat", "santa_sys"):       # all keys + union of sampled V rows
                H_kv = K.shape[1]
                v_union = unique_counts(info.idx.reshape(H_kv, -1)).float().mean()
                kv_rows = n_k + float(v_union)
        elif impl == "sphere_skip_v1":
            r0, h0 = state.rebuilds, state.head_steps
            out, info = state.attend(qd, K, V, budget=float(cfg["budget"]),
                                     group=cfg.get("group", "sum_share"))
            stats.rebuilds += state.rebuilds - r0
            stats.head_steps += state.head_steps - h0
            reads = info.unique.float()
            kv_rows = 2 * float(info.kv_union.float().mean()) + info.overhead_rows
        elif impl == "sphere_skip":                                 # deterministic; skips K and V rows
            out, info = attn(qd, K, V, impl=impl, return_info=True, **call_cfg)
            reads = info.unique.float()
            kv_rows = 2 * float(info.kv_union.float().mean()) + info.overhead_rows
        else:
            out = attn(qd, K, V, impl=impl, **call_cfg)
            if impl == "topk":                                  # reads exactly the top-k rows
                kk = min(int(cfg.get("k", n_k)), n_k)
                reads = torch.full((qd.shape[0],), float(kk), device=qd.device)
            else:                                               # dense: reads everything
                reads = torch.full((qd.shape[0],), float(n_k), device=qd.device)
            kv_rows = 2.0 * n_k if impl == "dense" else None

        stats.record(n_k=n_k, reads_per_head=reads, kv_rows=kv_rows)
        counter["step"] += 1
        return out.unsqueeze(0).unsqueeze(2)                   # [1, H, 1, d]

    return op


def install(model, impl: str, *, base_seed: int = 0, **cfg) -> ReadStats:
    """Set the decode op on every ``Attention`` module; return a shared ReadStats."""
    stats = ReadStats()
    for m in model.modules():
        if isinstance(m, Attention):
            m.decode_attn_op = make_decode_op(impl, base_seed=base_seed, stats=stats, cfg=cfg)
    return stats


def uninstall(model) -> None:
    """Restore the engine's default (dense SDPA) decode path."""
    for m in model.modules():
        if isinstance(m, Attention):
            m.decode_attn_op = None
