"""Fused Triton bin maintenance and selection for `cluster_skip`.

Replaces ~60 small PyTorch kernels per layer-step (``SphereIndexGPU``) with three Triton
kernels:

  - ``_bin_kernel``: for each key leaving the recent window, add it to the running key sum,
    center it with ``μ_ref``, find the nearest of the C fixed directions, update that bin's
    direction sum, min and max length and count, write its label; then set the head's drift
    flag if ``‖μ_t − μ_ref‖ > δ r̄``.
  - ``_score_kernel`` + ``_pick_kernel`` (grid: KV head × chunk of 64 bins): score every bin for the G query heads (max or min length × projection
    on the bin's mean direction), form the shared score (sum over heads of each head's
    softmax over bins) or keep per-head scores, and choose bins in decreasing score until
    their key counts reach the budget, without sorting: a bin is taken iff the bins scoring
    strictly higher hold fewer keys than the budget (one C × C comparison).

Recentering (rare) stays in PyTorch (``SphereIndexGPU._build``), triggered when the drift
flags are read every ``check_every`` steps. Bins exactly tied at the selection threshold are
all taken, where the sort-based reference takes the first; with continuous scores this
differs by at most one bin.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ..attn.sphere_gpu import SphereIndexGPU

NEG_INF = tl.constexpr(float("-inf"))
POS_INF = tl.constexpr(float("inf"))


@triton.jit
def _bin_kernel(k_ptr, s_kh, s_kn, dirs_ptr, s_dh, nact_ptr, sumdir_ptr, mmax_ptr, mmin_ptr, cnt_ptr, ksum_ptr, mu_ptr,
                rbar_ptr, flag_ptr, cap_ptr, oflag_ptr, lab_ptr, s_lh, start, stop, nbinned, delta,
                C: tl.constexpr, D: tl.constexpr, BC: tl.constexpr):
    h = tl.program_id(0)
    offs_d = tl.arange(0, D)
    mu = tl.load(mu_ptr + h * D + offs_d)
    ks = tl.load(ksum_ptr + h * D + offs_d)
    n_act = tl.load(nact_ptr + h).to(tl.int32)             # clusters in use; later slots are spare
    cap = tl.load(cap_ptr + h)
    over = tl.load(oflag_ptr + h)
    for p in range(start, stop):
        k = tl.load(k_ptr + h * s_kh + p * s_kn + offs_d).to(tl.float32)
        ks += k
        kr = k - mu
        mag = tl.sqrt(tl.sum(kr * kr, axis=0))
        kn = kr / tl.maximum(mag, 1e-12)
        best = mag * 0.0 - 2.0                             # below any cosine
        bi = (mag * 0.0).to(tl.int32)
        for c0 in range(0, n_act, BC):                     # only the blocks of slots in use
            offs_c = c0 + tl.arange(0, BC)
            dv = tl.load(dirs_ptr + h * s_dh + offs_c[:, None] * D + offs_d[None, :])
            sc = tl.where(offs_c < n_act, tl.sum(dv * kn[None, :], axis=1), NEG_INF)
            m = tl.max(sc, axis=0)
            am = tl.argmax(sc, axis=0)
            upd = m > best
            bi = tl.where(upd, c0 + am, bi)
            best = tl.where(upd, m, best)
        row = h * C + bi
        sd = tl.load(sumdir_ptr + row * D + offs_d)
        tl.store(sumdir_ptr + row * D + offs_d, sd + kn)
        tl.store(mmax_ptr + row, tl.maximum(tl.load(mmax_ptr + row), mag))
        tl.store(mmin_ptr + row, tl.minimum(tl.load(mmin_ptr + row), mag))
        cn = tl.load(cnt_ptr + row) + 1.0
        tl.store(cnt_ptr + row, cn)
        over = tl.maximum(over, ((cn > cap) & (n_act < C)).to(tl.int32))    # this cluster needs splitting
        tl.store(lab_ptr + h * s_lh + p, bi.to(tl.int16))
        tl.debug_barrier()
    tl.store(oflag_ptr + h, over)
    tl.store(ksum_ptr + h * D + offs_d, ks)
    mu_t = ks / nbinned
    dist = tl.sqrt(tl.sum((mu_t - mu) * (mu_t - mu), axis=0))
    rb = tl.load(rbar_ptr + h)
    tl.store(flag_ptr + h, tl.maximum(tl.load(flag_ptr + h), (dist > delta * rb).to(tl.int32)))


@triton.jit
def _score_kernel(q_ptr, s_qh, sumdir_ptr, mmax_ptr, mmin_ptr, cnt_ptr, nact_ptr, scratch_ptr,
                  G: tl.constexpr, G_PAD: tl.constexpr, C: tl.constexpr, D: tl.constexpr, BC: tl.constexpr):
    """Grid (H_kv, C / BC): estimated max score of each bin for each query head -> scratch.
    A program whose slots are all spare only writes −∞."""
    h = tl.program_id(0)
    c0 = tl.program_id(1) * BC
    offs_g = tl.arange(0, G_PAD)
    offs_c = c0 + tl.arange(0, BC)
    out = scratch_ptr + (h * G_PAD + offs_g)[:, None] * C + offs_c[None, :]
    if c0 < tl.load(nact_ptr + h):
        gm = offs_g < G
        offs_d = tl.arange(0, D)
        q = tl.load(q_ptr + (h * G + offs_g)[:, None] * s_qh + offs_d[None, :], mask=gm[:, None], other=0.0)
        sd = tl.load(sumdir_ptr + (h * C + offs_c)[:, None] * D + offs_d[None, :])            # [BC, D]
        nrm = tl.maximum(tl.sqrt(tl.sum(sd * sd, axis=1)), 1e-12)
        p = tl.dot(q.to(tl.float32), tl.trans(sd), input_precision="ieee") / nrm[None, :]    # [G_PAD, BC]
        mx = tl.load(mmax_ptr + h * C + offs_c)
        mn = tl.load(mmin_ptr + h * C + offs_c)
        cn = tl.load(cnt_ptr + h * C + offs_c)
        e = tl.where(p >= 0, mx[None, :] * p, mn[None, :] * p)
        tl.store(out, tl.where(cn[None, :] > 0, e, NEG_INF))
    else:
        tl.store(out, tl.full([G_PAD, BC], NEG_INF, tl.float32))


@triton.jit
def _shared_pick(scratch_ptr, cnt_ptr, w_ptr, s_wr, h, c0, n_act, need, scale,
                 G: tl.constexpr, G_PAD: tl.constexpr, C: tl.constexpr, BC: tl.constexpr):
    """Shared selection for slots ``[c0, c0 + BC)`` of head ``h``: rank by the sum over query heads
    of each head's softmax over clusters, and take a cluster iff those ranked strictly higher hold
    fewer keys than the budget. Reads only the blocks of slots in use, so the cost follows the
    number of clusters rather than the number of slots."""
    offs_g = tl.arange(0, G_PAD)
    gm = offs_g < G
    offs_b = tl.arange(0, BC)
    row = scratch_ptr + (h * G_PAD + offs_g)[:, None] * C
    zmax = tl.full([G_PAD], NEG_INF, tl.float32)
    for j0 in range(0, n_act, BC):
        z = tl.where(gm[:, None], tl.load(row + (j0 + offs_b)[None, :]) * scale, 0.0)
        zmax = tl.maximum(zmax, tl.max(z, axis=1))
    tot = tl.zeros([G_PAD], tl.float32)
    for j0 in range(0, n_act, BC):
        z = tl.where(gm[:, None], tl.load(row + (j0 + offs_b)[None, :]) * scale, 0.0)
        cn = tl.load(cnt_ptr + h * C + j0 + offs_b)
        tot += tl.sum(tl.where(cn[None, :] > 0, tl.exp(z - zmax[:, None]), 0.0), axis=1)
    inv = 1.0 / tot
    z = tl.where(gm[:, None], tl.load(row + (c0 + offs_b)[None, :]) * scale, 0.0)
    cn_i = tl.load(cnt_ptr + h * C + c0 + offs_b)
    ex = tl.where(cn_i[None, :] > 0, tl.exp(z - zmax[:, None]), 0.0)
    si = tl.where(cn_i > 0, tl.sum(tl.where(gm[:, None], ex * inv[:, None], 0.0), axis=0), NEG_INF)
    higher = tl.zeros([BC], tl.float32)
    for j0 in range(0, n_act, BC):
        z = tl.where(gm[:, None], tl.load(row + (j0 + offs_b)[None, :]) * scale, 0.0)
        cn = tl.load(cnt_ptr + h * C + j0 + offs_b)
        ex = tl.where(cn[None, :] > 0, tl.exp(z - zmax[:, None]), 0.0)
        sh = tl.where(cn > 0, tl.sum(tl.where(gm[:, None], ex * inv[:, None], 0.0), axis=0), NEG_INF)
        above = (sh[None, :] > si[:, None]) & ((j0 + offs_b)[None, :] != (c0 + offs_b)[:, None])   # never itself
        higher += tl.sum(tl.where(above, cn[None, :], 0.0), axis=1)
    tl.store(w_ptr + h * s_wr + c0 + offs_b, ((higher < need) & (si > NEG_INF)).to(tl.float32))


@triton.jit
def _pick_kernel(scratch_ptr, cnt_ptr, nact_ptr, w_ptr, s_wr, need, scale,
                 G: tl.constexpr, G_PAD: tl.constexpr, C: tl.constexpr, BC: tl.constexpr,
                 PER_HEAD: tl.constexpr):
    """Grid (H_kv, C / BC): select this program's bins. A bin is taken iff the bins scoring
    strictly higher hold fewer keys than the budget (exact, no sort; ties all taken).
    A program whose slots are all spare does nothing (their weights stay 0)."""
    h = tl.program_id(0)
    c0 = tl.program_id(1) * BC
    n_act = tl.load(nact_ptr + h).to(tl.int32)
    if c0 < n_act:
        if PER_HEAD:
            offs_c = tl.arange(0, C)
            offs_i = c0 + tl.arange(0, BC)
            cn = tl.load(cnt_ptr + h * C + offs_c)
            for r in tl.static_range(G):
                srow = tl.load(scratch_ptr + (h * G_PAD + r) * C + offs_c)
                si = tl.load(scratch_ptr + (h * G_PAD + r) * C + offs_i)
                higher = tl.sum(tl.where(srow[None, :] > si[:, None], cn[None, :], 0.0), axis=1)
                tl.store(w_ptr + (h * G + r) * s_wr + offs_i, ((higher < need) & (si > NEG_INF)).to(tl.float32))
        else:
            _shared_pick(scratch_ptr, cnt_ptr, w_ptr, s_wr, h, c0, n_act, need, scale, G, G_PAD, C, BC)


class SphereIndexFused(SphereIndexGPU):
    """``SphereIndexGPU`` with the per-step work in Triton kernels. Needs float32 state
    (BF16 or FP32 keys) on CUDA. ``labels_and_weights`` returns persistent buffers that the
    next call overwrites.

    ``async_check=True`` (default) reads the per-head flags (mean has drifted; a cluster has
    grown past its cap) without a host synchronization: every ``check_every`` steps the flags are
    copied to pinned host memory, and the copy is acted on at a later check once it has landed.
    Recentering and splitting therefore lag by at least one check period; attention stays exact
    over the rows read whatever the lag. A caller that never waits for the GPU (a timing loop fed
    known tokens) can queue many steps ahead of it, so a copy still outstanding after ``max_lag``
    further checks is waited for; a caller that reads each token before the next step never waits.
    Flagged heads are split every ``split_every`` checks, all their clusters over the cap together,
    since a split pass costs about a millisecond of small PyTorch operations however few it splits."""

    def __init__(self, *args, async_check: bool = True, max_lag: int = 2, split_every: int = 4, **kw):
        super().__init__(*args, **kw)
        self.async_check, self.max_lag, self.split_every = async_check, max_lag, split_every
        self._waited = 0
        self._checks = 0
        self._over_heads = None                 # heads flagged for a split since the last split pass

    def _bc(self) -> int:
        """Bin chunk per program: 64, or C if smaller. C must be a power of two ≥ 16."""
        assert self.C >= 16 and self.C & (self.C - 1) == 0, "C must be a power of two ≥ 16"
        return min(64, self.C)

    def _alloc(self, H_kv, d, dtype, device):
        super()._alloc(H_kv, d, torch.float32, device)
        self.flags = torch.zeros(H_kv, dtype=torch.int32, device=device)
        self.oflags = torch.zeros(H_kv, dtype=torch.int32, device=device)   # a cluster of this head is over its cap
        self._flags_host = torch.zeros(2, H_kv, dtype=torch.int32, pin_memory=True)
        self._pending = None
        self._w = {}
        self._scratch = None

    def _init(self, K, n):
        super()._init(K, n)

    def _dirs_arg(self):
        """Directions and their per-head stride for the binning kernels: the shared random
        directions (stride 0) or each head's fitted centroids."""
        return (self.cent, self.C * self.d) if self.partition == "kmeans" else (self.dirs, 0)

    def _advance(self, K, n):
        new_end = max(1, n - self.window)
        if new_end > self.end:
            _bin_kernel[(self.H_kv,)](
                K, K.stride(0), K.stride(1), *self._dirs_arg(), self.n_c, self.sum_dir, self.mmax, self.mmin, self.count,
                self.ksum, self.mu_ref, self.rbar, self.flags, self.cap, self.oflags, self.labels,
                self.labels.stride(0), self.end, new_end, float(new_end - 1), float(self.delta),
                C=self.C, D=self.d, BC=self._bc(), num_warps=4)
            self.end = new_end
        self.head_steps += self.H_kv
        if self.end - 1 <= 0:
            return
        self._steps += 1
        if self._steps % self.check_every != 0:
            return
        self._check_flags(K)

    def _host_end(self) -> None:
        """Bring ``self.end`` up to date before host-side work (kept on the GPU by subclasses)."""

    def _split_flagged(self, K, heads) -> None:
        if heads.numel():
            self._host_end()
            cand = torch.zeros(self.H_kv, self.C, dtype=torch.bool, device=self.device)
            cand[heads] = True
            self._split_overflow(K, cand)

    def _check_flags(self, K) -> None:
        if not self.async_check:
            self._split_flagged(K, self.oflags.nonzero().squeeze(1))       # host sync
            self._rebuild(K, self.flags.nonzero().squeeze(1))
            self.flags.zero_()
            self.oflags.zero_()
            return
        if self._pending is not None:
            self._waited += 1
            if self._waited > self.max_lag and not self._pending.query():  # the host has run far ahead of the GPU
                self._pending.synchronize()
        if self._pending is not None and self._pending.query():            # earlier copy has landed
            drift = self._flags_host[0].nonzero().squeeze(1)
            self._over_heads = self._flags_host[1].clone() if self._over_heads is None \
                else self._over_heads | self._flags_host[1]
            self._pending = None
            self._rebuild(K, drift.to(self.device))
        self._checks += 1
        if self._checks % self.split_every == 0 and self._over_heads is not None:
            self._split_flagged(K, self._over_heads.nonzero().squeeze(1).to(self.device))
            self._over_heads = None
        if self._pending is None:
            self._flags_host[0].copy_(self.flags, non_blocking=True)
            self._flags_host[1].copy_(self.oflags, non_blocking=True)
            self._pending = torch.cuda.Event()
            self._pending.record()
            self._waited = 0
            self.flags.zero_()
            self.oflags.zero_()

    def _rebuild(self, K, heads):
        if heads.numel():
            self._host_end()
            mu_t = self.ksum[heads] / (self.end - 1)
            self._build(K, heads, mu_t)
            self.rebuilds += int(heads.numel())

    def labels_and_weights(self, q, *, n, budget, group="sum_share"):
        if group not in ("sum_share", "per_head"):
            return super().labels_and_weights(q, n=n, budget=budget, group=group)
        H, d = q.shape
        G = H // self.H_kv
        G_PAD = max(16, triton.next_power_of_2(G))
        rows = H if group == "per_head" else self.H_kv
        key = (group, rows)
        if key not in self._w:
            w = torch.zeros(rows, self.C + 1, dtype=torch.float32, device=self.device)
            w[:, self.C] = 1.0
            self._w[key] = w
        w = self._w[key]
        need = math.ceil(budget * (self.end - 1))
        if self.end <= 1 or need <= 0:
            w[:, :self.C] = 0.0
            return self.labels[:, :n], w
        if self._scratch is None:
            self._scratch = torch.empty(self.H_kv, G_PAD, self.C, dtype=torch.float32, device=self.device)
        q = q.contiguous()
        bc = self._bc()
        grid = (self.H_kv, self.C // bc)
        _score_kernel[grid](q, q.stride(0), self.sum_dir, self.mmax, self.mmin, self.count, self.n_c, self._scratch,
                            G=G, G_PAD=G_PAD, C=self.C, D=d, BC=bc, num_warps=4)
        _pick_kernel[grid](self._scratch, self.count, self.n_c, w, w.stride(0), float(need), 1.0 / math.sqrt(d),
                           G=G, G_PAD=G_PAD, C=self.C, BC=bc, PER_HEAD=(group == "per_head"), num_warps=4)
        return self.labels[:, :n], w

    def attend(self, q, K, V, *, n, budget, group="sum_share"):
        """One decode step for this layer: update bins, select, and run the compacted kernel.
        ``K, V [H_kv, ≥n, d]`` (engine cache layout, in place). Returns ``(out, labels, w)``."""
        from .labeled_attn import label_weighted_attention_compact
        self.observe(K, n)
        labels, w = self.labels_and_weights(q, n=n, budget=budget, group=group)
        out = label_weighted_attention_compact(q, K[:, :n], V[:, :n], labels, w)
        return out, labels, w

    def attend_sampled(self, q, K, V, *, n, budget, S, alpha, generator=None):
        """``attend`` plus S bins drawn from the unselected ones (``ssa.attn.tail_sample``);
        drawn bins are read in full and weighted ``c_b/(S π_b)``. Returns ``(out, labels, w)``."""
        from ..attn.tail_sample import proposal_from_scores, tail_sample_weights
        from .labeled_attn import label_weighted_attention_compact
        self.observe(K, n)
        labels, w = self.labels_and_weights(q, n=n, budget=budget, group="sum_share")
        w2 = w.clone()
        if S > 0 and self._scratch is not None and self.end > 1 and math.ceil(budget * (self.end - 1)) > 0:
            G = q.shape[0] // self.H_kv
            e = self._scratch[:, :G, :]
            prop = proposal_from_scores(e, self.count, scale=1.0 / math.sqrt(q.shape[1]))
            w2[:, :self.C] = tail_sample_weights(prop, self.count, w[:, :self.C] > 0, S=S, alpha=alpha,
                                                 generator=generator)
        out = label_weighted_attention_compact(q, K[:, :n], V[:, :n], labels, w2)
        return out, labels, w2

