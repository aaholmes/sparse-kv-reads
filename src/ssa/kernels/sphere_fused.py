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
def _member_block(k_ptr, s_kh, s_kn, mem_ptr, mu, h, j0, m, D: tl.constexpr, MS: tl.constexpr, BM: tl.constexpr):
    """Members ``[j0, j0 + BM)`` of the cluster being split: unit directions, lengths, validity, positions."""
    offs_m = j0 + tl.arange(0, BM)
    vm = offs_m < m
    pos = tl.load(mem_ptr + h * MS + offs_m, mask=vm, other=0)
    k = tl.load(k_ptr + h * s_kh + pos[:, None] * s_kn + tl.arange(0, D)[None, :], mask=vm[:, None], other=0.0)
    kr = k.to(tl.float32) - mu[None, :]
    mag = tl.sqrt(tl.sum(kr * kr, axis=1))
    return kr / tl.maximum(mag, 1e-12)[:, None], mag, vm, pos


@triton.jit
def _split_cluster(k_ptr, s_kh, s_kn, dirs_ptr, cent_ptr, s_dh, nact_ptr, sumdir_ptr, qdir_ptr, mmax_ptr, mmin_ptr,
                   cnt_ptr, mu, lab_ptr, s_lh, mem_ptr, proj_ptr, side_ptr, nsplit_ptr, h, c, new, stop,
                   C: tl.constexpr, D: tl.constexpr, MS: tl.constexpr, BL: tl.constexpr, BM: tl.constexpr,
                   Q8: tl.constexpr):
    """Split cluster ``c`` of head ``h`` in two, the second half taking slot ``new``; the rule is
    ``ssa.attn.sphere_gpu._bisect_padded``'s. Scans the head's labels ``[1, stop)`` for the
    members, runs 2-means on their directions (from the first member and the one farthest from
    it), cuts at the median projection on the line between the two centroids when a half would
    hold under a quarter of the members, then rewrites both summaries, centroids and labels.
    ``cent_ptr`` holds the float32 centroids; with ``Q8``, ``dirs_ptr`` and ``qdir_ptr`` hold the
    8-bit centroids and mean directions."""
    offs_d = tl.arange(0, D)
    offs_l = tl.arange(0, BL)
    offs_b = tl.arange(0, BM)
    m = c * 0
    for b0 in range(1, stop, BL):                                         # member positions, in order
        offs = b0 + offs_l
        inb = offs < stop
        hit = inb & (tl.load(lab_ptr + h * s_lh + offs, mask=inb, other=-1).to(tl.int32) == c)
        hi = hit.to(tl.int32)
        slot = tl.cumsum(hi, 0) - 1 + m
        tl.store(mem_ptr + h * MS + slot, offs, mask=hit & (slot < MS))
        m += tl.sum(hi, 0)
    tl.debug_barrier()
    m = tl.minimum(m, MS)
    p0 = tl.load(mem_ptr + h * MS)
    a = tl.load(k_ptr + h * s_kh + p0 * s_kn + offs_d).to(tl.float32) - mu
    a = a / tl.maximum(tl.sqrt(tl.sum(a * a, axis=0)), 1e-12)
    low = tl.sum(a * 0.0, axis=0) + 2.0
    far = c * 0
    for j0 in range(0, m, BM):                                            # the member farthest from the first
        kn, mag, vm, pos = _member_block(k_ptr, s_kh, s_kn, mem_ptr, mu, h, j0, m, D, MS, BM)
        sim = tl.where(vm, tl.sum(kn * a[None, :], axis=1), 2.0)
        mn = tl.min(sim, axis=0)
        upd = mn < low
        far = tl.where(upd, tl.load(mem_ptr + h * MS + j0 + tl.argmin(sim, axis=0)), far)
        low = tl.where(upd, mn, low)
    b = tl.load(k_ptr + h * s_kh + far * s_kn + offs_d).to(tl.float32) - mu
    b = b / tl.maximum(tl.sqrt(tl.sum(b * b, axis=0)), 1e-12)
    mf = m.to(tl.float32)
    for it in range(4):                                                   # 2-means on cosine similarity
        s0 = tl.zeros([D], tl.float32)
        s1 = tl.zeros([D], tl.float32)
        n1 = mf * 0.0
        for j0 in range(0, m, BM):
            kn, mag, vm, pos = _member_block(k_ptr, s_kh, s_kn, mem_ptr, mu, h, j0, m, D, MS, BM)
            one = vm & (tl.sum(kn * b[None, :], axis=1) > tl.sum(kn * a[None, :], axis=1))
            s1 += tl.sum(tl.where(one[:, None], kn, 0.0), axis=0)
            s0 += tl.sum(tl.where((vm & (one == 0))[:, None], kn, 0.0), axis=0)
            n1 += tl.sum(one.to(tl.float32), axis=0)
        a = tl.where(mf - n1 > 0, s0 / tl.maximum(tl.sqrt(tl.sum(s0 * s0, axis=0)), 1e-12), a)
        b = tl.where(n1 > 0, s1 / tl.maximum(tl.sqrt(tl.sum(s1 * s1, axis=0)), 1e-12), b)
    n1 = mf * 0.0
    for j0 in range(0, m, BM):                                            # final sides and projections
        kn, mag, vm, pos = _member_block(k_ptr, s_kh, s_kn, mem_ptr, mu, h, j0, m, D, MS, BM)
        one = vm & (tl.sum(kn * b[None, :], axis=1) > tl.sum(kn * a[None, :], axis=1))
        tl.store(side_ptr + h * MS + j0 + offs_b, one.to(tl.int32), mask=vm)
        tl.store(proj_ptr + h * MS + j0 + offs_b, tl.sum(kn * (b - a)[None, :], axis=1), mask=vm)
        n1 += tl.sum(one.to(tl.float32), axis=0)
    tl.debug_barrier()
    if (n1 < 0.25 * mf) | (n1 > 0.75 * mf):                               # lopsided: cut at the median projection
        half = m // 2
        for i0 in range(0, m, BM):
            vi = (i0 + offs_b) < m
            pi = tl.load(proj_ptr + h * MS + i0 + offs_b, mask=vi, other=0.0)
            rank = tl.zeros([BM], tl.int32)
            for j0 in range(0, m, BM):
                vj = (j0 + offs_b) < m
                pj = tl.load(proj_ptr + h * MS + j0 + offs_b, mask=vj, other=0.0)
                less = (pj[None, :] < pi[:, None]) | ((pj[None, :] == pi[:, None])
                                                      & ((j0 + offs_b)[None, :] < (i0 + offs_b)[:, None]))
                rank += tl.sum((less & vj[None, :]).to(tl.int32), axis=1)
            tl.store(side_ptr + h * MS + i0 + offs_b, (rank >= half).to(tl.int32), mask=vi)
        tl.debug_barrier()
    s0 = tl.zeros([D], tl.float32)
    s1 = tl.zeros([D], tl.float32)
    n1 = mf * 0.0
    x0 = mf * 0.0
    x1 = mf * 0.0
    y0 = mf * 0.0 + POS_INF
    y1 = mf * 0.0 + POS_INF
    for j0 in range(0, m, BM):                                            # both halves' summaries; relabel one
        kn, mag, vm, pos = _member_block(k_ptr, s_kh, s_kn, mem_ptr, mu, h, j0, m, D, MS, BM)
        one = vm & (tl.load(side_ptr + h * MS + j0 + offs_b, mask=vm, other=0) > 0)
        zero = vm & (one == 0)
        s1 += tl.sum(tl.where(one[:, None], kn, 0.0), axis=0)
        s0 += tl.sum(tl.where(zero[:, None], kn, 0.0), axis=0)
        n1 += tl.sum(one.to(tl.float32), axis=0)
        x1 = tl.maximum(x1, tl.max(tl.where(one, mag, 0.0), axis=0))
        x0 = tl.maximum(x0, tl.max(tl.where(zero, mag, 0.0), axis=0))
        y1 = tl.minimum(y1, tl.min(tl.where(one, mag, POS_INF), axis=0))
        y0 = tl.minimum(y0, tl.min(tl.where(zero, mag, POS_INF), axis=0))
        tl.store(lab_ptr + h * s_lh + pos, (pos * 0 + new).to(tl.int16), mask=one)
    r0 = h * C + c
    r1 = h * C + new
    tl.store(sumdir_ptr + r0 * D + offs_d, s0)
    tl.store(sumdir_ptr + r1 * D + offs_d, s1)
    tl.store(mmax_ptr + r0, x0)
    tl.store(mmax_ptr + r1, x1)
    tl.store(mmin_ptr + r0, y0)
    tl.store(mmin_ptr + r1, y1)
    tl.store(cnt_ptr + r0, mf - n1)
    tl.store(cnt_ptr + r1, n1)
    u0 = s0 / tl.maximum(tl.sqrt(tl.sum(s0 * s0, axis=0)), 1e-12)
    u1 = s1 / tl.maximum(tl.sqrt(tl.sum(s1 * s1, axis=0)), 1e-12)
    tl.store(cent_ptr + h * s_dh + c * D + offs_d, u0)
    tl.store(cent_ptr + h * s_dh + new * D + offs_d, u1)
    if Q8:                                                                # the 8-bit copies the per-step reads use
        b0 = tl.floor(u0 * 127.0 + 0.5).to(tl.int8)
        b1 = tl.floor(u1 * 127.0 + 0.5).to(tl.int8)
        tl.store(dirs_ptr + h * s_dh + c * D + offs_d, b0)
        tl.store(dirs_ptr + h * s_dh + new * D + offs_d, b1)
        tl.store(qdir_ptr + r0 * D + offs_d, b0)
        tl.store(qdir_ptr + r1 * D + offs_d, b1)
    tl.store(nact_ptr + h, (new + 1).to(tl.int64))
    done = tl.load(nsplit_ptr + h) + 1
    tl.debug_barrier()                                                    # all threads read before any writes
    tl.store(nsplit_ptr + h, done)
    tl.debug_barrier()


@triton.jit
def _bin_kernel(k_ptr, s_kh, s_kn, dirs_ptr, cent_ptr, s_dh, nact_ptr, sumdir_ptr, qdir_ptr, mmax_ptr, mmin_ptr, cnt_ptr, ksum_ptr, mu_ptr,
                rbar_ptr, flag_ptr, cap_ptr, vthr_ptr, vmin, mem_ptr, proj_ptr, side_ptr, nsplit_ptr, lab_ptr, s_lh, start, stop, nbinned,
                delta, capk, reset_at, C: tl.constexpr, D: tl.constexpr, BC: tl.constexpr, MS: tl.constexpr,
                Q8: tl.constexpr):
    h = tl.program_id(0)
    offs_d = tl.arange(0, D)
    mu = tl.load(mu_ptr + h * D + offs_d)
    ks = tl.load(ksum_ptr + h * D + offs_d)
    n_act = tl.load(nact_ptr + h).to(tl.int32)             # clusters in use; later slots are spare
    cap = tl.load(cap_ptr + h)
    vthr = tl.load(vthr_ptr + h)
    for p in range(start, stop):
        k = tl.load(k_ptr + h * s_kh + p * s_kn + offs_d).to(tl.float32)
        ks += k
        kr = k - mu
        mag = tl.sqrt(tl.sum(kr * kr, axis=0))
        kn = kr / tl.maximum(mag, 1e-12)
        best = mag * 0.0 - 1000.0                          # below any score
        bi = (mag * 0.0).to(tl.int32)
        for c0 in range(0, n_act, BC):                     # only the blocks of slots in use
            offs_c = c0 + tl.arange(0, BC)
            dv = tl.load(dirs_ptr + h * s_dh + offs_c[:, None] * D + offs_d[None, :]).to(tl.float32)
            # 8-bit centroids have length 127 to within about 0.3%, so they are compared without normalizing
            sc = tl.where(offs_c < n_act, tl.sum(dv * kn[None, :], axis=1), NEG_INF)
            m = tl.max(sc, axis=0)
            am = tl.argmax(sc, axis=0)
            upd = m > best
            bi = tl.where(upd, c0 + am, bi)
            best = tl.where(upd, m, best)
        row = h * C + bi
        sd = tl.load(sumdir_ptr + row * D + offs_d) + kn
        tl.store(sumdir_ptr + row * D + offs_d, sd)
        if Q8:                                             # refresh this cluster's 8-bit mean direction
            un = sd / tl.maximum(tl.sqrt(tl.sum(sd * sd, axis=0)), 1e-12)
            tl.store(qdir_ptr + row * D + offs_d, tl.floor(un * 127.0 + 0.5).to(tl.int8))
        tl.store(mmax_ptr + row, tl.maximum(tl.load(mmax_ptr + row), mag))
        tl.store(mmin_ptr + row, tl.minimum(tl.load(mmin_ptr + row), mag))
        # Every thread of the program evaluates this scalar. The barrier makes them all read the old
        # count before any writes the new one; otherwise a thread that reads late sees the count
        # already incremented, and the threads then disagree on whether to split.
        cn = tl.load(cnt_ptr + row) + 1.0
        tl.debug_barrier()
        tl.store(cnt_ptr + row, cn)
        tl.store(lab_ptr + h * s_lh + p, bi.to(tl.int16))
        tl.debug_barrier()
        wide = (1.0 - tl.sum(sd * sd, axis=0) / (cn * cn) > vthr) & (cn >= vmin)       # spread over the threshold
        if ((cn > tl.maximum(cap, capk * p)) | wide) & (n_act < C) & (cn >= 2.0) & (cn <= MS):   # split it now
            _split_cluster(k_ptr, s_kh, s_kn, dirs_ptr, cent_ptr, s_dh, nact_ptr, sumdir_ptr, qdir_ptr, mmax_ptr,
                           mmin_ptr, cnt_ptr, mu, lab_ptr, s_lh, mem_ptr, proj_ptr, side_ptr, nsplit_ptr, h, bi,
                           n_act, p + 1, C, D, MS, 1024, 64, Q8)
            n_act += 1
    tl.store(ksum_ptr + h * D + offs_d, ks)
    mu_t = ks / nbinned
    dist = tl.sqrt(tl.sum((mu_t - mu) * (mu_t - mu), axis=0))
    rb = tl.load(rbar_ptr + h)
    # bit 0: the mean has drifted; bit 1: the cluster count has reached the reset threshold
    tl.store(flag_ptr + h, tl.load(flag_ptr + h) | (dist > delta * rb).to(tl.int32) | ((n_act >= reset_at).to(tl.int32) * 2))


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
        sd = tl.load(sumdir_ptr + (h * C + offs_c)[:, None] * D + offs_d[None, :]).to(tl.float32)   # [BC, D]
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

    ``async_check=True`` (default) reads the drift flags without a host synchronization:
    every ``check_every`` steps the flags are copied to pinned host memory, and the copy is
    acted on at a later check once it has landed. Recentering therefore lags by at least one
    check period; attention stays exact over the rows read whatever the lag. A caller that never waits for the GPU (a timing loop fed
    known tokens) can queue many steps ahead of it, so a copy still outstanding after ``max_lag``
    further checks is waited for; a caller that reads each token before the next step never waits.

    Splitting during decoding happens inside the binning kernel: the program that inserts a key
    into a cluster and takes it past its cap splits that cluster before the step's selection, with
    no host work. A cluster of more than ``split_scratch`` keys is left alone.

    ``summary_bits=8`` (default): scoring and the nearest-centroid search, which touch every
    cluster in use at every step, read 8-bit copies of each cluster's mean direction and centroid
    (unit vectors × 127, rounded). The float32 direction sums remain the accumulators; a copy is
    rewritten when its cluster changes. ``summary_bits=32`` reads the float32 vectors."""

    def __init__(self, *args, async_check: bool = True, max_lag: int = 2, split_scratch: int = 8192,
                 summary_bits: int = 8, **kw):
        if summary_bits not in (8, 32):
            raise ValueError("summary_bits must be 8 or 32")
        self.summary_bits = summary_bits
        self.dir8 = self.cent8 = None
        self._splits_host = 0
        self.nsplit = None
        super().__init__(*args, **kw)
        self.async_check, self.max_lag, self.split_scratch = async_check, max_lag, split_scratch
        self._waited = 0

    @property
    def splits(self) -> int:
        """Clusters created by splitting (reads a GPU counter)."""
        return self._splits_host + (int(self.nsplit.sum()) if self.nsplit is not None else 0)

    @splits.setter
    def splits(self, v: int) -> None:
        self._splits_host = v - (int(self.nsplit.sum()) if self.nsplit is not None else 0)

    def _bc(self) -> int:
        """Bin chunk per program: 64, or C if smaller. C must be a power of two ≥ 16."""
        assert self.C >= 16 and self.C & (self.C - 1) == 0, "C must be a power of two ≥ 16"
        return min(64, self.C)

    def _alloc(self, H_kv, d, dtype, device):
        super()._alloc(H_kv, d, torch.float32, device)
        self.flags = torch.zeros(H_kv, dtype=torch.int32, device=device)
        self._flags_host = torch.zeros(H_kv, dtype=torch.int32, pin_memory=True)
        ms = min(self.split_scratch, self.capacity)                         # scratch for the in-kernel split
        self.MS = ms
        self._mem = torch.zeros(H_kv, ms, dtype=torch.int32, device=device)
        self._proj = torch.zeros(H_kv, ms, dtype=torch.float32, device=device)
        self._side = torch.zeros(H_kv, ms, dtype=torch.int32, device=device)
        self.nsplit = torch.zeros(H_kv, dtype=torch.int32, device=device)
        if self.summary_bits == 8:
            self.dir8 = torch.zeros(H_kv, self.C, d, dtype=torch.int8, device=device)
            self.cent8 = torch.zeros_like(self.dir8) if self.partition == "kmeans" else None
        self._pending = None
        self._w = {}
        self._scratch = None

    def _init(self, K, n):
        super()._init(K, n)

    def _dirs_arg(self):
        """For the binning kernels: the directions searched for the nearest one, the float32
        centroids a split rewrites, and their per-head stride. Shared random directions have stride 0."""
        if self.partition != "kmeans":
            return self.dirs, self.dirs, 0
        return (self.cent8 if self.cent8 is not None else self.cent), self.cent, self.C * self.d

    def _qdir(self) -> torch.Tensor:
        """What scoring reads for each cluster's mean direction."""
        return self.dir8 if self.dir8 is not None else self.sum_dir

    @staticmethod
    def _to8(x: torch.Tensor) -> torch.Tensor:
        return torch.floor(torch.nn.functional.normalize(x, dim=-1) * 127 + 0.5).to(torch.int8)

    def _refresh8(self, heads=None) -> None:
        """Rewrite the 8-bit copies of ``heads`` (all by default) after host-side changes."""
        if self.dir8 is None:
            return
        sel = slice(None) if heads is None else heads
        self.dir8[sel] = self._to8(self.sum_dir[sel])
        if self.cent8 is not None:
            self.cent8[sel] = self._to8(self.cent[sel])

    def _build(self, K, heads, mu):
        super()._build(K, heads, mu)
        self._refresh8(heads)

    def _split_overflow(self, K, cand=None, nmeans=False):
        super()._split_overflow(K, cand, nmeans)
        self._refresh8()

    def _advance(self, K, n):
        new_end = max(1, n - self.window)
        if new_end > self.end:
            _bin_kernel[(self.H_kv,)](
                K, K.stride(0), K.stride(1), *self._dirs_arg(), self.n_c, self.sum_dir, self._qdir(), self.mmax, self.mmin,
                self.count,
                self.ksum, self.mu_ref, self.rbar, self.flags, self.cap, self.vthr, float(self.var_min), self._mem,
                self._proj, self._side, self.nsplit,
                self.labels, self.labels.stride(0), self.end, new_end, float(new_end - 1), float(self.delta),
                float(self.capk), self.reset_at or 2 ** 30, C=self.C, D=self.d, BC=self._bc(), MS=self.MS, Q8=self.dir8 is not None, num_warps=4)
            self.end = new_end
        self.head_steps += self.H_kv
        if self.end - 1 <= 0:
            return
        self._steps += 1
        if self._steps % self.check_every != 0:
            return
        self._check_flags(K)

    def _check_flags(self, K) -> None:
        if not self.async_check:
            self._act_on(K, self.flags)                                    # host sync
            self.flags.zero_()
            return
        if self._pending is not None:
            self._waited += 1
            if self._waited > self.max_lag and not self._pending.query():  # the host has run far ahead of the GPU
                self._pending.synchronize()
        if self._pending is not None and self._pending.query():            # earlier copy has landed
            self._pending = None
            self._act_on(K, self._flags_host.to(self.device))
        if self._pending is None:
            self._flags_host.copy_(self.flags, non_blocking=True)
            self._pending = torch.cuda.Event()
            self._pending.record()
            self._waited = 0
            self.flags.zero_()

    def _act_on(self, K, flags) -> None:
        """Refit the heads whose cluster count reached the reset threshold; recenter the others that drifted."""
        reset = (flags & 2) != 0
        self._reset_heads(K, reset.nonzero().squeeze(1))
        self._rebuild(K, (((flags & 1) != 0) & ~reset).nonzero().squeeze(1))

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
        _score_kernel[grid](q, q.stride(0), self._qdir(), self.mmax, self.mmin, self.count, self.n_c, self._scratch,
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

