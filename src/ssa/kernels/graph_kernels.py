"""Kernels and per-layer state for decoding inside a CUDA graph.

A CUDA graph replays a fixed sequence of kernels with fixed arguments, so nothing that
changes from one decode step to the next may be a Python number. Here the current length
``n`` (after writing the new token's key), each KV head's count of binned keys, and the
selection budget in keys all live in GPU memory; grids are sized for the cache capacity
and programs mask out positions beyond ``n``.

  - ``DenseAttentionGraph``: exact split-K decode attention over positions ``[0, n)``.
  - ``SphereIndexGraph``: ``cluster_skip`` with the fused kernels (bin, score, pick, compact,
    list attention, merge), all reading lengths from GPU memory. Recentering, which is rare,
    runs between graph replays (``after_replay``), with the drift flags read asynchronously.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .labeled_attn import NEG, _list_kernel, _sm_count, merge_splits
from .sphere_fused import NEG_INF, SphereIndexFused, _score_kernel


@triton.jit
def _dense_dev_kernel(q_ptr, k_ptr, v_ptr, n_ptr, m_out, l_out, a_out, num_splits, scale,
                      s_qh, s_kh, s_kn, s_vh, s_vn, s_mh, s_ms, s_ah, s_as, s_ag,
                      G: tl.constexpr, G_PAD: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr):
    kv = tl.program_id(0)
    sp = tl.program_id(1)
    offs_g = tl.arange(0, G_PAD)
    offs_d = tl.arange(0, D)
    gmask = offs_g < G
    q = tl.load(q_ptr + (kv * G + offs_g)[:, None] * s_qh + offs_d[None, :], mask=gmask[:, None], other=0.0)
    n = tl.load(n_ptr)
    chunk = ((n + num_splits - 1) // num_splits + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    start = sp * chunk
    stop = tl.minimum(start + chunk, n)
    m_i = tl.full([G_PAD], NEG, tl.float32)
    l_i = tl.zeros([G_PAD], tl.float32)
    acc = tl.zeros([G_PAD, D], tl.float32)
    for t in range(start, stop, BLOCK_N):
        offs_n = t + tl.arange(0, BLOCK_N)
        inb = offs_n < stop
        k = tl.load(k_ptr + kv * s_kh + offs_n[:, None] * s_kn + offs_d[None, :], mask=inb[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        z = tl.where(inb[None, :] & gmask[:, None], s, NEG)
        m_new = tl.maximum(m_i, tl.max(z, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.where(inb[None, :], tl.exp(z - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + kv * s_vh + offs_n[:, None] * s_vn + offs_d[None, :], mask=inb[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    tl.store(m_out + kv * s_mh + sp * s_ms + offs_g, m_i, mask=gmask)
    tl.store(l_out + kv * s_mh + sp * s_ms + offs_g, l_i, mask=gmask)
    tl.store(a_out + kv * s_ah + sp * s_as + offs_g[:, None] * s_ag + offs_d[None, :], acc, mask=gmask[:, None])


@triton.jit
def _bin_dev_kernel(k_ptr, s_kh, s_kn, dirs_ptr, s_dh, sumdir_ptr, mmax_ptr, mmin_ptr, cnt_ptr, ksum_ptr, mu_ptr,
                    rbar_ptr, flag_ptr, lab_ptr, s_lh, end_ptr, need_ptr, n_ptr, window, budget, delta,
                    C: tl.constexpr, D: tl.constexpr, BC: tl.constexpr):
    """Bin the keys that left the recent window since the last call; update this head's
    binned count, budget in keys and drift flag. All lengths read from GPU memory."""
    h = tl.program_id(0)
    offs_d = tl.arange(0, D)
    n = tl.load(n_ptr)
    old_end = tl.load(end_ptr + h)
    new_end = tl.maximum(tl.maximum(n - window, 1), old_end)
    mu = tl.load(mu_ptr + h * D + offs_d)
    ks = tl.load(ksum_ptr + h * D + offs_d)
    for p in range(old_end, new_end):
        k = tl.load(k_ptr + h * s_kh + p * s_kn + offs_d).to(tl.float32)
        ks += k
        kr = k - mu
        mag = tl.sqrt(tl.sum(kr * kr, axis=0))
        kn = kr / tl.maximum(mag, 1e-12)
        best = NEG_INF
        bi = 0
        for c0 in tl.static_range(0, C, BC):
            offs_c = c0 + tl.arange(0, BC)
            dv = tl.load(dirs_ptr + h * s_dh + offs_c[:, None] * D + offs_d[None, :])
            sc = tl.sum(dv * kn[None, :], axis=1)
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
        tl.store(cnt_ptr + row, tl.load(cnt_ptr + row) + 1.0)
        tl.store(lab_ptr + h * s_lh + p, bi.to(tl.int16))
        tl.debug_barrier()
    tl.store(ksum_ptr + h * D + offs_d, ks)
    tl.store(end_ptr + h, new_end)
    nb = (new_end - 1).to(tl.float32)
    tl.store(need_ptr + h, -tl.floor(-(budget * nb)))                                  # ceil(budget * nb)
    mu_t = ks / tl.maximum(nb, 1.0)
    dist = tl.sqrt(tl.sum((mu_t - mu) * (mu_t - mu), axis=0))
    moved = (dist > delta * tl.load(rbar_ptr + h)) & (nb > 0)
    tl.store(flag_ptr + h, tl.maximum(tl.load(flag_ptr + h), moved.to(tl.int32)))


@triton.jit
def _pick_dev_kernel(scratch_ptr, cnt_ptr, w_ptr, s_wr, need_ptr, scale,
                     G: tl.constexpr, G_PAD: tl.constexpr, C: tl.constexpr, BC: tl.constexpr):
    """Shared (`sum_share`) selection, as ``_pick_kernel``, with the budget read from GPU memory."""
    h = tl.program_id(0)
    c0 = tl.program_id(1) * BC
    need = tl.load(need_ptr + h)
    offs_g = tl.arange(0, G_PAD)
    gm = offs_g < G
    offs_c = tl.arange(0, C)
    offs_i = c0 + tl.arange(0, BC)
    cn = tl.load(cnt_ptr + h * C + offs_c)
    live = cn > 0
    e = tl.load(scratch_ptr + (h * G_PAD + offs_g)[:, None] * C + offs_c[None, :])
    z = tl.where(gm[:, None], e * scale, 0.0)
    zmax = tl.max(z, axis=1)
    ex = tl.where(live[None, :], tl.exp(z - zmax[:, None]), 0.0)
    inv = 1.0 / tl.sum(ex, axis=1)
    sh = tl.sum(tl.where(gm[:, None], ex * inv[:, None], 0.0), axis=0)
    sh = tl.where(live, sh, NEG_INF)
    pick = offs_c[None, :] == offs_i[:, None]
    si = tl.max(tl.where(pick, sh[None, :], NEG_INF), axis=1)
    higher = tl.sum(tl.where(sh[None, :] > si[:, None], cn[None, :], 0.0), axis=1)
    tl.store(w_ptr + h * s_wr + offs_i, ((higher < need) & (si > NEG_INF)).to(tl.float32))


@triton.jit
def _compact_dev_kernel(lab_ptr, w_ptr, idx_ptr, cnt_ptr, n_ptr, s_lh, s_wr, s_ih, BLOCK: tl.constexpr):
    kv = tl.program_id(0)
    n = tl.load(n_ptr)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < n
    lab = tl.load(lab_ptr + kv * s_lh + offs, mask=inb, other=0).to(tl.int32)
    wv = tl.load(w_ptr + kv * s_wr + lab, mask=inb, other=0.0)
    need = tl.where(inb & (wv > 0), 1, 0)
    pos = tl.cumsum(need, 0) - 1
    base = tl.atomic_add(cnt_ptr + kv, tl.sum(need, 0))
    tl.store(idx_ptr + kv * s_ih + base + pos, offs, mask=need > 0)


class _SplitBuffers:
    def __init__(self, H_kv, G, d, num_splits, dtype, device):
        self.num_splits = num_splits
        self.m = torch.empty(H_kv, num_splits, G, device=device, dtype=torch.float32)
        self.l = torch.empty_like(self.m)
        self.a = torch.empty(H_kv, num_splits, G, d, device=device, dtype=torch.float32)
        self.out = torch.empty(H_kv * G, d, device=device, dtype=dtype)


def _merge_into(buf: _SplitBuffers) -> torch.Tensor:
    return merge_splits(buf.m, buf.l, buf.a, buf.out.dtype, out=buf.out)


class DenseAttentionGraph:
    """Exact decode attention over ``[0, n)`` with ``n`` in GPU memory (graph-safe)."""

    def __init__(self, *, H: int, H_kv: int, d: int, dtype, device, block_n: int = 32):
        self.G = H // H_kv
        self.G_PAD = max(16, triton.next_power_of_2(self.G))
        self.block_n = block_n
        self.buf = _SplitBuffers(H_kv, self.G, d, max(1, triton.cdiv(4 * _sm_count(device), H_kv)), dtype, device)
        self.d = d

    def __call__(self, q, K, V, n_dev):
        b = self.buf
        H_kv = K.shape[0]
        _dense_dev_kernel[(H_kv, b.num_splits)](
            q, K, V, n_dev, b.m, b.l, b.a, b.num_splits, 1.0 / math.sqrt(self.d),
            q.stride(0), K.stride(0), K.stride(1), V.stride(0), V.stride(1),
            b.m.stride(0), b.m.stride(1), b.a.stride(0), b.a.stride(1), b.a.stride(2),
            G=self.G, G_PAD=self.G_PAD, D=self.d, BLOCK_N=self.block_n, num_warps=4, num_stages=3)
        return _merge_into(b)


class SphereIndexGraph(SphereIndexFused):
    """``SphereIndexFused`` for graph capture: ``step`` issues only fixed-argument kernels."""

    def __init__(self, *, budget: float, block_n: int = 32, compact_block: int = 1024, **kw):
        super().__init__(**kw)
        self.budget = budget
        self.block_n = block_n
        self.compact_block = compact_block

    def prepare(self, K: torch.Tensor, n: int, H: int) -> None:
        """Build from the prefilled cache ``K [H_kv, capacity, d]`` holding ``n`` positions
        (eager, outside the graph) and allocate the step's fixed buffers."""
        self.observe(K, n)
        H_kv, d, dev = self.H_kv, self.d, self.device
        G = H // H_kv
        self.G, self.G_PAD = G, max(16, triton.next_power_of_2(G))
        self.end_dev = torch.full((H_kv,), self.end, dtype=torch.int32, device=dev)
        self.need_dev = torch.zeros(H_kv, dtype=torch.float32, device=dev)
        self.wbuf = torch.zeros(H_kv, self.C + 1, dtype=torch.float32, device=dev)
        self.wbuf[:, self.C] = 1.0
        self.scratch = torch.empty(H_kv, self.G_PAD, self.C, dtype=torch.float32, device=dev)
        self.idx = torch.empty(H_kv, self.capacity, dtype=torch.int32, device=dev)
        self.cnt = torch.zeros(H_kv, dtype=torch.int32, device=dev)
        self.buf = _SplitBuffers(H_kv, G, d, max(1, triton.cdiv(4 * _sm_count(dev), H_kv)), K.dtype, dev)

    def step(self, q, K, V, n_dev):
        """One decode step; ``K, V [H_kv, capacity, d]`` with position ``n-1`` already written."""
        H_kv, C, d = self.H_kv, self.C, self.d
        bc = self._bc()
        _bin_dev_kernel[(H_kv,)](
            K, K.stride(0), K.stride(1), *self._dirs_arg(), self.sum_dir, self.mmax, self.mmin, self.count,
            self.ksum, self.mu_ref, self.rbar, self.flags, self.labels, self.labels.stride(0),
            self.end_dev, self.need_dev, n_dev, self.window, float(self.budget), float(self.delta),
            C=C, D=d, BC=bc, num_warps=4)
        grid = (H_kv, C // bc)
        _score_kernel[grid](q, q.stride(0), self.sum_dir, self.mmax, self.mmin, self.count, self.scratch,
                            G=self.G, G_PAD=self.G_PAD, C=C, D=d, BC=bc, num_warps=4)
        _pick_dev_kernel[grid](self.scratch, self.count, self.wbuf, self.wbuf.stride(0), self.need_dev,
                               1.0 / math.sqrt(d), G=self.G, G_PAD=self.G_PAD, C=C, BC=bc, num_warps=4)
        self.cnt.zero_()
        _compact_dev_kernel[(H_kv, triton.cdiv(self.capacity, self.compact_block))](
            self.labels, self.wbuf, self.idx, self.cnt, n_dev, self.labels.stride(0), self.wbuf.stride(0),
            self.idx.stride(0), BLOCK=self.compact_block, num_warps=4)
        b = self.buf
        _list_kernel[(H_kv, b.num_splits)](
            q, K, V, self.labels, self.wbuf, self.idx, self.cnt, b.m, b.l, b.a, b.num_splits, 1.0 / math.sqrt(d),
            q.stride(0), K.stride(0), K.stride(1), V.stride(0), V.stride(1), self.labels.stride(0),
            self.wbuf.stride(0), self.idx.stride(0), b.m.stride(0), b.m.stride(1), b.a.stride(0),
            b.a.stride(1), b.a.stride(2), G=self.G, G_PAD=self.G_PAD, D=d, BLOCK_N=self.block_n,
            PER_HEAD=False, num_stages=3, num_warps=4)
        return _merge_into(b)

    def after_replay(self, K: torch.Tensor) -> None:
        """Between replays: every ``check_every`` steps read the drift flags without a sync;
        once a copy has landed, recenter the flagged heads (eager, in place)."""
        self.head_steps += self.H_kv
        self._steps += 1
        if self._steps % self.check_every:
            return
        if self._pending is not None and self._pending.query():
            heads = self._flags_host.nonzero().squeeze(1)
            self._pending = None
            if heads.numel():
                self.end = int(self.end_dev[0])                                  # rare host sync
                self._rebuild(K, heads.to(self.device))
        if self._pending is None:
            self._flags_host.copy_(self.flags, non_blocking=True)
            self._pending = torch.cuda.Event()
            self._pending.record()
            self.flags.zero_()
