"""Triton label-weighted flash-decoding attention.

Computes, for query head ``h`` on KV head ``g`` (``G = H / H_kv`` query heads share a KV head),

    out[h] = Σ_j w[label_j] e^{s_j} v_j / Σ_j w[label_j] e^{s_j},    s_j = q_h·k_j / √d,

reading K and V in the engine's cache layout ``[H_kv, n, d]`` in place (any strides with a
unit last stride). Key and value rows whose weight is 0 for every query head of the group
are masked out of the loads, so they cost no memory traffic beyond their 2-byte label.

Split-K decoding: grid ``(H_kv, num_splits)``; each program scans a contiguous range of
positions in tiles of ``BLOCK_N``, loads each K row once for all G query heads (padded to
16 rows for ``tl.dot``), and keeps an online softmax with ``s + log w``. Partial results
``(max, sum, output)`` are merged across splits by log-sum-exp (in PyTorch for now).

The reference specification is ``ssa.attn.labeled.label_weighted_attention``.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

NEG = tl.constexpr(-1.0e30)  # finite "−∞" for the running max, so empty tiles give exp(NEG − NEG) = 1, never NaN


_SMS: dict = {}


def _sm_count(device) -> int:
    key = str(device)
    if key not in _SMS:
        _SMS[key] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SMS[key]


@triton.jit
def _split_kernel(q_ptr, k_ptr, v_ptr, lab_ptr, w_ptr, m_out, l_out, a_out,
                  n, chunk, scale,
                  s_qh, s_kh, s_kn, s_vh, s_vn, s_lh, s_wr,
                  s_mh, s_ms, s_ah, s_as, s_ag,
                  G: tl.constexpr, G_PAD: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
                  PER_HEAD: tl.constexpr):
    kv = tl.program_id(0)
    sp = tl.program_id(1)
    offs_g = tl.arange(0, G_PAD)
    offs_d = tl.arange(0, D)
    gmask = offs_g < G
    q = tl.load(q_ptr + (kv * G + offs_g)[:, None] * s_qh + offs_d[None, :], mask=gmask[:, None], other=0.0)
    if PER_HEAD:
        wrow = kv * G + offs_g
    else:
        wrow = kv + offs_g * 0
    start = sp * chunk
    stop = tl.minimum(start + chunk, n)
    m_i = tl.full([G_PAD], NEG, tl.float32)
    l_i = tl.zeros([G_PAD], tl.float32)
    acc = tl.zeros([G_PAD, D], tl.float32)
    for t in range(start, stop, BLOCK_N):
        offs_n = t + tl.arange(0, BLOCK_N)
        inb = offs_n < stop
        lab = tl.load(lab_ptr + kv * s_lh + offs_n, mask=inb, other=0).to(tl.int32)
        wt = tl.load(w_ptr + wrow[:, None] * s_wr + lab[None, :],
                     mask=gmask[:, None] & inb[None, :], other=0.0)              # [G_PAD, BLOCK_N]
        use = (wt > 0)
        row = tl.max(use.to(tl.int32), axis=0) > 0                               # any query head reads it
        k = tl.load(k_ptr + kv * s_kh + offs_n[:, None] * s_kn + offs_d[None, :],
                    mask=row[:, None], other=0.0)                                # [BLOCK_N, D]
        s = tl.dot(q, tl.trans(k)) * scale                                       # [G_PAD, BLOCK_N]
        z = tl.where(use, s + tl.log(tl.where(use, wt, 1.0)), NEG)
        m_new = tl.maximum(m_i, tl.max(z, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.where(use, tl.exp(z - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + kv * s_vh + offs_n[:, None] * s_vn + offs_d[None, :],
                    mask=row[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    tl.store(m_out + kv * s_mh + sp * s_ms + offs_g, m_i, mask=gmask)
    tl.store(l_out + kv * s_mh + sp * s_ms + offs_g, l_i, mask=gmask)
    tl.store(a_out + kv * s_ah + sp * s_as + offs_g[:, None] * s_ag + offs_d[None, :], acc,
             mask=gmask[:, None])


def label_weighted_attention_triton(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                    labels: torch.Tensor, w: torch.Tensor, *,
                                    block_n: int = 64, num_splits: int | None = None,
                                    num_stages: int | None = None, num_warps: int = 4) -> torch.Tensor:
    """``q [H, d]``, ``K, V [H_kv, n, d]`` (unit last stride), ``labels [H_kv, n]`` int16,
    ``w [H_kv or H, L]`` -> ``[H, d]`` in ``q``'s dtype."""
    H, d = q.shape
    H_kv, n, d2 = K.shape
    assert d == d2 and V.shape == K.shape and K.stride(-1) == 1 and V.stride(-1) == 1
    assert d & (d - 1) == 0 and d >= 16, "head dim must be a power of two ≥ 16"
    G = H // H_kv
    per_head = w.shape[0] == H and H != H_kv
    if not per_head:
        assert w.shape[0] == H_kv
    q = q.contiguous()
    labels = labels.contiguous()
    w = w.to(torch.float32).contiguous()
    if num_splits is None:
        num_splits = max(1, min(triton.cdiv(n, block_n), triton.cdiv(4 * _sm_count(q.device), H_kv)))
    chunk = triton.cdiv(triton.cdiv(n, num_splits), block_n) * block_n
    num_splits = triton.cdiv(n, chunk)
    G_PAD = max(16, triton.next_power_of_2(G))
    m = torch.empty(H_kv, num_splits, G, device=q.device, dtype=torch.float32)
    l = torch.empty_like(m)
    a = torch.empty(H_kv, num_splits, G, d, device=q.device, dtype=torch.float32)
    _split_kernel[(H_kv, num_splits)](
        q, K, V, labels, w, m, l, a, n, chunk, 1.0 / math.sqrt(d),
        q.stride(0), K.stride(0), K.stride(1), V.stride(0), V.stride(1), labels.stride(0), w.stride(0),
        m.stride(0), m.stride(1), a.stride(0), a.stride(1), a.stride(2),
        G=G, G_PAD=G_PAD, D=d, BLOCK_N=block_n, PER_HEAD=per_head,
        num_stages=num_stages if num_stages is not None else (2 if block_n >= 128 else 3),
        num_warps=num_warps)
    return merge_splits(m, l, a, q.dtype)


@triton.jit
def _merge_kernel(m_ptr, l_ptr, a_ptr, out_ptr, S, s_mh, s_ms, s_ah, s_as, s_ag, s_oh,
                  G: tl.constexpr, D: tl.constexpr):
    """Grid (H_kv,): combine the splits of each query head by log-sum-exp."""
    kv = tl.program_id(0)
    offs_d = tl.arange(0, D)
    one = tl.arange(0, 1)
    for g in tl.static_range(G):
        M = tl.full([1], NEG, tl.float32)
        for s in range(S):
            M = tl.maximum(M, tl.load(m_ptr + kv * s_mh + s * s_ms + g + one))
        den = tl.zeros([1], tl.float32)
        num = tl.zeros([D], tl.float32)
        for s in range(S):
            f = tl.exp(tl.load(m_ptr + kv * s_mh + s * s_ms + g + one) - M)
            den += f * tl.load(l_ptr + kv * s_mh + s * s_ms + g + one)
            num += f * tl.load(a_ptr + kv * s_ah + s * s_as + g * s_ag + offs_d)
        tl.store(out_ptr + (kv * G + g) * s_oh + offs_d, (num / den).to(out_ptr.dtype.element_ty))


def merge_splits(m: torch.Tensor, l: torch.Tensor, a: torch.Tensor, dtype, out: torch.Tensor | None = None) -> torch.Tensor:
    """Triton version of ``_merge``: one launch instead of ~8 PyTorch ops."""
    H_kv, S, G = m.shape
    d = a.shape[-1]
    if out is None:
        out = torch.empty(H_kv * G, d, device=m.device, dtype=dtype)
    _merge_kernel[(H_kv,)](m, l, a, out, S, m.stride(0), m.stride(1), a.stride(0), a.stride(1), a.stride(2),
                           out.stride(0), G=G, D=d, num_warps=4)
    return out


def _merge(m, l, a, H, d, dtype):
    """Combine per-split (max, sum, output) by log-sum-exp (PyTorch reference)."""
    M = m.max(dim=1, keepdim=True).values
    f = torch.exp(m - M)                                                 # [H_kv, S, G]
    den = (f * l).sum(1)                                                 # [H_kv, G]
    num = (f.unsqueeze(-1) * a).sum(1)                                   # [H_kv, G, d]
    return (num / den.unsqueeze(-1)).reshape(H, d).to(dtype)


# ---- compaction, then attention over the list of selected positions -----------------

@triton.jit
def _compact_kernel(lab_ptr, w_ptr, idx_ptr, cnt_ptr, n, s_lh, s_wr, s_ih,
                    G: tl.constexpr, PER_HEAD: tl.constexpr, BLOCK: tl.constexpr):
    kv = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < n
    lab = tl.load(lab_ptr + kv * s_lh + offs, mask=inb, other=0).to(tl.int32)
    if PER_HEAD:
        need = tl.zeros([BLOCK], tl.int32)
        for g in tl.static_range(G):
            wv = tl.load(w_ptr + (kv * G + g) * s_wr + lab, mask=inb, other=0.0)
            need = tl.maximum(need, (wv > 0).to(tl.int32))
    else:
        wv = tl.load(w_ptr + kv * s_wr + lab, mask=inb, other=0.0)
        need = (wv > 0).to(tl.int32)
    need = tl.where(inb, need, 0)
    pos = tl.cumsum(need, 0) - 1
    base = tl.atomic_add(cnt_ptr + kv, tl.sum(need, 0))
    tl.store(idx_ptr + kv * s_ih + base + pos, offs, mask=need > 0)


def compact_positions(labels: torch.Tensor, w: torch.Tensor, *, block: int = 1024):
    """Selected positions per KV head: ``idx [H_kv, n]`` int32 (first ``cnt[kv]`` valid, in no
    particular order) and ``cnt [H_kv]`` int32. A position is selected if its label has
    positive weight for any query head of its group."""
    H_kv, n = labels.shape
    per_head = w.shape[0] != H_kv
    G = w.shape[0] // H_kv if per_head else 1
    labels = labels.contiguous()
    w = w.to(torch.float32).contiguous()
    idx = torch.empty(H_kv, max(n, 1), device=labels.device, dtype=torch.int32)
    cnt = torch.zeros(H_kv, device=labels.device, dtype=torch.int32)
    _compact_kernel[(H_kv, triton.cdiv(n, block))](labels, w, idx, cnt, n, labels.stride(0), w.stride(0),
                                                  idx.stride(0), G=G, PER_HEAD=per_head, BLOCK=block)
    return idx, cnt


@triton.jit
def _list_kernel(q_ptr, k_ptr, v_ptr, lab_ptr, w_ptr, idx_ptr, cnt_ptr, m_out, l_out, a_out,
                 num_splits, scale,
                 s_qh, s_kh, s_kn, s_vh, s_vn, s_lh, s_wr, s_ih,
                 s_mh, s_ms, s_ah, s_as, s_ag,
                 G: tl.constexpr, G_PAD: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
                 PER_HEAD: tl.constexpr):
    kv = tl.program_id(0)
    sp = tl.program_id(1)
    offs_g = tl.arange(0, G_PAD)
    offs_d = tl.arange(0, D)
    gmask = offs_g < G
    q = tl.load(q_ptr + (kv * G + offs_g)[:, None] * s_qh + offs_d[None, :], mask=gmask[:, None], other=0.0)
    if PER_HEAD:
        wrow = kv * G + offs_g
    else:
        wrow = kv + offs_g * 0
    count = tl.load(cnt_ptr + kv)
    chunk = ((count + num_splits - 1) // num_splits + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    start = sp * chunk
    stop = tl.minimum(start + chunk, count)
    m_i = tl.full([G_PAD], NEG, tl.float32)
    l_i = tl.zeros([G_PAD], tl.float32)
    acc = tl.zeros([G_PAD, D], tl.float32)
    for t in range(start, stop, BLOCK_N):
        offs = t + tl.arange(0, BLOCK_N)
        inb = offs < stop
        pos = tl.load(idx_ptr + kv * s_ih + offs, mask=inb, other=0)
        lab = tl.load(lab_ptr + kv * s_lh + pos, mask=inb, other=0).to(tl.int32)
        wt = tl.load(w_ptr + wrow[:, None] * s_wr + lab[None, :], mask=gmask[:, None] & inb[None, :], other=0.0)
        use = wt > 0
        k = tl.load(k_ptr + kv * s_kh + pos[:, None] * s_kn + offs_d[None, :], mask=inb[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        z = tl.where(use, s + tl.log(tl.where(use, wt, 1.0)), NEG)
        m_new = tl.maximum(m_i, tl.max(z, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.where(use, tl.exp(z - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + kv * s_vh + pos[:, None] * s_vn + offs_d[None, :], mask=inb[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    tl.store(m_out + kv * s_mh + sp * s_ms + offs_g, m_i, mask=gmask)
    tl.store(l_out + kv * s_mh + sp * s_ms + offs_g, l_i, mask=gmask)
    tl.store(a_out + kv * s_ah + sp * s_as + offs_g[:, None] * s_ag + offs_d[None, :], acc, mask=gmask[:, None])


def label_weighted_attention_compact(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                     labels: torch.Tensor, w: torch.Tensor, *,
                                     block_n: int = 32, num_splits: int | None = None,
                                     num_stages: int | None = None, num_warps: int = 4,
                                     compact_block: int = 1024) -> torch.Tensor:
    """Same result as ``label_weighted_attention_triton``; work scales with the rows read.
    The count of selected rows stays on the GPU (no host synchronization)."""
    H, d = q.shape
    H_kv, n, d2 = K.shape
    assert d == d2 and V.shape == K.shape and K.stride(-1) == 1 and V.stride(-1) == 1
    assert d & (d - 1) == 0 and d >= 16, "head dim must be a power of two ≥ 16"
    G = H // H_kv
    per_head = w.shape[0] == H and H != H_kv
    q = q.contiguous()
    labels = labels.contiguous()
    w = w.to(torch.float32).contiguous()
    idx, cnt = compact_positions(labels, w, block=compact_block)
    if num_splits is None:
        num_splits = max(1, triton.cdiv(4 * _sm_count(q.device), H_kv))
    G_PAD = max(16, triton.next_power_of_2(G))
    m = torch.empty(H_kv, num_splits, G, device=q.device, dtype=torch.float32)
    l = torch.empty_like(m)
    a = torch.empty(H_kv, num_splits, G, d, device=q.device, dtype=torch.float32)
    _list_kernel[(H_kv, num_splits)](
        q, K, V, labels, w, idx, cnt, m, l, a, num_splits, 1.0 / math.sqrt(d),
        q.stride(0), K.stride(0), K.stride(1), V.stride(0), V.stride(1), labels.stride(0), w.stride(0),
        idx.stride(0), m.stride(0), m.stride(1), a.stride(0), a.stride(1), a.stride(2),
        G=G, G_PAD=G_PAD, D=d, BLOCK_N=block_n, PER_HEAD=per_head,
        num_stages=num_stages if num_stages is not None else (2 if block_n >= 128 else 3),
        num_warps=num_warps)
    return merge_splits(m, l, a, q.dtype)
