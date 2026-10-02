"""FlashInfer's exact decode attention inside a CUDA graph, on the engine's cache.

The engine keeps each layer's cache contiguous, ``[max_batch, H_kv, max_seq_len, d]``. A strided
view presents it to FlashInfer as a paged cache in the ``HND`` layout, ``[pages, H_kv, page_size, d]``,
without copying: page ``j`` of sequence ``b`` is positions ``[j·P, (j+1)·P)`` of that sequence.
Pages must be small (16 tokens by default), because FlashInfer splits a sequence's work across
thread blocks by pages; with one page per sequence a decode step runs on only ``H_kv`` blocks and
is ~3.7× slower at 32k tokens on this card.

``BatchDecodeWithPagedKVCacheWrapper`` in CUDA-graph mode keeps its index buffers fixed, so the
decode step is captured once and ``plan(n)`` is called on the host before each replay to set the
length. One plan serves every layer, since all layers have the same length.

Needs ``flashinfer-python``; on sm_120 its JIT needs a CUDA >= 12.9 compiler (see
``ssa.harness.kernel_bench.flashinfer_decode``).
"""

from __future__ import annotations

import torch


class FlashInferDecode:
    def __init__(self, *, H: int, H_kv: int, d: int, max_len: int, dtype, device, batch: int = 1,
                 page_size: int = 16, workspace_mb: int = 128):
        import flashinfer

        if max_len % page_size:
            raise ValueError(f"cache length {max_len} must be a multiple of page_size {page_size}")
        self.H, self.H_kv, self.d, self.P, self.batch, self.dtype = H, H_kv, d, page_size, batch, dtype
        self.max_len = max_len
        self.pages_per_seq = max_len // page_size
        self.seq_stride_pages = H_kv * self.pages_per_seq          # sequence b starts at page b·H_kv·pages_per_seq
        n_pages = batch * self.pages_per_seq
        self.ws = torch.empty(workspace_mb * 1024 * 1024, dtype=torch.uint8, device=device)
        self.w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.ws, "HND", use_cuda_graph=True,
            paged_kv_indptr_buffer=torch.zeros(batch + 1, dtype=torch.int32, device=device),
            paged_kv_indices_buffer=torch.zeros(n_pages, dtype=torch.int32, device=device),
            paged_kv_last_page_len_buffer=torch.ones(batch, dtype=torch.int32, device=device))
        self.out = torch.empty(batch, H, d, dtype=dtype, device=device)

    def plan(self, n: int) -> None:
        """Set the current length of every sequence (host side; call before each replay)."""
        used = (n + self.P - 1) // self.P
        indptr = torch.arange(self.batch + 1, dtype=torch.int32) * used
        indices = (torch.arange(self.batch, dtype=torch.int32).unsqueeze(1) * self.seq_stride_pages
                   + torch.arange(used, dtype=torch.int32)).reshape(-1)
        last = torch.full((self.batch,), n - (used - 1) * self.P, dtype=torch.int32)
        self.w.plan(indptr, indices, last, self.H, self.H_kv, self.d, self.P, pos_encoding_mode="NONE",
                    q_data_type=self.dtype, kv_data_type=self.dtype)

    def paged(self, cache: torch.Tensor) -> torch.Tensor:
        """``[batch, H_kv, max_len, d]`` contiguous -> strided ``[pages, H_kv, P, d]`` view."""
        B, Hk, L, d = cache.shape
        n_pages = (B - 1) * self.seq_stride_pages + self.pages_per_seq
        return cache.as_strided((n_pages, Hk, self.P, d), (self.P * d, L * d, d, 1))

    def __call__(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
        """``q [batch, H, d]``; ``k_cache, v_cache [batch, H_kv, max_len, d]`` (the engine's per-layer
        cache, read in place) -> ``[batch, H, d]``."""
        self.w.run(q, (self.paged(k_cache), self.paged(v_cache)), out=self.out)
        return self.out
