"""FlashInfer's exact decode attention inside a CUDA graph, on the engine's cache.

The engine keeps each layer's cache contiguous, ``[max_batch, H_kv, max_seq_len, d]``. Read as a
paged cache in FlashInfer's ``HND`` layout, that is one page per sequence, with the page as long
as the cache; the last page's length is the current sequence length. FlashInfer's
``BatchDecodeWithPagedKVCacheWrapper`` in CUDA-graph mode keeps its index buffers fixed, so the
decode step is captured once and ``plan(n)`` is called on the host before each replay to set the
length. One plan serves every layer, since all layers have the same length.

Needs ``flashinfer-python``; on sm_120 its JIT needs a CUDA >= 12.9 compiler (see
``ssa.harness.kernel_bench.flashinfer_decode``).
"""

from __future__ import annotations

import torch


class FlashInferDecode:
    def __init__(self, *, H: int, H_kv: int, d: int, page_len: int, dtype, device, batch: int = 1,
                 workspace_mb: int = 128):
        import flashinfer

        self.H, self.H_kv, self.d, self.page_len, self.batch, self.dtype = H, H_kv, d, page_len, batch, dtype
        ar = torch.arange(batch + 1, dtype=torch.int32)
        self._indptr_host, self._indices_host = ar, ar[:-1].clone()
        self.ws = torch.empty(workspace_mb * 1024 * 1024, dtype=torch.uint8, device=device)
        self.w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.ws, "HND", use_cuda_graph=True,
            paged_kv_indptr_buffer=ar.to(device), paged_kv_indices_buffer=ar[:-1].to(device),
            paged_kv_last_page_len_buffer=torch.ones(batch, dtype=torch.int32, device=device))
        self.out = torch.empty(batch, H, d, dtype=dtype, device=device)

    def plan(self, n: int) -> None:
        """Set the current length (host side; call before each replay)."""
        self.w.plan(self._indptr_host, self._indices_host, torch.full((self.batch,), n, dtype=torch.int32),
                    self.H, self.H_kv, self.d, self.page_len, pos_encoding_mode="NONE",
                    q_data_type=self.dtype, kv_data_type=self.dtype)

    def __call__(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
        """``q [batch, H, d]``; ``k_cache, v_cache [batch, H_kv, page_len, d]`` (the engine's per-layer
        cache, read in place) -> ``[batch, H, d]``."""
        self.w.run(q, (k_cache, v_cache), out=self.out)
        return self.out
