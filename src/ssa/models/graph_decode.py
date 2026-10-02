"""Decode step captured in a CUDA graph.

``GraphDecoder`` reruns the engine's single-token forward pass layer by layer, with the
engine's own modules, weights and KV cache, but with the position held in GPU memory:
RoPE angles are gathered at that position, the new key and value are written with
``index_copy_``, and attention reads the current length from GPU memory
(``DenseAttentionGraph`` or ``SphereIndexGraph``). Nothing in the step depends on a
Python number that changes between tokens, so it is captured once and replayed with a
single launch per token. Recentering of the hypersphere bins (rare) runs between replays.

The prompt is processed by the engine as usual; ``prepare(n)`` then builds the attention
state from the first ``n`` cached positions.
"""

from __future__ import annotations

import torch

from engine.attention import apply_rope

from ..kernels.graph_kernels import DenseAttentionGraph, SphereIndexGraph


class GraphDecoder:
    def __init__(self, model, cache, *, mode: str = "dense", budget: float = 0.2, C: int = 256,
                 window: int = 64, delta: float = 0.03, check_every: int = 16, **_):
        cfg = model.cfg
        self.model, self.cache, self.mode = model, cache, mode
        self.H, self.H_kv, self.d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        p = next(model.parameters())
        dev, dtype = p.device, p.dtype
        self.tok = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.n_dev = torch.zeros(1, dtype=torch.int32, device=dev)
        cap = cache.max_seq_len
        mode = "voronoi" if mode == "sphere" else mode           # old name
        self.mode = mode
        if mode == "dense":
            self.attn = [DenseAttentionGraph(H=self.H, H_kv=self.H_kv, d=self.d, dtype=dtype, device=dev)
                         for _ in model.layers]
        elif mode == "voronoi":
            self.attn = [SphereIndexGraph(budget=budget, C=C, window=window, delta=delta, capacity=cap,
                                          check_every=check_every) for _ in model.layers]
        else:
            raise ValueError(f"unknown mode {mode!r}")
        self.graph = None
        self.logits = None

    def prepare(self, n: int) -> None:
        """Build attention state from the first ``n`` cached positions (after prefill)."""
        if self.mode == "voronoi":
            for i, idx in enumerate(self.attn):
                idx.prepare(self.cache.k[i][0], n, self.H)

    def _body(self) -> torch.Tensor:
        m, H, H_kv, d = self.model, self.H, self.H_kv, self.d
        self.n_dev.copy_(self.pos + 1)
        h = m.embed(self.tok)                                              # [1, 1, hidden]
        cos = m.rope_cos.index_select(0, self.pos)
        sin = m.rope_sin.index_select(0, self.pos)
        for i, layer in enumerate(m.layers):
            a = layer.attn
            x = layer.norm1(h)
            q = a.q_norm(a.q(x).view(1, 1, H, d)).transpose(1, 2)
            k = a.k_norm(a.k(x).view(1, 1, H_kv, d)).transpose(1, 2)
            v = a.v(x).view(1, 1, H_kv, d).transpose(1, 2)
            q, k = apply_rope(q, k, cos, sin)
            kc, vc = self.cache.k[i], self.cache.v[i]
            kc.index_copy_(2, self.pos, k)
            vc.index_copy_(2, self.pos, v)
            out = self.attn[i](q[0, :, 0, :].contiguous(), kc[0], vc[0], self.n_dev) if self.mode == "dense" \
                else self.attn[i].step(q[0, :, 0, :].contiguous(), kc[0], vc[0], self.n_dev)
            h = h + a.o(out.view(1, 1, H * d))
            h = h + layer.ffn(layer.norm2(h))
        h = m.final_norm(h)
        return h @ m.embed.weight.T if m.lm_head is None else m.lm_head(h)

    def capture(self) -> None:
        """Record one decode step. Call after ``prepare`` and before the first ``step``; the
        warmup runs write a placeholder at the current position, which the first real step
        overwrites."""
        self.pos.fill_(self.cache.cur_len)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.logits = self._body()
        self.graph = g

    def step(self, token: torch.Tensor, pos: int) -> torch.Tensor:
        """Decode ``token [1, 1]`` at position ``pos``; returns logits ``[1, 1, vocab]`` (a
        buffer the next step overwrites)."""
        self.tok.copy_(token)
        self.pos.fill_(pos)
        if self.graph is not None:
            self.graph.replay()
        else:
            self.logits = self._body()
        if self.mode == "voronoi":
            for i, idx in enumerate(self.attn):
                idx.after_replay(self.cache.k[i][0])
        self.cache.cur_len = pos + 1
        return self.logits
