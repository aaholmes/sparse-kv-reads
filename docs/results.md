# Detailed results for `voronoi_skip`

Supporting detail for the [README](../README.md). Unless stated otherwise: Qwen3-4B in BF16, WikiText-103, 8 text chunks per setting (2040 scored decode steps), fidelity as total variation distance (TVD) from the exact model's next-token distribution, reads as key and value (K+V) rows including region summaries, and [95% bootstrap CI over chunks].

## Fidelity compared with systematic sampling

| context | method | K+V rows read | TVD from exact |
|---|---|---|---|
| 2048 | systematic sampling, 64 samples | 51.7% | 0.054 [0.048, 0.059] |
| 2048 | `voronoi_skip`, 20% budget | 30.2% | 0.041 [0.036, 0.045] |
| 2048 | systematic sampling, 256 samples | 54.4% | 0.023 [0.020, 0.026] |
| 2048 | `voronoi_skip`, 40% budget | 49.3% | 0.024 [0.022, 0.027] |
| 8192 | systematic sampling, 64 samples | 50.5% | 0.058 [0.049, 0.065] |
| 8192 | `voronoi_skip`, 10% budget | 13.1% | 0.063 [0.053, 0.074] |
| 8192 | systematic sampling, 256 samples | 51.4% | 0.026 [0.023, 0.028] |
| 8192 | `voronoi_skip`, 40% budget | 42.5% | 0.027 [0.023, 0.030] |
| 32768 | systematic sampling, 64 samples | 50.1% | 0.057 [0.046, 0.066] |
| 32768 | `voronoi_skip`, 20% budget | 21.2% | 0.064 [0.052, 0.074] |
| 32768 | systematic sampling, 256 samples | 50.4% | 0.027 [0.022, 0.031] |
| 32768 | `voronoi_skip`, 40% budget | 40.9% | 0.040 [0.031, 0.047] |

- At moderate fidelity (64-sample sampling's TVD), `voronoi_skip` needs about half the reads at 2048 tokens, 0.29× at 8192 and 0.50× at 32,768. At high fidelity (256 samples) it ties at 2048, reads ~17% less at 8192, and is not matched within a 40% budget at 32,768.
- At a fixed budget, TVD is 32–45% higher at 32,768 tokens than at 8192, while sampling's is unchanged: the dropped tail grows with the cache.
- Choosing random regions at a matched budget gives 3–4× the TVD (2048 tokens), so the ranking does the work.
- Updating regions incrementally, recentering a head only when its mean has moved more than 3% of the typical centered key length, matches rebuilding them every step (TVD 0.063 vs 0.062 at a 10% budget, 2048 tokens). The mean moves by about 1/n of that length per token, so recentering is rare after a long prompt.

## Other models, text and settings

- **Qwen3-0.6B.** At the TVD of 64-sample sampling (0.097 at 8192 tokens, 0.095 at 32,768), a 10% budget gives 0.093 and 0.091 with 0.26× and 0.23× the reads. At 30–40% budgets it matches 256-sample sampling (0.044) with 0.68× (8192) and 0.62× (32,768) of its reads. Unlike Qwen3-4B, its TVD does not rise with context.
- **Python code** (8192 tokens). Budgets of 5/10/20/30/40% give TVD 0.035 / 0.025 / 0.017 / 0.012 / 0.009, 2–3× lower than on WikiText. Sampling gives 0.021 (64 samples) and 0.010 (256); `voronoi_skip` matches them with 0.33× and 0.76× of the reads.
- **Number of regions** (8192 tokens). At matched reads, 512 regions give 8–13% lower TVD than 256, and 128 give 14–15% higher. Finer regions help despite their larger summary cost, so 512 is the better choice at long context.
- **The first token.** It takes 36–65% of all attention at layers 12–35 in each of 16 WikiText and Python contexts tested, whatever its text, so it is always read.

## Comparison with weight quantization

As reference points for what a given TVD means, Qwen3-4B with weight-only round-to-nearest quantization (embeddings and output layer kept in BF16), 8192 tokens:

| model | TVD from BF16 | top-1 agreement |
|---|---|---|
| 8-bit weights | 0.018 [0.015, 0.019] | 97.9% |
| 4-bit weights, groups of 128 | 0.144 [0.125, 0.159] | 84.0% |
| `voronoi_skip`, 20% budget | 0.044 | — |

`voronoi_skip` at 20% falls between the two, and on code at 20% (0.017) it matches 8-bit. Plain round-to-nearest 4-bit is much worse than calibrated 4-bit methods, so it is a loose bound.

## Sampling the skipped regions

After choosing the top regions, draw S of the remaining regions by systematic sampling from an estimate of their attention mass (with a uniform floor), read them in full, and weight each by the inverse of its inclusion probability. Both the numerator and the denominator of attention are then unbiased, and the output is a ratio estimator with O(1/S) bias.

At matched reads this loses to reading more top regions by 24–54% in TVD in every setting tested: Qwen3-4B at 8192 and 32,768 tokens and Qwen3-0.6B at 32,768, top-set sizes of 5–20% and S of 8 or 32 regions. At 8192 tokens, for example, the top 10% plus 8 sampled regions gives TVD 0.076 at 18% reads, compared with 0.050 for top regions alone at those reads; adding the samples makes it worse than the top 10% alone (0.063).

Why: split attention into the top set (weight f, mean value μ_H) and the rest (weight 1−f, mean μ_T). Dropping the rest gives error (1−f)(μ_H − μ_T), set by the tail's *mean*, which averages thousands of keys and lies close to the output. Sampling removes that bias but adds variance set by the *spread* of individual regions around the output, falling only as 1/S. An offline replay on captured queries and keys finds sampling's error is 84–100% variance, and 1.5–3× the top-k error at matched reads at layers 12, 24 and 35. The remaining mass is spread over ~200 regions with no dominant ones, so reading the next-largest regions exactly is the cheaper way to reduce error.

## Kernels and timing

Three Triton kernels (Triton is a Python-embedded language for writing GPU kernels) maintain and choose regions: they assign the key leaving the recent window, score the regions, and select up to the budget without sorting. Two more compact the selected positions into a list and run split attention over it (flash-decoding). One layer, Qwen3-4B head layout, BF16, RTX 5060 Ti; median of 100 calls with the L2 cache flushed, both sides in CUDA graphs; exact attention by PyTorch's scaled dot-product attention (SDPA) and by FlashInfer (`kernel_bench --step --flashinfer`):

| context | SDPA | FlashInfer | whole step, 20% of rows | whole step, 5% of rows |
|---|---|---|---|---|
| 8192 | 100 µs | 92 µs | 55 µs (1.67×) | 45 µs (2.06×) |
| 16384 | 180 µs | 174 µs | 74 µs (2.36×) | 49 µs (3.56×) |
| 32768 | 344 µs | 337 µs | 109 µs (3.11×) | 57 µs (5.89×) |
| 65536 | 670 µs | 660 µs | 182 µs (3.63×) | 80 µs (8.25×) |

Speedups are relative to FlashInfer, which reads the cache at 364–407 GB/s of the card's nominal 448 GB/s. To run it on this card (sm_120), FlashInfer's compiler must be CUDA 13; the `flashinfer_decode` docstring in `kernel_bench.py` gives the command.

Region maintenance and selection cost ~25 µs per step at any context length. Gathering scattered 256-byte rows costs at most ~1.5× per byte compared with contiguous reads on this card. About 50 µs of fixed cost per step makes the method slower than exact attention below ~4–5k tokens.

## End-to-end decoding

The kernels run inside the inference engine, one region index per layer, reading the KV cache in place, and match the offline fidelity (8192 tokens: TVD 0.063 [0.054, 0.073] at a 10% budget and 0.044 [0.037, 0.051] at 20%, compared with 0.063 and 0.043 offline). At batch 1 the engine's Python overhead per token exceeded the GPU time for Qwen3-0.6B, so the whole decode step is captured in a CUDA graph, with every step-dependent number (position, number of assigned keys, budget) kept in GPU memory. Median ms per token over 3 repeats of 64 steps:

| model | context | exact, engine's kernel | exact, FlashInfer | 20% budget | 5% budget |
|---|---|---|---|---|---|
| Qwen3-0.6B | 16384 | 9.6 ms | 9.8 ms | 6.7 ms (1.43×) | 6.1 ms (1.57×) |
| Qwen3-0.6B | 32768 | 14.0 ms | 14.5 ms | 7.8 ms (1.78×) | 6.3 ms (2.22×) |
| Qwen3-0.6B | 40448 | 16.0 ms | 16.6 ms | 8.3 ms (1.93×) | 6.5 ms (2.46×) |
| Qwen3-4B | 16384 | 29.1 ms | 29.4 ms | 25.5 ms (1.14×) | 24.6 ms (1.18×) |
| Qwen3-4B | 32768 | 34.8 ms | 35.1 ms | 26.8 ms (1.30×) | 25.0 ms (1.39×) |

Speedups are relative to the faster exact kernel, the engine's own. FlashInfer's paged decode (`ssa.kernels.flashinfer_graph`) reads the engine's contiguous cache in place, viewed as 16-token pages; per layer it matches FlashInfer's single-sequence decode, but being told the new length on the CPU before each token makes it 0.25–0.63 ms per token slower end to end (inferred from the per-layer timings; not profiled).

Reading the weights costs a fixed amount per token (8 GB for Qwen3-4B, ~22 ms; 1.2 GB for Qwen3-0.6B), and the cache's size reaches the weights' at ~54k tokens for Qwen3-4B and ~10k for Qwen3-0.6B, or proportionally sooner with batching. The method pays most when the weights are small relative to the cache: long context, batched serving, or smaller models. Graph-captured exact decoding differs from the engine's standard attention by TVD ~0.01 through BF16 rounding. During this work the engine's own exact-attention decode was made up to 2.6× faster by sharing KV heads across query heads instead of copying the cache.
