# Detailed results for `voronoi_skip`

Supporting detail for the [README](../README.md). Unless stated otherwise, results are for Qwen3-4B in bf16 on WikiText-103, with 8 text chunks per setting (2,040 scored decode steps). Fidelity is the total variation distance (TVD) between the model's next-token distribution and the exact model's; reads are key and value (K+V) rows, including region summaries; brackets are 95% bootstrap confidence intervals over chunks. Timings are on one RTX 5060 Ti GPU (16 GB, 448 GB/s).

**Fixed directions and fitted clusters.** The method first grouped keys by the nearest of 256 fixed random directions (`partition="random"`, called `voronoi_skip` in this file and in older result files); since October 2026 it groups them by centroids fitted with k-means at the end of the prompt (`partition="kmeans"`, the default). Every section below was measured with fixed directions unless it says otherwise. [Fitted clusters](#fitted-clusters) gives the results with the current method.

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

As reference points for what a given TVD means, Qwen3-4B with weight-only round-to-nearest quantization (embeddings and output layer kept in bf16), 8192 tokens:

| model | TVD from bf16 | top-1 agreement |
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

Three Triton kernels maintain and choose regions: they assign the key leaving the recent window, score the regions, and select up to the budget without sorting. Two more compact the selected positions into a list and run split attention over it (flash-decoding). One layer, Qwen3-4B head layout, bf16, RTX 5060 Ti; median of 100 calls with the L2 cache flushed, both sides captured in CUDA graphs (a recorded sequence of GPU operations replayed with one launch); exact attention by PyTorch's scaled dot-product attention (SDPA) and by FlashInfer (`kernel_bench --step --flashinfer`):

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
| Qwen3-0.6B | 16384 | 9.4 ms | 9.8 ms | 6.7 ms (1.40×) | 5.9 ms (1.58×) |
| Qwen3-0.6B | 32768 | 14.0 ms | 14.2 ms | 7.7 ms (1.81×) | 6.3 ms (2.22×) |
| Qwen3-0.6B | 40448 | 16.0 ms | 16.6 ms | 8.3 ms (1.93×) | 6.5 ms (2.46×) |
| Qwen3-4B | 16384 | 29.1 ms | 29.4 ms | 25.5 ms (1.14×) | 24.6 ms (1.18×) |
| Qwen3-4B | 32768 | 34.8 ms | 35.1 ms | 26.8 ms (1.30×) | 25.0 ms (1.39×) |

Rows at 16,384 and 32,768 tokens for Qwen3-0.6B come from the batched sweep's batch-1 run (`decode_speed_af689118_b1_Qwen3-0.6B.json`), the others from `decode_speed_7a026cd8_*_flashinfer.json`. Timings repeat closely: running a setting again in an independent run changes its median time by up to 2.5% for Qwen3-0.6B, whose 6–14 ms steps are sensitive to small delays on the CPU that launches them, and by under 0.1% for Qwen3-4B; speedups change by up to 0.03×. Within a run, the 3 repeats of a setting span a median 2.3% (at most 7.5%) for Qwen3-0.6B and 0.1% (at most 2.0%) for Qwen3-4B.

Speedups are relative to the faster exact kernel, the engine's own. FlashInfer's paged decode (`ssa.kernels.flashinfer_graph`) reads the engine's contiguous cache in place, viewed as 16-token pages; per layer it matches FlashInfer's single-sequence decode, but being told the new length by the CPU before each token makes it 0.2–0.6 ms per token slower end to end (inferred from the per-layer timings; not profiled).

Reading the weights costs a fixed amount per token (8 GB for Qwen3-4B, ~22 ms; 1.2 GB for Qwen3-0.6B), and the cache's size reaches the weights' at ~54k tokens for Qwen3-4B and ~10k for Qwen3-0.6B, or proportionally sooner with batching. The method pays most when the weights are small relative to the cache: long context, batched serving, or smaller models. Graph-captured exact decoding differs from the engine's standard attention by TVD ~0.01 through bf16 rounding. During this work the engine's own exact-attention decode was made up to 2.6× faster by sharing KV heads across query heads instead of copying the cache.

## Batched decoding

`GraphDecoder` (`src/ssa/models/graph_decode.py`), which runs the engine's per-token forward pass with the engine's own layers, weights and cache as one CUDA graph and lets the attention kernel be swapped, decodes B equal-length sequences together (B different WikiText chunks). Every attention mode treats the batch as B × 8 independent KV heads, viewing the engine's cache `[B, H_kv, L, d]` as `[B·H_kv, L, d]` without copying; tests check that a batch of 3 matches 3 separate runs. Speedups are over the faster of the two exact kernels (the engine's own and FlashInfer's paged decode) at the same batch size; bf16, CUDA graphs, median ms per decode step over 3 repeats of 64 steps (`decode_speed --graph --flashinfer --batch B`; `src/ssa/results/decode_speed_af689118_b1_Qwen3-0.6B.json` and `decode_speed_c3fa189d_b*_Qwen3-*.json`). The batch sizes and lengths are limited by the 16 GB card.

| model | batch | context | exact, ms per step | exact, tokens/s | 20% budget | 5% budget |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 1 | 8,192 | 7.2 | 139 | 1.16× | 1.25× |
| Qwen3-0.6B | 1 | 16,384 | 9.4 | 106 | 1.40× | 1.58× |
| Qwen3-0.6B | 1 | 32,768 | 14.0 | 71 | 1.81× | 2.22× |
| Qwen3-0.6B | 2 | 8,192 | 10.0 | 200 | 1.35× | 1.42× |
| Qwen3-0.6B | 2 | 16,384 | 14.2 | 141 | 1.63× | 1.93× |
| Qwen3-0.6B | 2 | 32,768 | 23.1 | 87 | 2.15× | 2.90× |
| Qwen3-0.6B | 4 | 8,192 | 14.4 | 278 | 1.58× | 1.94× |
| Qwen3-0.6B | 4 | 16,384 | 23.1 | 173 | 2.02× | 2.79× |
| Qwen3-0.6B | 8 | 8,192 | 23.4 | 342 | 1.91× | 2.55× |
| Qwen3-4B | 1 | 8,192 | 26.2 | 38 | 1.05× | 1.07× |
| Qwen3-4B | 1 | 16,384 | 29.1 | 34 | 1.14× | 1.18× |
| Qwen3-4B | 2 | 8,192 | 29.1 | 69 | 1.11× | 1.15× |
| Qwen3-4B | 2 | 16,384 | 34.4 | 58 | 1.26× | 1.34× |
| Qwen3-4B | 4 | 8,192 | 34.8 | 115 | 1.23× | 1.32× |

- **The speedup grows with batch size and tracks the batch's total cached tokens:** for Qwen3-0.6B with 32k cached tokens in total, 1.81× (1 × 32k), 1.63× (2 × 16k), 1.58× (4 × 8k); with 64k, 2.15×, 2.02×, 1.91×. Weights are read once per step for the whole batch, while every sequence's cache is read separately, so the cache's share of each step grows with the batch.
- **Fidelity is unchanged by batching:** TVD from the engine's exact decoding at a 20% budget, on one chunk per setting, is 0.054–0.076 for Qwen3-0.6B and 0.042–0.049 for Qwen3-4B, the same range as at batch 1.
- Exact throughput grows less than proportionally with batch size at long context (Qwen3-0.6B at 32k: 71 → 87 tokens/s from batch 1 to 2), because the cache reads grow with the batch.

## Long-context retrieval

Needle-in-a-haystack tasks modelled on RULER's (`ssa.harness.needle`, my own implementation, not official RULER scores). A sentence "The special magic number for KEY is: NNNNNNN." is inserted at a random depth (at a sentence boundary) in WikiText-103 text, and the prompt ends by asking for the number; `multi-key` inserts four such sentences with different keys and asks for one. The prompt is processed with exact attention except its last 64 tokens, which contain the question; those and the generated answer go through the decode path, so the answer depends on what sparse attention reads. Greedy decoding; an example is correct when the first number generated is the needle's. All conditions share the prompts and prefilled cache, so differences are paired. Qwen3-4B, 100 examples per cell (`src/ssa/results/needle_0b08ed17.json`).

| context | task | exact | 20% budget | 5% budget |
|---|---|---|---|---|
| 16,384 | single | 1.00 | 1.00 (+0.00) | 0.97 (−0.03 [−0.07, 0.00]) |
| 16,384 | multi-key | 1.00 | 1.00 (+0.00) | 0.95 (−0.05 [−0.10, −0.01]) |
| 32,768 | single | 1.00 | 0.99 (−0.01 [−0.03, 0.00]) | 0.92 (−0.08 [−0.14, −0.03]) |
| 32,768 | multi-key | 1.00 | 0.98 (−0.02 [−0.05, 0.00]) | 0.92 (−0.08 [−0.14, −0.03]) |

- K+V reads were 21–22% at the 20% budget and 6–7% at 5%.
- Of the 27 errors, 15 are near misses (the right number with one or two digits wrong, dropped or repeated: the needle was found but not every digit token was read) and 12 are a different number, 11 of them on multi-key. Whether those were the distractor needles was not recorded.
- Limits: exact attention scores 100%, so this test cannot separate conditions above the 20% budget; a needle is unusual text in its haystack and may be easier to find than other retrieval targets; RULER's harder task types (multi-value, variable tracking, aggregation) and LongBench were not run.

## Comparison with Quest

Quest (arXiv:2406.10774) splits the cache into 16-token pages, stores each page's element-wise minimum and maximum keys, and reads the pages with the highest bound `Σ_i max(q_i·min_i, q_i·max_i)` on `q·k` (`ssa.attn.quest`). `quest_matched` adds `voronoi_skip`'s always-read tokens (token 0 and the last 64) and one shared selection per KV head, so the remaining differences are pages by position compared with regions by direction, and Quest's bound compared with our score. Reads include each method's summaries: Quest's two keys per page cost ~6% of K+V rows.

End to end (`accept_sweep --preset quest_vs_voronoi`, both methods in PyTorch in one sweep; `src/ssa/results/tvd_4b_8k_quest_vs_voronoi.json`; `ssa.harness.matched_reads` interpolates each chunk's TVD along each curve and bootstraps over chunks):

| K+V reads | TVD, `quest_matched` | TVD, `voronoi_skip` | ratio [95% CI] |
|---|---|---|---|
| 20% | 0.0555 | 0.0487 | 1.14 [1.10, 1.18] |
| 23% | 0.0489 | 0.0434 | 1.13 [1.08, 1.17] |
| 27% | 0.0413 | 0.0393 | 1.05 [1.01, 1.10] |
| 30% | 0.0377 | 0.0365 | 1.03 [0.99, 1.08] |
| 37% | 0.0305 | 0.0307 | 1.00 [0.95, 1.04] |
| 42% | 0.0270 | 0.0271 | 1.00 [0.95, 1.04] |

`quest_matched` cannot read less than 17%; `voronoi_skip` reaches 8% (TVD 0.084).

Offline, per layer (`ssa.harness.quest_replay`, 8 WikiText contexts at 8k, single-layer attention-output error; `src/ssa/results/quest_replay_0a8e6747.json`), the comparison goes both ways: `quest_matched` has 1.3–1.9× our error at layer 12 at 10–30% reads, 0.27–0.88× at layer 24, and crosses over at layer 35 (1.58× at 10%, 0.74× at 30%). `quest_plain`, which pages every token and lets each query head choose its own pages, has 8–127× our error at 15–30% reads: heads that attend mostly to recent tokens lose them, because the bound over 128 dimensions is too loose to rank those pages near the top. The Quest paper keeps the first two layers dense and does not state whether recent tokens are always read or whether selection is per head; these runs apply sparsity to every layer for both methods.

## Fitted clusters

**What matters: partition, centering or score** (`ssa.harness.partition_ablation`, `src/ssa/results/partition_ablation_e26ca205.json`). On the 8 WikiText captures at 8,192 tokens, all eight combinations of partition (fixed random directions or cosine k-means), centering (mean-centered or raw keys) and score (key length × projection on the cluster's mean direction, or the projection alone) were run with the same always-read tokens, shared selection and 256 clusters. Single-layer attention-output error at matched K+V reads, divided by the fixed-direction method's (ranges over 10–30% reads; layer 0 at 30% only):

| combination | layer 0 | layer 12 | layer 24 | layer 35 |
|---|---|---|---|---|
| random / centered / length (fixed directions) | 1 | 1 | 1 | 1 |
| random / centered / projection | 1.02 | 1.01–1.08 | 0.97–1.01 | 0.96–1.01 |
| random / raw / length | — | 2.1–2.8 | 1.1–1.4 | 2.0–2.4 |
| k-means / centered / length (current method) | 0.16 | 0.42–0.52 | 0.55–0.62 | 0.28–0.46 |
| k-means / centered / projection | 0.41 | 0.52–0.63 | 0.58–0.63 | 0.31–0.51 |
| k-means / raw / length | 0.15 | 0.44–0.53 | 0.58–0.66 | 0.31–0.44 |
| k-means / raw / projection (ClusterKV-style scoring) | 0.16 | 0.54–0.62 | 0.59–0.62 | 0.34–0.49 |

The partition is what matters; the score changes little, and centering matters only with fixed directions. Tests check that the first row reproduces the fixed-direction method and the last reproduces the ClusterKV-style selector.

**End to end** (`accept_sweep --preset kmeans_vs_random`, both partitions in one sweep with the incremental PyTorch index, paired by chunk, `src/ssa/results/tvd_4b_8k_kmeans_vs_random.json`; and `--preset fused_kmeans` with the Triton kernels, `tvd_4b_8k_fused_kmeans.json`). Qwen3-4B, 8 WikiText chunks at 8,192 tokens:

| budget | fitted, kernels: K+V read, TVD, top-1 | fitted, PyTorch: TVD | fixed directions: K+V read, TVD, top-1 | TVD ratio, fitted / fixed [95% CI] |
|---|---|---|---|---|
| 5% | 7.6%, 0.064 [0.054, 0.075], 92.4% | 0.064 | 8.2%, 0.085 [0.072, 0.097], 90.9% | 0.76 [0.71, 0.80] |
| 10% | 12.6%, 0.046 [0.038, 0.055], 94.2% | 0.047 | 13.1%, 0.063 [0.053, 0.073], 92.2% | 0.74 [0.69, 0.79] |
| 20% | 22.5%, 0.032 [0.026, 0.038], 96.6% | 0.031 | 23.0%, 0.044 [0.037, 0.051], 94.9% | 0.72 [0.67, 0.74] |
| 40% | 42.3%, 0.020 [0.017, 0.024], 97.6% | 0.021 | 42.5%, 0.027 [0.023, 0.030], 97.4% | 0.77 [0.71, 0.82] |

The ratio is from the PyTorch sweep. Centroids are fitted once per KV head with 10 k-means iterations at the first build and then kept; later keys join the nearest centroid. Not yet measured: more than 255 decode steps after the fit, other models, and 32,768 tokens.

**Speed** (`decode_speed --graph --partition {kmeans,random}`, `decode_speed_39f45d3f_*`; ms per token at 16,384 / 32,768 tokens, 20% budget): Qwen3-4B 25.48 / 26.82 with fitted clusters and 25.50 / 26.85 with fixed directions; Qwen3-0.6B 6.64 / 7.71 and 6.68 / 7.75. The binning kernels read each KV head's centroids instead of one shared set of directions; the work per token is otherwise the same, and the fit runs once, before decoding.

**Compared with ClusterKV-style selection** (`ssa.attn.clusterkv`; `quest_replay --clusterkv`, `src/ssa/results/quest_replay_e26ca205_clusterkv.json`; same captures, error divided by the fixed-direction method's at 10–30% reads): k-means clusters with ClusterKV's scoring and our always-read tokens and shared selection have 0.53–0.62 (layer 12), 0.59–0.62 (24) and 0.34–0.49 (35), which is what led to adopting fitted clusters. ClusterKV's reported settings (about one cluster per 80 tokens, the first 16 tokens always read, each query head selecting for itself, and here the not-yet-clustered decode tokens always read, which is this implementation's assumption) have 1.2–1.5 (layer 12), 1.2–2.0 (24) and 0.61–0.65 (35): 1.4–3.6× the current method's error.

**Compared with Quest-style pages**: at matched reads, `quest_matched` has 1.59 [1.53, 1.66] times the fitted-cluster method's TVD at 20% of K+V rows read, 1.44 [1.39, 1.51] at 27%, and 1.32 [1.24, 1.41] at 42% (`ssa.harness.matched_reads` on `tvd_4b_8k_quest_vs_voronoi.json` and `tvd_4b_8k_fused_kmeans.json`: two runs on the same 8 chunks, whose fixed-direction controls agree to within 0.001 TVD).

