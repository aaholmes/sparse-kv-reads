# Efficient inference by reducing KV cache reads

Decoding at long context is limited by reading the KV cache: every new token re-reads the stored key and value vectors of every earlier token. After reading about the SANTA algorithm ([arXiv:2605.01910](https://arxiv.org/abs/2605.01910)), I noticed that it avoids most value reads by sampling, but still reads part of every key to compute the sampling probabilities. I asked whether most keys could be skipped entirely. I developed `voronoi_skip`, a training-free method that decides which parts of the cache to read from small running summaries of the keys, wrote Triton kernels for it, and run it inside a Qwen3 inference engine I wrote separately ([github.com/aaholmes/llms](https://github.com/aaholmes/llms)).

## Status (October 2026)

BF16, batch 1, one RTX 5060 Ti (16 GB, 448 GB/s). "Budget" is the fraction of keys the method reads.

**Attention only** (one layer, Qwen3-4B head layout, µs per decode step; ours includes choosing what to read):

| context | exact, FlashInfer | `voronoi_skip`, 20% budget | `voronoi_skip`, 5% budget |
|---|---|---|---|
| 8,192 | 92 | 55 (1.7× faster) | 45 (2.1×) |
| 32,768 | 337 | 109 (3.1×) | 57 (5.9×) |
| 65,536 | 660 | 182 (3.6×) | 80 (8.3×) |

**End-to-end decoding** (whole model, 32,768-token context, ms per token). The exact baseline is FlashInfer's paged decode, captured in the same CUDA graph and reading the same cache. Fidelity is the total variation distance (TVD) between the model's next-token distribution and the exact model's, with 95% intervals over 8 WikiText chunks, and how often both pick the same top token:

| model | exact | 20% budget | 5% budget |
|---|---|---|---|
| Qwen3-4B: speed | 35.1 ms | 26.8 ms (1.31× faster) | 25.0 ms (1.41×) |
| Qwen3-4B: fidelity | — | TVD 0.064 [0.052, 0.074]; top-1 92.3% | TVD 0.111 [0.093, 0.127]; top-1 86.9% |
| Qwen3-0.6B: speed | 14.5 ms | 7.8 ms (1.85×) | 6.3 ms (2.30×) |
| Qwen3-0.6B: fidelity | — | TVD 0.061 [0.051, 0.072]; top-1 92.0% | TVD 0.133 [0.116, 0.147]; top-1 84.8% |

- **Attention gets much faster, and more so at longer context.** FlashInfer already reads the cache at 81–91% of the card's bandwidth, so the gain comes from reading less, not from a faster kernel.
- **End to end, the weights limit the gain at batch 1.** Reading Qwen3-4B's 8 GB of weights takes ~22 ms of each token's 35.1 ms, so even free attention could not exceed about 1.6×. Batched serving reads the weights once per batch and each sequence's cache separately, so it should keep more of the attention speedup; that is not yet measured.
- **Fidelity costs something.** A 20% budget is about as close to the exact model as 64-sample SANTA-style sampling (TVD 0.057), which reads every key; a 5% budget lies between 8-bit and naive 4-bit weight quantization of Qwen3-4B (TVD 0.018 and 0.144, measured at 8,192 tokens).
- **Not yet done:** batched kernels and paged KV caches; comparison with FlashInfer's batched decode; task benchmarks (RULER, LongBench); direct comparison with Quest and ClusterKV; datacenter GPUs.

## Method

Attention weights each cached value by the softmax of the query's dot product with its key, so the few keys pointing along the query carry most of the weight. The method finds them without reading every key:

1. **Always read the first token and the 64 most recent tokens.** The first token acts as an attention sink, taking 36–65% of all attention at layers 12–35 of Qwen3-4B.
2. **Subtract the mean key** of each layer and KV head. This leaves attention unchanged, because softmax ignores a shift common to every score, and it is necessary: before centering, the keys at layer 0 all point almost the same way (average cosine similarity 0.99 with the mean).
3. **Group keys by direction** using 256 fixed random directions: each key joins the region of its nearest direction, so the regions are the Voronoi cells of those directions on the unit sphere (hence the name `voronoi_skip`). Each region keeps a running sum of its keys' directions, the minimum and maximum key length, and a count.
4. **Score each region without reading its keys**, as its maximum key length times the query's projection on its mean direction (minimum length when the projection is negative): an estimate of the largest attention score inside it.
5. **Choose once per KV head.** In Qwen3-4B four query heads share each KV head, so they rank regions jointly and read one set of rows.
6. **Read the top regions' keys and values up to a budget** and compute exact attention over them. The rest is dropped, so the result is slightly biased.
7. **Update the regions incrementally.** Each key is assigned once, when it leaves the recent window, and a head is recentered only when its mean has drifted noticeably, which after a long prompt is rare.

Fidelity is measured as total variation distance (TVD) between the model's next-token distribution and the exact model's, over 8 text chunks per setting. Perplexity is not used, because a biased method can push it *below* the exact model's.

## Results

*Fidelity compared with systematic sampling (SANTA's best variant).* Qwen3-4B, 8192-token context, WikiText-103; reads are key and value rows, including region summaries; [95% bootstrap CI over chunks].

| method | K+V rows read | TVD from exact |
|---|---|---|
| systematic sampling, 64 samples | 50.5% | 0.058 [0.049, 0.065] |
| `voronoi_skip`, 10% budget | 13.1% | 0.063 [0.053, 0.074] |
| systematic sampling, 256 samples | 51.4% | 0.026 [0.023, 0.028] |
| `voronoi_skip`, 40% budget | 42.5% | 0.027 [0.023, 0.030] |

![TVD from the exact model versus key and value rows read, for voronoi_skip at budgets of 2–40% and systematic sampling with 64 and 256 samples](docs/tvd_vs_reads_8192.png)

*The same setting across budgets of 2–40%; the two systematic-sampling points are 64 and 256 samples.*

At 32,768 tokens the advantage shrinks: matching 64-sample sampling takes 0.50× its reads, and 256-sample sampling is not matched within a 40% budget. On Qwen3-0.6B it matches 64-sample sampling with 0.23–0.26× the reads, and on Python code every TVD is 2–3× lower than on WikiText. Choosing random regions instead gives 3–4× the TVD (at 2048 tokens), so the ranking does the work.

*Attention per layer, compared with FlashInfer.* FlashInfer is the exact decode-attention library used by serving engines such as SGLang. One layer, Qwen3-4B head layout, BF16, batch 1, both sides in CUDA graphs, L2 cache flushed, median of 100 calls; our step includes region maintenance and selection:

| context | FlashInfer | `voronoi_skip`, 20% budget | `voronoi_skip`, 5% budget |
|---|---|---|---|
| 8192 | 92 µs | 55 µs (1.67×) | 45 µs (2.06×) |
| 16384 | 174 µs | 74 µs (2.36×) | 49 µs (3.56×) |
| 32768 | 337 µs | 109 µs (3.11×) | 57 µs (5.89×) |
| 65536 | 660 µs | 182 µs (3.63×) | 80 µs (8.25×) |

FlashInfer reads the cache at 364–407 GB/s, close to the card's nominal 448 GB/s, and PyTorch's own attention is within 2–8% of it: exact decode is limited by memory bandwidth, so the remaining gain has to come from reading less.

*End-to-end decoding.* The whole decode step runs as a CUDA graph (one recorded sequence of GPU operations replayed per token, removing Python overhead); the exact baseline is FlashInfer's paged decode, captured the same way (`decode_speed --graph --flashinfer`). BF16, one RTX 5060 Ti (16 GB), batch 1; median ms per token over 3 repeats of 64 steps:

| model | context | exact attention | 20% budget | 5% budget |
|---|---|---|---|---|
| Qwen3-0.6B | 16384 | 9.8 ms | 6.7 ms (1.46×) | 6.1 ms (1.61×) |
| Qwen3-0.6B | 32768 | 14.5 ms | 7.8 ms (1.85×) | 6.3 ms (2.30×) |
| Qwen3-0.6B | 40448 | 16.6 ms | 8.3 ms (2.01×) | 6.5 ms (2.55×) |
| Qwen3-4B | 16384 | 29.4 ms | 25.5 ms (1.15×) | 24.6 ms (1.19×) |
| Qwen3-4B | 32768 | 35.1 ms | 26.8 ms (1.31×) | 25.0 ms (1.41×) |

Fidelity at these settings: on Qwen3-4B at 32,768 tokens a 20% budget gives TVD 0.064 [0.052, 0.074] (64-sample sampling: 0.057); on Qwen3-0.6B a 10% budget already matches 64-sample sampling (0.091 compared with 0.095), and a 20% budget reads more. A 5% budget gives TVD 0.111 [0.093, 0.127] on Qwen3-4B (86.9% top-1 agreement with the exact model) and 0.133 [0.116, 0.147] on Qwen3-0.6B (84.8%), both at 32,768 tokens; fidelity at 40,448 tokens was not measured. The gain follows attention's share of the time per token: Qwen3-4B's 8 GB of weights cost ~22 ms to read, so the cache only matters at long context or when many sequences are batched.

*Sampling the skipped regions loses.* Reading the top regions exactly and sampling some of the rest, weighted by inverse inclusion probability, removes the bias. But at equal reads its TVD is 24–54% higher than reading more top regions, on both models and at 8192 and 32,768 tokens. Its error is almost all variance: after the top regions, the remaining attention is spread thinly over ~200 regions, and a few sampled regions estimate that tail less accurately than dropping it does.

More detail, including kernel timings, context-length and region-count scans, and a comparison with weight quantization, is in [docs/results.md](docs/results.md).

## Related work

After building this, I found that it is close in spirit to ClusterKV ([arXiv:2412.03213](https://arxiv.org/abs/2412.03213)), which also groups keys by direction (with k-means clustering) and reads the top groups exactly. The differences are that here keys are mean-centered first, the groups come from fixed random directions updated incrementally instead of periodic clustering, groups are scored by mean direction times maximum or minimum length, one selection is shared by the query heads of each KV head, and a recent window is always read. Quest ([arXiv:2406.10774](https://arxiv.org/abs/2406.10774)) selects fixed 16-token pages using per-page minimum and maximum keys. MagicPIG ([arXiv:2410.16179](https://arxiv.org/abs/2410.16179)) also centers keys, then samples them with locality-sensitive hashing (hashes that put similar vectors in the same bucket). SANTA++ ([arXiv:2609.35629](https://arxiv.org/abs/2609.35629)), from SANTA's authors, samples groups of keys.

## Earlier work: sampling the cache

Before the method above, I reproduced SANTA's sampling estimator in PyTorch and tested extensions of it: exact computation of the heaviest tokens, contiguous-block sampling, cluster-based key skipping, and a learned correction for sampling bias. Plain systematic sampling matches exact perplexity within +0.19% while reading 3.5% of cached values; none of the extensions beat it. Details are in [docs/sampling.md](docs/sampling.md).

## Repository

- `src/ssa/attn/` — attention methods behind one interface, `attn(q, K, V, impl=...)`; `src/ssa/sampling/` — index sampling.
- `src/ssa/kernels/` — Triton GPU kernels (region maintenance and selection, compaction, attention over selected rows).
- `src/ssa/models/` — the adapter into the inference engine, and the CUDA-graph decode step.
- `src/ssa/harness/` — experiments: `accept_sweep` measures TVD end to end, `decode_speed` times decoding, `kernel_bench` times the kernels; others cover the sampling work. Real-model harnesses need a CUDA GPU and download the model.
- `src/ssa/results/` — result files, each recording the git commit, GPU and library versions that produced it.

Setup: `uv sync`, then `uv run python -m pytest` (265 tests; the kernel tests need a CUDA GPU and the FlashInfer comparison needs FlashInfer, and both are skipped otherwise). The engine is installed from [github.com/aaholmes/llms](https://github.com/aaholmes/llms) at a pinned commit.

## License

Apache License 2.0; see [LICENSE](LICENSE).
