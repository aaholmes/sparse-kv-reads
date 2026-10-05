# Efficient inference by reducing KV cache reads

When a large language model generates text at long context, its speed is limited by memory traffic: every new token re-reads the cached key and value vectors of every earlier token (the KV cache) from GPU memory. A recent method, SANTA ([arXiv:2605.01910](https://arxiv.org/abs/2605.01910)), approximates attention by sampling which cached values to read, but it still reads part of every key to compute the sampling probabilities. I asked whether most keys could be skipped entirely. I developed a training-free method that clusters the cached keys by direction, decides which clusters to read from small running summaries, and reads only those. I wrote Triton GPU kernels for it and run it inside a Qwen3 inference engine I wrote separately ([github.com/aaholmes/llms](https://github.com/aaholmes/llms)).

## Method

Attention weights each cached value by the softmax of the query's dot product with its key, so the few keys pointing along the query carry most of the weight. The method finds them without reading every key:

1. **Always read the first token and the 64 most recent tokens.** The first token acts as an attention sink, taking 36–65% of all attention at layers 12–35 of Qwen3-4B.
2. **Subtract the mean key** of each layer and KV head (Qwen3 uses grouped-query attention: each set of keys and values, a "KV head", is shared by several query heads). Softmax ignores a shift common to every score, so attention is unchanged.
3. **Cluster the prompt's keys by direction.** At the end of the prompt, k-means on cosine similarity fits 256 centroids per KV head. Each cluster keeps a running sum of its keys' directions, the minimum and maximum key length, and a count.
4. **Score each cluster without reading its keys**, as its maximum key length times the query's projection on its mean direction (minimum length when the projection is negative): an estimate of the largest attention score inside it.
5. **Choose once per KV head.** In Qwen3-4B four query heads share each KV head, so they rank clusters jointly and read one set of rows.
6. **Read the top clusters' keys and values up to a budget** and compute exact attention over them. The rest is dropped, so the result is slightly biased.
7. **Update incrementally.** Each new key joins its nearest centroid when it leaves the recent window, at the cost of one comparison with the centroids. The centroids are not refitted during generation.

In the code this is `cluster_fused` (Triton kernels) with `partition="kmeans"`, the default. Until October 2026 the clusters came from 256 fixed random directions instead of fitted centroids (`partition="random"`); fitting them lowers next-token error by 23–28% at equal reads and equal speed.

## Status (October 2026)

All results use bf16 on one consumer GPU, an RTX 5060 Ti with 16 GB of memory and 448 GB/s of bandwidth, decoding one sequence at a time unless stated. "Budget" is the fraction of keys the method reads. Timings repeat closely: running a setting again in an independent run changes its median time by up to 2.5% for Qwen3-0.6B, whose 6–14 ms steps are sensitive to small delays on the CPU that launches them, and by under 0.1% for Qwen3-4B; speedups change by up to 0.03×.

**Which results use fitted clusters.** The fidelity table below does. Speed does not depend on how the clusters are chosen (the work per token is the same; end-to-end times with fitted and fixed-direction clusters agree within 0.6%). The retrieval test, the comparison with Quest's own run and the fidelity figures at 32,768 tokens were measured with fixed-direction clusters and are marked †; they have not yet been repeated with fitted clusters.

**Fidelity** (Qwen3-4B, 8,192-token context). Fidelity is the total variation distance (TVD) between the model's next-token distribution and the exact model's, on 8 WikiText-103 chunks with 95% bootstrap intervals over chunks, and the fraction of tokens where both pick the same top token ("top-1"). Reads are key and value (K+V) rows, including cluster summaries. Systematic sampling, SANTA's best variant, reads every key and samples S value rows:

| method | K+V rows read | TVD from exact | top-1 |
|---|---|---|---|
| this method, 5% budget | 7.6% | 0.064 [0.054, 0.075] | 92.4% |
| this method, 10% budget | 12.6% | 0.046 [0.038, 0.055] | 94.2% |
| this method, 20% budget | 22.5% | 0.032 [0.026, 0.038] | 96.6% |
| this method, 40% budget | 42.3% | 0.020 [0.017, 0.024] | 97.6% |
| systematic sampling, 64 samples | 50.5% | 0.058 [0.049, 0.065] | 92.9% |
| systematic sampling, 256 samples | 51.4% | 0.026 [0.023, 0.028] | 96.9% |

![TVD from the exact model versus key and value rows read, for this method at budgets of 5–40% and systematic sampling with 64 and 256 samples](docs/tvd_vs_reads_8192.png)

**Attention only** (one layer, Qwen3-4B head layout, µs per decode step; ours includes choosing what to read). FlashInfer, the exact attention library used by serving engines such as SGLang, runs its single-sequence decode on a contiguous copy of the cache, its fastest case:

| context | exact, FlashInfer | this method, 20% budget | this method, 5% budget |
|---|---|---|---|
| 8,192 | 92 | 55 (1.7× faster) | 45 (2.1×) |
| 32,768 | 337 | 109 (3.1×) | 57 (5.9×) |
| 65,536 | 660 | 182 (3.6×) | 80 (8.3×) |

**End-to-end decoding** (whole model, 32,768-token context, ms per token). Only the attention kernel differs between columns. The exact baseline is the faster of two exact kernels: the engine's own, and FlashInfer (0.2–0.6 ms per token slower here; see [How speed is measured](#how-speed-is-measured)):

| model | exact | 20% budget | 5% budget |
|---|---|---|---|
| Qwen3-4B | 34.8 ms | 26.8 ms (1.30× faster) | 25.0 ms (1.39×) |
| Qwen3-0.6B | 14.0 ms | 7.7 ms (1.81×) | 6.3 ms (2.22×) |

**Batched decoding** (B sequences of equal length decoded together; speedup over the faster exact kernel at each batch size; full table in [docs/results.md](docs/results.md#batched-decoding)):

| model | batch × context | exact, tokens/s | 20% budget | 5% budget |
|---|---|---|---|---|
| Qwen3-0.6B | 1 × 32,768 | 71 | 1.81× faster | 2.22× |
| Qwen3-0.6B | 2 × 32,768 | 87 | 2.15× | 2.90× |
| Qwen3-0.6B | 8 × 8,192 | 342 | 1.91× | 2.55× |
| Qwen3-4B | 2 × 16,384 | 58 | 1.26× | 1.34× |
| Qwen3-4B | 4 × 8,192 | 115 | 1.23× | 1.32× |

The speedup tracks the total number of tokens cached across the batch, relative to the size of the weights. With 16 GB, Qwen3-4B fits only ~32k cached tokens beside its 8 GB of weights, which caps its gain on this card.

**Long-context retrieval** † (needle-in-a-haystack tasks modelled on RULER, in my own implementation: a 7-digit number is hidden at a random depth in WikiText-103 text and the prompt asks for it; "multi-key" hides four numbers under different keys and asks for one). Qwen3-4B, 100 examples per cell; the question and the answer go through the sparse decode path. Accuracy, with the paired difference from exact and its 95% interval:

| context | task | exact | 20% budget | 5% budget |
|---|---|---|---|---|
| 16,384 | single | 1.00 | 1.00 | 0.97 (−0.03 [−0.07, 0.00]) |
| 16,384 | multi-key | 1.00 | 1.00 | 0.95 (−0.05 [−0.10, −0.01]) |
| 32,768 | single | 1.00 | 0.99 (−0.01 [−0.03, 0.00]) | 0.92 (−0.08 [−0.14, −0.03]) |
| 32,768 | multi-key | 1.00 | 0.98 (−0.02 [−0.05, 0.00]) | 0.92 (−0.08 [−0.14, −0.03]) |

More than half of the errors are near misses: the right number with one or two digits wrong.

- **Attention gets much faster, and more so at longer context.** FlashInfer already reads the cache at 81–91% of the card's bandwidth, so the gain comes from reading less, not from a faster kernel.
- **End to end, the weights limit the gain at batch 1.** Reading Qwen3-4B's 8 GB of weights takes ~22 ms of each token's 34.8 ms, so even free attention could not exceed about 1.6×. Batching helps, because the weights are read once per batch while each sequence's cache is read separately.
- **Fidelity costs something, less than sampling does.** Reading 8% of the cache is about as close to the exact model as 64-sample sampling, which reads 51%; reading 42% is closer than 256-sample sampling. For scale, 8-bit and naive 4-bit weight quantization of Qwen3-4B give TVD 0.018 and 0.144.
- **Not yet done:** repeating the † results with fitted clusters; fidelity after long generations, when clusters fitted to the prompt may go stale; sequences of different lengths in one batch; harder long-context tasks (RULER's multi-value, tracking and aggregation tasks, LongBench); datacenter GPUs; integration into a serving engine such as SGLang.

## Results

<a id="how-speed-is-measured"></a>*How speed is measured.* The model runs in my own Qwen3 inference engine. `GraphDecoder` (`src/ssa/models/graph_decode.py`) reruns the engine's per-token forward pass, with the engine's own layers, weights and cache, as one CUDA graph (a recorded sequence of GPU operations replayed with one launch), so only the attention kernel differs between conditions. Two exact kernels are timed: the engine's own Triton kernel, which splits each sequence across thread blocks and merges the parts (flash-decoding), and FlashInfer's paged decode, which reads the engine's contiguous cache in place as 16-token pages and is told the new length by the CPU before each token. Per layer FlashInfer is as fast as its single-sequence decode, but end to end it is 0.2–0.6 ms per token slower, most likely because of that per-token step, so speedups are reported relative to the faster of the two. Per-layer timings flush the L2 cache before each call and take the median of 100 calls; end-to-end timings take the median of 3 repeats of 64 steps. Full tables, with every context length and both exact kernels, are in [docs/results.md](docs/results.md).

*What matters in the design.* An ablation on captured keys (8 WikiText contexts at 8,192 tokens, single-layer attention error at matched reads) varied three choices with everything else fixed. How keys are grouped is the one that matters: clusters fitted by k-means have 0.28–0.66× the error of clusters from fixed random directions at layers 12–35. The scoring rule changes the error by under 10% with fixed directions, and subtracting the mean key matters only with fixed directions.

*Other settings* †. At 32,768 tokens a 20% budget gives TVD 0.064 [0.052, 0.074] on Qwen3-4B and 0.061 [0.051, 0.072] on Qwen3-0.6B, and a 5% budget 0.111 and 0.133. On Python code every TVD is 2–3× lower than on WikiText. Choosing clusters at random instead of by score gives 3–4× the TVD (at 2,048 tokens), so the ranking does the work.

*Sampling the skipped clusters loses* †. Reading the top clusters exactly and sampling some of the rest, weighted by inverse inclusion probability, removes the bias. But at equal reads its TVD is 24–54% higher than reading more top clusters, on both models and at 8,192 and 32,768 tokens. Its error is almost all variance: after the top clusters, the remaining attention is spread thinly over ~200 clusters, and a few sampled clusters estimate that tail less accurately than dropping it does.

More detail, including context-length and cluster-count scans and a comparison with weight quantization, is in [docs/results.md](docs/results.md).

## Related work

The method is closest to ClusterKV ([arXiv:2412.03213](https://arxiv.org/abs/2412.03213)), which also clusters keys by direction with k-means and reads the top clusters exactly. I first built it with fixed random directions, then compared it with a ClusterKV-style selector and adopted fitted clusters when they proved clearly better. What remains different: one selection is shared by the query heads of each KV head, a recent window is always read, clusters are fitted once and then updated key by key instead of being re-clustered periodically, and clusters are scored using key lengths as well as directions. On captured keys, ClusterKV's settings as reported (about one cluster per 80 tokens, each query head selecting for itself) have 1.4–3.6× this method's single-layer error at matched reads.

Quest ([arXiv:2406.10774](https://arxiv.org/abs/2406.10774)) selects fixed 16-token pages using per-page minimum and maximum keys. At matched reads on Qwen3-4B at 8,192 tokens, Quest-style pages given this method's always-read tokens and shared selection have 1.3–1.6× its TVD (20–42% of K+V rows read, the same 8 chunks in two runs), and cannot read less than ~17% because their per-page summaries cost ~6% of reads. Without those additions, Quest misses heads that attend to recent tokens and does far worse. Details are in [docs/results.md](docs/results.md#comparison-with-quest).

MagicPIG ([arXiv:2410.16179](https://arxiv.org/abs/2410.16179)) centers keys, then samples them with locality-sensitive hashing. SANTA++ ([arXiv:2609.35629](https://arxiv.org/abs/2609.35629)), from SANTA's authors, samples groups of keys.

## Earlier work: sampling the cache

Before the method above, I reproduced SANTA's sampling estimator in PyTorch and tested extensions of it: exact computation of the heaviest tokens, contiguous-block sampling, cluster-based key skipping, and a learned correction for sampling bias. Plain systematic sampling matches exact perplexity within +0.19% while reading 3.5% of cached values; none of the extensions beat it. Details are in [docs/sampling.md](docs/sampling.md).

## Repository

- `src/ssa/attn/` — attention methods behind one interface, `attn(q, K, V, impl=...)`; `src/ssa/sampling/` — index sampling.
- `src/ssa/kernels/` — Triton GPU kernels (cluster maintenance and selection, compaction, attention over selected rows).
- `src/ssa/models/` — `patch.py` plugs attention methods into the engine's decode hook (used for the fidelity runs); `graph_decode.py` (`GraphDecoder`) runs the whole decode step as one CUDA graph with a choice of attention kernel (used for the speed runs).
- `src/ssa/harness/` — experiments: `accept_sweep` measures TVD end to end, `decode_speed` times decoding, `kernel_bench` times the kernels; others cover the sampling work. Real-model harnesses need a CUDA GPU and download the model.
- `src/ssa/results/` — result files, each recording the git commit, GPU and library versions that produced it.

Setup: `uv sync`, then `uv run python -m pytest` (302 tests; the kernel tests need a CUDA GPU and the FlashInfer comparison needs FlashInfer, and both are skipped otherwise). The engine is installed from [github.com/aaholmes/llms](https://github.com/aaholmes/llms) at a pinned commit.

## License

Apache License 2.0; see [LICENSE](LICENSE).
