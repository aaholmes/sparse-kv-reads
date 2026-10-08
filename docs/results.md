# Detailed results

Supporting detail for the [README](../README.md). All runs use bf16 on one RTX 5060 Ti (16 GB, 448 GB/s of memory bandwidth), with text from the WikiText-103 test set unless stated.

**Configuration.** Unless a section says otherwise, results use the default configuration: 256 clusters per set of keys fitted by k-means at the end of the prompt, 1,024 cluster slots, a cluster split in two when it exceeds twice the mean cluster size at the fit, 8-bit cluster vectors, the first token and the last 64 always read. "Budget" is the fraction of keys read. "Bytes read" is everything read per decode step as a fraction of the key and value (K+V) cache: the selected key and value rows, both cluster vectors, the lengths and count per cluster, a label per cached token, a position and label per row read, and the inserted key (`ssa.attn.accounting`). Intervals are 95% bootstrap intervals over text chunks or examples.

**Fidelity** is the total variation distance (TVD) between the model's next-token distribution and the exact model's; "top-1" is how often both pick the same token.

## Fidelity

Qwen3-4B, 8,192-token context, 8 chunks, 256 decode steps each (`tvd_4b_8k_default.json`). Systematic sampling, the best variant of SANTA ([arXiv:2605.01910](https://arxiv.org/abs/2605.01910)), reads every key and samples S value rows:

| method | K+V bytes read | TVD from exact | top-1 |
|---|---|---|---|
| this method, 5% budget | 8.2% | 0.063 [0.053, 0.074] | 92.2% |
| this method, 10% budget | 13.2% | 0.046 [0.039, 0.054] | 94.4% |
| this method, 20% budget | 23.3% | 0.032 [0.026, 0.038] | 95.8% |
| this method, 40% budget | 43.3% | 0.020 [0.016, 0.024] | 97.4% |
| systematic sampling, 64 samples | 50.5% | 0.058 [0.049, 0.065] | 92.9% |
| systematic sampling, 256 samples | 51.4% | 0.026 [0.023, 0.028] | 96.9% |

![TVD from the exact model versus bytes of the cache read, for this method at budgets of 5–40% and systematic sampling with 64 and 256 samples](tvd_vs_reads_8192.png)

At 32,768 tokens (`tvd_4b_32k_default.json`, `tvd_06b_32k_default.json`; 8 chunks):

| budget | K+V bytes read | Qwen3-4B TVD | Qwen3-0.6B TVD |
|---|---|---|---|
| 5% | 6.3% | 0.089 [0.071, 0.105] | 0.094 [0.079, 0.108] |
| 10% | 11.3% | 0.075 [0.059, 0.089] | 0.064 [0.052, 0.076] |
| 20% | 21.4% | 0.057 [0.042, 0.072] | 0.042 [0.032, 0.055] |
| 30% | 31.5% | 0.044 [0.031, 0.057] | 0.029 [0.021, 0.041] |

## Task accuracy

LongBench ([arXiv:2308.14508](https://arxiv.org/abs/2308.14508)) with the benchmark's prompts, middle truncation and scoring (`ssa.harness.longbench`; `longbench_4a5c5ba5_default.json`). Qwen3-4B, the first 100 examples of each set, greedy decoding of at most 32 tokens, chat format with thinking off. The prompt is processed with exact attention except its last 64 tokens; those and the answer use the sparse path. Score, with the paired difference from exact:

| set (mean prompt tokens) | exact | 20% budget | 10% budget | 5% budget |
|---|---|---|---|---|
| HotpotQA (13.7k) | 0.543 | 0.541 (−0.002 [−0.027, +0.026]) | 0.551 (+0.008 [−0.032, +0.055]) | 0.545 (+0.003 [−0.038, +0.045]) |
| 2WikiMQA (7.5k) | 0.427 | 0.428 (+0.001 [−0.018, +0.027]) | 0.432 (+0.005 [−0.017, +0.032]) | 0.430 (+0.003 [−0.045, +0.050]) |
| MuSiQue (16.3k) | 0.310 | 0.330 (+0.020 [−0.028, +0.068]) | 0.305 (−0.005 [−0.057, +0.049]) | 0.321 (+0.012 [−0.059, +0.086]) |
| passage retrieval (13.0k) | 0.930 | 0.990 (+0.060 [+0.020, +0.110]) | 1.000 (+0.070 [+0.020, +0.120]) | 1.000 (+0.070 [+0.020, +0.120]) |
| all 400 | 0.552 | 0.572 (+0.020 [+0.002, +0.039]) | 0.572 (+0.020 [−0.002, +0.043]) | 0.574 (+0.022 [−0.005, +0.050]) |
| K+V bytes read | 100% | 22.8% | 12.8% | 7.8% |

The gain on passage retrieval comes from skipping rows, not from a numerical difference between the two attention implementations: reading every row through the sparse kernels scores 0.94 there and +0.005 [−0.002, +0.014] over all 400 (`longbench_c95748ab_full_budget.json`, run with 256 clusters and no splitting). It is narrow, though. On the examples exact attention gets wrong, it answers with a bare two-digit number ("28") where the sparse runs answer "Paragraph 8" or "Paragraph 18"; I have not established why. The supportable reading is no measurable loss, not an improvement.

## Retrieval

Needle-in-a-haystack tasks modelled on RULER, in my own implementation (`ssa.harness.needle`): a 7-digit number is hidden at a random depth in WikiText-103 text and the prompt asks for it; "multi-key" hides four numbers under different keys and asks for one. Qwen3-4B, 50 examples per cell at 16,384 and 32,768 tokens (`needle_4a5c5ba5_default.json`). Exact attention and the 20% budget are correct on all 200 examples. The 5% budget is correct on 199; the miss is in the multi-key task at 32,768 tokens (0.98, −0.02 [−0.06, 0.00]). An earlier run with 256 clusters and no splitting, 100 examples per cell, missed 1 of 400 at the 20% budget and 4 of 400 at 5%, all in that task and length.

<a id="long-generations"></a>
## Long generations

Clusters fitted to the prompt go out of date as the model generates, a drift first described by DynaKV ([arXiv:2511.07427](https://arxiv.org/abs/2511.07427)). In these runs (`ssa.harness.long_gen_tvd`) a prompt is followed by the document's own next tokens, fed to the model, and TVD from exact attention is tracked as the context grows. They do not test text the model generates itself: one attempt at that, sampling at temperature 1, produced near-deterministic text and showed nothing.

**Qwen3-4B, 8,192-token prompt continued to 32,768 tokens** (4 chunks, 20% budget; `long_gen_tvd_4a5c5ba5_4b_32k_default.json`). TVD by generated tokens; the ratio is splitting ÷ fixed clusters at equal budget:

| generated tokens | clusters fixed at the prompt | splitting (default) | ratio | splitting, no recentering | splitting, cap of 32 keys |
|---|---|---|---|---|---|
| 0–4k | 0.038 | 0.033 | 0.87 [0.83, 0.91] | 0.034 | 0.028 |
| 4k–8k | 0.051 | 0.039 | 0.78 [0.75, 0.81] | 0.040 | 0.032 |
| 8k–12k | 0.053 | 0.040 | 0.76 [0.72, 0.79] | 0.042 | 0.031 |
| 12k–16k | 0.046 | 0.030 | 0.66 [0.59, 0.72] | 0.033 | 0.024 |
| 16k–20k | 0.059 | 0.032 | 0.54 [0.50, 0.60] | 0.039 | 0.023 |
| 20k–25k | 0.059 | 0.030 | 0.50 [0.45, 0.56] | 0.040 | 0.022 |
| K+V bytes read (last row) | 21.6% | 22.3% | | 22.2% | 23.6% |
| clusters per set of keys at the end | 256 | 880 | | 820 | 1,696 |
| ms per step, mean over the run | 26.4 | 27.3 | | 26.6 | 28.7 |

- Splitting halves the error by the end, and the gap widens steadily.
- The mean key has to keep being re-estimated: with the mean held at its value at the prompt, the error is 1.35× [1.18, 1.56] the default's by the end. Over 4,096 generated tokens that difference is not visible (1.00–1.02×).
- Smaller clusters do better: a cap of 32 keys per cluster (the default cap here is 64) has 0.73× the default's TVD by the end, reading 1.3 points more of the cache and taking 1.4 ms more per step, with nearly twice the clusters. I kept the default at the configuration every other result uses.

**Controls and variations**, measured before the 8-bit cluster vectors (a separate check found those leave TVD unchanged: 1.00 [0.98, 1.02]). Ratios are at matched bytes read, in the last part of each run:

| question | setting | result |
|---|---|---|
| Does a second model show the drift? | Qwen3-0.6B, 8k prompt → 40,896, 6 chunks | splitting ÷ fixed clusters 0.60 [0.57, 0.62] |
| Is it just having more clusters? | Qwen3-4B, 4k + 4k, 8 chunks | 512 clusters fixed at the prompt drift as fast as 256 (TVD +16% from first to last quarter); splitting ÷ 512 fixed = 0.87 [0.81, 0.94] |
| Is a periodic full re-fit better? | same | re-fit every 320 tokens ÷ fixed 0.83 [0.79, 0.87]; splitting ÷ fixed 0.79 [0.74, 0.83] |
| Can the number of clusters be bounded? | Qwen3-4B → 32k and Qwen3-0.6B → 40k | a cap that grows with the context: 1.47× and 1.22× the fixed cap's TVD; a full re-fit when the count doubles: 1.59× and 1.24× |
| Does the split trigger matter? | Qwen3-4B, 4k + 4k, 8 chunks, 8-bit vectors | variance trigger as in DynaKV ÷ size cap, at a comparable cluster count: 0.98 [0.94, 1.02] |
| How large should clusters be? | Qwen3-4B, 4k + 8k, 4 chunks, 8-bit vectors | cap 32 ÷ cap 64 = 0.91 [0.89, 0.92]; cap 128 ÷ cap 64 = 1.04 [1.01, 1.07] |

The variance trigger is my reading of DynaKV, which does not give its threshold or split routine: a cluster that has just received a key is split when the variance of its keys' directions exceeds a multiple of its head's mean at the fit. The error follows the number of clusters, not the trigger.

## Speed

<a id="how-speed-is-measured"></a>*How speed is measured.* The model runs in my own Qwen3 inference engine. `GraphDecoder` (`src/ssa/models/graph_decode.py`) reruns the engine's per-token forward pass, with the engine's own layers, weights and cache, as one CUDA graph (a recorded sequence of GPU operations replayed with one launch), so only the attention kernel differs between conditions. Two exact kernels are timed: the engine's own Triton kernel, which splits each sequence across thread blocks and merges the parts (flash-decoding), and FlashInfer's paged decode, which reads the engine's contiguous cache in place as 16-token pages and is told the new length by the CPU before each token. Per layer FlashInfer is as fast as its single-sequence decode, but end to end it is 0.2–0.6 ms per token slower, most likely because of that per-token step, so speedups are reported relative to the faster of the two. Per-layer timings flush the L2 cache before each call and take the median of 100 calls; end-to-end timings take the median of 3 repeats of 64 steps. Full tables, with every context length and both exact kernels, are in the sections below.

**Attention only** (one layer, Qwen3-4B's head layout, µs per decode step, including choosing what to read; `kernel_bench_step_4a5c5ba5.json`, 256 clusters). FlashInfer is the exact attention library used by serving engines such as SGLang:

| context | exact, FlashInfer | 20% budget | 5% budget |
|---|---|---|---|
| 2,048 | 29 | 43 (0.67×) | 39 (0.73×) |
| 8,192 | 92 | 55 (1.67× faster) | 43 (2.14×) |
| 32,768 | 338 | 108 (3.12×) | 57 (5.89×) |
| 65,536 | 662 | 180 (3.67×) | 80 (8.32×) |

Choosing what to read costs about 25 µs per step at any context length, so the method is slower than exact attention below a few thousand tokens.

**Whole model** (B sequences of equal length decoded together; ms per step for the batch; "exact" is the faster of FlashInfer and the engine's own exact kernel at each setting; `decode_speed_4a5c5ba5_b*_default.json`):

| model | batch × context | exact | 20% budget | 5% budget |
|---|---|---|---|---|
| Qwen3-0.6B | 1 × 32,768 | 14.14 | 8.13 (1.74×) | 6.51 (2.17×) |
| Qwen3-0.6B | 2 × 8,192 | 10.00 | 7.83 (1.28×) | 7.11 (1.41×) |
| Qwen3-0.6B | 2 × 16,384 | 14.15 | 8.77 (1.61×) | 7.39 (1.91×) |
| Qwen3-0.6B | 2 × 32,768 | 23.05 | 10.73 (2.15×) | 7.98 (2.89×) |
| Qwen3-0.6B | 4 × 8,192 | 14.34 | 9.52 (1.51×) | 7.95 (1.80×) |
| Qwen3-0.6B | 4 × 16,384 | 23.21 | 11.67 (1.99×) | 8.59 (2.70×) |
| Qwen3-0.6B | 8 × 8,192 | 23.42 | 12.45 (1.88×) | 9.60 (2.44×) |
| Qwen3-4B | 1 × 16,384 | 29.08 | 25.61 (1.14×) | 24.74 (1.18×) |
| Qwen3-4B | 1 × 32,768 | 34.81 | 26.98 (1.29×) | 25.11 (1.39×) |
| Qwen3-4B | 2 × 8,192 | 28.60 | 26.29 (1.09×) | 25.44 (1.12×) |
| Qwen3-4B | 2 × 16,384 | 34.49 | 27.49 (1.25×) | 25.78 (1.34×) |
| Qwen3-4B | 4 × 8,192 | 34.86 | 28.50 (1.22×) | 26.51 (1.31×) |

The last row uses 512 cluster slots: the cluster state for 1,024 slots (about 0.4 GB per sequence on Qwen3-4B) does not fit beside four 8k caches and the weights in 16 GB.

**Where the time goes** (Qwen3-4B, 16k context, 256 clusters, 20% budget, GPU time per step summed over 36 layers; `kernel_profile_clusters_4b_16k_8bit.json`): everything outside attention 22.6 ms; attention over the rows read 1.33 ms; scoring clusters 0.28; inserting the key 0.25; selecting clusters 0.15; listing the selected rows 0.11. With 1,024 clusters in use the three cluster kernels together rise from 0.68 to 2.30 ms. Inserting the key for all 36 layers in one kernel launch, which is possible because the key leaving the 64-token window was written 64 steps earlier at every layer, saves 0.30 ms per step on Qwen3-4B at 32k.

## Sampling the skipped clusters

Reading the top clusters exactly and sampling some of the rest, weighted by inverse inclusion probability, removes the bias of dropping them. At matched bytes read it does not help (ratio of TVD, sampling ÷ reading more clusters; 8 chunks; 256 clusters, no splitting; `tvd_*_report_long.json`):

| | 20% of bytes read | 25% | 30% |
|---|---|---|---|
| Qwen3-4B, 8k | 1.14 [1.12, 1.16] | 1.13 [1.10, 1.17] | 1.15 [1.09, 1.20] |
| Qwen3-4B, 32k | 1.15 [1.12, 1.18] | 1.05 [1.02, 1.08] | 1.11 [1.07, 1.14] |
| Qwen3-0.6B, 8k | 1.01 [0.96, 1.06] | 0.93 [0.87, 1.00] | 0.95 [0.89, 1.00] |
| Qwen3-0.6B, 32k | 1.08 [1.01, 1.15] | 0.98 [0.90, 1.06] | 0.99 [0.89, 1.09] |

## Earlier measurements

The three sections below were measured with earlier versions of the method (clusters from fixed random directions, or fitted clusters before the 8-bit vectors and the full byte accounting). In them, `cluster_skip` names the version with fixed random directions. They are kept because nothing later replaces them; their read fractions use the earlier accounting, which counts about 0.6 points less at 8,192 tokens.

## Comparison with weight quantization

As reference points for what a given TVD means, Qwen3-4B with weight-only round-to-nearest quantization (embeddings and output layer kept in bf16), 8192 tokens:

| model | TVD from bf16 | top-1 agreement |
|---|---|---|
| 8-bit weights | 0.018 [0.015, 0.019] | 97.9% |
| 4-bit weights, groups of 128 | 0.144 [0.125, 0.159] | 84.0% |
| `cluster_skip`, 20% budget | 0.044 | — |

`cluster_skip` at 20% falls between the two, and on code at 20% (0.017) it matches 8-bit. Plain round-to-nearest 4-bit is much worse than calibrated 4-bit methods, so it is a loose bound.


## Comparison with Quest

Quest (arXiv:2406.10774) splits the cache into 16-token pages, stores each page's element-wise minimum and maximum keys, and reads the pages with the highest bound `Σ_i max(q_i·min_i, q_i·max_i)` on `q·k` (`ssa.attn.quest`). `quest_matched` adds `cluster_skip`'s always-read tokens (token 0 and the last 64) and one shared selection per KV head, so the remaining differences are pages by position compared with regions by direction, and Quest's bound compared with our score. Reads include each method's summaries: Quest's two keys per page cost ~6% of K+V rows.

End to end (`accept_sweep --preset quest_vs_cluster`, both methods in PyTorch in one sweep; `src/ssa/results/tvd_4b_8k_quest_vs_voronoi.json`; `ssa.harness.matched_reads` interpolates each chunk's TVD along each curve and bootstraps over chunks):

| K+V reads | TVD, `quest_matched` | TVD, `cluster_skip` | ratio [95% CI] |
|---|---|---|---|
| 20% | 0.0555 | 0.0487 | 1.14 [1.10, 1.18] |
| 23% | 0.0489 | 0.0434 | 1.13 [1.08, 1.17] |
| 27% | 0.0413 | 0.0393 | 1.05 [1.01, 1.10] |
| 30% | 0.0377 | 0.0365 | 1.03 [0.99, 1.08] |
| 37% | 0.0305 | 0.0307 | 1.00 [0.95, 1.04] |
| 42% | 0.0270 | 0.0271 | 1.00 [0.95, 1.04] |

`quest_matched` cannot read less than 17%; `cluster_skip` reaches 8% (TVD 0.084).

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
