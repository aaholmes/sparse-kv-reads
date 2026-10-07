# Related work: reading part of the KV cache per decode step

A decoder caches a key and a value vector for every earlier token at every layer (the KV cache), and exact
attention reads all of them at each step. The methods below keep the whole cache but read only the part a
given query needs. Most of them group keys ahead of time, score each group cheaply from a summary such as
its centroid (the mean of its keys), and read the best-scoring groups exactly.

This note records what each method does, with attention to one question the papers answer differently:
what happens to the groups as the model generates new tokens. It then lists which parts of this repository
are shared with that work and which are not.

**How this was compiled (2026-10-06).** I read DynaKV in full. The other eight papers were read in full by
automated readers working from the arXiv text; I compared the statements marked ✓ with the paper text
myself. Unmarked statements come from the readers' reports and should be rechecked before being quoted
elsewhere. I have not opened any of the code repositories. The search was not exhaustive: PQCache, FreeKV,
ArkVale and Squeezed Attention appear as baselines in these papers and are not covered here.

## The methods

### ClusterKV ([arXiv:2412.03213](https://arxiv.org/abs/2412.03213), Dec 2024)

- **Groups.** k-means on the keys using cosine distance, separately for each attention head, with one
  cluster per 80 tokens.
- **Each step.** Clusters are ranked by the inner product of the query and the centroid and read in order
  until a fixed number of tokens (256–2,048) is reached. The first 16 tokens are always read. Clusters not
  selected contribute nothing.
- **New tokens.** Every 320 generated tokens, those 320 keys alone are clustered into 4 new clusters,
  which are appended ✓ ("we instead apply clustering within the generated tokens only ... we set C+ and m
  to 4 and 320"). Clusters fitted to the prompt are never changed, and a generated token never joins one.
- **Setting.** Index on the GPU, cache in CPU memory. Accuracy on LongBench with GLM4-9B; the longest
  generation tested is 1,024 tokens, for latency only.

### RetroInfer ([arXiv:2505.02922](https://arxiv.org/abs/2505.02922), May 2025)

- **Groups.** Spherical k-means within segments of 8,000 tokens, on keys with their mean subtracted ✓ ("a
  classic centering technique ... inspired by MagicPIG"), one centroid per 16 tokens.
- **Each step.** The first 4 and last 64 tokens are always read ✓. The top 1.8% of clusters are read
  exactly. The next ~23% are not read; their contribution is estimated from the centroid's score times the
  cluster's stored sum of values. The rest contribute nothing.
- **New tokens.** Generated tokens are read exactly until 1,024 have accumulated; those are then clustered
  alone and appended ✓. Nothing is re-fitted.
- **Setting.** Cache in CPU memory, index on the GPU. RULER to 128k tokens, a needle test to 1M, reasoning
  tasks with up to 32k generated tokens. Llama and Qwen2.5 models up to 72B.

### Multipole Attention ([arXiv:2506.13059](https://arxiv.org/abs/2506.13059), Jun 2025)

- **Groups.** k-means within blocks of 8,000 tokens, one centroid per 16 tokens, with the position rotation
  of the keys (RoPE) handled by assuming a fixed distance between query and key.
- **Each step.** 10 first tokens and 128–256 recent tokens are always read. Clusters are ranked by their
  estimated share of attention, averaged over the query heads that share a set of keys ✓, and read until a
  budget of 128–512 tokens. Unselected clusters are replaced by their key and value centroids.
- **New tokens.** Every 128 steps the oldest 128 buffered tokens enter the final block: new centroids are
  sampled from them, each token joins its nearest centroid, and three k-means iterations run over that
  block only ✓ ("we only need to recluster the final block"). Earlier blocks stay fixed.
- **Setting.** All on the GPU, Triton kernels. Qwen3-8B and a 14B reasoning model, LongBench v2 and a
  synthetic arithmetic benchmark. Speed is reported for the attention kernel alone.

### LouisKV ([arXiv:2510.11292](https://arxiv.org/abs/2510.11292), Oct 2025)

- **Groups.** k-means on the prompt's keys, about 16 tokens per cluster. Generated tokens are not clustered
  by content: they form contiguous segments, cut wherever the query direction changes sharply from one step
  to the next.
- **Each step.** A new selection is made only at such a change; otherwise the previous one is reused. One
  selection is made per set of shared keys, from softmax scores averaged over its query heads
  ("group-consistent selection") ✓. First tokens and a recent buffer are always read.
- **New tokens.** Segments are appended. Nothing is re-clustered.
- **Setting.** Cache in CPU memory. Llama-3.1-8B and Qwen3-8B; reasoning tasks with up to 16k–32k generated
  tokens.

### DynaKV ([arXiv:2511.07427](https://arxiv.org/abs/2511.07427), Oct 2025)

- **Observation.** As more tokens are decoded, the clustering that would be best for the whole cache moves
  away from the one fitted at the prompt. The paper calls this KVCache distribution shift, and classifies
  earlier practice as static update (each new key appended to its nearest existing cluster) or local update
  (new keys clustered separately, as ClusterKV does).
- **New tokens.** Each new key joins its nearest cluster, and the cluster's internal variance is updated.
  When the variance exceeds a threshold set per head, the cluster is split in two.
- **Setting.** Smartphones, with the cache stored on flash and clusters fetched into memory when selected,
  so a split waits until its cluster is in memory. Built on llama.cpp on the CPU with 8-bit models of 1–8B
  parameters. Task accuracy and latency compared with ClusterKV and PQCache, at decode lengths of 1k–16k.
  It does not report error as a function of the number of generated tokens.

### LycheeCluster ([arXiv:2603.08453](https://arxiv.org/abs/2603.08453), Mar 2026)

- **Groups.** Text is cut into chunks of 8–16 tokens at sentence and paragraph boundaries. Chunk means are
  clustered by k-means, and clusters are grouped again into at most 64 coarse units; each node stores a
  centroid and a radius.
- **Each step.** Nodes are ranked by an upper bound (centroid score plus query length times radius), coarse
  units first and then clusters within them, up to a budget of 1,024 tokens. 16 first tokens are always
  read.
- **New tokens.** Each new chunk joins the nearest existing cluster; the centroid is updated by a moving
  average and the radius only grows. There are no splits. An appendix on long generation reports that
  retrieval "shows signs of decay after 6k decode steps" ✓ and suggests re-clustering as future work.
- **Setting.** All on the GPU, CUDA C++ built on ClusterKV's code. Llama-3.1-8B and two reasoning models.

### Louver ([arXiv:2605.06763](https://arxiv.org/abs/2605.06763), May 2026)

- **Goal.** Return every key whose score with the query exceeds a threshold, with none missed, instead of a
  fixed number of keys.
- **Groups.** The key's coordinates are divided into a few blocks; within each block, keys are put in
  groups of about 4 with a bounding ball. A group is skipped when its bound falls below the threshold, and
  the surviving keys are checked exactly.
- **New tokens.** New groups are built from each buffer of recent keys and appended.

### CommunityKV ([arXiv:2610.00418](https://arxiv.org/abs/2610.00418), Sep 2026)

- **Groups.** While processing the prompt, the top 8 attention scores of every query are kept and turned
  into a graph over tokens. The graph is partitioned by community detection (the Leiden algorithm) into
  communities of mean size 16, one partition per query head by default.
- **Each step.** Communities are ranked by a query–centroid score and read up to a fixed token budget
  (4,096 for the main results). The first 10 tokens are always read; there is no recent window.
- **New tokens.** A new token either joins the community of one of the tokens it attended to or becomes a
  community of one. Earlier assignments are fixed. Rebuilding the partition periodically is offered as an
  option.
- **Drift.** In the paper's own test with the text fed to the model, the divergence from the exact model on
  PG-19 rises from 0.15 at 2k generated tokens to 0.44 at 16k with the default rule, and to 0.15 when the
  partition is rebuilt every 512 tokens ✓.
- **Setting.** Qwen3 4B–14B and Llama-3.1-8B, contexts of 64k–128k, batch size 1, H100/H200 GPUs.

### CentroidKV ([arXiv:2506.11418](https://arxiv.org/abs/2506.11418), Jun 2025)

Not a selective reader. It merges similar keys and values permanently into weighted centroids to shrink the
cache, and every step attends to all of them.

## What happens to groups during generation, side by side

| rule | used by |
|---|---|
| New tokens are clustered among themselves and appended | ClusterKV (every 320 → 4 clusters), RetroInfer (every 1,024), Louver, LouisKV (contiguous segments) |
| New tokens join existing groups, which never change | LycheeCluster, CommunityKV (default) |
| The most recent block is re-fitted; older blocks are fixed | Multipole Attention |
| New keys join existing clusters; a cluster is split when it degrades | DynaKV (variance threshold), this repository (size cap) |

## This repository compared with that work

**Shared with earlier work.** These are not contributions of this repository:

- k-means on keys with cosine distance (ClusterKV, RetroInfer).
- Subtracting the mean key before clustering (RetroInfer, following MagicPIG).
- Always reading the first token and a recent window (RetroInfer, Multipole Attention and others).
- One selection per set of shared keys, from softmax shares combined over its query heads (Multipole
  Attention, LouisKV).
- Splitting a cluster as new keys arrive (DynaKV).
- Estimating unselected clusters from a centroid and a stored sum of values (RetroInfer, Multipole
  Attention). I tested a version of this and it lowered the distance from the exact model by 0–11%.
- Triton kernels for selection and gathering, and counting cluster summaries in what is read (Multipole
  Attention).

**Not found in the nine papers above:**

- One set of clusters over the whole context that grows by splitting, with each new key joining it as the
  key leaves the recent window. The others append clusters built from recent tokens only, keep assignments
  fixed, or re-fit only the last block; DynaKV is the exception.
- Cluster size as the trigger for a split, and the split carried out inside the GPU kernel that inserts the
  key, with no work on the host.
- The distance from the exact model's next-token distribution measured as a function of generated tokens,
  out to 25–33 thousand, with controls: clusters fixed at the prompt, twice as many fixed clusters, a full
  re-fit every 320 tokens, and two ways of bounding the number of clusters.
- The result that bounding the number of clusters as the context grows costs 22–59% in that distance
  relative to letting it grow.
- A measured negative result for sampling the unselected clusters.

**Where the other work is ahead.**

- They report task accuracy close to the exact model while reading 128–4,096 tokens, 1–6% of a 32k–128k
  context. This repository reports distance from the exact model at budgets of 5–40% and has no
  task-accuracy result yet.
- Most keep the cache in CPU memory or on flash, test several model families, and reach contexts of 128k
  to 1M tokens. This repository keeps the cache on one 16 GB GPU and tests Qwen3 models up to 40k tokens.
