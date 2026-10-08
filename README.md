# Efficient inference by reducing KV cache reads

When a large language model generates text at long context, its speed is limited by memory traffic: every new token re-reads the cached key and value vectors of every earlier token (the KV cache). After reproducing SANTA ([arXiv:2605.01910](https://arxiv.org/abs/2605.01910)), which samples which cached values to read but still reads part of every key, I asked whether most keys could be skipped too. My idea was to group the keys by direction, choose which groups to read from small summaries of them, and split groups as generation adds to them. I built it as Triton GPU kernels inside a Qwen3 inference engine I wrote separately ([github.com/aaholmes/llms](https://github.com/aaholmes/llms)), and it worked.

I then found that the main ideas had already been published: clustering keys and reading the best clusters by ClusterKV ([arXiv:2412.03213](https://arxiv.org/abs/2412.03213)), and splitting clusters during generation by DynaKV ([arXiv:2511.07427](https://arxiv.org/abs/2511.07427)). So this repository is an independent implementation of that approach, with measurements of how it behaves on a GPU.

On one consumer GPU (RTX 5060 Ti, 16 GB):

- **Attention is 3× faster than FlashInfer**, the exact attention library used by SGLang, at 32,768 tokens of context when reading 20% of the keys, and 6× faster when reading 5%.
- **Whole-model decoding is up to 2.1× faster** (2.9× at the 5% setting) with two 32k-token sequences batched on Qwen3-0.6B. On Qwen3-4B the gain is 1.2–1.4×, because reading its 8 GB of weights dominates each step on this card.
- **Task accuracy is not measurably lower** than exact attention's on four LongBench sets while reading 8–23% of the cache.
- **It holds up over long generations**, out to 25 thousand generated tokens, because the clusters are updated as tokens are generated.

## How it works

Attention weights each cached value by the softmax of the query's dot product with its key, so the few keys pointing along the query carry most of the weight. The method finds them without reading every key:

1. **Cluster the prompt's keys by direction** (k-means on cosine similarity, after subtracting the mean key), 256 clusters per set of keys. Each cluster keeps a small summary: its mean direction, its longest and shortest key, and a count.
2. **Score clusters from the summaries alone**, once for all the query heads that share a set of keys, and read the best clusters' keys and values up to a budget. The first token and the 64 most recent tokens are always read.
3. **Compute exact attention over what was read.** The rest is dropped.
4. **Keep the clusters current.** Each new key joins its nearest cluster, and a cluster that grows past a size cap is split in two, inside the GPU kernel that inserted the key. Splitting matters for long generations.

## Results

"Budget" is the fraction of keys read. "Bytes read" counts everything read per decode step as a fraction of the cache, including the cluster summaries. Details, intervals and methods for everything below are in [docs/results.md](docs/results.md).

**Speed** (whole model; speedup over the faster of FlashInfer and the engine's own exact kernel):

| model | batch × context | exact, tokens/s | 20% budget | 5% budget |
|---|---|---|---|---|
| Qwen3-0.6B | 1 × 32,768 | 71 | 1.74× faster | 2.17× |
| Qwen3-0.6B | 2 × 32,768 | 87 | 2.15× | 2.89× |
| Qwen3-0.6B | 8 × 8,192 | 342 | 1.88× | 2.44× |
| Qwen3-4B | 1 × 32,768 | 29 | 1.29× | 1.39× |
| Qwen3-4B | 4 × 8,192 | 115 | 1.22× | 1.31× |

The gain follows the number of tokens cached across the batch relative to the size of the weights, so it should be larger on a GPU with room for bigger batches; that is not yet measured.

**Accuracy** (Qwen3-4B; LongBench question answering and retrieval, 400 examples with prompts of 7–16 thousand tokens):

| | exact | 20% budget | 10% budget | 5% budget |
|---|---|---|---|---|
| LongBench score | 0.552 | 0.572 | 0.572 | 0.574 |
| bytes of the cache read | 100% | 23% | 13% | 8% |

The small gains come mostly from six examples in one set, and I read the result as no measurable loss. In a needle-in-a-haystack test at 16k and 32k tokens, the 20% budget finds all 200 hidden numbers and the 5% budget 199. The model's next-token distribution does move: its total variation distance from the exact model's is 0.03 at the 20% budget and 0.06 at 5% (8k context), which is between the effects of 8-bit and 4-bit weight quantization.

**Long generations.** Clusters fitted to the prompt go out of date as the model writes. With an 8k-token prompt continued to 32k tokens on Qwen3-4B, splitting clusters as they grow leaves half the error of clusters fixed at the prompt (0.50×, interval 0.45–0.56), and the gap widens with length. The number of clusters has to grow with the context for this to work; limiting it gives up much of the gain. Smaller clusters do better still, at some cost in time and memory.

## Relation to other work

Reading the cache selectively by clustering keys is an active area; [docs/related_work.md](docs/related_work.md) describes nine methods and what this one shares with each. Two points of contact are direct. My first version grouped keys by fixed random directions; after finding ClusterKV I compared the two and adopted its k-means clustering, which was clearly better. And I compared my rule for splitting a cluster (its size) with DynaKV's (its variance) and found no difference between them.

What this repository adds is measurement: speed relative to FlashInfer, alone and batched; task accuracy; error tracked over tens of thousands of generated tokens; and the finding that the number of clusters has to grow with the context. The earlier sampling work is in [docs/sampling.md](docs/sampling.md).

## Limits

- One model family (Qwen3), one 16 GB GPU, contexts up to 40k tokens.
- Other methods report near-exact task accuracy while reading 1–6% of the cache; this one has been tested at 5–40%.
- In the long-generation runs the text is fed to the model, not generated by it.
- The cluster summaries take memory: about 0.4 GB per sequence on Qwen3-4B at the default 1,024 slots.
- Not integrated into a serving engine; sequences in a batch must have equal length.

## Repository

- `src/ssa/kernels/` — Triton GPU kernels: inserting keys and splitting clusters, scoring and selecting clusters, attention over the selected rows.
- `src/ssa/attn/` — reference implementations in PyTorch, and baselines.
- `src/ssa/models/` — the hook into the inference engine, and the decode step captured as one CUDA graph.
- `src/ssa/harness/` — experiments; `src/ssa/results/` — their output, each file recording the commit, GPU and library versions that produced it.

Setup: `uv sync`, then `uv run python -m pytest` (356 tests; the kernel tests need a CUDA GPU). The engine is installed from [github.com/aaholmes/llms](https://github.com/aaholmes/llms) at a pinned commit.

## License

Apache License 2.0; see [LICENSE](LICENSE).
