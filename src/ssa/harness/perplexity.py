"""Teacher-forced decode perplexity (the real-model accuracy harness).

The engine's ``evaluate_ppl`` runs a single prefill forward and never touches the
decode path, so it cannot exercise the sampled attention op. This harness instead
prefills a prefix densely, then feeds the *true* tokens one at a time through the
decode path (which is sparse when an ssa op is installed), scoring only those
decode-step predictions. Perplexity (PPL) therefore reflects the sampled attention.

Whatever attention is installed on the model is what gets measured — install via
``ssa.models.patch.install`` and read the avoided-bytes metric off the returned
``ReadStats``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.inference_mode()
def decode_ppl(model, chunks: list[torch.Tensor], *, prefill_len: int) -> dict:
    """Perplexity over teacher-forced decode steps.

    chunks: list of ``(1, T)`` long tensors. For each, positions
    ``[prefill_len+1, T)`` are predicted one token at a time through the decode
    path and contribute to the NLL. Returns ``{ppl, avg_nll, token_count, chunk_count}``.

    The NLL computation mirrors ``engine.bench.eval_ppl.chunk_nll`` (float32
    log-softmax of next-token logits), specialised to one prediction per step.
    """
    device = next(model.parameters()).device
    total_nll, total_count, used = 0.0, 0, 0
    chunk_nll, chunk_tokens = [], []

    for ids in chunks:
        ids = ids.to(device)
        T = ids.shape[1]
        if T <= prefill_len + 1:
            continue  # nothing to score under the decode path
        used += 1
        cache = model.alloc_cache(T)
        model(ids[:, :prefill_len], cache, start_pos=0)  # dense prefill

        preds, targets = [], []
        for t in range(prefill_len, T - 1):
            logits = model(ids[:, t:t + 1], cache)        # decode step (sparse if installed)
            preds.append(logits[:, -1, :])
            targets.append(ids[:, t + 1])
        preds = torch.stack(preds, dim=1)                 # [1, n, vocab]
        targets = torch.stack(targets, dim=1)             # [1, n]

        log_probs = F.log_softmax(preds.float(), dim=-1)
        nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        total_nll += float(nll.sum().item())
        total_count += int(nll.numel())
        chunk_nll.append(float(nll.sum().item()))
        chunk_tokens.append(int(nll.numel()))
        del cache

    if total_count == 0:
        raise ValueError("no chunk long enough to score under the decode path")
    avg_nll = total_nll / total_count
    return {
        "ppl": math.exp(avg_nll),
        "avg_nll": avg_nll,
        "token_count": total_count,
        "chunk_count": used,
        "chunk_nll": chunk_nll,          # per-chunk NLL sums, for paired intervals
        "chunk_tokens": chunk_tokens,
    }
