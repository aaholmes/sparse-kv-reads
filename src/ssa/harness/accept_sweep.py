"""Acceptance / total variation distance (TVD) of stochastic-attention decode vs the dense model.

The ppl_sweep measures *perplexity* under sampled attention. This measures the
quantity a speculative-decode verifier (or a bias-correcting low-rank adapter, LoRA) actually
cares about: how close the sampled-attention next-token distribution is to the
*dense* model's, token for token, as a function of read budget.

For each decode step (teacher-forced, dense prefill then true tokens one at a
time through the sparse decode seam) we compare the sampled-attention logits to
the dense logits of the unmodified model at that decode step:

    acceptance = Σ_i min(p_dense_i, p_stoch_i) = 1 − TVD(p_dense, p_stoch)

This is the single-token spec-decode accept probability (Leviathan et al.) with
the dense model as target and the stochastic model as draft — i.e. exactly the
bias a debiasing LoRA would try to remove. Sampling is unbiased in *attention
output* but biased in *logits* (softmax + MLP are nonlinear), so acceptance < 1
even in expectation; this sweep quantifies that gap vs read-fraction.

Imports the model engine from the `specd` package (github.com/aaholmes/llms).

Run (GPU, sequential decode — slow):
    uv run python -m ssa.harness.accept_sweep --model Qwen/Qwen3-4B \
        --corpus code --max-chunks 8 --chunk-len 288 --prefill-len 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ..models.patch import install, uninstall
from .stamp import stamp

# santa_sys read-budget sweep (S = value rows sampled per head per step).
DEFAULT_S = [4, 8, 16, 32, 64, 128, 256]


@torch.inference_mode()
def _decode_logits(model, ids: torch.Tensor, *, prefill_len: int, prefill_chunk: int | None = None) -> torch.Tensor:
    """Dense prefill, then feed true tokens one at a time through the decode seam
    (sparse iff an ssa op is installed). Returns [n_steps, vocab] next-token
    logits for the scored steps ``[prefill_len, T-1)``. ``prefill_chunk`` splits the
    prefill (the engine returns logits for every prompt position, ~10 GB at 32k)."""
    T = ids.shape[1]
    cache = model.alloc_cache(T + 4)
    step = prefill_chunk or prefill_len
    for s0 in range(0, prefill_len, step):                  # dense prefill (T>1 branch)
        model(ids[:, s0:min(s0 + step, prefill_len)], cache, start_pos=s0)
    out = []
    for t in range(prefill_len, T - 1):
        logits = model(ids[:, t:t + 1], cache)       # decode step (sparse if installed)
        out.append(logits[0, -1, :])
    return torch.stack(out)                            # [steps, vocab]


def _compare(dense: torch.Tensor, stoch: torch.Tensor, true_next: torch.Tensor) -> dict:
    """Per-step acceptance / TVD / top-1 agreement (stoch vs dense) + student NLL."""
    p_d = dense.float().softmax(-1)
    p_s = stoch.float().softmax(-1)
    accept = torch.minimum(p_d, p_s).sum(-1)               # [steps]
    top1 = (dense.argmax(-1) == stoch.argmax(-1)).float()  # [steps]
    nll = F.cross_entropy(stoch.float(), true_next, reduction="none")  # student PPL
    return {
        "accept_sum": float(accept.sum()), "top1_sum": float(top1.sum()),
        "nll_sum": float(nll.sum()), "n": int(accept.numel()),
    }


def _load_chunks(model_id: str, corpus: str, *, max_chunks: int, chunk_len: int, device: str):
    if corpus == "code":
        from mla.calibrate import load_code_chunks
        chunks = load_code_chunks(n_samples=max_chunks, chunk_tokens=chunk_len,
                                  tokenizer_id=model_id, skip=20000)
    else:
        from mla.calibrate import load_wikitext103_chunks
        chunks = load_wikitext103_chunks(n_samples=max_chunks, chunk_tokens=chunk_len,
                                         tokenizer_id=model_id, split="validation")
    return [c.to(device) for c in chunks]


def run(model, chunks, *, s_values, prefill_len: int, base_seed: int = 0) -> list[dict]:
    """Dense reference once, then santa_sys at each S; acceptance/TVD/top1/PPL vs dense."""
    # 1) Dense teacher logits per chunk (store on CPU to free GPU for the S loop).
    uninstall(model)
    dense_logits = [_decode_logits(model, ids, prefill_len=prefill_len).cpu() for ids in chunks]
    true_next = [ids[0, prefill_len + 1:].cpu() for ids in chunks]

    results = []
    for S in s_values:
        stats = install(model, "santa_sys", base_seed=base_seed, S=S)
        agg = {"accept_sum": 0.0, "top1_sum": 0.0, "nll_sum": 0.0, "n": 0}
        for ids, d_cpu, tn in zip(chunks, dense_logits, true_next):
            s_logits = _decode_logits(model, ids, prefill_len=prefill_len)
            c = _compare(d_cpu.to(s_logits.device), s_logits, tn.to(s_logits.device))
            for k in agg:
                agg[k] += c[k]
        uninstall(model)
        n = max(agg["n"], 1)
        acc = agg["accept_sum"] / n
        results.append({
            "S": S, "read_fraction": stats.read_fraction,
            "acceptance": acc, "tvd": 1.0 - acc,
            "top1_agree": agg["top1_sum"] / n,
            "ppl": float(torch.tensor(agg["nll_sum"] / n).exp()),
            "n_steps": agg["n"],
        })
        r = results[-1]
        print(f"[accept] S={S:<4d} reads={100*r['read_fraction']:5.1f}%  "
              f"accept={100*acc:5.1f}%  tvd={r['tvd']:.4f}  top1={100*r['top1_agree']:5.1f}%  "
              f"ppl={r['ppl']:.3f}", flush=True)
    return results


def run_conditions(model, chunks, *, conditions, prefill_len: int, base_seed: int = 0,
                   on_condition=None, prefill_chunk: int | None = None) -> list[dict]:
    """Dense logits once, then each ``(impl, cfg)`` condition (one run; deterministic impls
    need no more, and a single draw is what decoding sees for stochastic ones).

    Per chunk records sums of TVD, KL(dense‖cond), top-1 agreement and NLL, so the
    summary can bootstrap over chunks with every condition paired to dense.
    """
    uninstall(model)
    dense_logits = [_decode_logits(model, ids, prefill_len=prefill_len, prefill_chunk=prefill_chunk).cpu()
                    for ids in chunks]
    true_next = [ids[0, prefill_len + 1:].cpu() for ids in chunks]
    results = []
    for impl, cfg in conditions:
        if impl == "quant":                          # weight-only quantization, applied in place (irreversible:
            from quant.swap import apply_weight_quant  # put it last); decoding then runs the quantized model
            apply_weight_quant(model, n_bits=int(cfg["n_bits"]), group_size=cfg.get("group_size"))
            impl_run = "dense"
        else:
            impl_run = impl
        stats = install(model, impl_run, base_seed=base_seed, **cfg) if impl_run != "dense" else None
        per = {"chunk_tvd": [], "chunk_kl": [], "chunk_top1": [], "chunk_nll": [], "chunk_n": []}
        for ids, d_cpu, tn in zip(chunks, dense_logits, true_next):
            lg = d_cpu if impl == "dense" else _decode_logits(model, ids, prefill_len=prefill_len,
                                                              prefill_chunk=prefill_chunk).cpu()
            ld = d_cpu.float().log_softmax(-1)
            ls = lg.float().log_softmax(-1)
            tvd = 0.5 * (ld.exp() - ls.exp()).abs().sum(-1)
            kl = (ld.exp() * (ld - ls)).sum(-1)
            per["chunk_tvd"].append(float(tvd.sum()))
            per["chunk_kl"].append(float(kl.sum()))
            per["chunk_top1"].append(float((d_cpu.argmax(-1) == lg.argmax(-1)).float().sum()))
            per["chunk_nll"].append(float(F.cross_entropy(lg.float(), tn, reduction="sum")))
            per["chunk_n"].append(int(tn.numel()))
        uninstall(model)
        n = sum(per["chunk_n"])
        r = {"impl": impl, "cfg": cfg,
             "read_fraction": 1.0 if stats is None else stats.read_fraction,
             "kv_read_fraction": 1.0 if stats is None else stats.kv_read_fraction,
             "rebuild_rate": None if stats is None else stats.rebuild_rate,
             "tvd": sum(per["chunk_tvd"]) / n, "kl": sum(per["chunk_kl"]) / n,
             "top1_agree": sum(per["chunk_top1"]) / n,
             "ppl": float(torch.tensor(sum(per["chunk_nll"]) / n).exp()), "n_steps": n, **per}
        results.append(r)
        if on_condition is not None:
            on_condition(r, results)
    return results


def paired_summary(results: list[dict], *, n_boot: int = 2000, seed: int = 0) -> list[dict]:
    """Per condition: TVD, KL, top-1 and Δppl% vs dense, with bootstrap CIs over chunks."""
    dense = next(r for r in results if r["impl"] == "dense")
    n = torch.tensor(dense["chunk_n"], dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    boots = torch.randint(0, len(n), (n_boot, len(n)), generator=g)
    nll_d = torch.tensor(dense["chunk_nll"], dtype=torch.float64)

    def ci(x):
        return [float(x.quantile(0.025)), float(x.quantile(0.975))]

    out = []
    for r in results:
        row = {"impl": r["impl"], "cfg": r["cfg"], "read_fraction": r["read_fraction"],
               "kv_read_fraction": r["kv_read_fraction"], "rebuild_rate": r.get("rebuild_rate")}
        for key in ("tvd", "kl", "top1"):
            v = torch.tensor(r[f"chunk_{key}"], dtype=torch.float64)
            name = "top1_agree" if key == "top1" else key
            row[name] = float(v.sum() / n.sum())
            row[f"{name}_ci"] = ci(v[boots].sum(1) / n[boots].sum(1))
        nll = torch.tensor(r["chunk_nll"], dtype=torch.float64)
        d = lambda idx: 100 * ((nll[idx].sum(-1) - nll_d[idx].sum(-1)) / n[idx].sum(-1)).exp() - 100
        row["dppl_pct"] = float(d(torch.arange(len(n))))
        row["dppl_pct_ci"] = ci(d(boots))
        out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--corpus", default="code", choices=["code", "wikitext", "wikitext_test"])
    p.add_argument("--max-chunks", type=int, default=8)
    p.add_argument("--chunk-len", type=int, default=288)
    p.add_argument("--prefill-len", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lora", default=None, help="trained debias-LoRA checkpoint to load before sweeping")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--out", default="src/ssa/results/accept_sweep.json")
    p.add_argument("--prefill-chunk", type=int, default=None, help="split the prefill into chunks of this size")
    p.add_argument("--preset", default=None,
                   help="ppl_sweep condition preset (e.g. sphere); default is the santa_sys S sweep")
    args = p.parse_args()

    from engine.model import Qwen3Model
    from engine.weights import load_weights

    print(f"[accept] loading {args.model}", flush=True)
    loaded = load_weights(args.model, dtype=torch.bfloat16, device="cpu")
    model = Qwen3Model.from_loaded(loaded).to(dtype=torch.bfloat16, device=args.device).eval()
    del loaded
    torch.cuda.empty_cache()

    if args.lora:
        from mla.heal import load_trainable, merge_lora, wrap_lora
        wrap_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        load_trainable(model, args.lora)
        merge_lora(model)  # fold in → plain model, unchanged decode path
        print(f"[accept] loaded + merged debias-LoRA {args.lora}", flush=True)

    if args.corpus == "wikitext_test":   # the chunks ppl_sweep uses
        from .ppl_sweep import _wikitext_chunks
        chunks = _wikitext_chunks(args.model, max_chunks=args.max_chunks, chunk_len=args.chunk_len,
                                  device=args.device)
    else:
        chunks = _load_chunks(args.model, args.corpus, max_chunks=args.max_chunks,
                              chunk_len=args.chunk_len, device=args.device)
    print(f"[accept] {len(chunks)} {args.corpus} chunks × {args.chunk_len} tok "
          f"(prefill {args.prefill_len})", flush=True)

    out = Path(args.out)
    if args.preset:
        from .ppl_sweep import CONDITION_PRESETS

        def save(r, rs):
            summ = paired_summary(rs) if any(x["impl"] == "dense" for x in rs) else []
            row = next((x for x in summ if x["cfg"] == r["cfg"] and x["impl"] == r["impl"]), None)
            if row:
                kvf = row["kv_read_fraction"]
                print(f"[accept] {r['impl']:12s} {str(r['cfg']):60s} rows/head {100*r['read_fraction']:5.1f}% "
                      f"K+V {'' if kvf is None else f'{100*kvf:5.1f}%'}  "
                      f"TVD {row['tvd']:.4f} [{row['tvd_ci'][0]:.4f},{row['tvd_ci'][1]:.4f}]  "
                      f"top1 {100*row['top1_agree']:5.1f}%  "
                      + ("" if row.get("rebuild_rate") is None else f"rebuild/step {row['rebuild_rate']:.4f}  ")
                      + f"dppl {row['dppl_pct']:+.2f}% [{row['dppl_pct_ci'][0]:+.2f},{row['dppl_pct_ci'][1]:+.2f}]",
                      flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(stamp({"model": args.model, "corpus": args.corpus,
                                             "chunk_len": args.chunk_len, "prefill_len": args.prefill_len,
                                             "preset": args.preset, "complete": len(rs) == len(conds),
                                             "summary": summ, "results": rs}), indent=1))

        conds = CONDITION_PRESETS[args.preset]
        run_conditions(model, chunks, conditions=conds, prefill_len=args.prefill_len, on_condition=save,
                       prefill_chunk=args.prefill_chunk)
        print(f"[accept] wrote {out}", flush=True)
        return

    results = run(model, chunks, s_values=DEFAULT_S, prefill_len=args.prefill_len)

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = stamp({"model": args.model, "corpus": args.corpus,
                     "chunk_len": args.chunk_len, "prefill_len": args.prefill_len,
                     "results": results})
    out.write_text(json.dumps(payload, indent=2))
    print(f"[accept] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
