"""Real-model sweep: perplexity vs read budget across deterministic/stochastic splits.

Main output: perplexity (PPL) increase vs value-read fraction for `dense`, `santa_sys`
at several budgets, and `santa_hybrid` at several (k_h, S_tail) splits at matched
total budget. The end-to-end analogue of the synthetic variance-vs-budget result.

Run (user-launched; downloads ~8 GB, GPU, sequential decode is slow):
    uv run python -m ssa.harness.ppl_sweep --model Qwen/Qwen3-4B --max-chunks 16

The actual Qwen3-4B run is intentionally left to the user; ``run_sweep`` is the
testable core and is exercised on a tiny CPU model in the test suite.
"""

from __future__ import annotations

import argparse

import torch

from ..attn import canonical
from ..models.patch import install, uninstall
from .perplexity import decode_ppl
from .stamp import stamp

# (impl, cfg). santa_hybrid takes `S` as the tail budget; total budget = k_h + S.
# Frontier set: santa_sys swept across budgets, plus santa_hybrid at two matched
# total budgets (64 and 256) with several head/tail splits — to test whether the
# hybrid overtakes plain systematic at long context.
DEFAULT_CONDITIONS = [
    ("dense", {}),
    ("santa_sys", {"S": 8}),
    ("santa_sys", {"S": 16}),
    ("santa_sys", {"S": 32}),
    ("santa_sys", {"S": 64}),
    ("santa_sys", {"S": 128}),
    ("santa_sys", {"S": 256}),
    ("santa_sys", {"S": 512}),
    ("santa_hybrid", {"k_h": 8, "S": 56}),     # total 64
    ("santa_hybrid", {"k_h": 16, "S": 48}),    # total 64
    ("santa_hybrid", {"k_h": 32, "S": 32}),    # total 64
    ("santa_hybrid", {"k_h": 32, "S": 224}),   # total 256
    ("santa_hybrid", {"k_h": 64, "S": 192}),   # total 256
    ("santa_hybrid", {"k_h": 128, "S": 128}),  # total 256
]

# Cheap-end set: probes the low-budget regime where the deterministic head should
# pay off, plus a det:stoch ratio sweep and the biased top-k baseline. Deliberately
# omits santa_sys (reuse the existing de-risk frontier — overlay at plot time).
CHEAP_CONDITIONS = [
    ("dense", {}),
    # top-k biased baseline (deterministic; reads exactly k rows)
    ("topk", {"k": 1}), ("topk", {"k": 2}), ("topk", {"k": 4}), ("topk", {"k": 8}),
    ("topk", {"k": 16}), ("topk", {"k": 32}), ("topk", {"k": 64}),
    # semistoch "1 exact + rest stochastic" across budgets — the cheap-end frontier
    ("santa_hybrid", {"k_h": 1, "S": 3}),     # total 4
    ("santa_hybrid", {"k_h": 1, "S": 7}),     # total 8
    ("santa_hybrid", {"k_h": 1, "S": 15}),    # total 16
    ("santa_hybrid", {"k_h": 1, "S": 31}),    # total 32
    ("santa_hybrid", {"k_h": 1, "S": 63}),    # total 64
    # det:stoch ratio sweep at fixed total budgets 16 and 32
    ("santa_hybrid", {"k_h": 4, "S": 12}),    # total 16
    ("santa_hybrid", {"k_h": 8, "S": 8}),     # total 16
    ("santa_hybrid", {"k_h": 8, "S": 24}),    # total 32
    ("santa_hybrid", {"k_h": 16, "S": 16}),   # total 32
]

# Skip-K set: magnitude-ranked cluster selection (read only selected clusters' K/V)
# at several mass-coverage targets, vs dense and a couple of santa_sys reference points.
SKIPK_CONDITIONS = [
    ("dense", {}),
    ("skip_k", {"coverage": 0.999, "B": 16}),
    ("skip_k", {"coverage": 0.99, "B": 16}),
    ("skip_k", {"coverage": 0.95, "B": 16}),
    ("skip_k", {"coverage": 0.90, "B": 16}),
    ("skip_k", {"coverage": 0.80, "B": 16}),
    ("santa_sys", {"S": 64}),
    ("santa_sys", {"S": 256}),
]

# Sphere-skip set: fixed hypersphere partition, token 0 + recent window exact, top regions
# by estimated max score up to a key budget, tail dropped; random-region controls at
# matched budget; santa_sys references (read every key).
SPHERE_CONDITIONS = (
    [("dense", {})]
    + [("voronoi_skip", {"budget": b, "C": 256, "center": True}) for b in (0.05, 0.1, 0.2, 0.3, 0.5)]
    + [("voronoi_skip", {"budget": b, "C": 256, "center": True, "rank": "random"}) for b in (0.1, 0.3)]
    + [("santa_sys", {"S": 64}), ("santa_sys", {"S": 256})]
)

# Shared selection (one region set per KV head, ranked by summed per-head mass shares).
SPHERE_SHARED_CONDITIONS = (
    [("dense", {})]
    + [("voronoi_skip", {"budget": b, "C": 256, "center": True, "group": "sum_share"})
       for b in (0.05, 0.1, 0.2, 0.3, 0.4)]
)

# Long-context check of shared selection: C=256 (regions ~32 keys at 8k) and C=1024
# (~8 keys, matching the 2k region size) to separate context length from region size.
SPHERE_SHARED_8K_CONDITIONS = (
    [("dense", {})]
    + [("voronoi_skip", {"budget": b, "C": 256, "center": True, "group": "sum_share"})
       for b in (0.02, 0.05, 0.1, 0.2)]
    + [("voronoi_skip", {"budget": b, "C": 1024, "center": True, "group": "sum_share"})
       for b in (0.05, 0.1)]
    + [("santa_sys", {"S": 64}), ("santa_sys", {"S": 256})]
)

# 8k completion: the budgets needed to reach santa_sys S=256's TVD.
SPHERE_SHARED_8K_HI_CONDITIONS = (
    [("dense", {})]
    + [("voronoi_skip", {"budget": b, "C": 256, "center": True, "group": "sum_share"}) for b in (0.3, 0.4)]
)

# v1 incremental bins end to end: delta=0.03 (recommended) and delta=inf (never recenter).
SPHERE_V1_CONDITIONS = (
    [("dense", {})]
    + [("voronoi_skip_v1", {"budget": b, "C": 256, "group": "sum_share", "delta": dl})
       for dl in (0.03, float("inf")) for b in (0.1, 0.2)]
)

# Fused Triton kernels in the engine: TVD compared with the simulator results.
SPHERE_FUSED_CONDITIONS = (
    [("dense", {})]
    + [("voronoi_fused", {"budget": b, "C": 256, "window": 64, "delta": 0.03, "group": "sum_share",
                         "check_every": 16, "partition": "random"}) for b in (0.1, 0.2)]
)

# Fused kernels at three budgets with systematic-sampling references (for other models / contexts).
SPHERE_FUSED_REFS_CONDITIONS = (
    [("dense", {})]
    + [("voronoi_fused", {"budget": b, "C": 256, "window": 64, "delta": 0.03, "group": "sum_share",
                         "check_every": 16, "partition": "random"}) for b in (0.05, 0.1, 0.2)]
    + [("santa_sys", {"S": 64}), ("santa_sys", {"S": 256})]
)

def _fused(b, C=256):
    return ("voronoi_fused", {"budget": b, "C": C, "window": 64, "delta": 0.03, "group": "sum_share",
                             "check_every": 16, "partition": "random"})


# Top-k-style selection, thorough checks.
FUSED_HI_CONDITIONS = ([("dense", {})] + [_fused(b) for b in (0.05, 0.1, 0.2, 0.3, 0.4)]
                       + [("santa_sys", {"S": 64}), ("santa_sys", {"S": 256})])
FUSED_HI_ONLY_CONDITIONS = [("dense", {})] + [_fused(b) for b in (0.3, 0.4)]
FUSED_C_SCAN_CONDITIONS = [("dense", {})] + [_fused(b, C) for C in (128, 512) for b in (0.1, 0.2, 0.4)]

def _sample(h, S, a, C=256):
    return ("voronoi_sample", {"budget": h, "S": S, "alpha": a, "C": C, "window": 64, "delta": 0.03,
                              "check_every": 16, "partition": "random"})


# Sampling the unselected bins: grid at 8k, focused set at 32k; top-k 15% for matched reads.
SAMPLE_GRID_CONDITIONS = ([("dense", {}), _fused(0.15)]
                          + [_sample(h, S, a) for h in (0.05, 0.1, 0.2) for S in (8, 32) for a in (0.1, 0.5)])
SAMPLE_FOCUS_CONDITIONS = [("dense", {})] + [_sample(h, S, 0.5) for h in (0.1, 0.2) for S in (8, 32)]
def _tail(b, order=1, C=256):
    return ("voronoi_tail", {"budget": b, "order": order, "C": C, "window": 64, "delta": 0.03,
                            "group": "sum_share", "check_every": 16, "partition": "random"})


# Estimating the dropped bins from per-bin sums; dropping with the same bins as a matched control.
TAIL_CONDITIONS = ([("dense", {})] + [_tail(b) for b in (0.05, 0.1, 0.2, 0.4)]
                   + [_tail(b, "drop") for b in (0.1, 0.2)])
# Convergence at large budgets, and the estimate on every layer but the first.
TAIL_CONV_CONDITIONS = [("dense", {})] + [_tail(b, o) for b in (0.7, 1.0) for o in (1, "drop")]
TAIL_NO_L0_CONDITIONS = [("dense", {})] + [(i, {**c, "drop_layers": [0]}) for i, c in (_tail(0.1), _tail(0.2))]
# Quest-style pages (our exact set, shared selection) and voronoi_skip in one sweep, both in PyTorch.
QUEST_VS_VORONOI_CONDITIONS = (
    [("dense", {})]
    + [("quest_matched", {"budget": b, "page": 16, "window": 64}) for b in (0.1, 0.2, 0.3, 0.4)]
    + [("voronoi_skip", {"budget": b, "C": 256, "center": True, "group": "sum_share"}) for b in (0.05, 0.1, 0.2, 0.4)]
)
# Fitted (k-means) centroids compared with fixed random directions: one sweep, the incremental
# PyTorch index for both, tail dropped.
KMEANS_VS_RANDOM_CONDITIONS = [("dense", {})] + [
    (i, {**c, "partition": part}) for part in ("kmeans", "random")
    for i, c in (_tail(b, "drop") for b in (0.05, 0.1, 0.2, 0.4))]
# Fitted centroids in the fused Triton kernels (compare with the PyTorch index in kmeans_vs_random).
FUSED_KMEANS_CONDITIONS = [("dense", {})] + [(i, {**c, "partition": "kmeans"})
                                             for i, c in (_fused(b) for b in (0.05, 0.1, 0.2, 0.4))]
# Reference: weight-only quantization of the same model, compared with BF16 on the same chunks.
QUANT8_CONDITIONS = [("dense", {}), ("quant", {"n_bits": 8})]
QUANT4_CONDITIONS = [("dense", {}), ("quant", {"n_bits": 4, "group_size": 128})]

CONDITION_PRESETS = {"full": DEFAULT_CONDITIONS, "cheap": CHEAP_CONDITIONS,
                     "skipk": SKIPK_CONDITIONS, "sphere": SPHERE_CONDITIONS,
                     "sphere_shared": SPHERE_SHARED_CONDITIONS,
                     "sphere_shared_8k": SPHERE_SHARED_8K_CONDITIONS,
                     "sphere_shared_8k_hi": SPHERE_SHARED_8K_HI_CONDITIONS,
                     "sphere_v1": SPHERE_V1_CONDITIONS,
                     "voronoi_fused": SPHERE_FUSED_CONDITIONS,
                     "sphere_fused_refs": SPHERE_FUSED_REFS_CONDITIONS,
                     "fused_hi": FUSED_HI_CONDITIONS, "fused_hi_only": FUSED_HI_ONLY_CONDITIONS,
                     "fused_C_scan": FUSED_C_SCAN_CONDITIONS, "sample_grid": SAMPLE_GRID_CONDITIONS,
                     "sample_focus": SAMPLE_FOCUS_CONDITIONS, "quant8": QUANT8_CONDITIONS,
                     "quant4": QUANT4_CONDITIONS, "tail": TAIL_CONDITIONS,
                     "tail_conv": TAIL_CONV_CONDITIONS, "fused_kmeans": FUSED_KMEANS_CONDITIONS, "kmeans_vs_random": KMEANS_VS_RANDOM_CONDITIONS, "quest_vs_voronoi": QUEST_VS_VORONOI_CONDITIONS, "tail_no_l0": TAIL_NO_L0_CONDITIONS}


def _total_budget(impl: str, cfg: dict) -> int | None:
    if impl == "dense":
        return None
    return int(cfg.get("k_h", 0)) + int(cfg.get("S", 0))


def _run_condition(model, chunks, impl, cfg, *, prefill_len, n_runs) -> dict:
    """Compute one condition's PPL (+ timing, read fraction). Dense uses 1 run."""
    import time
    t0 = time.time()
    if impl == "dense":
        uninstall(model)
        r = decode_ppl(model, chunks, prefill_len=prefill_len)
        dt = time.time() - t0
        return {
            "impl": impl, "cfg": cfg, "total_budget": None,
            "ppl_mean": r["ppl"], "ppl_std": 0.0, "read_fraction": 1.0,
            "kv_read_fraction": 1.0, "chunk_nll": r["chunk_nll"], "chunk_tokens": r["chunk_tokens"],
            "token_count": r["token_count"], "n_runs": 1,
            "wall_seconds": dt, "sec_per_token": dt / max(r["token_count"], 1),
        }
    if canonical(impl) in ("topk", "voronoi_skip", "voronoi_skip_v1", "voronoi_fused", "voronoi_sample"):  # single run
        stats = install(model, impl, **cfg)
        r = decode_ppl(model, chunks, prefill_len=prefill_len)
        uninstall(model)
        dt = time.time() - t0
        return {
            "impl": impl, "cfg": cfg, "total_budget": int(cfg.get("k", 0)) if impl == "topk" else None,
            "ppl_mean": r["ppl"], "ppl_std": 0.0, "read_fraction": stats.read_fraction,
            "kv_read_fraction": stats.kv_read_fraction,
            "chunk_nll": r["chunk_nll"], "chunk_tokens": r["chunk_tokens"],
            "token_count": r["token_count"], "n_runs": 1,
            "wall_seconds": dt, "sec_per_token": dt / max(r["token_count"], 1),
        }
    ppls, frac, tokens, kvf, chunk_runs = [], 1.0, 0, None, []
    for run in range(n_runs):
        stats = install(model, impl, base_seed=run, **cfg)
        r = decode_ppl(model, chunks, prefill_len=prefill_len)
        ppls.append(r["ppl"])
        tokens += r["token_count"]
        frac = stats.read_fraction
        kvf = stats.kv_read_fraction
        chunk_runs.append(r["chunk_nll"])
        uninstall(model)
    dt = time.time() - t0
    t = torch.tensor(ppls)
    return {
        "impl": impl, "cfg": cfg, "total_budget": _total_budget(impl, cfg),
        "ppl_mean": float(t.mean()), "ppl_std": float(t.std(unbiased=False)),
        "read_fraction": frac, "kv_read_fraction": kvf, "n_runs": n_runs,
        "chunk_nll": [sum(x) / len(x) for x in zip(*chunk_runs)], "chunk_tokens": r["chunk_tokens"],
        "wall_seconds": dt, "sec_per_token": dt / max(tokens, 1),
    }


def run_sweep(
    model,
    chunks: list[torch.Tensor],
    *,
    conditions=DEFAULT_CONDITIONS,
    prefill_len: int = 32,
    n_runs: int = 3,
    on_condition=None,
) -> list[dict]:
    """Run each condition; sampling conditions are averaged over ``n_runs`` seeds.

    Each result carries ``wall_seconds``/``sec_per_token``. A failing condition
    is recorded as an ``error`` entry and the sweep continues (so one OOM doesn't
    sink an overnight run). ``on_condition(result, results_so_far)`` is called
    after every condition — used for crash-safe checkpointing + live progress.
    """
    results: list[dict] = []
    for impl, cfg in conditions:
        try:
            result = _run_condition(model, chunks, impl, cfg, prefill_len=prefill_len, n_runs=n_runs)
        except Exception as exc:  # keep going; record what failed
            uninstall(model)
            result = {"impl": impl, "cfg": cfg, "total_budget": _total_budget(impl, cfg),
                      "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        if on_condition is not None:
            on_condition(result, results)
    return results


def _row(r: dict, base: float | None) -> str:
    label = r["impl"] + (f" {r['cfg']}" if r["cfg"] else "")
    if "error" in r:
        return f"{label:28s} {'ERROR: ' + r['error']}"
    budget = "" if r["total_budget"] is None else str(r["total_budget"])
    dppl = "" if base is None else f"{100 * (r['ppl_mean'] - base) / base:+.2f}"
    return (
        f"{label:28s} {budget:>7s} {100*r['read_fraction']:>6.1f}% "
        f"{r['ppl_mean']:>10.4f} {dppl:>8s} "
        f"{r.get('wall_seconds', 0):>7.1f} {1000*r.get('sec_per_token', 0):>7.1f}"
    )


def _header() -> str:
    return (f"{'condition':28s} {'budget':>7s} {'read%':>7s} {'ppl':>10s} {'Δppl%':>8s} "
            f"{'sec':>7s} {'ms/tok':>7s}")


def _format_table(results: list[dict]) -> str:
    dense = next((r for r in results if r["impl"] == "dense" and "error" not in r), None)
    base = dense["ppl_mean"] if dense else None
    lines = [_header()] + [_row(r, base) for r in results]
    total = sum(r.get("wall_seconds", 0) for r in results)
    lines.append(f"{'TOTAL':28s} {'':>7s} {'':>7s} {'':>10s} {'':>8s} {total:>7.1f}")
    return "\n".join(lines)


def _load_model(model_id: str, device: str, dtype: torch.dtype):
    from engine.model import Qwen3Model
    from engine.weights import load_weights

    loaded = load_weights(model_id, dtype=dtype, device="cpu")
    model = Qwen3Model.from_loaded(loaded)
    return model.to(device=device, dtype=dtype).eval()


def _wikitext_chunks(model_id: str, *, max_chunks: int, chunk_len: int, device: str):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    ds = load_dataset("wikitext", "wikitext-103-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    chunks = []
    for i in range(0, ids.numel() - chunk_len, chunk_len):
        chunks.append(ids[i:i + chunk_len].unsqueeze(0).to(device))
        if len(chunks) >= max_chunks:
            break
    return chunks


def main() -> None:
    import json
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--max-chunks", type=int, default=16)
    p.add_argument("--chunk-len", type=int, default=512)
    p.add_argument("--prefill", type=int, default=128)
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--preset", default="full", choices=list(CONDITION_PRESETS),
                   help="condition set: 'full' frontier or 'cheap' low-budget probe")
    p.add_argument("--tag", default="", help="suffix for the output filename")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = _load_model(args.model, args.device, dtype)
    chunks = _wikitext_chunks(
        args.model, max_chunks=args.max_chunks, chunk_len=args.chunk_len, device=args.device
    )

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)

    def make_payload(results, complete):
        return stamp({
            "phase": "C", "model": args.model, "dataset": "wikitext-103-v1/test",
            "chunk_len": args.chunk_len, "prefill_len": args.prefill,
            "max_chunks": args.max_chunks, "n_runs": args.n_runs,
            "complete": complete, "results": results,
        })

    sha = make_payload([], False).get("git_sha", "nogit")[:8]
    tag = f"_{args.tag}" if args.tag else ""
    path = out_dir / f"phaseC_ppl_{sha}{tag}.json"

    base = {"v": None}
    print(_header(), flush=True)

    def on_condition(result, results_so_far):
        if result["impl"] == "dense" and "error" not in result:
            base["v"] = result["ppl_mean"]
        print(_row(result, base["v"]), flush=True)
        # crash-safe: persist after every condition
        path.write_text(json.dumps(make_payload(results_so_far, False), indent=2))

    results = run_sweep(
        model, chunks, conditions=CONDITION_PRESETS[args.preset],
        prefill_len=args.prefill, n_runs=args.n_runs, on_condition=on_condition
    )
    path.write_text(json.dumps(make_payload(results, True), indent=2))
    print("\n" + _format_table(results))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
