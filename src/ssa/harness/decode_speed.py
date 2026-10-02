"""End-to-end decode speed, fused `voronoi_skip` vs exact attention.

Qwen3-4B BF16 on the engine. For each context length n: prefill a WikiText-103 prompt in
chunks (the engine returns logits for every prompt position, so one 32k forward would need
~10 GB), then for each condition reset the cache to the prompt, run ``warmup`` decode steps
(the region index is built at the first one) and time ``steps`` teacher-forced decode steps.
Repeated ``repeats`` times; median and range of ms/token reported. One extra step per
condition is profiled to split GPU time into attention kernels and everything else.

Run:
    uv run python -m ssa.harness.decode_speed
"""

from __future__ import annotations

import time

import torch

from ..models.patch import install, uninstall

ATTN_KERNEL_KEYS = ("flash", "fmha", "attention", "sdpa", "_bin_kernel", "_score_kernel", "_pick_kernel",
                    "_compact_kernel", "_list_kernel")


def prefill(model, ids: torch.Tensor, n: int, *, chunk: int = 512):
    cache = model.alloc_cache(ids.shape[1] + 1)
    with torch.inference_mode():
        for s in range(0, n, chunk):
            model(ids[:, s:min(s + chunk, n)], cache, start_pos=s)
    torch.cuda.synchronize()
    return cache


def time_condition(model, ids, cache, n, impl, cfg, *, warmup: int, steps: int, repeats: int) -> dict:
    ms = []
    for _ in range(repeats):
        if impl == "dense":
            uninstall(model)
        else:
            install(model, impl, **cfg)
        cache.cur_len = n
        with torch.inference_mode():
            for t in range(n, n + warmup):
                model(ids[:, t:t + 1], cache)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for t in range(n + warmup, n + warmup + steps):
                model(ids[:, t:t + 1], cache)
            torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) / steps * 1e3)
        uninstall(model)
    ms_t = torch.tensor(ms)
    return {"ms_per_token_median": float(ms_t.median()), "ms_min": float(ms_t.min()),
            "ms_max": float(ms_t.max()), "ms_all": ms}


def time_graph(model, ids, cache, n, mode, cfg, *, warmup: int, steps: int, repeats: int, ref_logits=None) -> dict:
    """Decode with the whole step captured in a CUDA graph (``GraphDecoder``); optional
    per-step TVD compared with ``ref_logits`` (the engine's dense logits on the same tokens)."""
    from ..models.graph_decode import GraphDecoder

    ms, tvd = [], None
    for r in range(repeats):
        cache.cur_len = n
        with torch.inference_mode():
            dec = GraphDecoder(model, cache, mode=mode, **cfg)
            dec.prepare(n)
            dec.capture()
            for t in range(n, n + warmup):
                dec.step(ids[:, t:t + 1], t)
            torch.cuda.synchronize()
            got = []
            t0 = time.perf_counter()
            for t in range(n + warmup, n + warmup + steps):
                lg = dec.step(ids[:, t:t + 1], t)
                if ref_logits is not None and r == 0:
                    got.append(lg[0, -1].float().clone())
            torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) / steps * 1e3)
        if got:
            pg = torch.stack(got).softmax(-1)
            tvd = float(0.5 * (pg - ref_logits.softmax(-1)).abs().sum(-1).mean())
        del dec
    ms_t = torch.tensor(ms)
    return {"ms_per_token_median": float(ms_t.median()), "ms_min": float(ms_t.min()),
            "ms_max": float(ms_t.max()), "ms_all": ms, "tvd_vs_engine_dense": tvd}


def dense_logits(model, ids, cache, n, *, warmup: int, steps: int) -> torch.Tensor:
    uninstall(model)
    cache.cur_len = n
    out = []
    with torch.inference_mode():
        for t in range(n, n + warmup + steps):
            lg = model(ids[:, t:t + 1], cache)
            if t >= n + warmup:
                out.append(lg[0, -1].float().clone())
    return torch.stack(out)


def profile_step(model, ids, cache, n, impl, cfg) -> dict:
    from torch.profiler import ProfilerActivity, profile

    if impl != "dense":
        install(model, impl, **cfg)
    cache.cur_len = n
    with torch.inference_mode():
        for t in range(n, n + 4):                               # build the index, warm up
            model(ids[:, t:t + 1], cache)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            model(ids[:, n + 4:n + 5], cache)
            torch.cuda.synchronize()
    uninstall(model)
    tot = attn = 0.0
    for e in prof.key_averages():
        t = e.self_device_time_total
        if t <= 0:
            continue
        tot += t
        if any(k in e.key.lower() for k in ATTN_KERNEL_KEYS):
            attn += t
    return {"gpu_ms": tot / 1e3, "attention_kernels_ms": attn / 1e3}


CONDITIONS = [("dense", {}),
              ("voronoi_fused", {"budget": 0.2, "C": 256, "window": 64, "delta": 0.03, "group": "sum_share",
                                "check_every": 16, "track_reads": False}),
              ("voronoi_fused", {"budget": 0.05, "C": 256, "window": 64, "delta": 0.03, "group": "sum_share",
                                "check_every": 16, "track_reads": False})]


GRAPH_CONDITIONS = [("dense", {}),
                    ("voronoi", {"budget": 0.2, "C": 256, "window": 64, "delta": 0.03, "check_every": 16}),
                    ("voronoi", {"budget": 0.05, "C": 256, "window": 64, "delta": 0.03, "check_every": 16})]


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from .capture_qkv import random_wikitext_chunks
    from .ppl_sweep import _load_model
    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--ns", type=int, nargs="+", default=[8192, 16384, 32768])
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tag", default="")
    p.add_argument("--graph", action="store_true", help="also time CUDA-graph decoding (dense and sphere)")
    args = p.parse_args()

    model = _load_model(args.model, "cuda", torch.bfloat16)
    rows = []
    for n in args.ns:
        ids = random_wikitext_chunks(args.model, n=1, length=n + args.warmup + args.steps + 8, seed=n)[0].cuda()
        cache = prefill(model, ids, n)
        conds = CONDITIONS[:1] if args.graph else CONDITIONS
        for impl, cfg in conds:
            r = time_condition(model, ids, cache, n, impl, cfg, warmup=args.warmup, steps=args.steps,
                               repeats=args.repeats)
            r.update(profile_step(model, ids, cache, n, impl, cfg))
            r.update({"n": n, "impl": impl, "cfg": cfg})
            rows.append(r)
            print(f"n={n:6d} {impl:12s} budget {cfg.get('budget', 1.0):4.2f}: "
                  f"{r['ms_per_token_median']:6.2f} ms/token [{r['ms_min']:.2f}, {r['ms_max']:.2f}]  "
                  f"GPU {r['gpu_ms']:6.2f} ms (attention kernels {r['attention_kernels_ms']:5.2f})", flush=True)
        if args.graph:
            ref = dense_logits(model, ids, cache, n, warmup=args.warmup, steps=args.steps)
            for mode, cfg in GRAPH_CONDITIONS:
                r = time_graph(model, ids, cache, n, mode, cfg, warmup=args.warmup, steps=args.steps,
                               repeats=args.repeats, ref_logits=ref)
                r.update({"n": n, "impl": f"graph_{mode}", "cfg": cfg})
                rows.append(r)
                print(f"n={n:6d} graph_{mode:6s} budget {cfg.get('budget', 1.0):4.2f}: "
                      f"{r['ms_per_token_median']:6.2f} ms/token [{r['ms_min']:.2f}, {r['ms_max']:.2f}]  "
                      f"TVD vs engine dense {r['tvd_vs_engine_dense']:.4f}", flush=True)
        del cache
        torch.cuda.empty_cache()
    payload = stamp({"kind": "decode_speed", "model": args.model, "steps": args.steps, "warmup": args.warmup,
                     "repeats": args.repeats, "prefill_chunk": 512, "rows": rows})
    out = Path("src/ssa/results") / f"decode_speed_{payload['git_sha'][:8]}{args.tag}.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
