"""Fidelity over a long generation: do clusters fitted to the prompt go stale, and does refreshing help?

Each WikiText-103 test chunk is a prompt of ``--prompt`` tokens followed by ``--steps`` decode steps
fed the chunk's own next tokens. Every arm decodes with the CUDA-graph decoder and is compared, step
by step, with exact attention in that decoder (total variation distance between next-token
distributions). Arms:

  frozen  clusters fitted once, at the end of the prompt
  split   as frozen, with spare slots, splitting any cluster that passes its size cap
  refit   clusters fitted again on every key each ``--refit-every`` steps
  frozen512        as frozen with 512 clusters: more clusters, without tracking the new keys
  split_norecenter as split, never recentering (the mean stays the prompt's)

Collection writes, per chunk and arm, the mean TVD and the mean K+V read fraction (cluster summaries
included) in blocks of 64 steps; ``summarize`` reads that file. Ratios are sums over chunks, with a
95% bootstrap interval over chunks (the same resampled chunks for both arms).

Run:
    uv run python -m ssa.harness.long_gen_tvd --model Qwen/Qwen3-4B
"""

from __future__ import annotations

import math
import time

import torch

BLOCK = 64
ARMS = {"frozen": dict(C=256), "split": dict(C=512, C_init=256, split_factor=2.0), "refit": dict(C=256)}
EXTRA_ARMS = {"frozen512": dict(C=512), "split_norecenter": dict(C=512, C_init=256, split_factor=2.0, delta=math.inf)}


def _interp_log(x0, y0, x1, y1, x):
    """log y linear in log x through two points, evaluated at x (arrays over chunks)."""
    t = (torch.log(x) - torch.log(x0)) / (torch.log(x1) - torch.log(x0))
    return torch.exp(torch.log(y0) + t * (torch.log(y1) - torch.log(y0)))


def summarize(payload: dict, *, nbins: int = 4, n_boot: int = 4000, seed: int = 0) -> list[dict]:
    """Per budget and bin of generated tokens: each arm's TVD and read fraction, and the ratios
    split ÷ frozen, refit ÷ frozen and split ÷ refit with bootstrap intervals over chunks. With two
    budgets, also split ÷ frozen with split interpolated to frozen's read fraction."""
    tvd = {k: torch.tensor(v, dtype=torch.float64) for k, v in payload["tvd"].items()}        # [chunks, blocks]
    rd = {k: torch.tensor(v, dtype=torch.float64) for k, v in payload["reads"].items()}
    budgets = payload["budgets"]
    n_chunks, n_blocks = next(iter(tvd.values())).shape
    per = n_blocks // nbins
    boot = torch.randint(0, n_chunks, (n_boot, n_chunks), generator=torch.Generator().manual_seed(seed))

    def ratio(a, b):
        r = a[boot].sum(1) / b[boot].sum(1)
        return [float(a.sum() / b.sum()), float(r.quantile(0.025)), float(r.quantile(0.975))]

    rows = []
    for b in budgets:
        for i in range(nbins):
            sl = slice(i * per, (i + 1) * per)
            t = {a: tvd[f"{a}_{b}"][:, sl].mean(1) for a in ARMS}
            r = {a: rd[f"{a}_{b}"][:, sl].mean(1) for a in ARMS}
            row = {"budget": b, "generated": [i * per * BLOCK, (i + 1) * per * BLOCK],
                   "tvd": {a: float(t[a].mean()) for a in ARMS}, "reads": {a: float(r[a].mean()) for a in ARMS},
                   "split_over_frozen": ratio(t["split"], t["frozen"]),
                   "refit_over_frozen": ratio(t["refit"], t["frozen"]),
                   "split_over_refit": ratio(t["split"], t["refit"])}
            for a in EXTRA_ARMS:
                if f"{a}_{b}" in tvd:
                    t[a], r[a] = tvd[f"{a}_{b}"][:, sl].mean(1), rd[f"{a}_{b}"][:, sl].mean(1)
                    row["tvd"][a], row["reads"][a] = float(t[a].mean()), float(r[a].mean())
            if "frozen512" in t:
                row["split_over_frozen512"] = ratio(t["split"], t["frozen512"])
            if "split_norecenter" in t:
                row["split_norecenter_over_split"] = ratio(t["split_norecenter"], t["split"])
            if len(budgets) == 2:
                lo, hi = (f"split_{x}" for x in sorted(budgets))
                for base in ("frozen", "frozen512"):                       # split interpolated to the other arm's reads
                    if base in t:
                        at = _interp_log(rd[lo][:, sl].mean(1), tvd[lo][:, sl].mean(1), rd[hi][:, sl].mean(1),
                                         tvd[hi][:, sl].mean(1), r[base])
                        row[f"split_over_{base}_matched_reads"] = ratio(at, t[base])
            rows.append(row)
    return rows


def _decode(model, ids, cache, n, steps, mode, cfg, ref=None):
    """Decode ``steps`` tokens after a prompt of ``n``. Returns logits ``[steps, vocab]`` (float16,
    when ``ref`` is None) or per-step TVD from ``ref`` and per-step read fractions."""
    from ..models.graph_decode import GraphDecoder
    cache.cur_len = n
    dec = GraphDecoder(model, cache, mode=mode, **cfg)
    dec.prepare(n)
    dec.capture()
    d = dec.d
    out = [] if ref is None else None
    tvd = torch.zeros(steps, device="cuda", dtype=torch.float64)
    reads = torch.zeros(steps, device="cuda", dtype=torch.float64)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for j, t in enumerate(range(n, n + steps)):
        lg = dec.step(ids[:, t:t + 1], t)[0, -1]
        if ref is None:
            out.append(lg.to(torch.float16).clone())
        else:
            tvd[j] = 0.5 * (lg.float().softmax(-1) - ref[j].float().softmax(-1)).abs().sum()
            rows = torch.stack([i.cnt for i in dec.attn]).double().mean()
            ncl = torch.stack([i.n_c for i in dec.attn]).double().mean()
            reads[j] = (2 * rows + ncl * (1 + 2 / d)) / (2 * (t + 1))
        torch.cuda.synchronize()                                   # as a caller that reads each token would
    extra = {"clusters_end": float(torch.stack([i.n_c for i in dec.attn]).double().mean()),
             "splits": sum(i.splits for i in dec.attn), "recenterings": sum(i.rebuilds for i in dec.attn),
             "refits": sum(i.refits for i in dec.attn),
             "ms_per_step": (time.perf_counter() - t_start) / steps * 1e3} if mode == "cluster" else {}
    del dec
    if ref is None:
        return torch.stack(out)
    return tvd.view(-1, BLOCK).mean(1).tolist(), reads.view(-1, BLOCK).mean(1).tolist(), extra


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from .capture_qkv import random_wikitext_chunks
    from .decode_speed import prefill
    from .ppl_sweep import _load_model
    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--prompt", type=int, default=4096)
    p.add_argument("--steps", type=int, default=4096)
    p.add_argument("--chunks", type=int, default=8)
    p.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.2])
    p.add_argument("--refit-every", type=int, default=320)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS) + list(EXTRA_ARMS))
    p.add_argument("--merge", default=None, help="earlier result file (same chunks) whose other arms to include")
    args = p.parse_args()
    assert args.steps % BLOCK == 0

    model = _load_model(args.model, "cuda", torch.bfloat16)
    chunks = random_wikitext_chunks(args.model, n=args.chunks, length=args.prompt + args.steps + 1, seed=args.seed)
    tvd, reads, extras = {}, {}, {}
    t0 = time.time()
    for ci, ids in enumerate(chunks):
        ids = ids.cuda()
        cache = prefill(model, ids, args.prompt)
        with torch.inference_mode():
            ref = _decode(model, ids, cache, args.prompt, args.steps, "dense", {})
            for b in args.budgets:
                for arm in args.arms:
                    cfg = {**{**ARMS, **EXTRA_ARMS}[arm], "budget": b, "refit_every": args.refit_every if arm == "refit" else 0}
                    tv, rd, ex = _decode(model, ids, cache, args.prompt, args.steps, "cluster", cfg, ref)
                    tvd.setdefault(f"{arm}_{b}", []).append(tv)
                    reads.setdefault(f"{arm}_{b}", []).append(rd)
                    extras.setdefault(f"{arm}_{b}", []).append(ex)
                    print(f"chunk {ci} {arm:6s} budget {b}: TVD {sum(tv) / len(tv):.4f}  reads {sum(rd) / len(rd):.3f}  "
                          f"{ex}  [{time.time() - t0:.0f}s]", flush=True)
        del ref, cache
        torch.cuda.empty_cache()
    if args.merge:
        old = json.loads(Path(args.merge).read_text())
        assert all(old[k] == getattr(args, k) for k in ("prompt", "steps", "chunks", "budgets", "seed")) \
            and old["model"] == args.model, "the merged file must come from the same chunks and settings"
        for store, key in ((tvd, "tvd"), (reads, "reads"), (extras, "extras")):
            for k, v in old[key].items():
                store.setdefault(k, v)
    payload = stamp({"kind": "long_gen_tvd", "merged_from": args.merge, "arms_run": args.arms, "model": args.model, "corpus": "wikitext_test", "prompt": args.prompt,
                     "steps": args.steps, "chunks": args.chunks, "budgets": args.budgets, "block": BLOCK,
                     "refit_every": args.refit_every, "arms": {**ARMS, **{k: {a: str(b) for a, b in v.items()} for k, v in EXTRA_ARMS.items()}}, "seed": args.seed, "tvd": tvd, "reads": reads,
                     "extras": extras})
    payload["summary"] = summarize(payload)
    out = Path("src/ssa/results") / f"long_gen_tvd_{payload['git_sha'][:8]}{args.tag}.json"
    out.write_text(json.dumps(payload, indent=1))
    f = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    for r in payload["summary"]:
        print(f"budget {r['budget']} tokens {r['generated'][0]:>4}-{r['generated'][1]:<4} "
              + "  ".join(f"{a} {v:.4f}@{r['reads'][a]:.3f}" for a, v in r["tvd"].items()))
        print("    " + "  ".join(f"{k} {f(v)}" for k, v in r.items() if "_over_" in k))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
