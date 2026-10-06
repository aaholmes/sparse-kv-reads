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
  split_grow       as split, with a cap that grows with the context, which bounds the cluster count
  split_reset      as split with 512 slots, refitting a head to 256 clusters when its count reaches 512

  split_f32        as split, with scoring and insertion reading float32 cluster vectors (not 8-bit copies)

  split_cap32, split_cap64, split_cap128   split with an absolute cap of that many keys per cluster

``split``, ``split_grow`` and the ``split_cap`` arms get ``--slots`` cluster slots (spare slots cost nothing).

``--free T`` measures free-running generation instead: each arm samples its own continuation at
temperature ``T`` (the same uniform draw per step for every arm, so arms diverge only where their
distributions differ), and exact attention is then run over that arm's tokens; the TVD at each step
is between the two distributions given the arm's own history.

Collection writes, per chunk and arm, the mean TVD, rows read and clusters in use in blocks of 64 steps,
and the K+V read fraction with every per-step read counted (``ssa.attn.accounting``); ``summarize`` reads that file. Ratios are sums over chunks, with a
95% bootstrap interval over chunks (the same resampled chunks for both arms).

Run:
    uv run python -m ssa.harness.long_gen_tvd --model Qwen/Qwen3-4B
"""

from __future__ import annotations

import math
import time

import torch

from ..attn.accounting import kv_read_fraction

BLOCK = 64
ARMS = {"frozen": dict(C=256), "split": dict(C=512, C_init=256, split_factor=2.0), "refit": dict(C=256)}
EXTRA_ARMS = {"frozen512": dict(C=512), "split_norecenter": dict(C=512, C_init=256, split_factor=2.0, delta=math.inf),
              "split_grow": dict(C=512, C_init=256, split_factor=2.0, grow_cap=True),
              "split_reset": dict(C=512, C_init=256, split_factor=2.0, reset_at=512),
              "split_f32": dict(C=512, C_init=256, split_factor=2.0, summary_bits=32),
              **{f"split_cap{c}": dict(C=512, C_init=256, cap_keys=float(c)) for c in (32, 64, 128)}}
SLOTTED = ("split", "split_grow", "split_cap32", "split_cap64", "split_cap128")
PAIRS = [("split", "frozen"), ("refit", "frozen"), ("split", "refit"), ("split", "frozen512"),
         ("split_norecenter", "split"), ("split_grow", "split"), ("split_reset", "split"), ("split_grow", "frozen"),
         ("split_reset", "frozen"), ("split_reset", "split_grow"), ("split", "split_f32"),
         ("split_cap64", "frozen"), ("split_cap32", "split_cap64"), ("split_cap128", "split_cap64"),
         ("split_cap64", "split")]


def _interp_log(x0, y0, x1, y1, x):
    """log y linear in log x through two points, evaluated at x (arrays over chunks)."""
    t = (torch.log(x) - torch.log(x0)) / (torch.log(x1) - torch.log(x0))
    return torch.exp(torch.log(y0) + t * (torch.log(y1) - torch.log(y0)))


def summarize(payload: dict, *, nbins: int = 4, n_boot: int = 4000, seed: int = 0) -> list[dict]:
    """Per budget and bin of generated tokens: each arm's TVD and read fraction, and the ratios in
    ``PAIRS`` whose arms were run, with bootstrap intervals over chunks. With two budgets, each ratio
    is also given with its first arm interpolated (log-log between the two budgets) to the second
    arm's read fraction; pairs where that read fraction lies outside the first arm's two are listed
    under ``extrapolated``."""
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
        arms = [a for a in list(ARMS) + list(EXTRA_ARMS) if f"{a}_{b}" in tvd]
        for i in range(nbins):
            sl = slice(i * per, (i + 1) * per)
            t = {a: tvd[f"{a}_{b}"][:, sl].mean(1) for a in arms}
            r = {a: rd[f"{a}_{b}"][:, sl].mean(1) for a in arms}
            row = {"budget": b, "generated": [i * per * BLOCK, (i + 1) * per * BLOCK],
                   "tvd": {a: float(t[a].mean()) for a in arms}, "reads": {a: float(r[a].mean()) for a in arms}}
            two = sorted(budgets) if len(budgets) == 2 else None
            for x, y in PAIRS:
                if x in t and y in t:
                    row[f"{x}_over_{y}"] = ratio(t[x], t[y])
                    if two:                                                # x interpolated to y's reads
                        r0, r1 = (rd[f"{x}_{v}"][:, sl].mean(1) for v in two)
                        t0, t1 = (tvd[f"{x}_{v}"][:, sl].mean(1) for v in two)
                        row[f"{x}_over_{y}_matched_reads"] = ratio(_interp_log(r0, t0, r1, t1, r[y]), t[y])
                        if not float(r0.mean()) <= float(r[y].mean()) <= float(r1.mean()):
                            row.setdefault("extrapolated", []).append(f"{x}_over_{y}")
            rows.append(row)
    return rows


def _tvd_blocks(a: torch.Tensor, b: torch.Tensor) -> list:
    """Mean TVD per block of ``BLOCK`` steps between two logit arrays ``[steps, vocab]`` held in host memory."""
    out = []
    for i in range(0, a.shape[0], BLOCK):
        pa, pb = a[i:i + BLOCK].cuda().float().softmax(-1), b[i:i + BLOCK].cuda().float().softmax(-1)
        out.append(float((0.5 * (pa - pb).abs().sum(-1)).mean()))
    return out


def _decode(model, ids, cache, n, steps, mode, cfg, ref=None, free=None):
    """Decode ``steps`` tokens after a prompt of ``n``. Teacher-forced on ``ids`` by default; with
    ``free = (temperature, u)`` each next token is sampled from this decode's own distribution using
    the uniform draws ``u [steps]``. Returns a dict: ``logits`` ``[steps, vocab]`` (float16, host
    memory; when ``ref`` is None), ``tvd`` per block compared with ``ref`` (when given), ``fed`` (the
    tokens fed, ``[1, steps]``), and for the cluster mode ``reads`` per block and ``extra``."""
    from ..models.graph_decode import GraphDecoder
    cache.cur_len = n
    dec = GraphDecoder(model, cache, mode=mode, **cfg)
    dec.prepare(n)
    dec.capture()
    d = dec.d
    out = torch.empty(steps, model.cfg.vocab_size, dtype=torch.float16) if ref is None else None   # host memory
    tvd = torch.zeros(steps, device="cuda", dtype=torch.float64)
    reads = torch.zeros(steps, device="cuda", dtype=torch.float64)
    rows_t, ncl_t = torch.zeros_like(reads), torch.zeros_like(reads)
    fed = torch.empty(1, steps, dtype=torch.long, device="cuda")
    tok = ids[:, n:n + 1]
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for j, t in enumerate(range(n, n + steps)):
        fed[:, j:j + 1] = tok
        lg = dec.step(tok, t)[0, -1]
        if ref is None:
            out[j] = lg.to(torch.float16)
        else:
            tvd[j] = 0.5 * (lg.float().softmax(-1) - ref[j].cuda().float().softmax(-1)).abs().sum()
        if mode == "cluster":
            rows = torch.stack([i.cnt for i in dec.attn]).double().mean()
            ncl = torch.stack([i.n_c for i in dec.attn]).double().mean()
            rows_t[j], ncl_t[j] = rows, ncl
            reads[j] = kv_read_fraction(n=t + 1, rows=rows, clusters=ncl, d=d,
                                        summary_bits=cfg.get("summary_bits", 8), heads=dec.H_kv)
        if free is None:
            tok = ids[:, t + 1:t + 2]
        else:                                                      # inverse-CDF sampling with a shared draw
            cdf = (lg.double() / free[0]).softmax(-1).cumsum(-1)
            tok = torch.searchsorted(cdf, free[1][j:j + 1].double() * cdf[-1]).clamp(max=cdf.numel() - 1).view(1, 1)
        torch.cuda.synchronize()                                   # as a caller that reads each token would
    res = {"logits": out, "fed": fed, "tvd": tvd.view(-1, BLOCK).mean(1).tolist() if ref is not None else None}
    if mode == "cluster":
        res["reads"] = reads.view(-1, BLOCK).mean(1).tolist()
        res["extra"] = {
            "clusters_end": float(torch.stack([i.n_c for i in dec.attn]).double().mean()),
            "splits": sum(i.splits for i in dec.attn), "recenterings": sum(i.rebuilds for i in dec.attn),
            "refits": sum(i.refits for i in dec.attn), "resets": sum(i.resets for i in dec.attn),
            "clusters_max": int(max(int(i.n_c.max()) for i in dec.attn)),
            "rows": rows_t.view(-1, BLOCK).mean(1).tolist(), "clusters": ncl_t.view(-1, BLOCK).mean(1).tolist(),
            "ms_per_step": (time.perf_counter() - t_start) / steps * 1e3}
    del dec
    return res


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
    p.add_argument("--slots", type=int, default=512, help="cluster slots for the split and split_grow arms")
    p.add_argument("--nbins", type=int, default=4)
    p.add_argument("--free", type=float, default=0.0,
                   help="free-running generation at this temperature (0: feed the chunk's own tokens)")
    p.add_argument("--merge", default=None, help="earlier result file (same chunks) whose other arms to include")
    args = p.parse_args()
    assert args.steps % BLOCK == 0

    model = _load_model(args.model, "cuda", torch.bfloat16)
    chunks = random_wikitext_chunks(args.model, n=args.chunks, length=args.prompt + args.steps + 1, seed=args.seed)
    tvd, reads, extras, first_fed = {}, {}, {}, {}
    t0 = time.time()
    for ci, ids in enumerate(chunks):
        ids = ids.cuda()
        cache = prefill(model, ids, args.prompt)
        with torch.inference_mode():
            ref = None if args.free else _decode(model, ids, cache, args.prompt, args.steps, "dense", {})["logits"]
            u = torch.rand(args.steps, generator=torch.Generator().manual_seed(args.seed * 1000 + ci)).cuda()
            for b in args.budgets:
                for arm in args.arms:
                    cfg = {**{**ARMS, **EXTRA_ARMS}[arm], "budget": b, **({"C": args.slots} if arm in SLOTTED else {}),
                           "refit_every": args.refit_every if arm == "refit" else 0}
                    if args.free:                                  # the arm's own continuation, then exact attention on it
                        r = _decode(model, ids, cache, args.prompt, args.steps, "cluster", cfg, free=(args.free, u))
                        own = torch.cat([ids[:, :args.prompt], r["fed"]], 1)
                        exact = _decode(model, own, cache, args.prompt, args.steps, "dense", {})["logits"]
                        r["tvd"] = _tvd_blocks(r["logits"], exact)
                        r["extra"]["top_prob"] = [float(r["logits"][i:i + BLOCK].cuda().float().softmax(-1).amax(-1).mean())
                                                  for i in range(0, args.steps, BLOCK)]     # near 1: the text has collapsed
                        r["extra"]["top_prob_mean"] = sum(r["extra"]["top_prob"]) / len(r["extra"]["top_prob"])
                        r["extra"]["same_tokens_as_first_arm"] = float(
                            (r["fed"] == first_fed.setdefault(b, r["fed"])).double().mean())
                        del exact
                    else:
                        r = _decode(model, ids, cache, args.prompt, args.steps, "cluster", cfg, ref)
                    tv, rd, ex = r["tvd"], r["reads"], r["extra"]
                    tvd.setdefault(f"{arm}_{b}", []).append(tv)
                    reads.setdefault(f"{arm}_{b}", []).append(rd)
                    extras.setdefault(f"{arm}_{b}", []).append(ex)
                    print(f"chunk {ci} {arm:6s} budget {b}: TVD {sum(tv) / len(tv):.4f}  reads {sum(rd) / len(rd):.3f}  "
                          f"{ {k: v for k, v in ex.items() if not isinstance(v, list)} }  [{time.time() - t0:.0f}s]",
                          flush=True)
                    del r
            first_fed.clear()
        del ref, cache
        torch.cuda.empty_cache()
        Path(f"src/ssa/results/long_gen_tvd_partial{args.tag}.json").write_text(      # survives a crash; gitignored name
            json.dumps({"chunks_done": ci + 1, "args": vars(args), "tvd": tvd, "reads": reads, "extras": extras}))
    if args.merge:
        old = json.loads(Path(args.merge).read_text())
        assert all(old[k] == getattr(args, k) for k in ("prompt", "steps", "chunks", "budgets", "seed")) \
            and old["model"] == args.model and not args.free, "the merged file must come from the same chunks and settings"
        for store, key in ((tvd, "tvd"), (reads, "reads"), (extras, "extras")):
            for k, v in old[key].items():
                store.setdefault(k, v)
    payload = stamp({"kind": "long_gen_tvd", "merged_from": args.merge, "arms_run": args.arms, "model": args.model, "corpus": "wikitext_test", "prompt": args.prompt,
                     "steps": args.steps, "chunks": args.chunks, "budgets": args.budgets, "block": BLOCK,
                     "refit_every": args.refit_every, "slots": args.slots, "free_temperature": args.free, "arms": {**ARMS, **{k: {a: str(b) for a, b in v.items()} for k, v in EXTRA_ARMS.items()}}, "seed": args.seed, "tvd": tvd, "reads": reads,
                     "extras": extras})
    payload["summary"] = summarize(payload, nbins=args.nbins)
    out = Path("src/ssa/results") / f"long_gen_tvd_{payload['git_sha'][:8]}{args.tag}.json"
    out.write_text(json.dumps(payload, indent=1))
    f = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    for r in payload["summary"]:
        print(f"budget {r['budget']} tokens {r['generated'][0]:>4}-{r['generated'][1]:<4} "
              + "  ".join(f"{a} {v:.4f}@{r['reads'][a]:.3f}" for a, v in r["tvd"].items()))
        print("    " + "  ".join(f"{k} {f(v)}" for k, v in r.items() if "_over_" in k)
              + (f"  extrapolated: {r['extrapolated']}" if "extrapolated" in r else ""))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
