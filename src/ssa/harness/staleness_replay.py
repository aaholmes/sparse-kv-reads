"""How clusters fitted to the prompt age as tokens are generated, and how many clusters to use.

On captured keys (8,192-token contexts, queries at the last decode steps), a context is treated as
a prompt of ``n − G`` tokens followed by ``G`` generated tokens: centroids are fitted by k-means on
the first ``n − G`` binned keys, and the remaining ``G`` keys join their nearest centroid, as the
method does during generation. ``G = 0`` is a fresh fit on every key. Also run: fixed random
directions, and fresh fits with more clusters. All use the method's always-read tokens, centering,
length-based score and shared selection. Single-layer attention-output error is interpolated to
common K+V read fractions (summaries counted) and divided by the fresh 256-cluster fit's, with a
95% bootstrap interval over contexts.

The generated tokens here are the same document continued, so a change of topic during real
generation is not represented; and a larger ``G`` also means fewer keys to fit on.

Run:
    uv run python -m ssa.harness.staleness_replay --captures 'src/ssa/results/multictx8k/*.pt'
"""

from __future__ import annotations

import math

import torch

from ..attn.labeled import label_weighted_attention
from .partition_ablation import BUDGETS, TARGETS, group, select
from .quest_replay import interp_log

STALE = (512, 1024, 2048, 4096, 6144)
MORE = (512, 1024)
BASE = "kmeans_256_fresh"


def conditions():
    yield "fixed_256", dict(partition="random", C=256)
    yield BASE, dict(partition="kmeans", C=256)
    for g in STALE:
        yield f"kmeans_256_stale_{g}", dict(partition="kmeans", C=256, stale=g)
    for c in MORE:
        yield f"kmeans_{c}_fresh", dict(partition="kmeans", C=c)


def eval_step(q, K, V, n: int, *, window: int = 64) -> dict:
    H, d = q.shape
    G = H // K.shape[0]
    s = torch.einsum("hd,hnd->hn", q, K.repeat_interleave(G, 0)) / math.sqrt(d)
    exact = torch.einsum("hn,hnd->hd", torch.softmax(s, -1), V.repeat_interleave(G, 0))
    den = float((exact ** 2).sum())
    binned = max(1, n - window) - 1
    out = {}
    for name, cfg in conditions():
        fit = None if "stale" not in cfg else binned - cfg["stale"]
        g = group(K, n, partition=cfg["partition"], center=True, C=cfg["C"], window=window, fit_tokens=fit)
        for b in BUDGETS:
            labels, w, summary = select(q, K, n=n, budget=b, partition=cfg["partition"], center=True, score="length",
                                        C=cfg["C"], window=window, groups=g)
            rows = (torch.gather(w, 1, labels) > 0).sum(1).double().mean()
            o = label_weighted_attention(q, K, V, labels, w)
            out[(name, b)] = (float(((o - exact) ** 2).sum()), den, float((2 * rows + summary) / (2 * n)))
    return out


def main() -> None:
    import argparse
    import glob
    import json
    from pathlib import Path

    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--captures", default="src/ssa/results/multictx8k/*.pt")
    p.add_argument("--step-stride", type=int, default=16)
    args = p.parse_args()
    files = sorted(glob.glob(args.captures))
    if not files:
        raise SystemExit(f"no captures match {args.captures}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    acc: dict = {}
    for fi, f in enumerate(files):
        cap = torch.load(f, weights_only=False)
        P = cap["prefill"]
        for li, L in enumerate(cap["layers"]):
            for t in range(1, cap["q"].shape[1], args.step_stride):
                n = P + t + 1
                r = eval_step(cap["q"][li, t].to(dev).double(), cap["k"][li, :n].permute(1, 0, 2).to(dev).double(),
                              cap["v"][li, :n].permute(1, 0, 2).to(dev).double(), n)
                for (name, b), (e, dn, rd) in r.items():
                    a = acc.setdefault((L, name, b), {}).setdefault(fi, [0.0, 0.0, 0.0, 0])
                    a[0] += e
                    a[1] += dn
                    a[2] += rd
                    a[3] += 1
        print(f"done {Path(f).name}", flush=True)
    layers = sorted({k[0] for k in acc})
    names = [nm for nm, _ in conditions()]
    ctxs = range(len(files))

    def curve(L, nm, c):
        pts = sorted((acc[(L, nm, b)][c][2] / acc[(L, nm, b)][c][3], acc[(L, nm, b)][c][0] / acc[(L, nm, b)][c][1])
                     for b in BUDGETS)
        return [x for x, _ in pts], [y for _, y in pts]

    boot = torch.randint(0, len(files), (4000, len(files)), generator=torch.Generator().manual_seed(0))
    summary = []
    for L in layers:
        for nm in names:
            row = {"layer": L, "condition": nm, "ratio_to_fresh_256": {}}
            for x in TARGETS:
                em = [interp_log(*curve(L, nm, c), x) for c in ctxs]
                eb = [interp_log(*curve(L, BASE, c), x) for c in ctxs]
                if any(v is None for v in em + eb):
                    row["ratio_to_fresh_256"][str(x)] = None
                    continue
                a, b = torch.tensor(em), torch.tensor(eb)
                r = a[boot].sum(1) / b[boot].sum(1)
                row["ratio_to_fresh_256"][str(x)] = [float(a.sum() / b.sum()), float(r.quantile(0.025)),
                                                     float(r.quantile(0.975))]
            summary.append(row)
    payload = stamp({"kind": "staleness_replay", "captures": [Path(f).name for f in files], "budgets": BUDGETS,
                     "stale": STALE, "more_clusters": MORE, "window": 64, "step_stride": args.step_stride,
                     "summary": summary})
    out = Path("src/ssa/results") / f"staleness_replay_{payload['git_sha'][:8]}.json"
    out.write_text(json.dumps(payload, indent=1))
    for r in summary:
        cells = "  ".join(f"{float(x):.0%}: " + (f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" if v else "—")
                          for x, v in r["ratio_to_fresh_256"].items())
        print(f"L{r['layer']:>2} {r['condition']:22s} {cells}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
