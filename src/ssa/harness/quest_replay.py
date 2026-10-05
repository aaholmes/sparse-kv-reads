"""Offline comparison of `cluster_skip` with Quest- and ClusterKV-style selection at matched reads.

On captured Qwen3-4B tensors (``capture_qkv --n-contexts``), for sampled decode steps and all heads:
single-layer attention-output error ``Σ‖out − exact‖² / Σ‖exact‖²`` against K+V rows read, for
``cluster_skip`` (256 regions, window 64, shared selection), ``quest_plain`` (16-token pages, per-head
selection, nothing always read) and ``quest_matched`` (``cluster_skip``'s exact set and shared selection);
optionally ``clusterkv_plain`` (cosine k-means, ~1 cluster per 80 tokens, first 16 tokens and
not-yet-clustered decode tokens always read, per-head selection) and ``clusterkv_matched_256`` /
``clusterkv_matched_n80`` (our exact set and shared selection, 256 or ~n/80 clusters).
Reads count each method's summaries: our regions' directions, lengths and counts; Quest's
per-page minimum and maximum keys. Each context's error-vs-reads curve is interpolated
(log-linearly) to common read fractions, and the ratio to ``cluster_skip`` is reported with a
95% bootstrap interval over contexts.

Run:
    uv run python -m ssa.harness.quest_replay --captures 'src/ssa/results/multictx8k/*.pt'
"""

from __future__ import annotations

import math

import torch

from ..attn.labeled import label_weighted_attention
from ..attn.clusterkv import clusterkv_labels_and_weights, spherical_kmeans
from ..attn.quest import quest_labels_and_weights
from ..attn.sphere_gpu import SphereIndexGPU

BUDGETS = (0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
METHODS = ("cluster_skip", "quest_plain", "quest_matched")
CLUSTERKV = ("clusterkv_plain", "clusterkv_matched_256", "clusterkv_matched_n80")
TARGETS = (0.10, 0.15, 0.20, 0.30)


def eval_step(q, K, V, n: int, *, C: int = 256, window: int = 64, page: int = 16, methods=METHODS,
              prefill: int | None = None) -> dict:
    """``q [H, d]``, ``K, V [H_kv, n, d]`` float64 -> {(method, budget): (err, den, reads)}."""
    H, d = q.shape
    G = H // K.shape[0]
    s = torch.einsum("hd,hnd->hn", q, K.repeat_interleave(G, 0)) / math.sqrt(d)
    exact = torch.einsum("hn,hnd->hd", torch.softmax(s, -1), V.repeat_interleave(G, 0))
    den = float((exact ** 2).sum())
    idx = SphereIndexGPU(C=C, window=window, delta=math.inf, capacity=n)
    idx.observe(K, n)
    out = {}
    ckv = {}                                          # clustering per ClusterKV variant, shared by budgets
    for m in methods:
        if m == "clusterkv_plain":
            st, en = 16, (n if prefill is None else min(prefill, n))
            ckv[m] = dict(variant="plain", clustered_end=en,
                          clusters=spherical_kmeans(K[:, st:en], C=max(1, round((en - st) / 80))))
        elif m.startswith("clusterkv_matched"):
            st, en = 1, max(1, n - window)
            Cm = 256 if m.endswith("256") else max(1, round((en - st) / 80))
            ckv[m] = dict(variant="matched", C=Cm, window=window, clusters=spherical_kmeans(K[:, st:en], C=Cm))
    for b in BUDGETS:
        for m in methods:
            if m == "cluster_skip":
                labels, w = idx.labels_and_weights(q, n=n, budget=b)
                summary = C * (1 + 2 / d)
            elif m in ckv:
                labels, w = clusterkv_labels_and_weights(q, K, n=n, budget=b, **ckv[m])
                summary = int(w.shape[1]) - 1                            # one centroid per cluster
            else:
                labels, w = quest_labels_and_weights(q, K, n=n, budget=b, variant=m, page=page, window=window)
                summary = 2 * (int(w.shape[1]) - 1)                     # min and max key per page
            wk = w if w.shape[0] == K.shape[0] else w.view(K.shape[0], G, -1).amax(1)   # union over the group
            rows = (torch.gather(wk, 1, labels.long()) > 0).sum(1).double().mean()
            o = label_weighted_attention(q, K, V, labels, w)
            out[(m, b)] = (float(((o - exact) ** 2).sum()), den, float((2 * rows + summary) / (2 * n)))
    return out


def interp_log(xs, ys, x):
    """Log-linear interpolation of ``ys`` at ``x`` (``xs`` increasing); None outside the range."""
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x0 <= x <= x1:
            if y0 <= 0 or y1 <= 0:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
            return math.exp(math.log(y0) + (math.log(y1) - math.log(y0)) * (x - x0) / (x1 - x0))
    return None


def main() -> None:
    import argparse
    import glob
    import json
    from pathlib import Path

    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--captures", default="src/ssa/results/multictx8k/*.pt")
    p.add_argument("--step-stride", type=int, default=16)
    p.add_argument("--tag", default="")
    p.add_argument("--clusterkv", action="store_true", help="compare ClusterKV-style variants instead of Quest-style")
    args = p.parse_args()
    files = sorted(glob.glob(args.captures))
    if not files:
        raise SystemExit(f"no captures match {args.captures}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    methods = ("cluster_skip",) + CLUSTERKV if args.clusterkv else METHODS
    acc: dict = {}                                   # (layer, method, budget) -> per-context [err, den, reads, steps]
    for fi, f in enumerate(files):
        cap = torch.load(f, weights_only=False)
        P = cap["prefill"]
        for li, L in enumerate(cap["layers"]):
            for t in range(1, cap["q"].shape[1], args.step_stride):
                n = P + t + 1
                r = eval_step(cap["q"][li, t].to(dev).double(), cap["k"][li, :n].permute(1, 0, 2).to(dev).double(),
                              cap["v"][li, :n].permute(1, 0, 2).to(dev).double(), n, methods=methods, prefill=P)
                for (m, b), (e, dn, rd) in r.items():
                    a = acc.setdefault((L, m, b), {}).setdefault(fi, [0.0, 0.0, 0.0, 0])
                    a[0] += e
                    a[1] += dn
                    a[2] += rd
                    a[3] += 1
        print(f"done {Path(f).name}", flush=True)

    layers = sorted({k[0] for k in acc})
    ctxs = range(len(files))
    curves = {}                                      # (layer, method, ctx) -> (reads list, err list)
    for L in layers:
        for m in methods:
            for c in ctxs:
                pts = sorted((acc[(L, m, b)][c][2] / acc[(L, m, b)][c][3], acc[(L, m, b)][c][0] / acc[(L, m, b)][c][1])
                             for b in BUDGETS)
                curves[(L, m, c)] = ([x for x, _ in pts], [y for _, y in pts])
    g = torch.Generator().manual_seed(0)
    boot = torch.randint(0, len(files), (4000, len(files)), generator=g)
    summary = []
    for L in layers:
        for m in methods:
            rows = {"layer": L, "method": m,
                    "curve": [{"budget": b, "reads": sum(acc[(L, m, b)][c][2] / acc[(L, m, b)][c][3] for c in ctxs) / len(files),
                               "rel_err": sum(acc[(L, m, b)][c][0] for c in ctxs) / sum(acc[(L, m, b)][c][1] for c in ctxs)}
                              for b in BUDGETS]}
            if m != "cluster_skip":
                rows["ratio_to_fixed_directions"] = {}
                for x in TARGETS:
                    em = [interp_log(*curves[(L, m, c)], x) for c in ctxs]
                    ev = [interp_log(*curves[(L, "cluster_skip", c)], x) for c in ctxs]
                    if any(v is None for v in em + ev):
                        rows["ratio_to_fixed_directions"][str(x)] = None
                        continue
                    em_t, ev_t = torch.tensor(em), torch.tensor(ev)
                    r = em_t[boot].sum(1) / ev_t[boot].sum(1)
                    rows["ratio_to_fixed_directions"][str(x)] = [float(em_t.sum() / ev_t.sum()),
                                                        float(r.quantile(0.025)), float(r.quantile(0.975))]
            summary.append(rows)
    payload = stamp({"kind": "quest_replay", "methods": methods, "captures": [Path(f).name for f in files], "budgets": BUDGETS,
                     "step_stride": args.step_stride, "page": 16, "C": 256, "window": 64, "summary": summary})
    out = Path("src/ssa/results") / f"quest_replay_{payload['git_sha'][:8]}{args.tag}.json"
    out.write_text(json.dumps(payload, indent=1))
    for r in summary:
        pts = "  ".join(f"{c['reads']:.3f}:{c['rel_err']:.2e}" for c in r["curve"])
        print(f"L{r['layer']:>2} {r['method']:13s} {pts}")
        for x, v in (r.get("ratio_to_fixed_directions") or {}).items():
            print(f"      at {float(x):.0%} reads: error / cluster_skip's (fixed directions) = "
                  + (f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" if v else "outside measured range"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
