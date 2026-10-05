"""Which choice separates `cluster_skip` from ClusterKV-style selection: partition, centering or score.

All eight combinations of
  partition: ``random`` (nearest of C fixed random directions) or ``kmeans`` (cosine k-means)
  center:    keys mean-centered before grouping and scoring, or raw
  score:     ``length`` (max key length × projection on the group's mean direction; min length when
             the projection is negative) or ``qcentroid`` (projection on the mean direction alone)
share `cluster_skip`'s always-read set (token 0 and the last ``window`` tokens), shared selection per
KV head and C groups. ``random/centered/length`` is `cluster_skip`; ``kmeans/raw/qcentroid`` is
``clusterkv_matched`` (both checked by tests). On captured tensors, single-layer attention-output
error is interpolated to common K+V read fractions and divided by `cluster_skip`'s, with a 95%
bootstrap interval over contexts.

Run:
    uv run python -m ssa.harness.partition_ablation --captures 'src/ssa/results/multictx8k/*.pt'
"""

from __future__ import annotations

import math

import torch

from ..attn.clusterkv import spherical_kmeans
from ..attn.labeled import label_weighted_attention
from ..attn.sphere_skip import fixed_directions
from .quest_replay import interp_log

BUDGETS = (0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
TARGETS = (0.10, 0.15, 0.20, 0.30)
COMBOS = [(p, c, s) for p in ("random", "kmeans") for c in (True, False) for s in ("length", "qcentroid")]
BASE = ("random", True, "length")


def name(combo) -> str:
    p, c, s = combo
    return f"{p}/{'centered' if c else 'raw'}/{s}"


def group(K: torch.Tensor, n: int, *, partition: str, center: bool, C: int, window: int, seed: int = 0,
          fit_tokens: int | None = None):
    """Assign binned keys ``K[:, 1:n-window]`` to groups; returns the per-group statistics.
    With ``partition="kmeans"`` and ``fit_tokens``, the centroids are fitted on the first
    ``fit_tokens`` binned keys only, and every key then joins its nearest centroid (a stale fit)."""
    H_kv, _, d = K.shape
    end = max(1, n - window)
    X = K[:, 1:end]
    if center:
        X = X - X.mean(1, keepdim=True)
    mag = X.norm(dim=-1)
    Xn = X / mag.clamp_min(1e-12).unsqueeze(-1)
    if partition == "random":
        dirs = fixed_directions(C, d, seed=seed, dtype=K.dtype, device=K.device)
        assign = (Xn @ dirs.t()).argmax(-1)
    elif partition == "kmeans":
        if fit_tokens is None or fit_tokens >= X.shape[1]:
            assign, _ = spherical_kmeans(X, C=C, seed=seed)
        else:
            _, cent = spherical_kmeans(X[:, :fit_tokens], C=C, seed=seed)
            assign = torch.einsum("hmd,hcd->hmc", Xn, cent).argmax(-1)
    else:
        raise ValueError(f"unknown partition {partition!r}")
    kw = dict(dtype=K.dtype, device=K.device)
    sum_dir = torch.zeros(H_kv, C, d, **kw).scatter_add_(1, assign.unsqueeze(-1).expand_as(Xn), Xn)
    count = torch.zeros(H_kv, C, **kw).scatter_add_(1, assign, torch.ones_like(mag))
    mmax = torch.zeros(H_kv, C, **kw).scatter_reduce_(1, assign, mag, reduce="amax")
    mmin = torch.full((H_kv, C), math.inf, **kw).scatter_reduce_(1, assign, mag, reduce="amin")
    return {"assign": assign, "sum_dir": sum_dir, "count": count, "mmax": mmax, "mmin": mmin, "end": end}


def select(q: torch.Tensor, K: torch.Tensor, *, n: int, budget: float, partition: str, center: bool, score: str,
           C: int = 256, window: int = 64, seed: int = 0, groups: dict | None = None):
    """``labels [H_kv, n]`` (group id; label ``C`` = always read), ``w [H_kv, C+1]`` and the summary
    cost in rows (``C(1+2/d)`` with lengths, ``C`` for directions alone)."""
    H, d = q.shape
    H_kv = K.shape[0]
    G = H // H_kv
    g = groups if groups is not None else group(K, n, partition=partition, center=center, C=C, window=window, seed=seed)
    end = g["end"]
    labels = torch.full((H_kv, n), C, dtype=torch.long, device=K.device)
    labels[:, 1:end] = g["assign"]
    w = torch.zeros(H_kv, C + 1, dtype=q.dtype, device=q.device)
    w[:, C] = 1.0
    need = math.ceil(budget * (end - 1))
    if end > 1 and need > 0:
        cdir = g["sum_dir"] / g["sum_dir"].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        p = torch.einsum("hgd,hcd->hgc", q.view(H_kv, G, d), cdir)
        if score == "length":
            e = torch.where(p >= 0, g["mmax"].unsqueeze(1) * p, g["mmin"].unsqueeze(1) * p)
        elif score == "qcentroid":
            e = p
        else:
            raise ValueError(f"unknown score {score!r}")
        empty = g["count"] == 0
        e = e.masked_fill(empty.unsqueeze(1), -math.inf)
        shared = torch.softmax(e / math.sqrt(d), dim=2).sum(1).masked_fill(empty, -math.inf)
        order = shared.argsort(1, descending=True)
        cum = g["count"].gather(1, order).cumsum(1)
        n_sel = (cum < need).sum(1) + 1
        rank = torch.empty_like(order).scatter_(1, order, torch.arange(C, device=K.device).expand(H_kv, C))
        w[:, :C] = (rank < n_sel.unsqueeze(1)).to(w.dtype)
    return labels, w, (C * (1 + 2 / d) if score == "length" else C)


def eval_step(q, K, V, n: int, *, C: int = 256, window: int = 64) -> dict:
    H, d = q.shape
    G = H // K.shape[0]
    s = torch.einsum("hd,hnd->hn", q, K.repeat_interleave(G, 0)) / math.sqrt(d)
    exact = torch.einsum("hn,hnd->hd", torch.softmax(s, -1), V.repeat_interleave(G, 0))
    den = float((exact ** 2).sum())
    out = {}
    for partition in ("random", "kmeans"):
        for center in (True, False):
            g = group(K, n, partition=partition, center=center, C=C, window=window)
            for score in ("length", "qcentroid"):
                for b in BUDGETS:
                    labels, w, summary = select(q, K, n=n, budget=b, partition=partition, center=center, score=score,
                                                C=C, window=window, groups=g)
                    rows = (torch.gather(w, 1, labels) > 0).sum(1).double().mean()
                    o = label_weighted_attention(q, K, V, labels, w)
                    out[((partition, center, score), b)] = (float(((o - exact) ** 2).sum()), den,
                                                            float((2 * rows + summary) / (2 * n)))
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
                for key, (e, dn, rd) in r.items():
                    a = acc.setdefault((L,) + key, {}).setdefault(fi, [0.0, 0.0, 0.0, 0])
                    a[0] += e
                    a[1] += dn
                    a[2] += rd
                    a[3] += 1
        print(f"done {Path(f).name}", flush=True)
    layers = sorted({k[0] for k in acc})
    ctxs = range(len(files))

    def curve(L, combo, c):
        pts = sorted((acc[(L, combo, b)][c][2] / acc[(L, combo, b)][c][3], acc[(L, combo, b)][c][0] / acc[(L, combo, b)][c][1])
                     for b in BUDGETS)
        return [x for x, _ in pts], [y for _, y in pts]

    boot = torch.randint(0, len(files), (4000, len(files)), generator=torch.Generator().manual_seed(0))
    summary = []
    for L in layers:
        for combo in COMBOS:
            row = {"layer": L, "combo": name(combo), "ratio_to_fixed_directions": {}}
            for x in TARGETS:
                em = [interp_log(*curve(L, combo, c), x) for c in ctxs]
                ev = [interp_log(*curve(L, BASE, c), x) for c in ctxs]
                if any(v is None for v in em + ev):
                    row["ratio_to_fixed_directions"][str(x)] = None
                    continue
                a, b = torch.tensor(em), torch.tensor(ev)
                r = a[boot].sum(1) / b[boot].sum(1)
                row["ratio_to_fixed_directions"][str(x)] = [float(a.sum() / b.sum()), float(r.quantile(0.025)), float(r.quantile(0.975))]
            summary.append(row)
    payload = stamp({"kind": "partition_ablation", "captures": [Path(f).name for f in files], "budgets": BUDGETS,
                     "C": 256, "window": 64, "step_stride": args.step_stride, "summary": summary})
    out = Path("src/ssa/results") / f"partition_ablation_{payload['git_sha'][:8]}.json"
    out.write_text(json.dumps(payload, indent=1))
    for r in summary:
        cells = "  ".join(f"{float(x):.0%}: " + (f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" if v else "—")
                          for x, v in r["ratio_to_fixed_directions"].items())
        print(f"L{r['layer']:>2} {r['combo']:26s} {cells}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
