"""Per-layer correction factor on the estimated tail mass.

``collect`` (expensive): for each captured context, layer, sampled step, budget and query head,
save the exactly read head's log-mass ``log Z_H`` and mean value ``μ_H``, the first-order
estimate's tail log-mass ``log Ẑ_T`` and mean value ``μ̂_T`` (``t0 = 0``), and exact attention.
``analyze`` (cheap): with ``out(κ) = (Z_H μ_H + κ Ẑ_T μ̂_T) / (Z_H + κ Ẑ_T)``, fit ``κ`` per layer and
budget on one half of the contexts (alternating, so each half mixes corpora), evaluate on the
other, swap, and report the pooled out-of-fold error relative to ``κ = 1`` (uncalibrated) and
``κ = 0`` (dropping), with 95% bootstrap intervals over contexts.

Run:
    uv run python -m ssa.harness.tail_calibrate collect --captures '<dir>/*.pt'
    uv run python -m ssa.harness.tail_calibrate analyze
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from ..attn.tail_estimate import SphereIndexTail, attend_with_tail, tail_log_estimates

BUDGETS = (0.05, 0.1, 0.2, 0.4)
COMPONENTS = Path("src/ssa/results/tail_components.pt")


def components(q, K, V, n, *, budget, C=256, window=64, check=False) -> dict:
    """Per query head: ``logZH, logZT [H]``, ``muH, muT, exact [H, d]``."""
    H, d = q.shape
    G = H // K.shape[0]
    idx = SphereIndexTail(C=C, window=window, delta=math.inf, capacity=n)
    idx.observe(K, V, n)
    labels, w = idx.labels_and_weights(q, n=n, budget=budget)
    read = torch.gather(w.repeat_interleave(G, 0), 1, labels.long().repeat_interleave(G, 0)) > 0
    Vx = V.repeat_interleave(G, 0)
    s = torch.einsum("hd,hnd->hn", q, K.repeat_interleave(G, 0)) / math.sqrt(d)
    exact = torch.einsum("hn,hnd->hd", torch.softmax(s, -1), Vx)
    sh = s.masked_fill(~read, -math.inf)
    muH = torch.einsum("hn,hnd->hd", torch.softmax(sh, -1), Vx)
    logz, vbar, _ = tail_log_estimates(idx, q, w, order=1)
    muT = torch.einsum("hc,hcd->hd", torch.softmax(logz, -1), vbar)
    out = {"logZH": torch.logsumexp(sh, -1), "logZT": torch.logsumexp(logz, -1), "muH": muH, "muT": muT,
           "exact": exact}
    if check:
        ref = attend_with_tail(idx, q, K, V, labels, w, order=1)
        torch.testing.assert_close(mix(out, torch.tensor(1.0, dtype=q.dtype)), ref)
    return out


def mix(c: dict, kappa: torch.Tensor) -> torch.Tensor:
    """``out(κ)`` for components ``c`` (any leading shape, heads last-but-one)."""
    a = torch.sigmoid(c["logZT"] + kappa.log() - c["logZH"]).unsqueeze(-1)       # κẐ_T / (Z_H + κẐ_T)
    return (1 - a) * c["muH"] + a * c["muT"]


def collect(files: list[str], stride: int) -> None:
    from .stamp import stamp
    recs = []
    for fi, f in enumerate(files):
        cap = torch.load(f, weights_only=False)
        P = cap["prefill"]
        for li, L in enumerate(cap["layers"]):
            for t in range(1, cap["q"].shape[1], stride):
                n = P + t + 1
                q = cap["q"][li, t].double()
                K = cap["k"][li, :n].permute(1, 0, 2).double()
                V = cap["v"][li, :n].permute(1, 0, 2).double()
                for b in BUDGETS:
                    c = components(q, K, V, n, budget=b, check=(fi == 0 and t == 1))
                    recs.append({"ctx": fi, "layer": L, "step": t, "budget": b, **c})
        print(f"done {Path(f).name}", flush=True)
    torch.save({"meta": stamp({"kind": "tail_components", "captures": [Path(f).name for f in files],
                               "step_stride": stride, "budgets": BUDGETS}), "records": recs}, COMPONENTS)
    print(f"wrote {COMPONENTS}")


def _boot(num, den, reps=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, len(num), (reps, len(num)), generator=g)
    r = num[i].sum(1) / den[i].sum(1)
    return [float(r.quantile(0.025)), float(r.quantile(0.975))]


def analyze() -> None:
    import json
    from .stamp import stamp
    data = torch.load(COMPONENTS, weights_only=False)
    recs = data["records"]
    n_ctx = 1 + max(r["ctx"] for r in recs)
    grid = torch.exp(torch.linspace(math.log(0.5), math.log(20.0), 241)).double()
    folds = [[i for i in range(n_ctx) if i % 2 == k] for k in (0, 1)]
    rows = []
    for L in sorted({r["layer"] for r in recs}):
        for b in BUDGETS:
            err = torch.zeros(n_ctx, len(grid), dtype=torch.float64)          # Σ‖out(κ) − exact‖² per context
            e0 = torch.zeros(n_ctx, dtype=torch.float64)
            for r in recs:
                if r["layer"] != L or r["budget"] != b:
                    continue
                o = mix({k: v.unsqueeze(0) for k, v in r.items() if torch.is_tensor(v)}, grid.view(-1, 1))
                err[r["ctx"]] += ((o - r["exact"]) ** 2).sum((1, 2))
                e0[r["ctx"]] += ((r["muH"] - r["exact"]) ** 2).sum()
            one = int((grid - 1.0).abs().argmin())
            oof = torch.zeros(n_ctx, dtype=torch.float64)
            kap = []
            for k in (0, 1):
                train, test = folds[1 - k], folds[k]
                j = int(err[train].sum(0).argmin())
                kap.append(float(grid[j]))
                oof[test] = err[test, j]
            j_all = int(err.sum(0).argmin())
            rows.append({"layer": L, "budget": b, "kappa_folds": kap, "kappa_all": float(grid[j_all]),
                         "oof_over_uncal": float(oof.sum() / err[:, one].sum()),
                         "oof_over_uncal_ci": _boot(oof, err[:, one]),
                         "insample_over_uncal": float(err[:, j_all].sum() / err[:, one].sum()),
                         "uncal_over_drop": float(err[:, one].sum() / e0.sum()),
                         "oof_over_drop": float(oof.sum() / e0.sum()), "oof_over_drop_ci": _boot(oof, e0)})
    payload = stamp({"kind": "tail_calibrate", "components": str(COMPONENTS), "source": data["meta"],
                     "folds": folds, "grid": [0.5, 20.0, len(grid)], "summary": rows})
    out = Path("src/ssa/results") / f"tail_calibrate_{payload['git_sha'][:8]}.json"
    out.write_text(json.dumps(payload, indent=1))
    for r in rows:
        print(f"L{r['layer']:>2} b={r['budget']:.2f} κ folds {r['kappa_folds'][0]:.2f}/{r['kappa_folds'][1]:.2f} "
              f"all {r['kappa_all']:.2f}  held-out/uncal {r['oof_over_uncal']:.3f} "
              f"[{r['oof_over_uncal_ci'][0]:.3f}, {r['oof_over_uncal_ci'][1]:.3f}]  in-sample {r['insample_over_uncal']:.3f}  "
              f"uncal/drop {r['uncal_over_drop']:.3f}  held-out/drop {r['oof_over_drop']:.3f} "
              f"[{r['oof_over_drop_ci'][0]:.3f}, {r['oof_over_drop_ci'][1]:.3f}]")
    print(f"wrote {out}")


def main() -> None:
    import argparse
    import glob
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("collect", "analyze"))
    p.add_argument("--captures", default="src/ssa/results/multictx/*.pt")
    p.add_argument("--step-stride", type=int, default=16)
    args = p.parse_args()
    if args.mode == "collect":
        files = sorted(glob.glob(args.captures))
        if not files:
            raise SystemExit(f"no captures match {args.captures}")
        collect(files, args.step_stride)
    else:
        analyze()


if __name__ == "__main__":
    main()
