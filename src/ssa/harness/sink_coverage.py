"""Multi-context sink and coverage check.

Reads per-context capture files (``capture_qkv --n-contexts``). For each query, keys are
split into an exact set (token 0 and the ``window`` most recent positions, always read)
and a clustered set (the rest), which is re-clustered by direction. Reports token-0
mass, exact-set mass, and the fraction of clustered keys needed to reach each target
fraction of the clustered mass, ranking clusters by mean direction × max/min length
(``est``) or by true mass (``oracle``).

Run:
    uv run python -m ssa.harness.sink_coverage --captures 'src/ssa/results/multictx/*.pt'
"""

from __future__ import annotations

import math

import torch

from .bound_order import cluster_summary, spherical_kmeans_best


def exact_mask(n: int, window: int) -> torch.Tensor:
    """Positions always read exactly: token 0 and the ``window`` most recent."""
    m = torch.zeros(n, dtype=torch.bool)
    m[0] = True
    m[max(0, n - window):] = True
    return m


def query_coverage(q, K, *, scale: float, window: int = 64, B: int = 8, restarts: int = 10,
                   seed: int = 0, targets=(0.9, 0.99)) -> dict:
    """Per-query statistics for queries ``q [G, d]`` over keys ``K [n, d]``; each value is ``[G]``."""
    q = q.to(torch.float64)
    K64 = K.to(torch.float64)
    n = K.shape[0]
    A = torch.softmax((q @ K64.t()) * scale, dim=-1)
    ex = exact_mask(n, window).to(K.device)
    out = {"sink": A[:, 0], "exact_mass": A[:, ex].sum(1)}

    Kc = K[~ex]
    n_cl = Kc.shape[0]
    Kn = (Kc / Kc.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(torch.float32)
    labels, _ = spherical_kmeans_best(Kn, k=math.ceil(n_cl / B), restarts=restarts, seed=seed)
    st = cluster_summary(Kc.to(torch.float64), labels)
    m = torch.zeros(q.shape[0], st["k"], dtype=A.dtype, device=A.device).index_add_(1, labels, A[:, ~ex])
    p = q @ st["cdir"].t()
    ranks = {"est": torch.where(p >= 0, st["mmax"] * p, st["mmin"] * p), "oracle": m}
    size = st["size"].expand(q.shape[0], -1)
    for name, key in ranks.items():
        order = key.argsort(dim=1, descending=True)
        cm = m.gather(1, order).cumsum(1) / m.sum(1, keepdim=True)
        cs = size.gather(1, order).cumsum(1)
        for t in targets:
            i = (cm >= t - 1e-12).to(torch.int8).argmax(1)
            out[f"nclu_{name}_{t}"] = (i + 1).to(torch.float64)
            out[f"frac_{name}_{t}"] = cs.gather(1, i.unsqueeze(1)).squeeze(1) / n_cl
    return out


def main() -> None:
    import argparse
    import glob
    import json
    from pathlib import Path

    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--captures", default="src/ssa/results/multictx/*.pt")
    p.add_argument("--step-stride", type=int, default=3)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--restarts", type=int, default=10)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    targets = (0.9, 0.99)
    files = sorted(glob.glob(args.captures))
    stats = ["sink", "exact_mass"] + [f"frac_{r}_{t}" for r in ("est", "oracle") for t in targets]
    per_ctx = []                                            # one dict per context
    for f in files:
        cap = torch.load(f, weights_only=False)
        P, scale, layers = cap["prefill"], cap["scale"], cap["layers"]
        T, H = cap["q"].shape[1], cap["q"].shape[2]
        H_kv = cap["k"].shape[2]
        G = H // H_kv
        row = {"file": Path(f).name, "corpus": cap["corpus"], "first_tokens": cap.get("first_text"),
               "layers": {}}
        for li, L in enumerate(layers):
            acc = {s: [] for s in stats}
            for t in range(0, T, args.step_stride):
                n = P + t + 1
                for hkv in range(H_kv):
                    K = cap["k"][li, :n, hkv].to(dev)
                    q = cap["q"][li, t, hkv * G:(hkv + 1) * G].to(dev)
                    r = query_coverage(q, K, scale=scale, window=args.window, B=args.B,
                                       restarts=args.restarts, seed=L * 1000 + t * 10 + hkv,
                                       targets=targets)
                    for s in stats:
                        acc[s].append(r[s].cpu())
            v = {s: torch.cat(x) for s, x in acc.items()}   # per-query values
            row["layers"][str(L)] = {
                s: {"mean": float(v[s].mean()), "median": float(v[s].median()),
                    "p90": float(v[s].quantile(0.9))} for s in stats}
        per_ctx.append(row)
        print(f"{row['file']} ({row['corpus']}, starts {row['first_tokens']!r}): " + "  ".join(
            f"L{L} sink {row['layers'][str(L)]['sink']['mean']:.2f} "
            f"f90 {row['layers'][str(L)]['frac_est_0.9']['median']:.3f}" for L in layers), flush=True)

    # across contexts: the context is the unit
    g = torch.Generator().manual_seed(0)
    summary = {}
    layers = [int(L) for L in per_ctx[0]["layers"]]
    for group in ("all", "wikitext", "code"):
        rows = [r for r in per_ctx if group == "all" or r["corpus"] == group]
        if not rows:
            continue
        boots = torch.randint(0, len(rows), (args.n_boot, len(rows)), generator=g)
        summary[group] = {}
        for L in layers:
            summary[group][str(L)] = {}
            for s in stats:
                for agg in ("mean", "median", "p90"):
                    x = torch.tensor([r["layers"][str(L)][s][agg] for r in rows])
                    bm = x[boots].mean(1)
                    summary[group][str(L)][f"{s}.{agg}"] = {
                        "mean_over_contexts": float(x.mean()),
                        "ci": [float(bm.quantile(0.025)), float(bm.quantile(0.975))],
                        "min": float(x.min()), "max": float(x.max())}
    payload = stamp({"kind": "sink_coverage", "captures": files, "window": args.window, "B": args.B,
                     "restarts": args.restarts, "step_stride": args.step_stride, "targets": targets,
                     "per_context": per_ctx, "summary": summary})
    out = Path("src/ssa/results") / f"sink_coverage_{payload['git_sha'][:8]}.json"
    out.write_text(json.dumps(payload, indent=1))

    def fmt(group, L, key):
        d = summary[group][str(L)][key]
        return f"{d['mean_over_contexts']:.3f} [{d['ci'][0]:.3f},{d['ci'][1]:.3f}]"
    for group in summary:
        print(f"\n== {group}")
        for L in layers:
            print(f"L{L:>2} sink {fmt(group, L, 'sink.mean')} | exact {fmt(group, L, 'exact_mass.mean')} | "
                  f"f90 est med {fmt(group, L, 'frac_est_0.9.median')} p90 {fmt(group, L, 'frac_est_0.9.p90')} | "
                  f"f99 est med {fmt(group, L, 'frac_est_0.99.median')} | "
                  f"f90 oracle med {fmt(group, L, 'frac_oracle_0.9.median')}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
