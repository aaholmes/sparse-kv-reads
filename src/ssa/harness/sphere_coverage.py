"""Partition comparison for `sphere_skip`: captured mass vs key budget.

For each query, token 0 and the last ``window`` keys form the exact set; the rest are
grouped by a partition, regions are ranked by mean direction × max/min length, and
regions are taken until ``budget`` of the non-exact keys is reached. Reports the
fraction of the non-exact attention mass captured. Partitions: fixed direction sets
(random / Sobol / cross-polytope), 8-key spherical k-means re-run per step, and the
per-key best (top keys by true weight; no partition can beat it).

Run:
    uv run python -m ssa.harness.sphere_coverage
"""

from __future__ import annotations

import math

import torch

from ..attn.sphere_skip import _region_rank_mask, fixed_directions, region_stats
from .bound_order import cluster_summary, spherical_kmeans_best
from .sink_coverage import exact_mask


def captured_fraction(q, K, *, scale: float, window: int, budgets, method: str, seed: int = 0,
                      kind: str = "random", C: int = 64, B: int = 8, restarts: int = 10) -> dict:
    """``captured [G, len(budgets)]``: share of non-exact attention mass in the selected keys."""
    q = q.to(torch.float64)
    K = K.to(torch.float64)
    n = K.shape[0]
    A = torch.softmax((q @ K.t()) * scale, dim=-1)
    ex = exact_mask(n, window).to(K.device)
    Ac = A[:, ~ex]
    Kc = K[~ex]
    n_cl = Kc.shape[0]
    tot = Ac.sum(1, keepdim=True)
    out = {"exact_mass": A[:, ex].sum(1)}
    caps = []
    if method == "oracle":
        s = Ac.sort(dim=1, descending=True).values.cumsum(1)
        reads = []
        for b in budgets:
            k = max(1, math.ceil(b * n_cl))
            caps.append(s[:, k - 1:k] / tot)
            reads.append(torch.full((q.shape[0], 1), k / n_cl, dtype=A.dtype, device=A.device))
        out["captured"] = torch.cat(caps, 1)
        out["read"] = torch.cat(reads, 1)
        return out
    if method == "fixed":
        dirs = fixed_directions(C, K.shape[1], seed=seed, kind=kind, dtype=torch.float64, device=K.device)
        st = region_stats(Kc, dirs)
        labels, count = st["labels"], st["count"]
        cdir, mmax, mmin = st["cdir"], st["mmax"], st["mmin"]
        out["occupancy"] = count
    elif method == "kmeans":
        Kn = (Kc / Kc.norm(dim=1, keepdim=True)).to(torch.float32)
        labels, _ = spherical_kmeans_best(Kn, k=math.ceil(n_cl / B), restarts=restarts, seed=seed)
        st = cluster_summary(Kc, labels)
        count, cdir, mmax, mmin = st["size"], st["cdir"], st["mmax"], st["mmin"]
        out["occupancy"] = count
    else:
        raise ValueError(method)
    p = q @ cdir.t()
    e = torch.where(p >= 0, mmax * p, mmin * p).masked_fill(count == 0, -math.inf)
    m = torch.zeros(q.shape[0], count.numel(), dtype=A.dtype, device=A.device).index_add_(1, labels, Ac)
    reads = []
    for b in budgets:
        reg = _region_rank_mask(e, count, math.ceil(b * n_cl))
        caps.append(((m * reg).sum(1, keepdim=True)) / tot)
        reads.append((reg * count).sum(1, keepdim=True) / n_cl)      # actual keys read (overshoot)
    out["captured"] = torch.cat(caps, 1)
    out["read"] = torch.cat(reads, 1)
    return out


PARTITIONS = {
    "random C=64": ("fixed", {"kind": "random", "C": 64}),
    "random C=256": ("fixed", {"kind": "random", "C": 256}),
    "sobol C=256": ("fixed", {"kind": "sobol", "C": 256}),
    "cross C=256": ("fixed", {"kind": "cross", "C": 256}),
    "kmeans 8-key": ("kmeans", {"B": 8, "restarts": 10}),
    "per-key best": ("oracle", {}),
}


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
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    budgets = (0.05, 0.1, 0.2, 0.3, 0.5)
    files = sorted(glob.glob(args.captures))
    per_ctx = {nm: {} for nm in PARTITIONS}          # nm -> layer -> [ctx][budget] median captured
    occ = {nm: {} for nm in PARTITIONS if PARTITIONS[nm][0] != "oracle"}
    exact = {}
    for f in files:
        cap = torch.load(f, weights_only=False)
        P, scale, layers = cap["prefill"], cap["scale"], cap["layers"]
        T, H, H_kv = cap["q"].shape[1], cap["q"].shape[2], cap["k"].shape[2]
        G = H // H_kv
        for li, L in enumerate(layers):
            acc = {nm: [] for nm in PARTITIONS}
            rd = {nm: [] for nm in PARTITIONS}
            oc = {nm: [] for nm in occ}
            ex_m = []
            for t in range(0, T, args.step_stride):
                n = P + t + 1
                for hkv in range(H_kv):
                    K = cap["k"][li, :n, hkv].to(dev)
                    q = cap["q"][li, t, hkv * G:(hkv + 1) * G].to(dev)
                    for nm, (method, kw) in PARTITIONS.items():
                        r = captured_fraction(q, K, scale=scale, window=args.window, budgets=budgets,
                                              method=method, seed=L * 1000 + t * 10 + hkv
                                              if method == "kmeans" else 0, **kw)
                        acc[nm].append(r["captured"].cpu())
                        rd[nm].append(r["read"].cpu())
                        if nm in oc:
                            c = r["occupancy"].cpu()
                            pr = c / c.sum()
                            oc[nm].append(torch.tensor([float((c > 0).double().mean()),
                                                        float(pr.max()), float(1 / (pr ** 2).sum())]))
                    ex_m.append(r["exact_mass"].cpu())
            for nm in PARTITIONS:
                v = torch.cat(acc[nm])                                   # [queries, budgets]
                per_ctx[nm].setdefault(str(L), []).append(
                    {"median": v.median(0).values.tolist(), "p10": v.quantile(0.1, 0).tolist(),
                     "read_mean": torch.cat(rd[nm]).mean(0).tolist()})
            for nm in occ:
                occ[nm].setdefault(str(L), []).append(torch.stack(oc[nm]).mean(0).tolist())
            exact.setdefault(str(L), []).append(float(torch.cat(ex_m).mean()))
        print(f"done {Path(f).name}", flush=True)

    g = torch.Generator().manual_seed(0)
    nctx = len(files)
    boots = torch.randint(0, nctx, (args.n_boot, nctx), generator=g)
    summary = {}
    for nm in PARTITIONS:
        summary[nm] = {}
        for L, rows in per_ctx[nm].items():
            med = torch.tensor([r["median"] for r in rows])             # [ctx, budgets]
            p10 = torch.tensor([r["p10"] for r in rows])
            bm = med[boots].mean(1)
            summary[nm][L] = {"median_captured": med.mean(0).tolist(),
                              "ci_lo": bm.quantile(0.025, 0).tolist(), "ci_hi": bm.quantile(0.975, 0).tolist(),
                              "p10_captured": p10.mean(0).tolist(),
                              "read_mean": torch.tensor([r["read_mean"] for r in rows]).mean(0).tolist()}
    occ_summary = {nm: {L: dict(zip(("frac_nonempty", "largest_share", "effective_regions"),
                                    torch.tensor(v).mean(0).tolist())) for L, v in d.items()}
                   for nm, d in occ.items()}
    payload = stamp({"kind": "sphere_coverage", "captures": files, "window": args.window,
                     "budgets": budgets, "step_stride": args.step_stride,
                     "partitions": {k: [m, kw] for k, (m, kw) in PARTITIONS.items()},
                     "summary": summary, "occupancy": occ_summary,
                     "exact_mass": {L: sum(v) / len(v) for L, v in exact.items()},
                     "per_context": per_ctx})
    out = Path("src/ssa/results") / f"sphere_coverage_{payload['git_sha'][:8]}.json"
    out.write_text(json.dumps(payload, indent=1))
    for L in summary["per-key best"]:
        print(f"\n== layer {L}  exact-set mass {payload['exact_mass'][L]:.3f}   budgets {budgets}")
        for nm in PARTITIONS:
            s = summary[nm][L]
            o = occ_summary.get(nm, {}).get(L)
            ostr = (f"  nonempty {o['frac_nonempty']:.2f} largest {o['largest_share']:.2f} "
                    f"eff {o['effective_regions']:.0f}") if o else ""
            print(f"  {nm:14s} median " + " ".join(f"{x:.3f}" for x in s["median_captured"])
                  + "  p10 " + " ".join(f"{x:.3f}" for x in s["p10_captured"]) + ostr)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
