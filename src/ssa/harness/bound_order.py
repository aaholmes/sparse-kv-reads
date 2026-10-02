"""Bound-ordered cluster coverage: how much attention mass the top clusters by an upper bound hold.

At each sampled decode step, cluster all cached keys by direction (best of several
k-means++ restarts of spherical k-means, run fresh for that step). Each cluster stores
its unit mean direction ``ĉ_b``, angular radius ``ρ_b = max ‖k̂_j − ĉ_b‖``, and the max
and min key length ``M_b``, ``μ_b``. For a query ``q`` the score of any key in ``b`` obeys

    q·k_j ≤ U_b = M_b·x_b if x_b ≥ 0 else μ_b·x_b,     x_b = q·ĉ_b + ‖q‖ρ_b.

Visit clusters in decreasing order of a ranking key and track cumulative softmax
attention mass versus the fraction of clusters visited. Rankings: ``bound`` (U_b),
``meandir_maxmag`` (M_b·q·ĉ_b, no radius: an estimate, not a bound), ``oracle``
(true cluster mass, the best possible order), ``random``.

Run:
    uv run python -m ssa.harness.bound_order --layers 0 12 24 35
"""

from __future__ import annotations

import math

import torch

ORDERS = ("oracle", "bound", "meandir_maxmag", "random")


def _kmeanspp(Kn: torch.Tensor, k: int, R: int, seed: int) -> torch.Tensor:
    """Batched spherical k-means++ seeding. Kn [n, d] unit rows -> centers [R, k, d].

    Restart ``r`` draws all its random numbers from its own generator, so the first
    ``R`` restarts are identical whatever the batch size.
    """
    n, d = Kn.shape
    u = torch.stack([torch.rand(k, generator=torch.Generator().manual_seed(seed * 7919 + r),
                                dtype=torch.float64) for r in range(R)]).to(Kn.device, Kn.dtype)
    idx = torch.empty(R, k, dtype=torch.long, device=Kn.device)
    idx[:, 0] = (u[:, 0] * n).long().clamp_(max=n - 1)
    best = Kn[idx[:, 0]] @ Kn.t()                                    # [R, n] max cosine so far
    for c in range(1, k):
        dist = (1 - best).clamp_min(0)
        p = dist / dist.sum(1, keepdim=True).clamp_min(1e-30)
        pick = torch.searchsorted(p.cumsum(1), u[:, c:c + 1].contiguous()).clamp_(max=n - 1).squeeze(1)
        idx[:, c] = pick
        best = torch.maximum(best, Kn[pick] @ Kn.t())
    return Kn[idx]


def spherical_kmeans_best(Kn: torch.Tensor, *, k: int, restarts: int = 10, iters: int = 100,
                          seed: int = 0):
    """Best-of-``restarts`` spherical k-means. Returns (labels [n], objective Σ cos to center)."""
    C = _kmeanspp(Kn, k, restarts, seed)                             # [R, k, d]
    R = restarts
    labels = None
    for _ in range(iters):
        new = torch.einsum("nd,rkd->rnk", Kn, C).argmax(2)           # [R, n]
        if labels is not None and torch.equal(new, labels):
            break
        labels = new
        csum = torch.zeros_like(C).scatter_add_(
            1, labels.unsqueeze(-1).expand(-1, -1, Kn.shape[1]), Kn.expand(R, -1, -1))
        norm = csum.norm(dim=2, keepdim=True)
        C = torch.where(norm > 1e-12, csum / norm.clamp_min(1e-12), C)   # keep empty centers
    obj = torch.einsum("nd,rnd->rn", Kn, torch.gather(
        C, 1, labels.unsqueeze(-1).expand(-1, -1, Kn.shape[1]))).sum(1)
    r = int(obj.argmax())
    _, lab = torch.unique(labels[r], return_inverse=True)            # drop empty clusters
    return lab, float(obj[r])


def cluster_summary(K: torch.Tensor, labels: torch.Tensor) -> dict:
    """Per-cluster unit mean direction, angular radius, max/min length, size (float64)."""
    K = K.to(torch.float64)
    k = int(labels.max()) + 1
    mag = K.norm(dim=1)
    Kn = K / mag.clamp_min(1e-12).unsqueeze(1)
    csum = torch.zeros(k, K.shape[1], dtype=K.dtype, device=K.device).index_add_(0, labels, Kn)
    cdir = csum / csum.norm(dim=1, keepdim=True).clamp_min(1e-12)
    ang = (Kn - cdir[labels]).norm(dim=1)
    red = lambda v, how, init: torch.full((k,), init, dtype=K.dtype, device=K.device).scatter_reduce(
        0, labels, v, reduce=how)
    return {"k": k, "cdir": cdir, "rho": red(ang, "amax", 0.0), "mmax": red(mag, "amax", 0.0),
            "mmin": red(mag, "amin", math.inf),
            "size": torch.zeros(k, dtype=K.dtype, device=K.device).index_add_(0, labels, torch.ones_like(mag))}


def coverage_curves(q, K, labels, st, *, scale: float, grid: torch.Tensor, seed: int = 0) -> dict:
    """Cumulative attention mass vs fraction of clusters visited, per ordering: ``[G, len(grid)]``."""
    q = q.to(torch.float64)
    K = K.to(torch.float64)
    G = q.shape[0]
    k = st["k"]
    A = torch.softmax((q @ K.t()) * scale, dim=-1)
    m = torch.zeros(G, k, dtype=A.dtype, device=A.device).index_add_(1, labels, A)
    proj = q @ st["cdir"].t()
    x = proj + q.norm(dim=1, keepdim=True) * st["rho"]
    keys = {
        "oracle": m,
        "bound": torch.where(x >= 0, st["mmax"] * x, st["mmin"] * x),
        "meandir_maxmag": st["mmax"] * proj,
        "random": torch.rand(G, k, generator=torch.Generator().manual_seed(seed),
                             dtype=A.dtype).to(A.device),
    }
    n_vis = torch.ceil(grid.to(A.device) * k).long()                   # clusters visited at each grid point
    out = {}
    for name, key in keys.items():
        order = key.argsort(dim=1, descending=True)
        cum = torch.cat([torch.zeros(G, 1, dtype=A.dtype, device=A.device),
                         m.gather(1, order).cumsum(1)], 1)             # [G, k+1]
        out[name] = cum[:, n_vis].clamp(max=1.0)
        out[name][:, -1] = 1.0 if grid[-1] == 1 else out[name][:, -1]
    return out


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--capture", default=None)
    p.add_argument("--layers", type=int, nargs="+", default=[0, 12, 24, 35])
    p.add_argument("--steps", type=int, nargs="+", default=[0, 9, 18, 27, 36, 45, 54, 63])
    p.add_argument("--B", type=int, default=8, help="mean keys per cluster (k = ceil(n/B))")
    p.add_argument("--restarts", type=int, default=10)
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res_dir = Path(__file__).resolve().parent.parent / "results"
    cap_path = Path(args.capture) if args.capture else res_dir / "qkv_capture_p2048_s64.pt"
    cap = torch.load(cap_path, weights_only=False)
    P, scale = cap["prefill"], cap["scale"]
    H, H_kv = cap["q"].shape[2], cap["k"].shape[2]
    G = H // H_kv
    grid = torch.linspace(0, 1, 401, dtype=torch.float64)
    targets = (0.9, 0.99, 0.999)

    per_layer = {}
    for L in args.layers:
        units = {o: [] for o in ORDERS}                                # per (step, kv head): [G, grid]
        for t in args.steps:
            n = P + t + 1
            for hkv in range(H_kv):
                K = cap["k"][L, :n, hkv].to(dev, torch.float32)
                Kn = K / K.norm(dim=1, keepdim=True).clamp_min(1e-12)
                labels, _ = spherical_kmeans_best(Kn, k=math.ceil(n / args.B),
                                                  restarts=args.restarts, seed=L * 1000 + t * 10 + hkv)
                K64 = K.double()
                st = cluster_summary(K64, labels)
                q = cap["q"][L, t, hkv * G:(hkv + 1) * G].to(dev, torch.float64)
                cur = coverage_curves(q, K64, labels, st, scale=scale, grid=grid, seed=t * 10 + hkv)
                for o in ORDERS:
                    units[o].append(cur[o].cpu())
            print(f"layer {L} step {t} done", flush=True)
        U = {o: torch.stack(v) for o, v in units.items()}             # [n_units, G, grid]
        gb = torch.Generator().manual_seed(L)
        boots = torch.randint(0, U["oracle"].shape[0], (args.n_boot, U["oracle"].shape[0]), generator=gb)
        summ = {}
        for o in ORDERS:
            per_unit = U[o].mean(1)                                    # [n_units, grid]
            mean = per_unit.mean(0)
            bm = per_unit[boots].mean(1)
            summ[o] = {"mean": mean.tolist(), "lo": bm.quantile(0.025, 0).tolist(),
                       "hi": bm.quantile(0.975, 0).tolist(),
                       "frac_clusters_to": {str(x): float(grid[int((mean >= x).double().argmax())]) for x in targets}}
        per_layer[str(L)] = summ
        print(f"layer {L}: " + "  ".join(
            f"{o}: " + "/".join(f"{summ[o]['frac_clusters_to'][str(x)]:.3f}" for x in targets) for o in ORDERS)
            + "   (fraction of clusters to reach 90/99/99.9% mass)", flush=True)

    payload = stamp({"kind": "bound_order", "capture": str(cap_path), "capture_meta": cap.get("meta"),
                     "layers": args.layers, "steps": args.steps, "B": args.B, "restarts": args.restarts,
                     "n_queries_per_layer": len(args.steps) * H, "grid": grid.tolist(),
                     "per_layer": per_layer})
    tag = payload["git_sha"][:8]
    (res_dir / f"bound_order_{tag}.json").write_text(json.dumps(payload))

    colors = {"oracle": "#3d3d3a", "bound": "#2a78d6", "meandir_maxmag": "#eb6834", "random": "#a8a79f"}
    labels_txt = {"oracle": "true cluster mass (best order)", "bound": "upper bound (with radius)",
                  "meandir_maxmag": "mean direction × max length", "random": "random order"}
    styles = {"oracle": (0, (4, 2)), "bound": "-", "meandir_maxmag": "-", "random": (0, (1, 2))}
    fig, axes = plt.subplots(1, len(args.layers), figsize=(11, 3.2), sharey=True, squeeze=False)
    axes = axes[0]
    g = grid.numpy()
    for ax, L in zip(axes, args.layers):
        for o in ORDERS:
            s = per_layer[str(L)][o]
            ax.fill_between(g, s["lo"], s["hi"], color=colors[o], alpha=0.2, lw=0)
            ax.plot(g, s["mean"], color=colors[o], lw=1.8, ls=styles[o], label=labels_txt[o])
        ax.text(0.97, 0.05, f"layer {L}", transform=ax.transAxes, ha="right", va="bottom", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("fraction of clusters visited")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(alpha=0.15, lw=0.5)
    axes[0].set_ylabel("cumulative attention mass")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, fontsize=9, loc="upper center", ncol=4)
    fig.text(0.01, 0.01, f"n = {len(args.steps) * H} queries per layer; bands: 95% CI",
             ha="left", va="bottom", fontsize=8, color="#5f5e5a")
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    png = res_dir / f"bound_order_{tag}.png"
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
