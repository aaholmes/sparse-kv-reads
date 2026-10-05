"""Offline `cluster_skip` sweep: attention-output error vs K+V reads.

For each captured context, layer, sampled decode step and KV head: center the non-exact
keys, partition them (flat: ``C`` fixed random directions; tree: ``C1`` coarse directions,
each cell split by its own ``C2`` fixed directions), rank regions by mean direction ×
max/min length, select up to a key budget (per query head or shared across the 4-head
group), and compare the exact softmax over the read keys with dense attention.

Reads per KV head = 2 × (union of rows read by the group) + summary rows, where summary
rows are the direction vectors consulted (flat: all ``C``; tree: ``C1`` plus ``C2`` for
each expanded coarse cell) plus their length scalars; reported as a fraction of all 2n
K and V rows. ``santa_sys`` single draws are the reference (all keys + distinct values).

Tree search: rank coarse cells, expand them in order until the expanded cells hold
``gamma × need`` keys, rank the fine regions inside the expanded cells, select until the
budget is met; if still short, take whole unexpanded coarse cells in coarse order.

Run:
    uv run python -m ssa.harness.sphere_sweep
"""

from __future__ import annotations

import math

import torch

from ..attn.sphere_skip import (
    GROUP_MODES,
    _region_rank_mask,
    estimate_max_score,
    fixed_directions,
    select_regions,
)
from ..sampling.draws import systematic_indices, unique_counts


def _stats_from_labels(Kr: torch.Tensor, labels: torch.Tensor, R: int) -> dict:
    mag = Kr.norm(dim=1)
    Kn = Kr / mag.clamp_min(1e-12).unsqueeze(1)
    kw = dict(dtype=Kr.dtype, device=Kr.device)
    s = torch.zeros(R, Kr.shape[1], **kw).index_add_(0, labels, Kn)
    return {"cdir": s / s.norm(dim=1, keepdim=True).clamp_min(1e-12),
            "mmax": torch.zeros(R, **kw).scatter_reduce_(0, labels, mag, reduce="amax"),
            "mmin": torch.full((R,), math.inf, **kw).scatter_reduce_(0, labels, mag, reduce="amin"),
            "count": torch.zeros(R, **kw).index_add_(0, labels, torch.ones_like(mag)),
            "labels": labels}


def flat_partition(Kr: torch.Tensor, C: int, seed: int = 0) -> dict:
    dirs = fixed_directions(C, Kr.shape[1], seed=seed, dtype=Kr.dtype, device=Kr.device)
    Kn = Kr / Kr.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return _stats_from_labels(Kr, (Kn @ dirs.t()).argmax(1), C)


def tree_partition(Kr: torch.Tensor, *, C1: int, C2: int, seed: int = 0) -> dict:
    """Two-level fixed partition: coarse cell by ``C1`` directions, then ``C2`` per cell."""
    d = Kr.shape[1]
    dirs1 = fixed_directions(C1, d, seed=seed, dtype=Kr.dtype, device=Kr.device)
    dirs2 = fixed_directions(C1 * C2, d, seed=seed + 1, dtype=Kr.dtype, device=Kr.device).view(C1, C2, d)
    Kn = Kr / Kr.norm(dim=1, keepdim=True).clamp_min(1e-12)
    coarse = (Kn @ dirs1.t()).argmax(1)
    fine_local = torch.einsum("nd,ncd->nc", Kn, dirs2[coarse]).argmax(1)
    fine = coarse * C2 + fine_local
    return {"coarse": _stats_from_labels(Kr, coarse, C1), "fine": _stats_from_labels(Kr, fine, C1 * C2),
            "coarse_labels": coarse, "fine_labels": fine, "C1": C1, "C2": C2}


def tree_select(qg, t: dict, need: int, *, gamma: float, group: str, scale: float):
    """Fine-region mask ``[G, C1*C2]`` and summary rows consulted (mean over the group)."""
    G = qg.shape[0]
    C1, C2 = t["C1"], t["C2"]
    co, fi = t["coarse"], t["fine"]
    e1 = estimate_max_score(qg, co)
    # coarse cells to expand: in coarse rank order until they hold gamma*need keys
    exp_mask = select_regions(e1, co["count"], max(1, math.ceil(gamma * need)), group=group, scale=scale)
    e2 = estimate_max_score(qg, fi)
    e2 = e2.masked_fill(~exp_mask.repeat_interleave(C2, dim=1), -math.inf)
    count2 = fi["count"]
    reg = select_regions(e2, count2, need, group=group, scale=scale)
    reg = reg & exp_mask.repeat_interleave(C2, dim=1)                   # never select unexpanded fines
    got = (reg * count2).sum(1)
    short = got < need
    if short.any():                                                      # fall back to whole coarse cells
        e1b = e1.masked_fill(exp_mask, -math.inf)
        for g in short.nonzero(as_tuple=True)[0].tolist():
            rem = need - int(got[g])
            m = _region_rank_mask(e1b[g:g + 1], co["count"], rem)[0] & ~exp_mask[g]
            reg[g] |= m.repeat_interleave(C2)
    overhead = C1 + exp_mask.float().sum(1).mean() * C2
    if group != "per_head":
        overhead = C1 + exp_mask[0].float().sum() * C2
    return reg, float(overhead)


def _metrics(Ag, Vh, dense_g, sel):
    """Renormalized dense weights over ``sel`` = softmax over the read keys."""
    As = Ag * sel
    out = (As @ Vh) / As.sum(1, keepdim=True)
    return float(((out - dense_g) ** 2).sum()), float((1 - As.sum(1)).mean())


def step_eval(q, K, V, *, scale: float, window: int, Cs, groups, budgets, trees, santa_S,
              seed: int = 0) -> dict:
    """One decode step, all KV heads. Returns ``{config: {err_num, err_den, kv_frac, dropped, ...}}``."""
    q, K, V = (x.to(torch.float64) for x in (q, K, V))
    H, d = q.shape
    n, H_kv, _ = K.shape
    G = H // H_kv
    ex = torch.zeros(n, dtype=torch.bool, device=q.device)
    ex[0] = True
    ex[max(0, n - window):] = True
    cl = (~ex).nonzero(as_tuple=True)[0]
    need = math.ceil
    res: dict = {}

    def acc(key, err, dropped, rows, overhead=None):
        r = res.setdefault(key, {"err_num": 0.0, "err_den": 0.0, "kv_rows": 0.0, "dropped": 0.0,
                                 "overhead_rows": 0.0})
        r["err_num"] += err
        r["kv_rows"] += rows / H_kv
        r["dropped"] += dropped / H_kv
        if overhead is not None:
            r["overhead_rows"] += overhead / H_kv

    den = 0.0
    for hkv in range(H_kv):
        Kh, Vh = K[:, hkv], V[:, hkv]
        qg = q[hkv * G:(hkv + 1) * G]
        Ag = torch.softmax((qg @ Kh.t()) * scale, dim=-1)
        dense_g = Ag @ Vh
        den += float((dense_g ** 2).sum())
        Kr = Kh[cl] - Kh[cl].mean(0)
        for C in Cs:
            st = flat_partition(Kr, C, seed=seed)
            e = estimate_max_score(qg, st)
            ov = C + 2 * C / d
            for group in groups:
                for b in budgets:
                    reg = select_regions(e, st["count"], need(b * cl.numel()), group=group, scale=scale)
                    sel = ex.expand(G, n).clone()
                    sel[:, cl] = reg[:, st["labels"]]
                    err, dr = _metrics(Ag, Vh, dense_g, sel)
                    acc(f"flat C={C} {group} b={b}", err, dr, 2 * float(sel.any(0).sum()) + ov, ov)
        for (C1, C2, gamma) in trees:
            t = tree_partition(Kr, C1=C1, C2=C2, seed=seed)
            for group in groups:
                for b in budgets:
                    reg, ov_dirs = tree_select(qg, t, need(b * cl.numel()), gamma=gamma, group=group,
                                               scale=scale)
                    ov = ov_dirs * (1 + 2 / d)
                    sel = ex.expand(G, n).clone()
                    sel[:, cl] = reg[:, t["fine_labels"]]
                    err, dr = _metrics(Ag, Vh, dense_g, sel)
                    acc(f"tree {C1}x{C2} g={gamma} {group} b={b}", err, dr,
                        2 * float(sel.any(0).sum()) + ov, ov)
        for S in santa_S:
            g = torch.Generator(device="cpu").manual_seed(seed * 7919 + hkv * 31 + S)
            idx = systematic_indices(Ag.cpu(), S, generator=g).to(q.device)   # [G, S]
            out = Vh[idx].mean(1)
            err = float(((out - dense_g) ** 2).sum())
            vu = float(unique_counts(idx.reshape(1, -1))[0])
            acc(f"santa_sys S={S}", err, 0.0, n + vu)
    for r in res.values():
        r["err_den"] = den
        r["kv_frac"] = r["kv_rows"] / (2 * n)
    return res


def main() -> None:
    import argparse
    import glob
    import json
    from pathlib import Path

    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--captures", default="src/ssa/results/multictx/*.pt")
    p.add_argument("--step-stride", type=int, default=9)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    Cs = (32, 64, 128, 256, 512, 1024)
    budgets = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
    trees = ((16, 16, 2.0), (32, 32, 2.0), (16, 64, 2.0), (32, 32, 4.0))
    santa_S = (16, 64, 256)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(args.captures))
    per_ctx: dict = {}                       # layer -> config -> list over contexts of dict
    for f in files:
        cap = torch.load(f, weights_only=False)
        P, scale, layers = cap["prefill"], cap["scale"], cap["layers"]
        T = cap["q"].shape[1]
        for li, L in enumerate(layers):
            tot: dict = {}
            steps = list(range(0, T, args.step_stride))
            for t in steps:
                n = P + t + 1
                r = step_eval(cap["q"][li, t].to(dev), cap["k"][li, :n].to(dev), cap["v"][li, :n].to(dev),
                              scale=scale, window=args.window, Cs=Cs, groups=GROUP_MODES, budgets=budgets,
                              trees=trees, santa_S=santa_S, seed=0)
                for k, v in r.items():
                    a = tot.setdefault(k, {"err_num": 0.0, "err_den": 0.0, "kv_frac": 0.0, "dropped": 0.0,
                                           "overhead_frac": 0.0})
                    a["err_num"] += v["err_num"]
                    a["err_den"] += v["err_den"]
                    a["kv_frac"] += v["kv_frac"] / len(steps)
                    a["dropped"] += v["dropped"] / len(steps)
                    a["overhead_frac"] += v["overhead_rows"] / (2 * n) / len(steps)
            for k, a in tot.items():
                per_ctx.setdefault(str(L), {}).setdefault(k, []).append(
                    {"rel_err": a["err_num"] / a["err_den"], "kv_frac": a["kv_frac"],
                     "dropped": a["dropped"], "overhead_frac": a["overhead_frac"]})
        print(f"done {Path(f).name}", flush=True)

    g = torch.Generator().manual_seed(0)
    nctx = len(files)
    boots = torch.randint(0, nctx, (args.n_boot, nctx), generator=g)
    summary = {}
    for L, cfgs in per_ctx.items():
        summary[L] = {}
        for k, rows in cfgs.items():
            e = torch.tensor([r["rel_err"] for r in rows])
            be = e[boots].mean(1)
            summary[L][k] = {"rel_err": float(e.mean()),
                             "rel_err_ci": [float(be.quantile(0.025)), float(be.quantile(0.975))],
                             "kv_frac": float(torch.tensor([r["kv_frac"] for r in rows]).mean()),
                             "dropped": float(torch.tensor([r["dropped"] for r in rows]).mean()),
                             "overhead_frac": float(torch.tensor([r["overhead_frac"] for r in rows]).mean())}
    payload = stamp({"kind": "sphere_sweep", "captures": files, "window": args.window, "Cs": Cs,
                     "budgets": budgets, "trees": trees, "groups": GROUP_MODES, "santa_S": santa_S,
                     "step_stride": args.step_stride, "summary": summary, "per_context": per_ctx})
    out = Path("src/ssa/results") / f"sphere_sweep_{payload['git_sha'][:8]}.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
