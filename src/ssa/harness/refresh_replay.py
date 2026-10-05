"""Ways to keep fitted clusters current as tokens are generated, compared offline at matched reads.

A captured context is treated as a prompt of ``n − G`` tokens, clustered by k-means into ``C0``
clusters, followed by ``G`` generated tokens (see ``staleness_replay``). Schemes for the new keys:

  frozen  each joins its nearest prompt centroid (the method as it stands)
  refit   all keys are re-clustered every ``block`` tokens; on average the last ``block/2`` keys
          joined after the latest refit
  append  every ``block`` new keys are clustered on their own into ``block/per_cluster`` extra
          clusters (as ClusterKV does); keys of an unfinished block join the nearest centroid
  split   each joins its nearest centroid, and a cluster that grows beyond twice the prompt's mean
          cluster size is split in two by 2-means over its own keys
  drift   each joins the cluster whose members' mean direction, updated as keys join, is nearest

plus a fresh fit on every key and fixed random directions. All use the method's always-read tokens,
centering, length-based score and shared selection; reads count each scheme's cluster summaries, so
extra clusters are paid for. Single-layer attention-output error is interpolated to common K+V read
fractions and divided by the fresh fit's, with a 95% bootstrap interval over contexts.

Run:
    uv run python -m ssa.harness.refresh_replay --captures 'src/ssa/results/multictx8k/*.pt'
"""

from __future__ import annotations

import math

import torch

from ..attn.clusterkv import spherical_kmeans
from ..attn.labeled import label_weighted_attention
from ..attn.sphere_skip import fixed_directions
from .partition_ablation import BUDGETS, TARGETS, select
from .quest_replay import interp_log

SCHEMES = ("frozen", "refit", "append", "split", "drift")
GS = (1024, 4096)


def _nearest(Xn: torch.Tensor, cent: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
    s = torch.einsum("hmd,hcd->hmc", Xn, cent)
    if active is not None:
        s = s.masked_fill(~active.unsqueeze(1), -math.inf)
    return s.argmax(-1)


def assign_scheme(X: torch.Tensor, *, G: int, scheme: str, C0: int = 256, block: int = 320, per_cluster: int = 32,
                  seed: int = 0):
    """Cluster ids ``[h, m]`` for centered keys ``X [h, m, d]`` whose last ``G`` arrived after the
    prompt fit, and the number of clusters (the mean over heads for ``split``)."""
    h, m, d = X.shape
    Xn = torch.nn.functional.normalize(X, dim=-1)
    mf = m - G
    if scheme == "refit":
        _, cent = spherical_kmeans(X[:, :max(1, m - block // 2)], C=C0, seed=seed)
        return _nearest(Xn, cent), C0
    a0, cent0 = spherical_kmeans(X[:, :mf], C=C0, seed=seed)
    if scheme == "frozen":
        return _nearest(Xn, cent0), C0
    if scheme == "append":
        k = max(1, block // per_cluster)
        cents, parts, pos = [cent0], [a0], mf
        while pos + block <= m:
            ab, cb = spherical_kmeans(X[:, pos:pos + block], C=k, seed=seed)
            parts.append(ab + sum(c.shape[1] for c in cents))
            cents.append(cb)
            pos += block
        cent = torch.cat(cents, 1)
        if pos < m:
            parts.append(_nearest(Xn[:, pos:], cent))
        return torch.cat(parts, 1), cent.shape[1]
    if scheme == "drift":
        sums = torch.zeros(h, C0, d, dtype=X.dtype, device=X.device).scatter_add_(
            1, a0.unsqueeze(-1).expand(h, mf, d), Xn[:, :mf])
        parts = [a0]
        for p in range(mf, m, 64):
            xb = Xn[:, p:p + 64]
            ab = _nearest(xb, torch.nn.functional.normalize(sums, dim=-1))
            sums.scatter_add_(1, ab.unsqueeze(-1).expand(*ab.shape, d), xb)
            parts.append(ab)
        return torch.cat(parts, 1), C0
    if scheme == "split":
        cap = 2 * (mf // C0)
        cmax = C0 + 2 * (G // max(1, cap // 2)) + 8
        cent = torch.zeros(h, cmax, d, dtype=X.dtype, device=X.device)
        cent[:, :C0] = cent0
        assign = torch.full((h, m), -1, dtype=torch.long, device=X.device)
        assign[:, :mf] = a0
        n_c = [C0] * h
        g = torch.Generator(device="cpu").manual_seed(seed)
        for p in range(mf, m, 64):
            e = min(p + 64, m)
            active = torch.arange(cmax, device=X.device).unsqueeze(0) < torch.tensor(n_c, device=X.device).unsqueeze(1)
            assign[:, p:e] = _nearest(Xn[:, p:e], cent, active)
            for hh in range(h):
                counts = torch.bincount(assign[hh, :e], minlength=cmax)
                for c in (counts > cap).nonzero().flatten().tolist():
                    if n_c[hh] >= cmax:
                        break
                    idx = (assign[hh, :e] == c).nonzero().flatten()
                    pts = Xn[hh, idx]
                    two = pts[torch.randperm(len(idx), generator=g)[:2].to(X.device)].clone()
                    for _ in range(5):                              # 2-means on cosine similarity
                        side = (pts @ two.t()).argmax(1)
                        for j in (0, 1):
                            if (side == j).any():
                                two[j] = torch.nn.functional.normalize(pts[side == j].sum(0), dim=0)
                    side = (pts @ two.t()).argmax(1)
                    if (side == 0).all() or (side == 1).all():
                        continue
                    cent[hh, c], cent[hh, n_c[hh]] = two[0], two[1]
                    assign[hh, idx[side == 1]] = n_c[hh]
                    n_c[hh] += 1
        return assign, sum(n_c) / h
    raise ValueError(f"unknown scheme {scheme!r}")


def _groups(X: torch.Tensor, assign: torch.Tensor, C: int, end: int) -> dict:
    h, m, d = X.shape
    mag = X.norm(dim=-1)
    Xn = X / mag.clamp_min(1e-12).unsqueeze(-1)
    kw = dict(dtype=X.dtype, device=X.device)
    return {"assign": assign, "end": end,
            "sum_dir": torch.zeros(h, C, d, **kw).scatter_add_(1, assign.unsqueeze(-1).expand_as(Xn), Xn),
            "count": torch.zeros(h, C, **kw).scatter_add_(1, assign, torch.ones_like(mag)),
            "mmax": torch.zeros(h, C, **kw).scatter_reduce_(1, assign, mag, reduce="amax"),
            "mmin": torch.full((h, C), math.inf, **kw).scatter_reduce_(1, assign, mag, reduce="amin")}


def eval_step(q, K, V, n: int, *, window: int = 64) -> dict:
    H, d = q.shape
    Gq = H // K.shape[0]
    s = torch.einsum("hd,hnd->hn", q, K.repeat_interleave(Gq, 0)) / math.sqrt(d)
    exact = torch.einsum("hn,hnd->hd", torch.softmax(s, -1), V.repeat_interleave(Gq, 0))
    den = float((exact ** 2).sum())
    end = max(1, n - window)
    X = K[:, 1:end]
    X = X - X.mean(1, keepdim=True)
    Xn = torch.nn.functional.normalize(X, dim=-1)
    conds = {"fresh": (spherical_kmeans(X, C=256)[0], 256),
             "fixed": ((Xn @ fixed_directions(256, d, dtype=K.dtype, device=K.device).t()).argmax(-1), 256)}
    for G in GS:
        for sch in SCHEMES:
            if sch == "refit" and G != GS[0]:
                continue                                           # does not depend on G
            conds[f"{sch}_{G}" if sch != "refit" else "refit"] = assign_scheme(X, G=G, scheme=sch)
    out = {}
    for name, (assign, n_c) in conds.items():
        C = int(assign.max()) + 1
        g = _groups(X, assign, C, end)
        for b in BUDGETS:
            labels, w, _ = select(q, K, n=n, budget=b, partition="kmeans", center=True, score="length", C=C,
                                  window=window, groups=g)
            rows = (torch.gather(w, 1, labels) > 0).sum(1).double().mean()
            o = label_weighted_attention(q, K, V, labels, w)
            out[(name, b)] = (float(((o - exact) ** 2).sum()), den, float((2 * rows + n_c * (1 + 2 / d)) / (2 * n)),
                              float(n_c))
    return out


def main() -> None:
    import argparse
    import glob
    import json
    from pathlib import Path

    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--captures", default="src/ssa/results/multictx8k/*.pt")
    p.add_argument("--step-stride", type=int, default=32)
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
                for (name, b), (e, dn, rd, nc) in r.items():
                    a = acc.setdefault((L, name, b), {}).setdefault(fi, [0.0, 0.0, 0.0, 0, 0.0])
                    a[0] += e
                    a[1] += dn
                    a[2] += rd
                    a[3] += 1
                    a[4] += nc
        print(f"done {Path(f).name}", flush=True)
    layers = sorted({k[0] for k in acc})
    names = sorted({k[1] for k in acc})
    ctxs = range(len(files))

    def curve(L, nm, c):
        pts = sorted((acc[(L, nm, b)][c][2] / acc[(L, nm, b)][c][3], acc[(L, nm, b)][c][0] / acc[(L, nm, b)][c][1])
                     for b in BUDGETS)
        return [x for x, _ in pts], [y for _, y in pts]

    boot = torch.randint(0, len(files), (4000, len(files)), generator=torch.Generator().manual_seed(0))
    summary = []
    for L in layers:
        for nm in names:
            b0 = BUDGETS[0]
            row = {"layer": L, "condition": nm, "ratio_to_fresh": {},
                   "clusters": sum(acc[(L, nm, b0)][c][4] / acc[(L, nm, b0)][c][3] for c in ctxs) / len(files)}
            for x in TARGETS:
                em = [interp_log(*curve(L, nm, c), x) for c in ctxs]
                eb = [interp_log(*curve(L, "fresh", c), x) for c in ctxs]
                if any(v is None for v in em + eb):
                    row["ratio_to_fresh"][str(x)] = None
                    continue
                a, b = torch.tensor(em), torch.tensor(eb)
                r = a[boot].sum(1) / b[boot].sum(1)
                row["ratio_to_fresh"][str(x)] = [float(a.sum() / b.sum()), float(r.quantile(0.025)), float(r.quantile(0.975))]
            summary.append(row)
    payload = stamp({"kind": "refresh_replay", "captures": [Path(f).name for f in files], "budgets": BUDGETS,
                     "G": GS, "schemes": SCHEMES, "block": 320, "per_cluster": 32, "window": 64,
                     "step_stride": args.step_stride, "summary": summary})
    out = Path("src/ssa/results") / f"refresh_replay_{payload['git_sha'][:8]}.json"
    out.write_text(json.dumps(payload, indent=1))
    for r in summary:
        cells = "  ".join(f"{float(x):.0%}: " + (f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" if v else "—")
                          for x, v in r["ratio_to_fresh"].items())
        print(f"L{r['layer']:>2} {r['condition']:12s} C={r['clusters']:6.0f}  {cells}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
