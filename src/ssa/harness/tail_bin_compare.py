"""Per-bin comparison of the dropped bins' sums.

For every dropped, non-empty bin at a 10% budget (Qwen3-4B captures, every 16th decode step,
all query heads): the exact ``Z_b = Σ e^{s_i}``, ``N_b = Σ e^{s_i} v_i`` and mean value ``N_b/Z_b``,
compared with (a) the formula ``Σ e^{t_b|k_i|}`` and ``Σ e^{t_b|k_i|} v_i`` computed exactly from each
key's length (bin-direction cosine only), and (b) the stored-sum estimate (first order, ``t0 = 0``:
``N̂_b = Ẑ_b · mean value at t0``). Also the same three for the whole tail of each query head.
Quantiles over bins are weighted by each bin's share of its head's exact tail mass.

Run:
    uv run python -m ssa.harness.tail_bin_compare --captures '<dir>/*.pt'
"""

from __future__ import annotations

import math

import torch

from ..attn.tail_estimate import SphereIndexTail, tail_log_estimates


def _wq(x: torch.Tensor, w: torch.Tensor, qs=(0.1, 0.5, 0.9)) -> list[float]:
    """Weighted quantiles of ``x``."""
    o = x.argsort()
    cw = w[o].cumsum(0) / w.sum()
    return [float(x[o][int((cw < q).sum().clamp(max=len(x) - 1))]) for q in qs]


def compare_step(q, K, V, n, *, budget=0.1, C=256, window=64) -> dict:
    H, d = q.shape
    H_kv = K.shape[0]
    G = H // H_kv
    scale = 1.0 / math.sqrt(d)
    idx = SphereIndexTail(C=C, window=window, delta=math.inf, capacity=n)
    idx.observe(K, V, n)
    labels, w = idx.labels_and_weights(q, n=n, budget=budget)
    logz, vbar, dropped = tail_log_estimates(idx, q, w, order=1)            # [H, C], [H, C, d], [H, C]
    lab = labels[:, :n].long().repeat_interleave(G, 0)                        # [H, n]
    read = torch.gather(w.repeat_interleave(G, 0), 1, lab) > 0
    Kx, Vx = K.repeat_interleave(G, 0), V.repeat_interleave(G, 0)
    s = torch.einsum("hd,hnd->hn", q, Kx) * scale
    # formula with exact lengths: t_b |k_i| + shift, t_b = q·ĉ_b/√d
    mu = idx.mu_ref.repeat_interleave(G, 0)
    m = (Kx - mu.unsqueeze(1)).norm(dim=-1)
    cdir = idx.sum_dir / idx.sum_dir.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    t = torch.einsum("hgd,hcd->hgc", q.view(H_kv, G, d), cdir).reshape(H, C) * scale
    shift = (q * mu).sum(-1, keepdim=True) * scale
    labc = lab.clamp(max=C - 1)
    f = torch.gather(t, 1, labc) * m + shift
    tail = ~read
    ref = s.masked_fill(read, -math.inf).max(1, keepdim=True).values          # common scale per head
    e_exact = torch.where(tail, (s - ref).exp(), torch.zeros_like(s))
    e_form = torch.where(tail, (f - ref).exp(), torch.zeros_like(s))

    def per_bin(e):
        Z = torch.zeros(H, C, dtype=s.dtype).scatter_add_(1, labc, e)
        N = torch.zeros(H * C, d, dtype=s.dtype).index_add_(0, (labc + torch.arange(H).unsqueeze(1) * C).reshape(-1),
                                                             (e.unsqueeze(-1) * Vx).reshape(-1, d)).view(H, C, d)
        return Z, N

    Ze, Ne = per_bin(e_exact)
    Zf, Nf = per_bin(e_form)
    Zs = torch.where(dropped, (logz - ref).exp(), torch.zeros_like(logz))
    Ns = Zs.unsqueeze(-1) * vbar
    keep = dropped & (Ze > 0)
    wt = (Ze / Ze.sum(1, keepdim=True).clamp_min(1e-300))[keep]

    def stats(Z, N):
        vb_e, vb = Ne[keep] / Ze[keep].unsqueeze(-1), N[keep] / Z[keep].clamp_min(1e-300).unsqueeze(-1)
        return {"log10_Z_ratio": (Z[keep].clamp_min(1e-300) / Ze[keep]).log10(),
                "N_rel_err": (N[keep] - Ne[keep]).norm(dim=-1) / Ne[keep].norm(dim=-1).clamp_min(1e-300),
                "mean_rel_err": (vb - vb_e).norm(dim=-1) / vb_e.norm(dim=-1).clamp_min(1e-300),
                "mean_cos": torch.nn.functional.cosine_similarity(vb, vb_e, dim=-1)}

    def tail_stats(Z, N):
        Zt, Nt, Zte, Nte = Z.sum(1), N.sum(1), Ze.sum(1), Ne.sum(1)
        return {"log10_Z_ratio": (Zt / Zte).log10(),
                "N_rel_err": (Nt - Nte).norm(dim=-1) / Nte.norm(dim=-1),
                "mean_rel_err": (Nt / Zt.unsqueeze(-1) - Nte / Zte.unsqueeze(-1)).norm(dim=-1)
                / (Nte / Zte.unsqueeze(-1)).norm(dim=-1),
                "mean_cos": torch.nn.functional.cosine_similarity(Nt / Zt.unsqueeze(-1), Nte / Zte.unsqueeze(-1), dim=-1)}

    return {"w": wt, "formula": stats(Zf, Nf), "stored": stats(Zs, Ns),
            "tail_formula": tail_stats(Zf, Nf), "tail_stored": tail_stats(Zs, Ns)}


def main() -> None:
    import argparse
    import glob
    import json
    from pathlib import Path

    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--captures", default="src/ssa/results/multictx/*.pt")
    p.add_argument("--step-stride", type=int, default=16)
    p.add_argument("--budget", type=float, default=0.1)
    args = p.parse_args()
    files = sorted(glob.glob(args.captures))
    acc: dict = {}
    for f in files:
        cap = torch.load(f, weights_only=False)
        P, layers = cap["prefill"], cap["layers"]
        for li, L in enumerate(layers):
            for t in range(1, cap["q"].shape[1], args.step_stride):
                n = P + t + 1
                r = compare_step(cap["q"][li, t].double(), cap["k"][li, :n].permute(1, 0, 2).double(),
                                 cap["v"][li, :n].permute(1, 0, 2).double(), n, budget=args.budget)
                a = acc.setdefault(L, {})
                a.setdefault("w", []).append(r["w"])
                for kind in ("formula", "stored", "tail_formula", "tail_stored"):
                    for k, v in r[kind].items():
                        a.setdefault((kind, k), []).append(v)
        print(f"done {Path(f).name}", flush=True)
    summary = {}
    for L, a in acc.items():
        w = torch.cat(a["w"])
        row = {}
        for (kind, k), vs in ((key, v) for key, v in a.items() if key != "w"):
            x = torch.cat(vs)
            if kind.startswith("tail"):
                row[f"{kind}:{k}"] = [float(x.quantile(q)) for q in (0.1, 0.5, 0.9)]
            else:
                row[f"{kind}:{k}"] = _wq(x, w)
        summary[str(L)] = row
    payload = stamp({"kind": "tail_bin_compare", "captures": [Path(f).name for f in files], "budget": args.budget,
                     "step_stride": args.step_stride, "summary": summary,
                     "note": "per-bin quantiles weighted by exact tail-mass share; tail rows unweighted over query heads"})
    out = Path("src/ssa/results") / f"tail_bin_compare_{payload['git_sha'][:8]}.json"
    out.write_text(json.dumps(payload, indent=1))
    for L, row in summary.items():
        print(f"== layer {L}  (p10 / p50 / p90)")
        for k, v in row.items():
            print(f"  {k:32s} {v[0]:+.3f} / {v[1]:+.3f} / {v[2]:+.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
