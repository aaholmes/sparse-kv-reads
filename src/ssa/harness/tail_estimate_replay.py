"""Offline check of estimating the dropped bins.

On captured Qwen3-4B tensors (``capture_qkv --n-contexts``), for sampled decode steps and all
KV heads: build ``SphereIndexTail`` bins, choose the top bins by the shared rule, then compare
dropping the rest with the zero- and first-order estimates of their contribution. ``t0`` per KV
head is the median ``t_b`` over non-empty bins at the previous step's query. Reports, per layer
and budget, the relative attention-output error ``Σ‖out − exact‖² / Σ‖exact‖²`` with a 95%
bootstrap interval over contexts, the paired error ratio to dropping, the spread of
``log(Ẑ_tail / Z_tail)``, and K+V reads including the new summaries.

Run:
    uv run python -m ssa.harness.tail_estimate_replay --captures '<dir>/*.pt'
"""

from __future__ import annotations

import math

import torch

from ..attn.labeled import label_weighted_attention
from ..attn.tail_estimate import SphereIndexTail, attend_with_tail, tail_log_estimates

BUDGETS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
ESTIMATE_BUDGETS = (0.05, 0.1, 0.2, 0.4)
# Estimates from the stored sums (t0 = 0), and diagnostics that compute an approximation
# exactly from the keys: "dir" keeps each key's length but gives it the bin's mean-direction
# cosine (what the first-order estimate expands), "dir2" uses the second-order cosine model
# with exact lengths, "len" keeps each key's cosine but gives it the bin's mean length.
ESTIMATES = {"order1": dict(order=1), "order2": dict(order=2), "order2_noshrink": dict(order=2, shrink=False)}
ORACLES = ("oracle_dir", "oracle_dir2", "oracle_len")


def _oracle_logits(idx: SphereIndexTail, q, K, dropped_pos, which: str) -> torch.Tensor:
    """Approximate log-weights ``[H, n]`` of the positions in dropped bins (−∞ elsewhere)."""
    H, d = q.shape
    H_kv, C = idx.H_kv, idx.C
    G = H // H_kv
    scale = 1.0 / math.sqrt(d)
    n = K.shape[1]
    lab = idx.labels[:, :n].long().clamp(max=C - 1)                         # exact-set label C unused here
    Kr = K - idx.mu_ref.unsqueeze(1)
    m = Kr.norm(dim=-1)                                                      # [H_kv, n]
    khat = Kr / m.clamp_min(1e-12).unsqueeze(-1)
    cdir = idx.sum_dir / idx.sum_dir.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    qv = q.view(H_kv, G, d)
    cos_b = torch.einsum("hgd,hcd->hgc", qv, cdir) / qv.norm(dim=-1, keepdim=True)   # q̂·ĉ_b
    cos_bj = torch.gather(cos_b, 2, lab.unsqueeze(1).expand(H_kv, G, n))     # per position
    sq = (qv.norm(dim=-1) * scale).unsqueeze(-1)                             # [H_kv, G, 1]
    mj = m.unsqueeze(1)
    if which == "oracle_dir":
        ell = sq * cos_bj * mj
    elif which == "oracle_dir2":
        rho = (idx.sum_dir.norm(dim=-1) / idx.count.clamp_min(1)).clamp(max=1.0)
        rho_j = torch.gather(rho, 1, lab).unsqueeze(1)
        v = (1 - rho_j ** 2) * (1 - cos_bj ** 2).clamp_min(0) / (d - 1)
        ell = sq * rho_j * cos_bj * mj + 0.5 * sq ** 2 * v * mj ** 2
    else:
        lsum = torch.zeros(H_kv, C, dtype=K.dtype).scatter_add_(1, lab[:, 1:idx.end], m[:, 1:idx.end])
        mean_len = torch.gather(lsum / idx.count.clamp_min(1), 1, lab).unsqueeze(1)
        ell = torch.einsum("hgd,hnd->hgn", qv, khat) * scale * mean_len
    shift = torch.einsum("hgd,hd->hg", qv, idx.mu_ref).unsqueeze(-1) * scale
    ell = (ell + shift).reshape(H, n)
    return torch.where(dropped_pos, ell, torch.full_like(ell, -math.inf))


def eval_step(q, K, V, n: int, *, C: int = 256, window: int = 64) -> dict:
    """``q [H, d]``, ``K, V [H_kv, n, d]`` float64. Per (method, budget): error sums, reads,
    and log(Ẑ_tail/Z_tail) per query head."""
    H, d = q.shape
    H_kv = K.shape[0]
    G = H // H_kv
    idx = SphereIndexTail(C=C, window=window, delta=math.inf, capacity=n)
    idx.observe(K, V, n)
    s = torch.einsum("hd,hnd->hn", q, K.repeat_interleave(G, 0)) / math.sqrt(d)
    Vx = V.repeat_interleave(G, 0)
    exact = torch.einsum("hn,hnd->hd", torch.softmax(s, -1), Vx)
    den = float((exact ** 2).sum())
    summary = C * (1 + 2 / d) / (2 * n)                   # stored directions + lengths + count
    extra = C * (1 + 3 / d) / (2 * n)                     # value sums + three scalars per bin
    out: dict = {}
    for b in BUDGETS:
        labels, w = idx.labels_and_weights(q, n=n, budget=b)
        wr = torch.gather(w.repeat_interleave(G, 0), 1, labels.long().repeat_interleave(G, 0))
        rows = (torch.gather(w, 1, labels.long()) > 0).sum(1).double().mean()
        reads = float(2 * rows / (2 * n)) + summary
        o_drop = label_weighted_attention(q, K, V, labels, w)
        out[("drop", b)] = {"err": float(((o_drop - exact) ** 2).sum()), "den": den, "reads": reads}
        if b not in ESTIMATE_BUDGETS:
            continue
        read = wr > 0
        dropped_pos = ~read
        ztail = torch.logsumexp(s.masked_fill(read, -math.inf), -1)          # exact log Z of dropped rows
        for m, kw in ESTIMATES.items():
            o = attend_with_tail(idx, q, K, V, labels, w, **kw)
            logz, _, _ = tail_log_estimates(idx, q, w, **kw)
            out[(m, b)] = {"err": float(((o - exact) ** 2).sum()), "den": den, "reads": reads + extra,
                           "log_ratio": (torch.logsumexp(logz, -1) - ztail).tolist()}
        for m in ORACLES:
            ell = _oracle_logits(idx, q, K, dropped_pos, m)
            logits = torch.where(read, s, ell)
            o = torch.einsum("hn,hnd->hd", torch.softmax(logits, -1), Vx)
            out[(m, b)] = {"err": float(((o - exact) ** 2).sum()), "den": den, "reads": reads + extra,
                           "log_ratio": (torch.logsumexp(ell, -1) - ztail).tolist()}
    return out


def _boot(num, den, reps=1000, seed=0):
    """95% bootstrap interval of Σnum/Σden, resampling contexts."""
    g = torch.Generator().manual_seed(seed)
    num, den = torch.tensor(num), torch.tensor(den)
    i = torch.randint(0, len(num), (reps, len(num)), generator=g)
    r = num[i].sum(1) / den[i].sum(1)
    return [float(r.quantile(0.025)), float(r.quantile(0.975))]


def main() -> None:
    import argparse
    import glob
    import json
    from pathlib import Path

    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--captures", default="src/ssa/results/multictx/*.pt")
    p.add_argument("--step-stride", type=int, default=16)
    p.add_argument("--tag", default="", help="appended to the output file name")
    args = p.parse_args()
    files = sorted(glob.glob(args.captures))
    if not files:
        raise SystemExit(f"no captures match {args.captures}")
    per: dict = {}                                        # (layer, method, budget) -> per-context sums
    for fi, f in enumerate(files):
        cap = torch.load(f, weights_only=False)
        P, layers = cap["prefill"], cap["layers"]
        T = cap["q"].shape[1]
        for li, L in enumerate(layers):
            for t in range(1, T, args.step_stride):
                n = P + t + 1
                K = cap["k"][li, :n].permute(1, 0, 2).double()
                V = cap["v"][li, :n].permute(1, 0, 2).double()
                r = eval_step(cap["q"][li, t].double(), K, V, n)
                for (m, b), v in r.items():
                    a = per.setdefault((L, m, b), {}).setdefault(fi, {"err": 0.0, "den": 0.0, "reads": 0.0,
                                                                      "steps": 0, "log_ratio": []})
                    a["err"] += v["err"]
                    a["den"] += v["den"]
                    a["reads"] += v["reads"]
                    a["steps"] += 1
                    a["log_ratio"] += v.get("log_ratio", [])
        print(f"done {Path(f).name}", flush=True)

    summary = []
    for (L, m, b), ctx in sorted(per.items()):
        fis = sorted(ctx)
        num = [ctx[i]["err"] for i in fis]
        den = [ctx[i]["den"] for i in fis]
        row = {"layer": L, "method": m, "budget": b, "rel_err": sum(num) / sum(den), "rel_err_ci": _boot(num, den),
               "reads": sum(ctx[i]["reads"] for i in fis) / sum(ctx[i]["steps"] for i in fis)}
        if m != "drop":
            dn = [per[(L, "drop", b)][i]["err"] for i in fis]
            row["ratio_to_drop"] = sum(num) / sum(dn)
            row["ratio_to_drop_ci"] = _boot(num, dn)
            lr = torch.tensor([x for i in fis for x in ctx[i]["log_ratio"]])
            row["zhat_over_z_quantiles"] = {str(qq): float(lr.quantile(qq).exp()) for qq in (0.1, 0.5, 0.9)}
        summary.append(row)
    payload = stamp({"kind": "tail_estimate_replay", "captures": [Path(f).name for f in files],
                     "step_stride": args.step_stride, "C": 256, "window": 64, "summary": summary})
    out = Path("src/ssa/results") / f"tail_estimate_replay{'_' + args.tag if args.tag else ''}_{payload['git_sha'][:8]}.json"
    out.write_text(json.dumps(payload, indent=1))
    for r in summary:
        extra = ""
        if "ratio_to_drop" in r:
            qz = r["zhat_over_z_quantiles"]
            extra = (f"  vs drop {r['ratio_to_drop']:.2f} [{r['ratio_to_drop_ci'][0]:.2f}, {r['ratio_to_drop_ci'][1]:.2f}]"
                     f"  Ẑ/Z p10/50/90 {qz['0.1']:.2f}/{qz['0.5']:.2f}/{qz['0.9']:.2f}")
        print(f"L{r['layer']:>2} {r['method']:14s} b={r['budget']:.2f} reads {r['reads']:.3f} "
              f"err {r['rel_err']:.2e} [{r['rel_err_ci'][0]:.2e}, {r['rel_err_ci'][1]:.2e}]{extra}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
