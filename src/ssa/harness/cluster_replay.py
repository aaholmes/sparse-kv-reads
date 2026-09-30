"""Offline cluster-sampling replay: error vs total K+V reads on captured real-model tensors.

The cheap half of the cluster-sampling replay. Reads a ``capture_qkv`` file and, per
layer and KV head, clusters the prefill keys once (the decode tokens are the recent
window, read exactly), then at every decode step compares:

  - ``santa_sys``      reads every key + the distinct sampled value rows;
  - ``cs_drop``        top-h clusters by the magnitude estimate, tail dropped (biased);
  - ``cluster_sample`` top-h head + S importance-sampled tail cluster draws (ratio estimator);
  - ``cs_oracle``      as ``cluster_sample`` but ranked/sampled by exact cluster masses
                       (unrealizable reference: a perfect read-free estimate).

Reads are per KV head (union over its query-head group), counted as
``(key rows + value rows + summary overhead) / (2 n)``; the overhead for the cluster
methods is one centroid row per cluster plus ``n_cl/d`` rows of per-key lengths.
Error per step is ``Σ_heads E‖out − dense‖²``, split into bias² and variance from
``R`` draws; the layer's relative error is the ratio of sums over steps and heads,
bootstrapped over decode steps.

Run:
    uv run python -m ssa.harness.cluster_replay --layers 0 12 24 35
"""

from __future__ import annotations

import math
import zlib

import torch

from ..attn.cluster_sample import build_clusters, certified_head, cluster_sample
from ..sampling.draws import systematic_indices, unique_counts


def cfg_name(impl: str, cfg: dict) -> str:
    return impl + "|" + ",".join(f"{k}={v}" for k, v in cfg.items())


def _err_terms(out: torch.Tensor, ref: torch.Tensor):
    """``out [R, G, d]``, ``ref [G, d]`` -> (bias², variance) summed over heads."""
    R = out.shape[0]
    mean = out.mean(0)
    if R > 1:
        var = ((out - mean) ** 2).sum(-1).sum(0) / (R - 1)          # [G]
        bias2 = ((mean - ref) ** 2).sum(-1) - var / R
    else:
        var = torch.zeros(out.shape[1], dtype=out.dtype)
        bias2 = ((mean - ref) ** 2).sum(-1)
    return float(bias2.sum()), float(var.sum())


def replay_layer(q, K, V, *, prefill: int, scale: float, configs, R: int = 64, B: int = 8,
                 alpha: float = 0.1, seed: int = 0) -> dict:
    """``q [T, H, d]``, ``K, V [prefill+T, H_kv, d]`` -> per-config per-step sums."""
    q, K, V = (t.to(torch.float64) for t in (q, K, V))
    T, H, d = q.shape
    H_kv = K.shape[1]
    G = H // H_kv
    names = [cfg_name(i, c) for i, c in configs]
    acc = {nm: {"bias2": [0.0] * T, "var": [0.0] * T, "reads": [0.0] * T} for nm in names}
    ref2 = [0.0] * T
    cert = {"frac_clusters": 0.0, "frac_keys": 0.0}

    for hkv in range(H_kv):
        st = build_clusters(K[:prefill, hkv], B=B, seed=seed + hkv)
        overhead = st.n_clusters + prefill / d
        for t in range(T):
            n = prefill + t + 1
            Kt, Vt = K[:n, hkv], V[:n, hkv]
            qg = q[t, hkv * G:(hkv + 1) * G]
            A = torch.softmax((qg @ Kt.t()) * scale, dim=-1)       # [G, n]
            ref = A @ Vt
            ref2[t] += float((ref ** 2).sum())
            cm = certified_head(qg, st, scale).to(torch.float64)
            cert["frac_clusters"] += float(cm.mean()) / (H_kv * T)
            cert["frac_keys"] += float((cm * st.sizes).sum(1).mean() / prefill) / (H_kv * T)

            for (impl, cfg), nm in zip(configs, names):
                gen = torch.Generator().manual_seed(
                    zlib.crc32(f"{seed}|{hkv}|{t}|{nm}".encode()))
                if impl == "santa_sys":
                    S = cfg["S"]
                    idx = systematic_indices(A.repeat(R, 1), S, generator=gen).view(R, G, S)
                    out = Vt[idx].mean(2)                            # [R, G, d]
                    vrows = unique_counts(idx.reshape(R, G * S)).double().mean()
                    reads = (n + float(vrows)) / (2 * n)
                else:
                    S = cfg.get("S", 0) if impl in ("cluster_sample", "cs_oracle") else 0
                    prop = "oracle" if impl == "cs_oracle" else "mag"
                    out, kread = cluster_sample(qg, Kt, Vt, st, h=cfg["h"], S=S,
                                                R=R if S > 0 else 1, alpha=alpha,
                                                generator=gen, scale=scale, proposal=prop)
                    rows = kread.any(1).sum(-1).double().mean()     # union over the group
                    reads = (2 * float(rows) + overhead) / (2 * n)
                b2, var = _err_terms(out, ref)
                a = acc[nm]
                a["bias2"][t] += b2
                a["var"][t] += var
                a["reads"][t] += reads / H_kv
    for a in acc.values():
        a["mse"] = [b + v for b, v in zip(a["bias2"], a["var"])]
    return {"configs": acc, "ref2": ref2, "certified_head": cert,
            "shape": {"T": T, "H": H, "H_kv": H_kv, "d": d, "prefill": prefill},
            "R": R, "B": B, "alpha": alpha}


def reads_to_reach(points, target: float) -> float:
    """Smallest read fraction at which the Pareto frontier of ``(reads, err)`` reaches ``target``.

    Interpolates log-err linearly in log-reads between frontier points; ``inf`` if never.
    """
    pts = sorted(points)
    front, best = [], math.inf
    for r, e in pts:
        if e < best:
            front.append((r, e))
            best = e
    if front[0][1] <= target:
        return front[0][0]
    for (r0, e0), (r1, e1) in zip(front, front[1:]):
        if e1 <= target:
            f = (math.log(target) - math.log(e0)) / (math.log(e1) - math.log(e0))
            return math.exp(math.log(r0) + f * (math.log(r1) - math.log(r0)))
    return math.inf


def summarize_layer(res: dict, *, n_boot: int = 1000, seed: int = 0,
                    ref_cfg: str = "santa_sys|S=256", family: str = "cluster_sample") -> dict:
    """Per-config relative MSE / bias share / reads with bootstrap CIs over decode steps."""
    T = len(res["ref2"])
    ref2 = torch.tensor(res["ref2"])
    arrs = {nm: {k: torch.tensor(v) for k, v in a.items()} for nm, a in res["configs"].items()}
    g = torch.Generator().manual_seed(seed)
    boots = torch.randint(0, T, (n_boot, T), generator=g)

    def stat(nm, idx):
        a = arrs[nm]
        den = ref2[idx].sum(-1)
        return (a["mse"][idx].sum(-1) / den, a["bias2"][idx].sum(-1) / a["mse"][idx].sum(-1).clamp_min(1e-300),
                a["reads"][idx].mean(-1))

    out = {"configs": {}}
    full = torch.arange(T).unsqueeze(0)
    for nm in arrs:
        e, bshare, r = (x[0] for x in stat(nm, full))
        eb, bb, rb = stat(nm, boots)
        q = lambda x: [float(x.quantile(0.025)), float(x.quantile(0.975))]
        out["configs"][nm] = {"rel_mse": float(e), "rel_mse_ci": q(eb),
                              "bias_share": float(bshare), "bias_share_ci": q(bb),
                              "reads": float(r), "reads_ci": q(rb)}

    fam = [nm for nm in arrs if nm.startswith(family + "|")]
    if ref_cfg in arrs and fam:
        def ratio(idx_row):
            idx = idx_row.unsqueeze(0)
            e_ref, _, r_ref = (float(x[0]) for x in stat(ref_cfg, idx))
            pts = [(float(stat(nm, idx)[2][0]), float(stat(nm, idx)[0][0])) for nm in fam]
            return reads_to_reach(pts, e_ref) / r_ref
        point = ratio(torch.arange(T))
        bs = torch.tensor([ratio(b) for b in boots])
        finite = torch.isfinite(bs)
        out["reads_to_match_ref"] = {
            "ref": ref_cfg, "family": family, "ratio": point,
            "ci": ([float(bs[finite].quantile(0.025)), float(bs[finite].quantile(0.975))]
                   if finite.all() else None),
            "frac_boot_unreached": float((~finite).double().mean()),
        }
    out["certified_head"] = res["certified_head"]
    return out


DEFAULT_CONFIGS = (
    [("santa_sys", {"S": S}) for S in (16, 32, 64, 128, 256, 512)]
    + [("cs_drop", {"h": h}) for h in (2, 4, 8, 16, 32, 64, 128)]
    + [("cluster_sample", {"h": h, "S": S}) for h in (0, 4, 16, 64) for S in (4, 16, 64, 128)]
    + [("cs_oracle", {"h": h, "S": S}) for h in (0, 4, 16) for S in (4, 16, 64)]
)


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--capture", default=None)
    p.add_argument("--layers", type=int, nargs="+", default=[0, 12, 24, 35])
    p.add_argument("--R", type=int, default=64)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--tag", default="")
    args = p.parse_args()

    res_dir = Path(__file__).resolve().parent.parent / "results"
    cap_path = Path(args.capture) if args.capture else res_dir / "qkv_capture_p2048_s64.pt"
    cap = torch.load(cap_path, weights_only=False)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    layers = {}
    for L in args.layers:
        res = replay_layer(cap["q"][L], cap["k"][L], cap["v"][L], prefill=cap["prefill"],
                           scale=cap["scale"], configs=DEFAULT_CONFIGS, R=args.R, B=args.B,
                           alpha=args.alpha, seed=L)
        summ = summarize_layer(res, n_boot=args.n_boot, seed=L)
        layers[str(L)] = {"raw": res, "summary": summ}
        m = summ.get("reads_to_match_ref", {})
        ref = summ["configs"]["santa_sys|S=256"]
        for fam in ("cs_drop", "cs_oracle"):
            o = summarize_layer(res, n_boot=args.n_boot, seed=L, family=fam)
            summ[f"reads_to_match_ref_{fam}"] = o.get("reads_to_match_ref")
        print(f"layer {L:>2}: sys256 rel_mse {ref['rel_mse']:.2e} reads {ref['reads']:.3f} | "
              f"cluster_sample reads ratio {m.get('ratio', float('nan')):.3f} ci {m.get('ci')} "
              f"| cs_drop {summ['reads_to_match_ref_cs_drop']['ratio']:.3f} "
              f"| oracle {summ['reads_to_match_ref_cs_oracle']['ratio']:.3f} "
              f"| certified head {summ['certified_head']['frac_keys']:.3f} of keys", flush=True)

    payload = stamp({"kind": "cluster_replay", "capture": str(cap_path),
                     "capture_meta": cap.get("meta"), "layers_run": args.layers, "R": args.R,
                     "B": args.B, "alpha": args.alpha, "n_boot": args.n_boot,
                     "configs": [cfg_name(i, c) for i, c in DEFAULT_CONFIGS], "layers": layers})
    path = res_dir / f"cluster_replay{args.tag}_{payload['git_sha'][:8]}.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
