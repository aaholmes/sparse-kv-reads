"""Offline check of sampling the unselected bins.

On captured Qwen3-4B tensors (``capture_qkv --n-contexts``), for sampled decode steps and
all KV heads: build the bins (``SphereIndexGPU``), choose head bins by the shared rule, then
compare, at matched K+V reads,
  - top-k: head only (budget b), and
  - head (budget h) + S sampled tail bins (``tail_sample_weights``, floor α), R draws,
using the label-weighted reference attention. Reports relative attention-output error
``Σ E‖out − dense‖² / Σ ‖dense‖²`` split into bias² and variance, and mean reads.

Run:
    uv run python -m ssa.harness.tail_replay
"""

from __future__ import annotations

import math

import torch

from ..attn.labeled import label_weighted_attention
from ..attn.sphere_gpu import SphereIndexGPU
from ..attn.tail_sample import proposal_from_scores, tail_sample_weights


def scores(idx: SphereIndexGPU, q: torch.Tensor) -> torch.Tensor:
    """Max-score estimates ``e [H_kv, G, C]`` (as used for selection)."""
    H, d = q.shape
    G = H // idx.H_kv
    cdir = idx.sum_dir / idx.sum_dir.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    p = torch.einsum("hgd,hcd->hgc", q.view(idx.H_kv, G, d).to(idx.dtype), cdir)
    e = torch.where(p >= 0, idx.mmax.unsqueeze(1) * p, idx.mmin.unsqueeze(1) * p)
    return e.masked_fill((idx.count == 0).unsqueeze(1), -math.inf)


def eval_step(q, K, V, n, *, configs, R: int, seed: int) -> dict:
    """``q [H, d]``, ``K, V [H_kv, n, d]``. Returns per-config error sums and reads."""
    q64, K64, V64 = q.double(), K.double(), V.double()
    H, d = q.shape
    H_kv = K.shape[0]
    idx = SphereIndexGPU(C=256, window=64, delta=math.inf, capacity=n)
    idx.observe(K64, n)
    A = torch.softmax(torch.einsum("hd,hnd->hn", q64, K64.repeat_interleave(H // H_kv, 0)) / math.sqrt(d), -1)
    dense = torch.einsum("hn,hnd->hd", A, V64.repeat_interleave(H // H_kv, 0))
    den = float((dense ** 2).sum())
    e = scores(idx, q64)
    prop = proposal_from_scores(e, idx.count, scale=1.0 / math.sqrt(d))
    g = torch.Generator().manual_seed(seed)
    out = {}
    for key, (h, S, alpha) in configs.items():
        labels, w = idx.labels_and_weights(q64, n=n, budget=h, group="sum_share")
        draws = R if S > 0 else 1
        outs, reads = [], []
        for _ in range(draws):
            w2 = w.clone()
            if S > 0:
                w2[:, :idx.C] = tail_sample_weights(prop, idx.count, w[:, :idx.C] > 0, S=S, alpha=alpha, generator=g)
            outs.append(label_weighted_attention(q64, K64, V64, labels, w2))
            rows = (torch.gather(w2, 1, labels.long()) > 0).sum(1).double().mean()
            reads.append((2 * float(rows) + idx.C * (1 + 2 / d)) / (2 * n))
        o = torch.stack(outs)
        mean = o.mean(0)
        var = float(((o - mean) ** 2).sum() / max(draws - 1, 1)) if draws > 1 else 0.0
        bias2 = float(((mean - dense) ** 2).sum()) - (var / draws if draws > 1 else 0.0)
        out[key] = {"bias2": bias2, "var": var, "den": den, "reads": sum(reads) / len(reads)}
    return out


def default_configs() -> dict:
    cfg = {f"topk b={b}": (b, 0, 0.0) for b in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)}
    for h in (0.05, 0.1, 0.2):
        for S in (4, 8, 16, 32):
            for a in (0.1, 0.5):
                cfg[f"sample h={h} S={S} a={a}"] = (h, S, a)
    return cfg


def main() -> None:
    import argparse
    import glob
    import json
    from pathlib import Path

    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--captures", default="src/ssa/results/multictx/*.pt")
    p.add_argument("--step-stride", type=int, default=16)
    p.add_argument("--R", type=int, default=32)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    configs = default_configs()
    files = sorted(glob.glob(args.captures))
    tot: dict = {}
    for fi, f in enumerate(files):
        cap = torch.load(f, weights_only=False)
        P, layers = cap["prefill"], cap["layers"]
        T = cap["q"].shape[1]
        for li, L in enumerate(layers):
            for t in range(0, T, args.step_stride):
                n = P + t + 1
                r = eval_step(cap["q"][li, t].to(dev), cap["k"][li, :n].permute(1, 0, 2).to(dev),
                              cap["v"][li, :n].permute(1, 0, 2).to(dev), n, configs=configs, R=args.R,
                              seed=fi * 1000 + L * 10 + t)
                for k, v in r.items():
                    a = tot.setdefault(str(L), {}).setdefault(k, {"bias2": 0.0, "var": 0.0, "den": 0.0,
                                                                  "reads": 0.0, "m": 0})
                    for kk in ("bias2", "var", "den", "reads"):
                        a[kk] += v[kk]
                    a["m"] += 1
        print(f"done {Path(f).name}", flush=True)
    summary = {L: {k: {"rel_err": (a["bias2"] + a["var"]) / a["den"], "bias_share": a["bias2"] / max(a["bias2"] + a["var"], 1e-300),
                       "reads": a["reads"] / a["m"]} for k, a in cfgs.items()} for L, cfgs in tot.items()}
    payload = stamp({"kind": "tail_replay", "captures": files, "R": args.R, "step_stride": args.step_stride,
                     "configs": {k: list(v) for k, v in configs.items()}, "summary": summary})
    out = Path("src/ssa/results") / f"tail_replay_{payload['git_sha'][:8]}.json"
    out.write_text(json.dumps(payload, indent=1))
    for L, cfgs in summary.items():
        print(f"\n== layer {L}")
        for k, v in sorted(cfgs.items(), key=lambda kv: kv[1]["reads"]):
            print(f"  {k:28s} reads {v['reads']:.3f}  rel_err {v['rel_err']:.3e}  bias share {v['bias_share']:.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
