"""Compare two methods' end-to-end TVD at matched K+V reads, paired by chunk.

Reads an ``accept_sweep`` result file. For each method, each chunk's TVD is interpolated
log-linearly along the method's TVD-vs-reads curve (reads are per condition) to common read
fractions inside both curves' ranges; the ratio ``Σ TVD_a / Σ TVD_b`` over chunks is reported with
a 95% bootstrap interval over chunks.

Run:
    uv run python -m ssa.harness.matched_reads src/ssa/results/tvd_4b_8k_quest_vs_voronoi.json \\
        --a quest_matched --b voronoi_skip --at 0.17 0.23 0.27 0.37 0.42
"""

from __future__ import annotations

import json

import torch

from ..attn import canonical
from .quest_replay import interp_log


def curves(payload: dict, impl: str):
    """``reads [k]`` and per-chunk mean TVD ``[k, chunks]`` for ``impl``, sorted by reads."""
    rows = [r for r in payload["results"] if canonical(r["impl"]) == impl]
    rows.sort(key=lambda r: r["kv_read_fraction"])
    reads = [r["kv_read_fraction"] for r in rows]
    tvd = [[t / n for t, n in zip(r["chunk_tvd"], r["chunk_n"])] for r in rows]
    return reads, tvd


def ratio_at(payload: dict, a: str, b: str, x: float, *, reps: int = 4000, seed: int = 0):
    ra, ta = curves(payload, a)
    rb, tb = curves(payload, b)
    if not (ra[0] <= x <= ra[-1] and rb[0] <= x <= rb[-1]):
        return None
    n_chunks = len(ta[0])
    va = torch.tensor([interp_log(ra, [t[c] for t in ta], x) for c in range(n_chunks)], dtype=torch.float64)
    vb = torch.tensor([interp_log(rb, [t[c] for t in tb], x) for c in range(n_chunks)], dtype=torch.float64)
    i = torch.randint(0, n_chunks, (reps, n_chunks), generator=torch.Generator().manual_seed(seed))
    r = va[i].sum(1) / vb[i].sum(1)
    return float(va.mean()), float(vb.mean()), float(va.sum() / vb.sum()), float(r.quantile(0.025)), float(r.quantile(0.975))


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--at", type=float, nargs="+", required=True)
    args = p.parse_args()
    payload = json.load(open(args.path))
    for x in args.at:
        r = ratio_at(payload, args.a, args.b, x)
        if r is None:
            print(f"{x:.0%} reads: outside one of the curves")
            continue
        print(f"{x:.0%} reads: TVD {args.a} {r[0]:.4f}, {args.b} {r[1]:.4f}; ratio {r[2]:.2f} [{r[3]:.2f}, {r[4]:.2f}]")


if __name__ == "__main__":
    main()
