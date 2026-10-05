"""Plot fidelity versus reads: TVD from exact vs percent of K+V rows read.

Reads one or more stamped ``tvd_*.json`` files (from ``ssa.harness.accept_sweep``),
keeps ``cluster_skip`` and ``cluster_tail`` (skipped regions estimated; its drop control
is left out) at one region count and ``santa_sys``, and draws each with its
95% bootstrap CI over chunks. Lower-left is better (fewer reads, closer to exact).

Run:
    uv run python -m ssa.harness.plot_tvd A.json [B.json ...] --regions 256 --out tvd.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..attn import canonical

LABELS = {"fitted": "this method", "cluster_skip": "fixed random directions",
          "cluster_tail": "fixed random directions + tail estimate", "santa_sys": "systematic sampling"}


def build_series(paths, regions: int = 256, only=None) -> dict[str, dict[str, list[float]]]:
    """Collect (reads %, TVD, CI) per method, sorted by reads. ``fitted`` is the fused kernels with
    fitted (k-means) centroids; ``only`` keeps the named series."""
    rows: dict[str, list[tuple[float, float, float, float]]] = {k: [] for k in LABELS}
    for p in paths:
        for r in json.loads(Path(p).read_text())["summary"]:
            impl = canonical(r["impl"])
            if impl == "cluster_fused":
                if r["cfg"].get("partition") != "kmeans" or r["cfg"].get("C") != regions:
                    continue
                impl = "fitted"
            if impl not in rows or "tvd_ci" not in r:
                continue
            if impl in ("cluster_skip", "cluster_tail") and r["cfg"].get("C") != regions:
                continue
            if impl == "cluster_tail" and r["cfg"].get("order") == "drop":       # matched control, not a series
                continue
            lo, hi = r["tvd_ci"]
            rows[impl].append((round(100 * r["kv_read_fraction"], 1), r["tvd"], lo, hi))
    out = {}
    for impl, pts in rows.items():
        if only is not None and impl not in only:
            continue
        pts.sort()
        out[impl] = {
            "x": [p[0] for p in pts],
            "y": [p[1] for p in pts],
            "lo": [p[2] for p in pts],
            "hi": [p[3] for p in pts],
        }
    return out


def plot_series(series, out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(4.4, 3.3), dpi=200)
    style = {"fitted": ("#1f5fbf", "o", "-"), "cluster_skip": ("#7f8c8d", "D", "-"), "cluster_tail": ("#2a9d8f", "^", "--"),
             "santa_sys": ("#c0392b", "s", "none")}
    for impl, s in series.items():
        if not s["x"]:
            continue
        c, m, ls = style[impl]
        yerr = [[y - lo for y, lo in zip(s["y"], s["lo"])], [hi - y for y, hi in zip(s["y"], s["hi"])]]
        ax.errorbar(s["x"], s["y"], yerr=yerr, color=c, marker=m, ms=4, ls=ls, lw=1.5,
                    capsize=2, elinewidth=1, label=LABELS[impl])
    ax.set_xlabel("Key and value rows read (%)")
    ax.set_ylabel("TVD from exact model")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="lower left" if series.get("fitted", {}).get("x") else "upper right", fontsize=9 if series.get("cluster_tail", {}).get("x") else None)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--regions", type=int, default=256)
    ap.add_argument("--out", default="tvd.png")
    ap.add_argument("--only", nargs="+", default=None, help="series to draw (fitted, cluster_skip, cluster_tail, santa_sys)")
    args = ap.parse_args()
    plot_series(build_series(args.paths, args.regions, args.only), args.out)


if __name__ == "__main__":
    main()
