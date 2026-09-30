"""v1 update rule: how fast the mean moves, and the δ tradeoff.

(a) ``mean_moves``: stream the cache one key at a time (keys enter the bins when they leave
    the recent window) and record the per-token move of the running mean of binned keys,
    relative to the RMS centered-key length. One key moves the mean by ``(k − μ)/n``.
(b) ``delta_sweep``: stream the same keys through ``SphereState`` for several δ and record
    the recentering rate (per KV head per token) and, at the steps with captured queries,
    the attention-output error of shared selection at several budgets.

Scenarios on the captured contexts (queries exist only for the last 64 positions):
  - ``prefill2048``: bins built at the 2048-token prefill, then 64 decode steps;
  - ``prefill256``: bins built at 256 tokens, then every key streamed through to 2112
    (a short prompt followed by long generation), evaluated at the last 64 steps.

Run:
    uv run python -m ssa.harness.sphere_update
"""

from __future__ import annotations

import math

import torch

from ..attn.sphere_state import SphereState


def mean_moves(K: torch.Tensor, *, window: int, start: int) -> dict:
    """Per-token mean move for each KV head. ``K [n_end, H_kv, d]`` -> tensors over (token, head)."""
    K = K.to(torch.float64)
    n_end, H_kv, _ = K.shape
    Kb = K[1:]                                          # binned candidates (token 0 is exact)
    csum = Kb.cumsum(0)                                 # running sums
    ns, moves = [], []
    for n in range(start, n_end + 1):
        end = n - window                                # keys [1, end) binned at length n
        c = end - 1
        if c < 2:
            continue
        mu = csum[c - 1] / c
        mu_prev = csum[c - 2] / (c - 1)
        Kr = Kb[:c] - mu
        rbar = Kr.pow(2).sum(-1).mean(0).sqrt()        # [H_kv]
        moves.append(((mu - mu_prev).norm(dim=-1) / rbar))
        ns.append(torch.full((H_kv,), float(c), dtype=torch.float64))
    return {"n": torch.stack(ns).flatten(), "move_rel": torch.stack(moves).flatten()}


def delta_sweep(Q, K, V, *, start: int, eval_ns, deltas, budgets, C: int = 256, window: int = 64,
                group: str = "sum_share") -> dict:
    """``Q [n_end, H, d]`` (only rows ``n-1`` for ``n`` in ``eval_ns`` are used), ``K, V [n_end, H_kv, d]``."""
    Q, K, V = (x.to(torch.float64) for x in (Q, K, V))
    n_end, H_kv, d = K.shape
    H = Q.shape[1]
    G = H // H_kv
    eval_ns = set(eval_ns)
    out = {}
    for delta in deltas:
        st = SphereState(C=C, window=window, delta=delta)
        st.observe(K[:start])
        st.rebuilds, st.head_steps = 0, 0
        r = {f"b={b}": {"err_num": 0.0, "err_den": 0.0, "kv_rows": 0.0, "evals": 0} for b in budgets}
        for n in range(start + 1, n_end + 1):
            st.observe(K[:n])
            if n not in eval_ns:
                continue
            q = Q[n - 1]
            Kexp = K[:n].repeat_interleave(G, 1)
            A = torch.softmax(torch.einsum("hd,nhd->hn", q, Kexp) / math.sqrt(d), dim=-1)
            dense = torch.einsum("hn,nhd->hd", A, V[:n].repeat_interleave(G, 1))
            for b in budgets:
                o, info = st.read(q, K[:n], V[:n], budget=b, group=group)
                x = r[f"b={b}"]
                x["err_num"] += float(((o - dense) ** 2).sum())
                x["err_den"] += float((dense ** 2).sum())
                x["kv_rows"] += (2 * float(info.kv_union.double().mean()) + info.overhead_rows) / (2 * n)
                x["evals"] += 1
        for x in r.values():
            x["kv_frac"] = x["kv_rows"] / max(x["evals"], 1)
        r["rebuild_rate"] = st.rebuilds / max(st.head_steps, 1)
        r["rebuilds"] = st.rebuilds
        out[f"delta={delta}"] = r
    return out


def main() -> None:
    import argparse
    import glob
    import json
    from pathlib import Path

    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--captures", default="src/ssa/results/multictx/*.pt")
    p.add_argument("--stream-contexts", type=int, default=4, help="contexts for the prefill256 scenario")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    deltas = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, math.inf)
    budgets = (0.1, 0.2)
    files = sorted(glob.glob(args.captures))
    stream_files = files[::max(1, len(files) // args.stream_contexts)][:args.stream_contexts]

    moves: dict = {}
    sweeps: dict = {"prefill2048": {}, "prefill256": {}}
    for f in files:
        cap = torch.load(f, weights_only=False)
        P, layers = cap["prefill"], cap["layers"]
        T = cap["q"].shape[1]
        n_end = P + T
        for li, L in enumerate(layers):
            K = cap["k"][li].to(dev)
            V = cap["v"][li].to(dev)
            m = mean_moves(K, window=64, start=128)
            moves.setdefault(str(L), []).append({"n": m["n"].cpu(), "move_rel": m["move_rel"].cpu()})
            Qfull = torch.zeros(n_end, cap["q"].shape[2], cap["q"].shape[3], dtype=torch.float64, device=dev)
            Qfull[P:] = cap["q"][li].to(dev, torch.float64)     # query at position P+t is row P+t
            scen = [("prefill2048", P, range(P + 1, n_end + 1))]
            if f in stream_files:
                scen.append(("prefill256", 256, range(n_end - 63, n_end + 1, 4)))
            for name, start, ev in scen:
                r = delta_sweep(Qfull, K, V, start=start, eval_ns=ev, deltas=deltas, budgets=budgets,
                                C=256, window=64)
                sweeps[name].setdefault(str(L), []).append(r)
        print(f"done {Path(f).name}", flush=True)

    # (a) mean moves: median relative per-token move in n bins, and median of move × n
    bins = [(128, 256), (256, 512), (512, 1024), (1024, 2200)]
    move_summary = {}
    for L, rows in moves.items():
        n = torch.cat([r["n"] for r in rows])
        mv = torch.cat([r["move_rel"] for r in rows])
        move_summary[L] = {f"{a}-{b}": {"median": float(mv[(n >= a) & (n < b)].median()),
                                        "p90": float(mv[(n >= a) & (n < b)].quantile(0.9))}
                           for a, b in bins}
        move_summary[L]["median_move_times_n"] = float((mv * n).median())
    # (b) δ sweep: pool over contexts (ratio of sums for errors, mean for rates)
    sweep_summary = {}
    for scen, per_layer in sweeps.items():
        sweep_summary[scen] = {}
        for L, rows in per_layer.items():
            sweep_summary[scen][L] = {}
            for dk in rows[0]:
                entry = {"rebuild_rate": sum(r[dk]["rebuild_rate"] for r in rows) / len(rows)}
                for b in budgets:
                    num = sum(r[dk][f"b={b}"]["err_num"] for r in rows)
                    den = sum(r[dk][f"b={b}"]["err_den"] for r in rows)
                    entry[f"b={b}"] = {"rel_err": num / den,
                                       "kv_frac": sum(r[dk][f"b={b}"]["kv_frac"] for r in rows) / len(rows)}
                sweep_summary[scen][L][dk] = entry
    payload = stamp({"kind": "sphere_update", "captures": files, "stream_files": stream_files,
                     "deltas": [str(x) for x in deltas], "budgets": budgets,
                     "mean_moves": move_summary, "delta_sweep": sweep_summary})
    out = Path("src/ssa/results") / f"sphere_update_{payload['git_sha'][:8]}.json"
    out.write_text(json.dumps(payload, indent=1))
    for L, v in move_summary.items():
        print(f"layer {L} per-token mean move / r̄ (median by n): "
              + "  ".join(f"{k}: {x['median']:.2e}" for k, x in v.items() if k != "median_move_times_n")
              + f"   median move×n = {v['median_move_times_n']:.2f}")
    for scen, per_layer in sweep_summary.items():
        print(f"\n== {scen}")
        for L, dd in per_layer.items():
            print(f" layer {L}")
            for dk, e in dd.items():
                print(f"   {dk:12s} rebuilds/head/token {e['rebuild_rate']:.4f}  "
                      + "  ".join(f"b={b}: err {e[f'b={b}']['rel_err']:.3e} K+V {e[f'b={b}']['kv_frac']:.3f}"
                                  for b in budgets))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
