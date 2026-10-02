"""Microbenchmark: label-weighted Triton decode attention vs PyTorch SDPA.

One layer's decode step in Qwen3-4B geometry (32 query heads, 8 KV heads, d=128, BF16).
Times each call with CUDA events (median of ``reps``), flushing L2 before every call so the
cache is read from DRAM as it would be in real decoding. Reports achieved bandwidth =
bytes the call must read (selected K and V rows, labels, weights) ÷ time.

Label layouts: ``scattered`` (bins assigned uniformly at random over positions),
``contiguous`` (each bin a contiguous run of positions: worst case for load balance),
``real`` (SphereState labels on captured keys; n ≈ 2k only).

Run:
    uv run python -m ssa.harness.kernel_bench
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

H, H_KV, D = 32, 8, 128
_FLUSH = None


def _flush():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(64 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    _FLUSH.zero_()


def time_call(fn, *, reps: int = 100, warmup: int = 5, flush: bool = True) -> float:
    """Median seconds per call."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        if flush:
            _flush()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) / 1e3)
    return float(torch.tensor(times).median())


def make_labels(H_kv: int, n: int, *, C: int, frac: float, layout: str, device, seed: int = 0,
                window: int = 64):
    """Labels ``[H_kv, n]`` (label C = exact set: token 0 + last ``window``) and shared weights
    ``[H_kv, C+1]`` selecting ≈ ``frac`` of the binned positions."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    nb = max(0, n - window - 1)
    if layout == "scattered":
        lab = torch.randint(0, C, (H_kv, nb), generator=g)
    elif layout == "contiguous":
        lab = (torch.arange(nb) * C // max(nb, 1)).expand(H_kv, nb)
    else:
        raise ValueError(layout)
    labels = torch.full((H_kv, n), C, dtype=torch.int16)
    labels[:, 1:1 + nb] = lab.to(torch.int16)
    w = torch.zeros(H_kv, C + 1)
    w[:, C] = 1.0
    k = max(0, min(C, round(frac * C)))
    for h in range(H_kv):
        w[h, torch.randperm(C, generator=g)[:k]] = 1.0
    return labels.to(device), w.to(device)


def bytes_needed(labels: torch.Tensor, w: torch.Tensor, *, d: int, elem: int) -> int:
    rows = int((torch.gather(w, 1, labels.long()) > 0).sum())       # shared weights: rows per KV head
    return rows * 2 * d * elem + labels.numel() * labels.element_size() + w.numel() * w.element_size()


def _cache(n, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(H, D, device="cuda", generator=g).to(torch.bfloat16)
    K = torch.randn(1, H_KV, n, D, device="cuda", generator=g).to(torch.bfloat16)
    V = torch.randn(1, H_KV, n, D, device="cuda", generator=g).to(torch.bfloat16)
    return q, K, V


def bench_n(n: int, *, fracs, layouts, reps: int, tiles, real=None,
            compact_cfgs=((32, 4, None), (64, 4, None), (32, 4, 72), (64, 8, 72))) -> list[dict]:
    from ..kernels.labeled_attn import label_weighted_attention_compact as lwc
    from ..kernels.labeled_attn import label_weighted_attention_triton as lwa

    q, K, V = _cache(n)
    dense_bytes = 2 * H_KV * n * D * 2
    rows = []

    def sdpa_gqa():
        return F.scaled_dot_product_attention(q.view(1, H, 1, D), K, V, enable_gqa=True)

    def sdpa_repeat():
        return F.scaled_dot_product_attention(q.view(1, H, 1, D), K.repeat_interleave(H // H_KV, 1),
                                              V.repeat_interleave(H // H_KV, 1))

    for name, fn in (("sdpa_gqa", sdpa_gqa), ("sdpa_repeat_interleave", sdpa_repeat)):
        t = time_call(fn, reps=reps)
        rows.append({"n": n, "method": name, "frac": 1.0, "layout": "-", "time_s": t,
                     "bytes": dense_bytes, "gbps": dense_bytes / t / 1e9})
        print(f"n={n:6d} {name:24s} {t*1e6:8.1f} us  {dense_bytes/t/1e9:6.1f} GB/s", flush=True)

    label_sets = []
    for layout in layouts:
        for f in fracs:
            label_sets.append((layout, f, *make_labels(H_KV, n, C=256, frac=f, layout=layout, device="cuda")))
    if real is not None and abs(real[0].shape[1] - n) == 0:
        for f, (lab, w) in real[1].items():
            label_sets.append(("real", f, lab, w))

    Kc, Vc = K[0], V[0]
    for layout, f, labels, w in label_sets:
        need = bytes_needed(labels, w, d=D, elem=2)
        sel = float((torch.gather(w, 1, labels.long()) > 0).float().mean())
        best = None
        for bn, nw in tiles:
            fn = lambda: lwa(q, Kc, Vc, labels, w, block_n=bn, num_warps=nw)
            try:
                t = time_call(fn, reps=reps)
            except Exception as exc:                                  # e.g. shared-memory limits
                print(f"   skip block_n={bn} warps={nw}: {type(exc).__name__}")
                continue
            if best is None or t < best[0]:
                best = (t, bn, nw)
        t, bn, nw = best
        rows.append({"n": n, "method": "triton_labeled", "frac": f, "rows_selected": sel, "layout": layout,
                     "time_s": t, "bytes": need, "gbps": need / t / 1e9, "block_n": bn, "num_warps": nw})
        print(f"n={n:6d} triton {layout:10s} target {f:4.2f} sel {sel:5.3f} {t*1e6:8.1f} us  "
              f"{need/t/1e9:6.1f} GB/s  (block {bn}, warps {nw})", flush=True)
        best = None
        for bn, nw, ns in compact_cfgs:
            fn = lambda: lwc(q, Kc, Vc, labels, w, block_n=bn, num_warps=nw, num_splits=ns)
            try:
                t = time_call(fn, reps=reps)
            except Exception as exc:
                print(f"   skip compact block_n={bn} warps={nw} splits={ns}: {type(exc).__name__}")
                continue
            if best is None or t < best[0]:
                best = (t, bn, nw, ns)
        t, bn, nw, ns = best
        rows.append({"n": n, "method": "triton_compact", "frac": f, "rows_selected": sel, "layout": layout,
                     "time_s": t, "bytes": need, "gbps": need / t / 1e9, "block_n": bn, "num_warps": nw,
                     "num_splits": ns})
        print(f"n={n:6d} compact {layout:10s} target {f:4.2f} sel {sel:5.3f} {t*1e6:8.1f} us  "
              f"{need/t/1e9:6.1f} GB/s  (block {bn}, warps {nw}, splits {ns})", flush=True)
    return rows


def flashinfer_decode(q, K, V, *, tensor_cores: bool = False):
    """FlashInfer's single-request decode attention on the engine layout ``K, V [H_kv, n, d]``.

    Needs ``flashinfer-python`` and, on this sm_120 card, a CUDA >= 12.9 compiler for its JIT; e.g.
    ``CUDA_HOME=<site-packages>/nvidia/cu13 uv run --with flashinfer-python==0.7.0.post1
    --with nvidia-cuda-nvcc==13.0.88 --with nvidia-nvvm==13.0.88 --with nvidia-cuda-crt==13.0.88 ...``."""
    import flashinfer
    return flashinfer.single_decode_with_kv_cache(q, K, V, kv_layout="HND", use_tensor_cores=tensor_cores)


def bench_step(n: int, *, budget: float, reps: int, fused: bool = False, flashinfer: bool = False) -> dict:
    """Per-layer decode step: bin maintenance + selection, attention, and both,
    eager and captured in a CUDA graph, compared with SDPA. Real Qwen3-4B keys are not
    needed for timing; random keys with a shared offset are used."""
    from ..attn.sphere_gpu import SphereIndexGPU
    from ..kernels.labeled_attn import label_weighted_attention_compact as lwc
    from ..kernels.sphere_fused import SphereIndexFused

    q, K, V = _cache(n + 1)
    K = K + 0.5
    Kc, Vc = K[0], V[0]
    cls = SphereIndexFused if fused else SphereIndexGPU
    idx = cls(C=256, window=64, delta=0.03, capacity=n + 1, check_every=10 ** 9)
    idx.observe(Kc, n)

    def maint_select():
        idx.observe(Kc, n + 1)
        idx.n = n                        # replay the same step each call (timing only)
        idx.end = max(1, n - 64)
        return idx.labels_and_weights(q, n=n + 1, budget=budget, group="sum_share")

    lab, w = maint_select()

    def attend():
        return lwc(q, Kc[:, :n + 1], Vc[:, :n + 1], lab, w)

    def full():
        l2, w2 = maint_select()
        return lwc(q, Kc[:, :n + 1], Vc[:, :n + 1], l2, w2)

    def sdpa():
        return F.scaled_dot_product_attention(q.view(1, H, 1, D), K[:, :, :n + 1], V[:, :, :n + 1], enable_gqa=True)

    fns = [("maint_select", maint_select), ("attention", attend), ("step", full), ("sdpa", sdpa)]
    if flashinfer:
        Kf, Vf = Kc[:, :n + 1].contiguous(), Vc[:, :n + 1].contiguous()
        fns += [("flashinfer", lambda: flashinfer_decode(q, Kf, Vf)),
                ("flashinfer_tc", lambda: flashinfer_decode(q, Kf, Vf, tensor_cores=True))]
    out = {"n": n, "budget": budget, "fused": fused}
    for name, fn in fns:
        out[f"{name}_eager_us"] = time_call(fn, reps=reps) * 1e6
        for _ in range(3):
            fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        out[f"{name}_graph_us"] = time_call(g.replay, reps=reps) * 1e6
    print("n={n:6d} budget {budget:.2f} fused {fused!s:5}: ".format(**out) + "  ".join(
        f"{k[:-3]} {v:6.1f}" for k, v in out.items() if k.endswith("_us")), flush=True)
    return out


def _real_labels(path: str, layer_index: int, fracs):
    """SphereState labels/weights on one captured layer at its last decode step (n ≈ 2k)."""
    from ..attn.sphere_state import SphereState

    cap = torch.load(path, weights_only=False)
    t = cap["q"].shape[1] - 1
    n = cap["prefill"] + t + 1
    K = cap["k"][layer_index, :n].cuda()
    q = cap["q"][layer_index, t].cuda()
    st = SphereState(C=256, window=64, delta=0.03)
    st.observe(K)
    out = {}
    for f in fracs:
        lab, w = st.labels_and_weights(q, n=n, budget=f, group="sum_share")
        out[f] = (lab, w.float())
    return n, out


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--ns", type=int, nargs="+", default=[2048, 8192, 32768, 65536])
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--real", default="src/ssa/results/multictx/wikitext03.pt")
    p.add_argument("--step", action="store_true", help="time maintenance + selection + attention per step")
    p.add_argument("--flashinfer", action="store_true", help="with --step: also time FlashInfer's exact decode")
    args = p.parse_args()

    if args.step:
        rows = [bench_step(n, budget=b, reps=args.reps, fused=fz, flashinfer=args.flashinfer)
                for n in args.ns for b in (0.2, 0.05) for fz in (False, True)]
        payload = stamp({"kind": "kernel_bench_step", "geometry": {"H": H, "H_kv": H_KV, "d": D, "dtype": "bf16"},
                         "reps": args.reps, "rows": rows})
        out = Path("src/ssa/results") / f"kernel_bench_step_{payload['git_sha'][:8]}.json"
        out.write_text(json.dumps(payload, indent=1))
        print(f"wrote {out}")
        return

    fracs = (1.0, 0.5, 0.2, 0.1, 0.05)
    tiles = ((32, 4),)                      # the masked scan's best tile at every size
    real = None
    if args.real and Path(args.real).exists():
        n_real, lw = _real_labels(args.real, 2, (0.5, 0.2, 0.1, 0.05))   # layer 24 of the capture
        real = (torch.empty(1, n_real), lw)
    rows = []
    for n in args.ns:
        rows += bench_n(n, fracs=fracs, layouts=("scattered", "contiguous"), reps=args.reps, tiles=tiles)
    if real is not None:
        rows += bench_n(real[0].shape[1], fracs=(), layouts=(), reps=args.reps, tiles=tiles, real=real)
    props = torch.cuda.get_device_properties(0)
    payload = stamp({"kind": "kernel_bench", "geometry": {"H": H, "H_kv": H_KV, "d": D, "dtype": "bf16"},
                     "reps": args.reps, "l2_flush_bytes": 64 * 1024 * 1024, "peak_gbps_nominal": 448,
                     "sms": props.multi_processor_count, "tiles": tiles, "rows": rows})
    out = Path("src/ssa/results") / f"kernel_bench_{payload['git_sha'][:8]}.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
