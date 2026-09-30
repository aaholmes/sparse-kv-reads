"""Capture post-RoPE decode queries and the K/V cache from every layer, for offline replay.

The expensive half of the cluster-sampling replay: run the
model once (prefill, then ``steps`` teacher-forced decode steps with exact attention),
record each layer's query at every decode step and the final K/V cache, and write them
to disk. Decode step ``i`` attends over cache positions ``[0, prefill + i]``.

Run:
    uv run python -m ssa.harness.capture_qkv --prefill 2048 --steps 64
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def capture(model, ids: torch.Tensor, *, prefill: int, steps: int) -> dict:
    """Return ``q [L, steps, H, d]``, ``k, v [L, prefill+steps, H_kv, d]``, decode logits."""
    from engine.attention import Attention

    mods = [m for m in model.modules() if isinstance(m, Attention)]
    qs = [[] for _ in mods]
    last = [None] * len(mods)
    scale_box = [None]

    def make_op(i):
        def op(q, full_k, full_v, *, scale: float, layer_idx: int):
            qs[i].append(q[0, :, 0, :].clone())
            last[i] = (full_k, full_v)
            scale_box[0] = scale
            G = q.shape[1] // full_k.shape[1]
            return F.scaled_dot_product_attention(
                q, full_k.repeat_interleave(G, 1), full_v.repeat_interleave(G, 1))
        return op

    for i, m in enumerate(mods):
        m.decode_attn_op = make_op(i)
    try:
        cache = model.alloc_cache(prefill + steps + 1)
        logits = []
        with torch.inference_mode():
            model(ids[:, :prefill], cache, start_pos=0)
            for t in range(prefill, prefill + steps):
                logits.append(model(ids[:, t:t + 1], cache))
        q = torch.stack([torch.stack(x) for x in qs])                        # [L, T, H, d]
        k = torch.stack([fk[0].permute(1, 0, 2).clone() for fk, _ in last])  # [L, n, H_kv, d]
        v = torch.stack([fv[0].permute(1, 0, 2).clone() for _, fv in last])
    finally:
        for m in mods:
            m.decode_attn_op = None
    return {"q": q, "k": k, "v": v, "logits": torch.cat(logits, 1),
            "prefill": prefill, "scale": scale_box[0]}


def random_wikitext_chunks(model_id: str, *, n: int, length: int, seed: int = 0) -> list[torch.Tensor]:
    """``n`` WikiText-103 test chunks of ``length`` tokens at random offsets (almost always mid-article)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    ds = load_dataset("wikitext", "wikitext-103-v1", split="test")
    ids = tok("\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(seed)
    offs = torch.randint(0, ids.numel() - length, (n,), generator=g)
    return [ids[o:o + length].unsqueeze(0) for o in offs.tolist()]


def main() -> None:
    import argparse
    from pathlib import Path

    from .ppl_sweep import _load_model, _wikitext_chunks
    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--prefill", type=int, default=2048)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--chunk", type=int, default=0, help="which WikiText-103 test chunk (single mode)")
    p.add_argument("--out", default=None)
    p.add_argument("--n-contexts", type=int, default=0,
                   help="multi-context mode: this many contexts from --corpus, one file each")
    p.add_argument("--corpus", default="wikitext", choices=["wikitext", "code"])
    p.add_argument("--layers", type=int, nargs="+", default=None, help="keep only these layers")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = _load_model(args.model, device, dtype)
    length = args.prefill + args.steps
    res_dir = Path(__file__).resolve().parent.parent / "results"

    if args.n_contexts:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        if args.corpus == "wikitext":
            ctxs = random_wikitext_chunks(args.model, n=args.n_contexts, length=length)
        else:
            from .accept_sweep import _load_chunks
            ctxs = _load_chunks(args.model, "code", max_chunks=args.n_contexts, chunk_len=length,
                                device="cpu")
        jobs = [(f"{args.corpus}{i:02d}", c) for i, c in enumerate(ctxs)]
        out_dir = Path(args.out_dir) if args.out_dir else res_dir / "multictx"
    else:
        tok = None
        ids = _wikitext_chunks(args.model, max_chunks=args.chunk + 1, chunk_len=length,
                               device="cpu")[args.chunk]
        jobs = [(None, ids)]
        out_dir = res_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, ids in jobs:
        ids = ids.to(device)
        cap = capture(model, ids, prefill=args.prefill, steps=args.steps)
        layers = args.layers if args.layers is not None else list(range(cap["q"].shape[0]))
        sel = lambda x: x[layers].cpu()
        meta = stamp({"kind": "qkv_capture", "model": args.model, "prefill": args.prefill,
                      "steps": args.steps, "chunk": args.chunk if name is None else name,
                      "corpus": args.corpus, "layers": layers,
                      "shapes": {k: list(sel(cap[k]).shape) for k in ("q", "k", "v")}})
        rec = {"q": sel(cap["q"]), "k": sel(cap["k"]), "v": sel(cap["v"]), "ids": ids.cpu(),
               "prefill": args.prefill, "scale": cap["scale"], "layers": layers,
               "corpus": args.corpus, "meta": meta}
        if tok is not None:
            rec["first_text"] = tok.decode(ids[0, :6].tolist())
        if name is None:
            path = Path(args.out) if args.out else out_dir / f"qkv_capture_p{args.prefill}_s{args.steps}.pt"
        else:
            path = out_dir / f"{name}.pt"
        torch.save(rec, path)
        print(f"wrote {path}  q{tuple(rec['q'].shape)} k{tuple(rec['k'].shape)}", flush=True)


if __name__ == "__main__":
    main()
