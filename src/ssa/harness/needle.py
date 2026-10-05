"""Needle-in-a-haystack retrieval, modelled on RULER's needle tasks (our implementation, not RULER).

A 7-digit "special magic number" for a random key is hidden at a random depth in WikiText-103
text, and the prompt ends by asking for it:

    single:   one needle
    multikey: four needles with different keys and numbers; the question asks for one

The prompt is processed with exact attention (as a serving engine's prefill would be), except its
last ``tail`` tokens, which contain the question: those, and the generated answer, go through the
decode path, so the answer depends on what the decode attention reads. Greedy decoding; an
example is correct when the first number generated is the needle's. Conditions share prompts and
the prefilled cache, so differences between them are paired.

Run:
    uv run python -m ssa.harness.needle --lengths 16384 32768 --n 100
"""

from __future__ import annotations

import random
import re

ADJ = ("amber", "brisk", "cobalt", "dusty", "eager", "fabled", "gilded", "hollow", "ivory", "jagged",
       "lucid", "mellow", "nimble", "opal", "placid", "quiet", "russet", "sable", "tidal", "umber",
       "velvet", "wistful", "zesty", "frosty", "silent", "crimson", "golden", "rustic", "stormy", "verdant")
NOUN = ("falcon", "harbor", "lantern", "meadow", "orchid", "quarry", "riddle", "saddle", "thimble", "valley",
        "walrus", "anchor", "beacon", "canyon", "dagger", "ember", "fjord", "glacier", "heron", "island",
        "juniper", "kettle", "lagoon", "marble", "nectar", "oasis", "pebble", "quill", "raven", "spruce")
N_NEEDLES = {"single": 1, "multikey": 4}


def _needle(key: str, value: str) -> str:
    return f" The special magic number for {key} is: {value}. "


def _question(key: str) -> str:
    return (f"\n\nQuestion: What is the special magic number for {key} mentioned in the text above?\n"
            f"Answer: The special magic number for {key} is:")


def _sentence_end(tok, ids: list[int], pos: int, lookahead: int = 200) -> int:
    """First position at or after ``pos`` that follows a token ending a sentence."""
    for j in range(pos, min(pos + lookahead, len(ids))):
        if tok.decode([ids[j - 1]]).rstrip().endswith(".") if j > 0 else False:
            return j
    return pos


def build_example(tok, hay: list[int], *, length: int, task: str, rng: random.Random) -> dict:
    """Prompt token ids of about ``length`` tokens with ``N_NEEDLES[task]`` needles."""
    keys = rng.sample([f"{a}-{n}" for a in ADJ for n in NOUN], N_NEEDLES[task])
    values = []
    while len(values) < len(keys):
        v = str(rng.randrange(1_000_000, 10_000_000))
        if v not in values:
            values.append(v)
    needles = [tok.encode(_needle(k, v), add_special_tokens=False) for k, v in zip(keys, values)]
    target = rng.randrange(len(keys))
    question = tok.encode(_question(keys[target]), add_special_tokens=False)
    hay_len = length - sum(map(len, needles)) - len(question)
    off = rng.randrange(0, len(hay) - hay_len)
    body = list(hay[off:off + hay_len])
    depths = [rng.random() for _ in keys]
    spots = sorted(((_sentence_end(tok, body, int(d * hay_len)), i) for i, d in enumerate(depths)), reverse=True)
    for p, i in spots:                                       # insert from the back so positions stay valid
        body[p:p] = needles[i]
    return {"ids": body + question, "task": task, "key": keys[target], "value": values[target],
            "depth": depths[target], "distractor_values": [v for i, v in enumerate(values) if i != target]}


def score(generated: str, value: str) -> bool:
    """Correct when the first number in the generated text is exactly the needle's value."""
    m = re.search(r"\d+", generated)
    return m is not None and m.group(0) == value


def _boot_diff(a: list[int], b: list[int], reps: int = 4000, seed: int = 0) -> list[float]:
    """95% bootstrap interval of mean(a) − mean(b) over paired examples."""
    import torch
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    i = torch.randint(0, len(x), (reps, len(x)), generator=g)
    m = x[i].mean(1)
    return [float(m.quantile(0.025)), float(m.quantile(0.975))]


def _boot_mean(a: list[int], reps: int = 4000, seed: int = 0) -> list[float]:
    import torch
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor(a, dtype=torch.float64)
    m = x[torch.randint(0, len(x), (reps, len(x)), generator=g)].mean(1)
    return [float(m.quantile(0.025)), float(m.quantile(0.975))]


CONDITIONS = [("exact", None, {}),
              ("voronoi_20", "voronoi_fused", {"budget": 0.2, "C": 256, "window": 64, "delta": 0.03,
                                               "group": "sum_share", "check_every": 16, "partition": "kmeans"}),
              ("voronoi_05", "voronoi_fused", {"budget": 0.05, "C": 256, "window": 64, "delta": 0.03,
                                               "group": "sum_share", "check_every": 16, "partition": "kmeans"})]


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from ..models.patch import install, uninstall
    from .ppl_sweep import _load_model
    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768])
    p.add_argument("--tasks", nargs="+", default=["single", "multikey"])
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--tail", type=int, default=64, help="final prompt tokens processed by the decode path")
    p.add_argument("--max-new", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset("wikitext", "wikitext-103-v1", split="test")
    hay = tok("\n\n".join(t for t in ds["text"] if t.strip()), add_special_tokens=False).input_ids
    model = _load_model(args.model, "cuda", torch.bfloat16)
    records = []
    for L in args.lengths:
        cache = model.alloc_cache(L + args.max_new + 16)
        for task in args.tasks:
            rng = random.Random(f"{args.seed}-{L}-{task}")
            for i in range(args.n):
                ex = build_example(tok, hay, length=L, task=task, rng=rng)
                ids = torch.tensor(ex["ids"], device="cuda").unsqueeze(0)
                P = ids.shape[1] - args.tail
                uninstall(model)
                cache.cur_len = 0
                with torch.inference_mode():
                    for s in range(0, P, 512):
                        model(ids[:, s:min(s + 512, P)], cache, start_pos=s)
                    for name, impl, cfg in CONDITIONS:
                        cache.cur_len = P
                        stats = install(model, impl, **cfg) if impl else None
                        for t in range(P, ids.shape[1] - 1):                 # question tokens, teacher-forced
                            model(ids[:, t:t + 1], cache)
                        lg = model(ids[:, -1:], cache)
                        out = []
                        for _ in range(args.max_new):
                            nxt = lg[0, -1].argmax().view(1, 1)
                            out.append(int(nxt))
                            if "\n" in tok.decode([out[-1]]):
                                break
                            lg = model(nxt, cache)
                        uninstall(model)
                        text = tok.decode(out)
                        records.append({"length": L, "task": task, "i": i, "condition": name, "depth": ex["depth"],
                                        "value": ex["value"], "generated": text, "correct": score(text, ex["value"]),
                                        "kv_read_fraction": stats.kv_read_fraction if stats else 1.0})
                if (i + 1) % 10 == 0:
                    acc = {c: sum(r["correct"] for r in records if r["length"] == L and r["task"] == task
                                  and r["condition"] == c) for c, _, _ in CONDITIONS}
                    print(f"L={L} {task} {i + 1}/{args.n}: correct {acc}", flush=True)
        del cache
        torch.cuda.empty_cache()

    summary = []
    for L in args.lengths:
        for task in args.tasks:
            by = {c: [int(r["correct"]) for r in records if r["length"] == L and r["task"] == task
                      and r["condition"] == c] for c, _, _ in CONDITIONS}
            for c, _, _ in CONDITIONS:
                row = {"length": L, "task": task, "condition": c, "n": len(by[c]),
                       "accuracy": sum(by[c]) / len(by[c]), "accuracy_ci": _boot_mean(by[c])}
                if c != "exact":
                    row["diff_vs_exact"] = row["accuracy"] - sum(by["exact"]) / len(by["exact"])
                    row["diff_vs_exact_ci"] = _boot_diff(by[c], by["exact"])
                    reads = [r["kv_read_fraction"] for r in records if r["length"] == L and r["task"] == task
                             and r["condition"] == c and r["kv_read_fraction"] is not None]
                    row["kv_read_fraction"] = sum(reads) / len(reads) if reads else None
                summary.append(row)
    payload = stamp({"kind": "needle", "model": args.model, "tail": args.tail, "max_new": args.max_new,
                     "seed": args.seed, "conditions": CONDITIONS, "summary": summary, "records": records})
    out = Path("src/ssa/results") / f"needle_{payload['git_sha'][:8]}{args.tag}.json"
    out.write_text(json.dumps(payload, indent=1))
    for r in summary:
        extra = (f"  diff {r['diff_vs_exact']:+.2f} [{r['diff_vs_exact_ci'][0]:+.2f}, {r['diff_vs_exact_ci'][1]:+.2f}]"
                 if "diff_vs_exact" in r else "")
        print(f"L={r['length']:6d} {r['task']:8s} {r['condition']:11s} acc {r['accuracy']:.2f} "
              f"[{r['accuracy_ci'][0]:.2f}, {r['accuracy_ci'][1]:.2f}]{extra}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
