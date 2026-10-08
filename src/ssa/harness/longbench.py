"""Task accuracy on LongBench question answering and retrieval (arXiv:2308.14508), exact compared with sparse decode.

Datasets (200 English examples each): ``hotpotqa``, ``2wikimqa``, ``musique`` (multi-document question
answering, scored by token F1) and ``passage_retrieval_en`` (which of 30 paragraphs an abstract
summarizes, scored by the benchmark's retrieval score). Prompts, answer lengths and scoring follow the
benchmark's repository (github.com/THUDM/LongBench, ``LongBench/config`` and ``metrics.py``); the
scoring functions below are ported from it. Prompts longer than ``--max-length`` tokens are truncated
in the middle, as the benchmark does. The prompt is wrapped in the model's chat template with thinking
turned off.

As in ``ssa.harness.needle``, the prompt is processed with exact attention except its last ``tail``
tokens; those and the generated answer go through the decode path, so the answer depends on what the
decode attention reads. Greedy decoding. Conditions share each example's prompt and prefilled cache,
so differences are paired; intervals are 95% bootstrap over examples.

Run:
    uv run python -m ssa.harness.longbench --model Qwen/Qwen3-4B --n 100
"""

from __future__ import annotations

import re
import string
from collections import Counter

QA_PROMPT = ("Answer the question based on the given passages. Only give me the answer and do not output any other "
             "words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given "
             "passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:")
PROMPTS = {
    "hotpotqa": QA_PROMPT, "2wikimqa": QA_PROMPT, "musique": QA_PROMPT,
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the "
        "abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of "
        "the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", "
        "etc.\n\nThe answer is: "),
}
MAX_NEW = {"hotpotqa": 32, "2wikimqa": 32, "musique": 32, "passage_retrieval_en": 32}
BUDGETS = (0.2, 0.1, 0.05)


def conditions(budgets=BUDGETS) -> list:
    """Exact attention, then the cluster kernels at each budget. A budget of 1.0 reads every row
    through the sparse path, which separates the effect of skipping rows from any numerical
    difference between the two attention implementations."""
    return [("exact", None, {})] + [
        (f"cluster_{int(round(b * 100)):02d}", "cluster_fused",
         {"budget": b, "window": 64, "delta": 0.03, "group": "sum_share", "check_every": 16,
          "partition": "kmeans"}) for b in budgets]


CONDITIONS = conditions()


def normalize_answer(s: str) -> str:
    """Lower-case, drop punctuation and articles, collapse whitespace (the benchmark's normalization)."""
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def qa_f1(prediction: str, truth: str) -> float:
    p, t = normalize_answer(prediction).split(), normalize_answer(truth).split()
    same = sum((Counter(p) & Counter(t)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(t)
    return 2 * precision * recall / (precision + recall)


def retrieval_score(prediction: str, truth: str) -> float:
    """Fraction of the numbers in the prediction that equal the true paragraph number."""
    target = re.findall(r"Paragraph (\d+)", truth)[0]
    numbers = re.findall(r"\d+", prediction)
    return 0.0 if not numbers else sum(n == target for n in numbers) / len(numbers)


def score(dataset: str, prediction: str, answers: list[str]) -> float:
    """Best score over the reference answers, as the benchmark's scorer takes."""
    fn = retrieval_score if dataset == "passage_retrieval_en" else qa_f1
    return max(fn(prediction, a) for a in answers)


def truncate_middle(ids: list[int], max_length: int) -> list[int]:
    """Keep the first and last ``max_length // 2`` tokens of a prompt that is too long."""
    if len(ids) <= max_length:
        return ids
    half = max_length // 2
    return ids[:half] + ids[-half:]


def build_prompt_ids(tok, dataset: str, example: dict, max_length: int) -> list[int]:
    """Token ids of the chat-formatted prompt, its body truncated in the middle to ``max_length`` tokens."""
    body = PROMPTS[dataset].format(context=example["context"], input=example["input"])
    body = tok.decode(truncate_middle(tok.encode(body, add_special_tokens=False), max_length))
    text = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return tok.encode(text, add_special_tokens=False)


def load_examples(dataset: str, n: int) -> list[dict]:
    import json
    import zipfile

    from huggingface_hub import hf_hub_download
    z = zipfile.ZipFile(hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset"))
    return [json.loads(line) for line in z.open(f"data/{dataset}.jsonl")][:n]


def summarize(records: list[dict]) -> list[dict]:
    """Per dataset and pooled: each condition's mean score, its paired difference from exact with a
    bootstrap interval, and its mean K+V read fraction."""
    from .needle import _boot_diff, _boot_mean
    out = []
    names = list(dict.fromkeys(r["condition"] for r in records))
    for ds in list(dict.fromkeys(r["dataset"] for r in records)) + ["all"]:
        rs = [r for r in records if ds == "all" or r["dataset"] == ds]
        by = {c: {(r["dataset"], r["i"]): r for r in rs if r["condition"] == c} for c in names}
        keys = sorted(by["exact"])
        ex = [by["exact"][k]["score"] for k in keys]
        row = {"dataset": ds, "n": len(keys), "exact": sum(ex) / len(ex), "exact_ci": _boot_mean(ex),
               "prompt_tokens": sum(by["exact"][k]["prompt_tokens"] for k in keys) / len(keys)}
        for c in names:
            if c == "exact":
                continue
            sc = [by[c][k]["score"] for k in keys]
            row[c] = {"score": sum(sc) / len(sc), "diff": sum(sc) / len(sc) - row["exact"], "diff_ci": _boot_diff(sc, ex),
                      "kv_read_fraction": sum(by[c][k]["kv_read_fraction"] for k in keys) / len(keys),
                      "same_text_as_exact": sum(by[c][k]["generated"] == by["exact"][k]["generated"] for k in keys) / len(keys)}
        out.append(row)
    return out


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    import torch
    from transformers import AutoTokenizer

    from ..models.patch import install, uninstall
    from .ppl_sweep import _load_model
    from .stamp import stamp

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--datasets", nargs="+", default=list(PROMPTS))
    p.add_argument("--n", type=int, default=100, help="examples per dataset (the first n)")
    p.add_argument("--max-length", type=int, default=31500, help="prompt tokens kept (middle truncated)")
    p.add_argument("--tail", type=int, default=64, help="final prompt tokens processed by the decode path")
    p.add_argument("--budgets", type=float, nargs="+", default=list(BUDGETS))
    p.add_argument("--tag", default="")
    args = p.parse_args()
    CONDITIONS = conditions(args.budgets)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = _load_model(args.model, "cuda", torch.bfloat16)
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    cache = model.alloc_cache(args.max_length + 256)
    records = []
    for ds in args.datasets:
        for i, ex in enumerate(load_examples(ds, args.n)):
            ids = torch.tensor(build_prompt_ids(tok, ds, ex, args.max_length), device="cuda").unsqueeze(0)
            P = ids.shape[1] - args.tail
            uninstall(model)
            cache.cur_len = 0
            with torch.inference_mode():
                for s in range(0, P, 512):
                    model(ids[:, s:min(s + 512, P)], cache, start_pos=s)
                for name, impl, cfg in CONDITIONS:
                    cache.cur_len = P
                    stats = install(model, impl, **cfg) if impl else None
                    for t in range(P, ids.shape[1] - 1):                     # end of the prompt, teacher-forced
                        model(ids[:, t:t + 1], cache)
                    lg = model(ids[:, -1:], cache)
                    out = []
                    for _ in range(MAX_NEW[ds]):
                        nxt = lg[0, -1].argmax().view(1, 1)
                        if int(nxt) in stop:
                            break
                        out.append(int(nxt))
                        lg = model(nxt, cache)
                    uninstall(model)
                    text = tok.decode(out, skip_special_tokens=True)
                    records.append({"dataset": ds, "i": i, "id": ex["_id"], "condition": name, "generated": text,
                                    "score": score(ds, text, ex["answers"]), "prompt_tokens": ids.shape[1],
                                    "kv_read_fraction": stats.kv_read_fraction if stats else 1.0})
            if (i + 1) % 10 == 0:
                done = [r for r in records if r["dataset"] == ds]
                print(f"{ds} {i + 1}: " + "  ".join(
                    f"{c} {sum(r['score'] for r in done if r['condition'] == c) / (i + 1):.3f}" for c, _, _ in CONDITIONS),
                    flush=True)
    summary = summarize(records)
    payload = stamp({"kind": "longbench", "model": args.model, "datasets": args.datasets, "n": args.n,
                     "max_length": args.max_length, "tail": args.tail, "conditions": CONDITIONS, "summary": summary,
                     "records": records})
    out = Path("src/ssa/results") / f"longbench_{payload['git_sha'][:8]}{args.tag}.json"
    out.write_text(json.dumps(payload, indent=1))
    for r in summary:
        print(f"{r['dataset']:22s} n={r['n']:3d} ~{r['prompt_tokens']:.0f} tokens  exact {r['exact']:.3f}  " + "  ".join(
            f"{c} {v['score']:.3f} ({v['diff']:+.3f} [{v['diff_ci'][0]:+.3f}, {v['diff_ci'][1]:+.3f}]) @{v['kv_read_fraction']:.3f}"
            for c, v in r.items() if isinstance(v, dict)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
