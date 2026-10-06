"""LongBench harness: scoring ported from the benchmark, middle truncation, prompt construction, summary."""

from __future__ import annotations

from ssa.harness.longbench import PROMPTS, build_prompt_ids, normalize_answer, qa_f1, retrieval_score, score, \
    summarize, truncate_middle


def test_f1_normalizes_case_punctuation_and_articles():
    assert normalize_answer("The  Miller v. California!") == "miller v california"
    assert qa_f1("Miller v. California", "miller v california") == 1.0
    assert qa_f1("the answer is Paris", "Paris") == 2 * (1 / 3) * 1 / (1 / 3 + 1)     # 'the' dropped: 3 tokens, 1 shared
    assert qa_f1("London", "Paris") == 0.0
    assert score("hotpotqa", "Paris", ["London", "paris"]) == 1.0                      # best over the references


def test_retrieval_score_counts_matching_numbers():
    assert retrieval_score("Paragraph 15", "Paragraph 15") == 1.0
    assert retrieval_score("Paragraph 15 or Paragraph 3", "Paragraph 15") == 0.5
    assert retrieval_score("I do not know", "Paragraph 15") == 0.0
    assert score("passage_retrieval_en", "The answer is: Paragraph 7", ["Paragraph 7"]) == 1.0


def test_truncation_keeps_both_ends():
    assert truncate_middle(list(range(10)), 20) == list(range(10))
    assert truncate_middle(list(range(100)), 10) == [0, 1, 2, 3, 4, 95, 96, 97, 98, 99]


class _Tok:
    """Whitespace tokenizer standing in for the model's."""
    def encode(self, text, add_special_tokens=False):
        self.vocab = getattr(self, "vocab", {})
        return [self.vocab.setdefault(w, len(self.vocab)) for w in text.split(" ")]

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return "<user> " + messages[0]["content"] + " <assistant>"


def test_prompt_is_truncated_in_the_middle_and_wrapped_for_chat():
    tok = _Tok()
    ex = {"context": " ".join(f"w{i}" for i in range(500)), "input": "Who?"}
    ids = build_prompt_ids(tok, "hotpotqa", ex, max_length=60)
    text = tok.decode(ids)
    assert len(ids) == 62 and text.startswith("<user> Answer the question") and text.endswith("Who?\nAnswer: <assistant>")
    assert "w250" not in text
    assert set(PROMPTS) == {"hotpotqa", "2wikimqa", "musique", "passage_retrieval_en"}


def test_summary_pairs_conditions_by_example():
    recs = []
    for i in range(20):
        for c, s in (("exact", 1.0), ("cluster_20", 1.0 if i % 4 else 0.0)):
            recs.append({"dataset": "hotpotqa", "i": i, "condition": c, "score": s, "generated": "x" if s else "y",
                         "prompt_tokens": 1000, "kv_read_fraction": 1.0 if c == "exact" else 0.25})
    rows = summarize(recs)
    assert [r["dataset"] for r in rows] == ["hotpotqa", "all"]
    r = rows[0]
    assert r["exact"] == 1.0 and abs(r["cluster_20"]["diff"] + 0.25) < 1e-12
    assert r["cluster_20"]["diff_ci"][0] <= -0.25 <= r["cluster_20"]["diff_ci"][1]
    assert r["cluster_20"]["same_text_as_exact"] == 0.75 and r["cluster_20"]["kv_read_fraction"] == 0.25
