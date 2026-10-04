"""Needle-in-a-haystack prompts and scoring (CPU, no model)."""

from __future__ import annotations

import random

from ssa.harness.needle import build_example, score


class _Tok:
    """Whitespace tokenizer stand-in: one token per word."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.setdefault(w, len(self.vocab)) for w in text.split()]

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids)


def _hay(tok, n=5000):
    return tok.encode(" ".join(f"word{i % 97}." if i % 13 == 0 else f"word{i % 97}" for i in range(n)))


def test_single_needle_prompt_has_the_requested_length_and_contains_the_answer():
    tok = _Tok()
    ex = build_example(tok, _hay(tok), length=800, task="single", rng=random.Random(0))
    assert abs(len(ex["ids"]) - 800) <= 2
    text = tok.decode(ex["ids"])
    assert f"{ex['key']} is: {ex['value']}" in text.replace(" .", ".")
    assert text.rstrip().endswith("is:")
    assert len(ex["value"]) == 7 and ex["value"].isdigit()


def test_multikey_prompt_has_four_needles_and_asks_for_one_of_them():
    tok = _Tok()
    ex = build_example(tok, _hay(tok), length=800, task="multikey", rng=random.Random(1))
    text = tok.decode(ex["ids"])
    assert text.count("special magic number for") == 6            # four needles + twice in the question
    assert len(set(ex["distractor_values"])) == 3 and ex["value"] not in ex["distractor_values"]
    assert f"number for {ex['key']} mentioned" in text


def test_needle_depths_spread_over_the_context():
    tok = _Tok()
    hay = _hay(tok)
    depths = [build_example(tok, hay, length=800, task="single", rng=random.Random(s))["depth"] for s in range(40)]
    assert min(depths) < 0.2 and max(depths) > 0.8


def test_score_takes_the_first_seven_digit_number():
    assert score(" 4829173.\nQuestion", "4829173")
    assert score("4829173", "4829173")
    assert not score(" 4829174.", "4829173")
    assert not score(" 48291735", "4829173")              # eight digits: not the answer
    assert not score(" the number is unknown", "4829173")
