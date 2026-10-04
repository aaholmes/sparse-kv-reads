"""Matched-reads comparison of two methods' TVD curves, paired by chunk."""

from __future__ import annotations

from ssa.harness.matched_reads import ratio_at


def _row(impl, reads, chunk_tvd):
    return {"impl": impl, "kv_read_fraction": reads, "chunk_tvd": [t * 10 for t in chunk_tvd], "chunk_n": [10] * len(chunk_tvd)}


def test_identical_curves_give_ratio_one_and_shifted_curves_interpolate():
    payload = {"results": [_row("a", 0.1, [0.08, 0.06]), _row("a", 0.3, [0.02, 0.03]),
                           _row("b", 0.1, [0.08, 0.06]), _row("b", 0.3, [0.02, 0.03])]}
    r = ratio_at(payload, "a", "b", 0.2)
    assert abs(r[2] - 1.0) < 1e-12 and abs(r[3] - 1.0) < 1e-12
    assert ratio_at(payload, "a", "b", 0.5) is None                     # outside the measured range


def test_a_twice_b_everywhere_gives_ratio_two():
    payload = {"results": [_row("a", 0.1, [0.2, 0.4]), _row("a", 0.3, [0.1, 0.2]),
                           _row("b", 0.1, [0.1, 0.2]), _row("b", 0.3, [0.05, 0.1])]}
    assert abs(ratio_at(payload, "a", "b", 0.2)[2] - 2.0) < 1e-9
