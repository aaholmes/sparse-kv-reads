"""Tail sampling over unselected bins: expected weight 1 per tail bin (unbiased sums)."""

from __future__ import annotations

import torch

from ssa.attn.tail_sample import proposal_from_scores, tail_sample_weights


def _case(seed=0, H_kv=3, C=40):
    g = torch.Generator().manual_seed(seed)
    count = torch.randint(0, 6, (H_kv, C), generator=g).double()
    head = torch.rand(H_kv, C, generator=g) < 0.2
    prop = torch.rand(H_kv, C, generator=g, dtype=torch.float64) ** 3
    return count, head, prop


def test_head_weight_one_empty_bins_never_drawn():
    count, head, prop = _case()
    w = tail_sample_weights(prop, count, head, S=8, alpha=0.1, generator=torch.Generator().manual_seed(1))
    assert torch.all(w[head & (count > 0)] == 1.0)
    assert torch.all(w[count == 0] == 0.0)


def test_expected_tail_weight_is_one():
    count, head, prop = _case(seed=2)
    R = 20000
    acc = torch.zeros_like(prop)
    g = torch.Generator().manual_seed(3)
    for _ in range(R):
        acc += tail_sample_weights(prop, count, head, S=5, alpha=0.2, generator=g)
    mean = acc / R
    tail = (~head) & (count > 0)
    torch.testing.assert_close(mean[tail], torch.ones_like(mean[tail]), rtol=0.08, atol=0.08)


def test_zero_draws_is_head_only():
    count, head, prop = _case(seed=4)
    w = tail_sample_weights(prop, count, head, S=0, alpha=0.1, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(w, (head & (count > 0)).double())


def test_draw_multiplicities_sum_to_S():
    count, head, prop = _case(seed=5)
    g = torch.Generator().manual_seed(6)
    tail = (~head) & (count > 0)
    w = tail_sample_weights(prop, count, head, S=7, alpha=0.3, generator=g, return_pi=True)
    wt, pi = w
    draws = (wt * pi * 7).where(tail, torch.zeros_like(wt)).sum(1)      # Σ multiplicity
    torch.testing.assert_close(draws, torch.full_like(draws, 7.0))


def test_proposal_from_scores_prefers_high_scores_and_counts():
    e = torch.tensor([[[0.0, 1.0, 1.0, -float("inf")]]], dtype=torch.float64)     # [H_kv=1, G=1, C=4]
    count = torch.tensor([[2.0, 1.0, 4.0, 0.0]], dtype=torch.float64)
    p = proposal_from_scores(e, count, scale=1.0)
    assert p[0, 3] == 0 and p[0, 2] > p[0, 1] > p[0, 0]
    torch.testing.assert_close(p.sum(1), torch.ones(1, dtype=torch.float64))
