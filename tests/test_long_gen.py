"""Summary of the long-generation fidelity run: bins, paired ratios, matched-reads interpolation."""

from __future__ import annotations

import torch

from ssa.harness.long_gen_tvd import ARMS, summarize


def test_summary_bins_and_ratios():
    g = torch.Generator().manual_seed(0)
    base = 0.02 + 0.01 * torch.rand(6, 8, generator=g, dtype=torch.float64)
    tvd, reads = {}, {}
    for b, scale in ((0.1, 2.0), (0.2, 1.0)):
        ramp = torch.linspace(1.0, 2.0, 8, dtype=torch.float64)                 # frozen degrades with generated tokens
        tvd[f"frozen_{b}"] = (base * scale * ramp).tolist()
        tvd[f"split_{b}"] = (base * scale).tolist()
        tvd[f"refit_{b}"] = (base * scale * 0.5).tolist()
        for a in ARMS:
            reads[f"{a}_{b}"] = torch.full((6, 8), b + (0.01 if a == "split" else 0.0), dtype=torch.float64).tolist()
    rows = summarize({"tvd": tvd, "reads": reads, "budgets": [0.1, 0.2]}, nbins=4)
    assert len(rows) == 8 and rows[0]["generated"] == [0, 128] and rows[3]["generated"] == [384, 512]
    first, last = rows[4], rows[7]                                              # budget 0.2
    assert abs(first["split_over_frozen"][0] - 1 / 1.0714) < 0.01
    assert abs(last["split_over_frozen"][0] - 1 / 1.9286) < 0.01
    assert last["split_over_frozen"][1] <= last["split_over_frozen"][0] <= last["split_over_frozen"][2]
    assert abs(last["split_over_refit"][0] - 2.0) < 1e-9 and abs(last["refit_over_frozen"][0] - 0.5 / 1.9286) < 0.01
    # split reads 0.21 at budget 0.2; at frozen's 0.20 it must do slightly worse than at 0.21
    assert last["split_over_frozen_matched_reads"][0] > last["split_over_frozen"][0]


def test_summary_includes_extra_arms_when_present():
    one = torch.full((5, 4), 0.04, dtype=torch.float64)
    tvd, reads = {}, {}
    for b in (0.1, 0.2):
        for a, k, rd in (("frozen", 1.0, 0.0), ("split", 0.5, 0.01), ("refit", 0.6, 0.0), ("frozen512", 0.8, 0.02),
                         ("split_norecenter", 0.55, 0.01)):
            tvd[f"{a}_{b}"] = (one * k / (b * 10)).tolist()
            reads[f"{a}_{b}"] = (torch.zeros(5, 4, dtype=torch.float64) + b + rd).tolist()
    row = summarize({"tvd": tvd, "reads": reads, "budgets": [0.1, 0.2]}, nbins=2)[-1]
    assert abs(row["split_over_frozen512"][0] - 0.5 / 0.8) < 1e-9
    assert abs(row["split_norecenter_over_split"][0] - 1.1) < 1e-9
    assert row["split_over_frozen512_matched_reads"][0] < row["split_over_frozen512"][0]    # split given 512's extra reads
    assert set(row["tvd"]) == {"frozen", "split", "refit", "frozen512", "split_norecenter"}
