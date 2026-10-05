"""Tests for the TVD-vs-reads plot builder."""

from __future__ import annotations

import json

from ssa.harness.plot_tvd import build_series, plot_series


def _payload(rows):
    return {"summary": rows}


def _row(impl, cfg, kv, tvd, lo, hi):
    return {"impl": impl, "cfg": cfg, "kv_read_fraction": kv, "tvd": tvd, "tvd_ci": [lo, hi]}


def test_build_series_filters_region_count_sorts_and_merges(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(_payload([
        {"impl": "dense", "cfg": {}, "kv_read_fraction": 1.0, "tvd": 0.0},
        _row("cluster_skip", {"budget": 0.2, "C": 256}, 0.23, 0.043, 0.037, 0.050),
        _row("cluster_skip", {"budget": 0.1, "C": 256}, 0.13, 0.063, 0.053, 0.074),
        _row("cluster_skip", {"budget": 0.1, "C": 1024}, 0.18, 0.050, 0.041, 0.059),
        _row("santa_sys", {"S": 64}, 0.50, 0.058, 0.049, 0.065),
    ])))
    b.write_text(json.dumps(_payload([
        _row("cluster_skip", {"budget": 0.4, "C": 256}, 0.42, 0.027, 0.023, 0.030),
    ])))

    s = build_series([a, b], regions=256)

    assert s["cluster_skip"]["x"] == [13.0, 23.0, 42.0]  # percent, sorted, C=1024 dropped
    assert s["cluster_skip"]["y"] == [0.063, 0.043, 0.027]
    assert s["cluster_skip"]["lo"][0] == 0.053 and s["cluster_skip"]["hi"][0] == 0.074
    assert s["santa_sys"]["x"] == [50.0]
    assert "dense" not in s


def test_rows_without_ci_are_skipped(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps(_payload([
        {"impl": "cluster_skip", "cfg": {"budget": 0.1, "C": 256}, "kv_read_fraction": 0.13, "tvd": 0.06},
    ])))
    assert build_series([a], regions=256)["cluster_skip"]["x"] == []


def test_plot_writes_png(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps(_payload([
        _row("cluster_skip", {"budget": 0.1, "C": 256}, 0.13, 0.063, 0.053, 0.074),
        _row("santa_sys", {"S": 64}, 0.50, 0.058, 0.049, 0.065),
    ])))
    out = tmp_path / "tvd.png"
    plot_series(build_series([a], regions=256), out)
    assert out.exists() and out.stat().st_size > 0


def test_tail_estimate_is_its_own_series_and_the_drop_control_is_left_out(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps(_payload([
        _row("cluster_tail", {"budget": 0.1, "C": 256, "order": 1}, 0.147, 0.045, 0.038, 0.052),
        _row("cluster_tail", {"budget": 0.1, "C": 256, "order": "drop"}, 0.131, 0.063, 0.053, 0.074),
        _row("cluster_skip", {"budget": 0.1, "C": 256}, 0.13, 0.063, 0.053, 0.074),
    ])))
    s = build_series([a], regions=256)
    assert s["cluster_tail"]["x"] == [14.7] and s["cluster_tail"]["y"] == [0.045]
    assert s["cluster_skip"]["x"] == [13.0]
    out = tmp_path / "tvd.png"
    plot_series(s, out)
    assert out.exists()


def test_fitted_cluster_results_form_their_own_series(tmp_path):
    from ssa.harness.plot_tvd import LABELS
    a = tmp_path / "a.json"
    a.write_text(json.dumps(_payload([
        _row("cluster_fused", {"budget": 0.1, "C": 256, "partition": "kmeans"}, 0.126, 0.046, 0.038, 0.055),
        _row("cluster_fused", {"budget": 0.1, "C": 256, "partition": "random"}, 0.131, 0.063, 0.053, 0.073),
        _row("sphere_skip", {"budget": 0.1, "C": 256}, 0.13, 0.063, 0.053, 0.074),
        _row("santa_sys", {"S": 64}, 0.50, 0.058, 0.049, 0.065),
    ])))
    s = build_series([a], regions=256, only=("fitted", "santa_sys"))
    assert s["fitted"]["x"] == [12.6] and s["fitted"]["y"] == [0.046]
    assert set(s) == {"fitted", "santa_sys"}                         # the random-direction rows are left out
    assert LABELS["fitted"] == "this method"
    out = tmp_path / "tvd.png"
    plot_series(s, out)
    assert out.exists()
