"""Tests for sphere_skip: fixed hypersphere partition + budgeted region selection."""

from __future__ import annotations

import math

import pytest
import torch

from ssa.attn import attn, available
from ssa.attn.sphere_skip import SphereIndex, fixed_directions, region_stats, sphere_skip
from _fixtures import Geom, make_qkv


def test_registered():
    assert "voronoi_skip" in available()


def test_fixed_directions_unit_and_deterministic():
    a = fixed_directions(64, 16, seed=3)
    b = fixed_directions(64, 16, seed=3)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(a.norm(dim=1), torch.ones(64, dtype=a.dtype))


@pytest.mark.parametrize("kind", ["random", "sobol", "cross"])
def test_direction_kinds(kind):
    x = fixed_directions(32, 16, seed=1, kind=kind, dtype=torch.float64)
    torch.testing.assert_close(x.norm(dim=1), torch.ones(32, dtype=torch.float64))
    if kind == "cross":
        assert torch.equal(x.abs().sum(1), torch.ones(32, dtype=torch.float64))


def test_region_stats_match_definition():
    g = torch.Generator().manual_seed(0)
    K = torch.randn(50, 8, generator=g, dtype=torch.float64) * 2
    dirs = fixed_directions(10, 8, seed=0, dtype=torch.float64)
    st = region_stats(K, dirs)
    Kn = K / K.norm(dim=1, keepdim=True)
    torch.testing.assert_close(st["labels"], (Kn @ dirs.t()).argmax(1))
    for r in range(10):
        m = st["labels"] == r
        assert int(st["count"][r]) == int(m.sum())
        if m.any():
            c = Kn[m].sum(0)
            torch.testing.assert_close(st["cdir"][r], c / c.norm())
            assert math.isclose(float(st["mmax"][r]), float(K[m].norm(dim=1).max()), rel_tol=1e-12)
            assert math.isclose(float(st["mmin"][r]), float(K[m].norm(dim=1).min()), rel_tol=1e-12)


def test_incremental_index_matches_batch():
    g = torch.Generator().manual_seed(1)
    K = torch.randn(40, 8, generator=g, dtype=torch.float64)
    dirs = fixed_directions(6, 8, seed=1, dtype=torch.float64)
    idx = SphereIndex(dirs)
    for chunk in K.split(7):
        idx.add(chunk)
    batch = region_stats(K, dirs)
    inc = idx.stats()
    for k in ("count", "mmax", "mmin"):
        torch.testing.assert_close(inc[k], batch[k])
    torch.testing.assert_close(inc["cdir"], batch["cdir"])


def test_full_budget_equals_dense():
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=120), seed=0)
    out = sphere_skip(q, K, V, budget=1.0, C=16, window=8)
    ref = attn(q, K, V, impl="dense")
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-12)


def test_budget_reads_and_exact_set():
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=200), seed=2)
    out, info = sphere_skip(q, K, V, budget=0.2, C=32, window=10, return_info=True)
    n_cl = 200 - 11
    assert torch.isfinite(out).all()
    assert info.selected.shape == (8, 200)
    assert info.selected[:, 0].all() and info.selected[:, -10:].all()
    chosen = info.selected[:, 1:190].sum(1)
    assert torch.all(chosen >= math.ceil(0.2 * n_cl))            # budget met (region granularity)
    torch.testing.assert_close(info.unique, info.selected.sum(1))
    assert info.kv_union.shape == (2,)
    assert torch.all(info.kv_union >= info.unique.view(2, 4).max(1).values)


def test_output_is_softmax_over_selected():
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=150), seed=3)
    out, info = sphere_skip(q, K, V, budget=0.3, C=16, window=5, return_info=True)
    for h in range(4):
        sel = info.selected[h]
        kv = h // 2
        A = torch.softmax(q[h] @ K[sel, kv].t() / 4.0, -1)
        torch.testing.assert_close(out[h], A @ V[sel, kv])


@pytest.mark.parametrize("rank", ["est", "random"])
def test_rank_modes_run(rank):
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=100), seed=4)
    out = sphere_skip(q, K, V, budget=0.25, C=16, window=5, rank=rank, seed=1)
    assert torch.isfinite(out).all()


def test_estimate_beats_random_on_concentrated_attention():
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=400), seed=5)
    q = q * 4
    ref = attn(q, K, V, impl="dense")
    err = {r: float((sphere_skip(q, K, V, budget=0.15, C=32, window=4, rank=r, seed=0) - ref)
                    .pow(2).sum()) for r in ("est", "random")}
    assert err["est"] < err["random"]


def test_centered_output_invariant_to_key_offset():
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=160), seed=6)
    shift = torch.randn(1, 2, 16, dtype=torch.float64) * 5
    a, ia = sphere_skip(q, K, V, budget=0.2, C=16, window=5, center=True, return_info=True)
    b, ib = sphere_skip(q, K + shift, V, budget=0.2, C=16, window=5, center=True, return_info=True)
    assert torch.equal(ia.selected, ib.selected)
    torch.testing.assert_close(a, b)


def test_centered_full_budget_equals_dense():
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=100), seed=7)
    out = sphere_skip(q, K, V, budget=1.0, C=16, window=5, center=True)
    torch.testing.assert_close(out, attn(q, K, V, impl="dense"), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("group", ["max", "max_rel", "sum_share"])
def test_shared_selection_is_identical_within_group(group):
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=300), seed=8)
    out, info = sphere_skip(q, K, V, budget=0.15, C=32, window=6, center=True, group=group,
                            return_info=True)
    sel = info.selected.view(2, 4, 300)
    assert torch.equal(sel, sel[:, :1].expand_as(sel))                 # one selection per KV head
    torch.testing.assert_close(info.kv_union, info.unique.view(2, 4)[:, 0])
    assert torch.all(sel[:, 0, 1:300 - 6].sum(-1) >= math.ceil(0.15 * (300 - 7)))


def test_shared_full_budget_equals_dense():
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=100), seed=9)
    out = sphere_skip(q, K, V, budget=1.0, C=16, window=4, center=True, group="max_rel")
    torch.testing.assert_close(out, attn(q, K, V, impl="dense"), rtol=1e-10, atol=1e-12)


def test_unknown_group_rejected():
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=60), seed=10)
    with pytest.raises(ValueError):
        sphere_skip(q, K, V, budget=0.2, C=8, window=4, group="nope")
