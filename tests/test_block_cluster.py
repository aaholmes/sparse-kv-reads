"""Content-clustered block layout: permutation validity and the unbiasedness test.

Reordering the key axis so similar keys sit contiguously is a pure efficiency move:
``santa_block`` is unbiased for *any* partition, so clustering can
change variance/bytes but must never change the expectation. These tests pin that —
the clustered layout stays unbiased vs ``dense`` — plus the mechanical guarantees
that ``apply_permutation`` is a genuine per-head permutation and leaves the exact
(dense) attention output invariant.
"""

from __future__ import annotations

import pytest
import torch

from ssa.attn.dense import dense
from ssa.harness.plot_blocks import apply_permutation, cluster_permutation
from ssa.harness.variance import collect_estimates, mc_mean_and_stderr
from _fixtures import Geom, make_qkv

METHODS = ["sys", "strat", "iid"]


def test_cluster_permutation_is_valid_per_head():
    _, K, _ = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=128), seed=0, dtype=torch.float64)
    perm = cluster_permutation(K, n_clusters=8, seed=0)
    assert perm.shape == (2, 128)
    ident = torch.arange(128)
    for h in range(2):
        assert torch.equal(perm[h].sort().values, ident)  # a true permutation


def test_permutation_leaves_dense_invariant():
    # Dense attention is order-free, so permuting the cache cannot move the exact output.
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=96), seed=1, dtype=torch.float64)
    q = q * 4.0
    perm = cluster_permutation(K, n_clusters=12, seed=1)
    K_c, V_c = apply_permutation(K, perm), apply_permutation(V, perm)
    torch.testing.assert_close(dense(q, K, V), dense(q, K_c, V_c), rtol=1e-12, atol=1e-12)


# --- unbiasedness test (clustered layout) ------------------------------------

@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("B", [2, 8])
def test_clustered_block_unbiased(B, method):
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=128), seed=0, dtype=torch.float64)
    q = q * 4.0  # concentrate the weights so clustering has hot/cold blocks to find
    perm = cluster_permutation(K, n_clusters=16, seed=0)
    K_c, V_c = apply_permutation(K, perm), apply_permutation(V, perm)

    ref = dense(q, K_c, V_c).to(torch.float64)  # == dense(q, K, V), order-free
    est = collect_estimates("santa_block", q, K_c, V_c, S=32, runs=2000, base_seed=500,
                            B=B, method=method)
    mean, stderr = mc_mean_and_stderr(est)
    z = (mean - ref).abs() / stderr.clamp_min(1e-300)
    assert z.max().item() < 5.0, f"B={B} method={method}: max|z|={z.max().item():.2f}"
