"""Ways to keep fitted clusters current during generation (offline schemes)."""

from __future__ import annotations

import torch

from ssa.attn.clusterkv import spherical_kmeans
from ssa.harness.refresh_replay import assign_scheme
from _fixtures import Geom, make_qkv

C0 = 16


def _X(m=900, seed=0):
    _, K, _ = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=m), seed=seed)
    X = K.permute(1, 0, 2).contiguous()
    return X - X.mean(1, keepdim=True)


def test_frozen_assigns_every_key_to_its_nearest_prompt_centroid():
    X = _X()
    a, n_c = assign_scheme(X, G=300, scheme="frozen", C0=C0)
    _, cent = spherical_kmeans(X[:, :600], C=C0)
    nearest = torch.einsum("hmd,hcd->hmc", torch.nn.functional.normalize(X, dim=-1), cent).argmax(-1)
    assert torch.equal(a, nearest) and n_c == C0


def test_append_gives_each_full_block_of_new_keys_its_own_clusters():
    X = _X(seed=1)
    a, n_c = assign_scheme(X, G=320, scheme="append", C0=C0, block=160, per_cluster=32)
    assert n_c == C0 + 2 * 5                                        # two blocks of 160 keys, 5 clusters each
    assert a[:, :580].max() < C0                                    # prompt keys stay in prompt clusters
    assert a[:, 580:740].min() >= C0 and a[:, 580:740].max() < C0 + 5
    assert a[:, 740:900].min() >= C0 + 5


def test_split_keeps_every_cluster_at_or_below_the_cap_and_adds_clusters():
    X = _X(seed=2)
    a, n_c = assign_scheme(X, G=500, scheme="split", C0=C0)
    cap = 2 * (400 // C0)
    for h in range(X.shape[0]):
        counts = torch.bincount(a[h])
        assert counts.max() <= cap + 64                             # checked after each block of 64 keys
    assert n_c > C0
    assert a.min() >= 0


def test_drift_and_refit_cover_every_key_with_the_prompt_cluster_count():
    X = _X(seed=3)
    for scheme in ("drift", "refit"):
        a, n_c = assign_scheme(X, G=300, scheme=scheme, C0=C0, block=64)
        assert n_c == C0 and a.shape == X.shape[:2] and a.max() < C0 and a.min() >= 0
