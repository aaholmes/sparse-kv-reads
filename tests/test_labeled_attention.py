"""Reference label-weighted attention (the specification the Triton kernel must match)."""

from __future__ import annotations

import math

import pytest
import torch

from ssa.attn import attn
from ssa.attn.labeled import label_weighted_attention
from ssa.attn.sphere_state import SphereState
from _fixtures import Geom, make_qkv


def _cache(q, K, V):
    """ssa layout [n, H_kv, d] -> engine cache layout [H_kv, n, d] (a view, no copy)."""
    return q, K.permute(1, 0, 2), V.permute(1, 0, 2)


def test_all_ones_equals_dense():
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=70), seed=0)
    qc, Kc, Vc = _cache(q, K, V)
    labels = torch.randint(0, 5, (2, 70), generator=torch.Generator().manual_seed(0)).to(torch.int16)
    w = torch.ones(2, 6, dtype=torch.float64)
    out = label_weighted_attention(qc, Kc, Vc, labels, w)
    torch.testing.assert_close(out, attn(q, K, V, impl="dense"), rtol=1e-10, atol=1e-12)


def _naive(q, Kc, Vc, labels, w):
    H, d = q.shape
    H_kv, n, _ = Kc.shape
    G = H // H_kv
    out = torch.zeros(H, d, dtype=torch.float64)
    for h in range(H):
        kv = h // G
        wr = w[h] if w.shape[0] == H else w[kv]
        num = torch.zeros(d, dtype=torch.float64)
        den = 0.0
        s = (Kc[kv].double() @ q[h].double()) / math.sqrt(d)
        m = float(s.max())
        for j in range(n):
            wj = float(wr[int(labels[kv, j])])
            e = wj * math.exp(float(s[j]) - m)
            num += e * Vc[kv, j].double()
            den += e
        out[h] = num / den
    return out


@pytest.mark.parametrize("per_head", [False, True])
def test_matches_naive_formula_with_general_weights(per_head):
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=40), seed=1)
    qc, Kc, Vc = _cache(q, K, V)
    g = torch.Generator().manual_seed(1)
    labels = torch.randint(0, 7, (2, 40), generator=g).to(torch.int16)
    w = torch.rand(8 if per_head else 2, 8, generator=g, dtype=torch.float64) * 3
    w[:, 2] = 0.0                                                   # a skipped bin
    w[:, 7] = 1.0                                                   # the exact-set label
    out = label_weighted_attention(qc, Kc, Vc, labels, w)
    torch.testing.assert_close(out, _naive(qc, Kc, Vc, labels, w), rtol=1e-10, atol=1e-12)


def test_zero_weight_rows_do_not_affect_output():
    q, K, V = make_qkv(Geom(H=4, H_kv=2, d=16, n_k=50), seed=2)
    qc, Kc, Vc = _cache(q, K, V)
    labels = (torch.arange(50) % 4).to(torch.int16).expand(2, -1).contiguous()
    w = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]] * 2, dtype=torch.float64)
    a = label_weighted_attention(qc, Kc, Vc, labels, w)
    K2, V2 = Kc.clone(), Vc.clone()
    skip = (labels[0] == 1) | (labels[0] == 3)
    K2[:, skip] = 1e3                                               # garbage in unread rows
    V2[:, skip] = float("nan")
    b = label_weighted_attention(qc, K2, V2, labels, w)
    torch.testing.assert_close(a, b)


def test_bf16_inputs_accumulate_in_float32():
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=64), seed=3, dtype=torch.bfloat16)
    qc, Kc, Vc = _cache(q, K, V)
    labels = torch.zeros(2, 64, dtype=torch.int16)
    w = torch.ones(2, 1)
    out = label_weighted_attention(qc, Kc, Vc, labels, w)
    assert out.dtype == torch.bfloat16
    ref = attn(q.double(), K.double(), V.double(), impl="dense")
    torch.testing.assert_close(out.double(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("group", ["sum_share", "per_head"])
def test_sphere_state_labels_and_weights_reproduce_read(group):
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=260), seed=4)
    K = K + 1.5
    st = SphereState(C=16, window=6, delta=0.03)
    st.observe(K[:200])
    out_ref, _ = st.read(q, K[:200], V[:200], budget=0.2, group=group)
    labels, w = st.labels_and_weights(q, n=200, budget=0.2, group=group)
    assert labels.shape == (2, 200) and labels.dtype == torch.int16
    assert w.shape == ((8 if group == "per_head" else 2), 17)
    qc, Kc, Vc = _cache(q, K[:200], V[:200])
    out = label_weighted_attention(qc, Kc, Vc, labels, w)
    torch.testing.assert_close(out, out_ref, rtol=1e-10, atol=1e-12)
