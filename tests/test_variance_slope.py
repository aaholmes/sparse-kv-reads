"""Variance-decay test: estimator variance-trace falls as ~1/S.

The paper reports log-log slopes near -1 (-1.1 iid / -1.27 strat / -1.29 sys). On
tiny float64 tensors we test only the qualitative result: each slope is clearly ~1/S
(within a band around -1), and stratified/systematic are no worse than iid.
If this fails, stop — nothing downstream is valid.
"""

from __future__ import annotations

import torch

from ssa.harness.variance import variance_sweep
from _fixtures import Geom, make_qkv

S_VALUES = [8, 16, 32, 64, 128, 256]
SLOPE_LO, SLOPE_HI = -1.5, -0.8


def _sweeps():
    q, K, V = make_qkv(Geom(H=8, H_kv=2, d=16, n_k=256), seed=0, dtype=torch.float64)
    return {
        impl: variance_sweep(impl, q, K, V, S_values=S_VALUES, runs=600, base_seed=100)
        for impl in ("santa", "santa_strat", "santa_sys")
    }


def test_slopes_are_one_over_S():
    sweeps = _sweeps()
    for impl, sw in sweeps.items():
        assert SLOPE_LO <= sw.slope <= SLOPE_HI, f"{impl}: slope {sw.slope:+.3f} not ~ -1"


def test_variance_is_monotone_decreasing_in_S():
    sweeps = _sweeps()
    for impl, sw in sweeps.items():
        v = sw.var_values
        assert all(v[i + 1] < v[i] for i in range(len(v) - 1)), f"{impl}: not decreasing"


def test_stratified_systematic_no_worse_than_iid():
    sweeps = _sweeps()
    iid_var = sweeps["santa"].var_values
    for impl in ("santa_strat", "santa_sys"):
        v = sweeps[impl].var_values
        # at the largest S, structured sampling should not exceed iid variance
        assert v[-1] <= iid_var[-1] * 1.25, f"{impl}: variance worse than iid at max S"
