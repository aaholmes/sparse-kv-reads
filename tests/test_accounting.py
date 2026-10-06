"""Bytes read per decode step by the cluster kernels, in units of one 16-bit key row."""

from __future__ import annotations

from ssa.attn.accounting import kv_read_fraction, overhead_rows


def test_overhead_counts_every_per_step_read():
    d, n, rows, c = 128, 32768, 6000.0, 1000.0
    row = 2 * d                                                            # bytes in one key or value row
    got = overhead_rows(n=n, rows=rows, clusters=c, d=d, summary_bits=8)
    want = (c * (d + 12)                                                   # 8-bit direction, two lengths and a count
            + c * d                                                        # 8-bit centroids, for the key inserted
            + 2 * n                                                        # a 16-bit label per cached token
            + 6 * rows                                                     # position and label of each row read
            + row + 2 * 4 * d) / row                                       # the inserted key; its cluster's float32 sum, read and written
    assert abs(got - want) < 1e-9
    f32 = overhead_rows(n=n, rows=rows, clusters=c, d=d, summary_bits=32)
    assert abs((f32 - got) - 2 * c * 3 * d / row) < 1e-9                   # both vectors four times the bytes


def test_shared_directions_are_read_once_for_all_heads():
    a = overhead_rows(n=1000, rows=100.0, clusters=256.0, d=128, summary_bits=8, fitted=False, heads=8)
    b = overhead_rows(n=1000, rows=100.0, clusters=256.0, d=128, summary_bits=8, fitted=True)
    assert a < b


def test_fraction_is_rows_plus_overhead_over_the_cache():
    f = kv_read_fraction(n=1000, rows=200.0, clusters=64.0, d=16, summary_bits=8)
    assert abs(f - (400 + overhead_rows(n=1000, rows=200.0, clusters=64.0, d=16, summary_bits=8)) / 2000) < 1e-12
