"""Bytes the cluster kernels read at each decode step, beyond the key and value rows they attend over.

Everything is expressed in key rows (``d`` components of ``key_bits`` each), per KV head per step,
so it adds to the count of K and V rows read:

  summaries   per cluster in use: its mean direction (``summary_bits`` per component) plus the
              longest and shortest key length and the key count (float32 each), read by scoring
  centroids   per cluster in use: its centroid (``summary_bits`` per component), read to place the
              one key inserted this step; with fixed random directions these are one float32
              table shared by all ``heads``
  labels      a 16-bit cluster label per cached token, scanned to list the rows to read
  rows read   a 32-bit position and a 16-bit label for each row read
  insertion   the inserted key, and its cluster's float32 direction sum (read and written)
"""

from __future__ import annotations


def overhead_rows(*, n, rows, clusters, d: int, summary_bits: int = 8, fitted: bool = True, heads: int = 1,
                  key_bits: int = 16):
    """Key-row equivalents read per KV head per step besides the ``rows`` K and V rows themselves.
    ``n`` cached tokens, ``clusters`` in use; numbers or tensors."""
    row = d * key_bits / 8
    vec = d * summary_bits / 8
    summaries = clusters * (vec + 12)
    centroids = clusters * vec if fitted else clusters * 4 * d / heads
    return (summaries + centroids + 2 * n + 6 * rows + row + 8 * d) / row


def kv_read_fraction(*, n, rows, clusters, d: int, **kw):
    """Fraction of the K+V cache's bytes read per step: ``rows`` rows of each, plus the overhead."""
    return (2 * rows + overhead_rows(n=n, rows=rows, clusters=clusters, d=d, **kw)) / (2 * n)
