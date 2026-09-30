"""The swappable attention interface ``attn(q, K, V, impl=..., **cfg)``.

Implementations register themselves in ``_REGISTRY`` via the ``@register`` decorator;
``attn`` dispatches by name. Implementations: dense, topk, santa, santa_strat, santa_sys,
santa_hybrid, santa_block, skip_k, sphere_skip.
"""

from __future__ import annotations

from typing import Callable

import torch

_REGISTRY: dict[str, Callable] = {}


def register(name: str) -> Callable:
    """Decorator: register an attention implementation under ``name``."""

    def deco(fn: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"attn impl {name!r} already registered")
        _REGISTRY[name] = fn
        return fn

    return deco


def available() -> list[str]:
    """Names of registered implementations."""
    return sorted(_REGISTRY)


def attn(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, *, impl: str, **cfg):
    """Dispatch to a registered attention implementation by name."""
    try:
        fn = _REGISTRY[impl]
    except KeyError:
        raise KeyError(f"unknown attn impl {impl!r}; available: {available()}") from None
    return fn(q, K, V, **cfg)


# Register implementations (import for side effects). Kept at the bottom to avoid
# an import cycle: the impl modules import ``register`` from this module.
from . import dense as _dense  # noqa: E402,F401
from . import santa as _santa  # noqa: E402,F401
from . import hybrid as _hybrid  # noqa: E402,F401
from . import block as _block  # noqa: E402,F401
from . import skip_k as _skip_k  # noqa: E402,F401
from . import sphere_skip as _sphere_skip  # noqa: E402,F401
