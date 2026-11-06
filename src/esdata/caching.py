"""Caching layers for dataset capabilities."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from esdata.layers import Transform


class CacheToRAM(Transform):
    """Cache selected parent capabilities in RAM.

    ``cache_size=None`` creates an unbounded cache.  A non-negative integer
    creates an LRU cache with that maximum size.  Cache keys are the complete
    method arguments, so this layer is intended for deterministic, callable
    per-sample capabilities such as ``features`` and ``label``.
    """

    def __init__(
        self,
        methods: tuple,
        cache_size: Optional[int] = None,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent=parent)
        if not isinstance(methods, tuple):
            raise TypeError("methods must be a tuple of method names")
        if any(not isinstance(method, str) for method in methods):
            raise TypeError("methods must contain only strings")
        if cache_size is not None and (
            isinstance(cache_size, bool)
            or not isinstance(cache_size, int)
            or cache_size < 0
        ):
            raise ValueError("cache_size must be None or a non-negative integer")

        self.methods = tuple(methods)
        self.cache_size = cache_size
        self._cache_wrappers = {}

    def bind(self, parent: Any):
        bound = super().bind(parent)
        bound._cache_wrappers = {}
        return bound

    def _get_cached_method(self, name: str):
        wrappers = object.__getattribute__(self, "_cache_wrappers")
        if name in wrappers:
            return wrappers[name]

        parent = object.__getattribute__(self, "_shadowed")
        if parent is None:
            self._require_parent()

        method = getattr(parent, name)
        if not callable(method):
            raise TypeError(f"cached attribute {name!r} is not callable")

        cached_method = lru_cache(
            maxsize=object.__getattribute__(self, "cache_size")
        )(method)
        wrappers[name] = cached_method
        return cached_method

    def __getattribute__(self, name: str) -> Any:
        if not name.startswith("_"):
            try:
                methods = object.__getattribute__(self, "methods")
            except AttributeError:
                methods = ()
            if name in methods:
                return object.__getattribute__(self, "_get_cached_method")(name)
        return object.__getattribute__(self, name)

    def cache_info(self):
        """Return cache statistics for each configured method."""
        wrappers = object.__getattribute__(self, "_cache_wrappers")
        return {
            name: method.cache_info()
            for name, method in wrappers.items()
        }

    def clear_cache(self) -> None:
        """Clear all method caches created by this layer."""
        wrappers = object.__getattribute__(self, "_cache_wrappers")
        for method in wrappers.values():
            method.cache_clear()
