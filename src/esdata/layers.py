"""Composable dataset layers for the dataset API.

This module defines the independent node API used by the dataset module:

``Source``
    Root dataset node.  Sources own an ordered collection of stable sample
    IDs and provide the small protocol needed by transforms.
``Transform``
    A lazy proxy node which delegates capabilities to its parent dataset.
``Chain``
    A reusable sequence of transforms.
"""

from __future__ import annotations

import copy
from functools import wraps
from contextlib import contextmanager
from typing import Any, Hashable, Iterable, Iterator, Optional


SampleId = Hashable


class LayerError(RuntimeError):
    """Raised when a bound layer cannot process one sample."""

    def __init__(self, layer: str, method: str, sample_id: SampleId, message: str) -> None:
        self.layer = layer
        self.method = method
        self.sample_id = sample_id
        super().__init__(f"{layer}.{method} failed for sample_id={sample_id!r}: {message}")


def sample_operation(method):
    """Decorate a bound layer method that processes one sample."""

    @wraps(method)
    def wrapped(self, sample_id, *args, **kwargs):
        with self._error_context(method.__name__, sample_id):
            return method(self, sample_id, *args, **kwargs)

    return wrapped


class Source:
    """Base class for a root node in the new dataset architecture.

    Concrete sources must populate ``self.ids`` and implement ``metadata`` and
    ``load_csv``.  The protocol is intentionally small: source-specific
    archive or storage operations stay in the concrete source implementation.

    IDs are the canonical way to address samples.  Sample payloads are exposed
    through explicitly named capabilities such as ``load_csv`` and ``label``;
    this base class does not define a generic payload method.
    """

    def __init__(self, ids: Optional[Iterable[SampleId]] = None) -> None:
        self._set_ids(()) if ids is None else self._set_ids(ids)

    def _set_ids(self, ids: Iterable[SampleId]) -> None:
        ordered_ids = tuple(ids)
        if len(set(ordered_ids)) != len(ordered_ids):
            raise ValueError("sample IDs must be unique")
        self.ids = ordered_ids

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[SampleId]:
        return iter(self.ids)

    def id_at(self, position: int) -> SampleId:
        """Return the stable ID at a positional index."""
        return self.ids[position]

    def metadata(self, sample_id: SampleId) -> dict:
        """Return metadata for ``sample_id``.

        Concrete sources must implement this method.
        """
        raise NotImplementedError

    def load_csv(self, sample_id: SampleId) -> Any:
        """Load the raw CSV payload for ``sample_id``.

        CSV is currently the source contract used by this project.  A future
        non-CSV source can introduce a narrower capability protocol while
        retaining the same node and transform machinery.
        """
        raise NotImplementedError

    def label(self, sample_id: SampleId) -> Any:
        """Return the target label for ``sample_id``.

        Concrete sources must implement this method.
        """
        raise NotImplementedError

    def raw_csv(self, sample_id: SampleId) -> Any:
        """Alias for the source's raw CSV capability."""
        return self.load_csv(sample_id)

    def iter_metadata(self) -> Iterator[dict]:
        for sample_id in self.ids:
            yield self.metadata(sample_id)

    def __rshift__(self, layer: Any):
        """Apply a :class:`Transform` or :class:`Chain` to this source."""
        return _apply_layer(self, layer)


class Transform(Source):
    """Base class for a lazy, delegating dataset transform.

    Transform instances are configured first and bound to a parent dataset
    when used with ``dataset >> transform``.  Binding makes a shallow copy so
    one configured transform can safely be reused in multiple branches.

    A subclass can add or override named capabilities.  Any capability not
    implemented by the subclass is delegated to ``self._shadowed``.
    """

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__()
        self._shadowed = parent
        if parent is not None:
            self._set_ids(parent.ids)

    @property
    def parent(self) -> Any:
        """The dataset wrapped by this bound transform."""
        self._require_parent()
        return self._shadowed

    def bind(self, parent: Any):
        """Return an independent bound copy of this transform."""
        bound = copy.copy(self)
        bound._shadowed = parent
        bound._set_ids(parent.ids)
        return bound

    def __call__(self, parent: Any):
        """Bind this transform to a dataset."""
        return self.bind(parent)

    def _require_parent(self) -> None:
        if self._shadowed is None:
            raise RuntimeError(
                f"{type(self).__name__} is not bound to a dataset; "
                "apply it with dataset >> transform"
            )

    @contextmanager
    def _error_context(self, method: str, sample_id: SampleId):
        """Add layer, method, and sample-ID context to processing failures."""
        self._require_parent()
        try:
            yield
        except LayerError:
            raise
        except Exception as exc:
            raise LayerError(type(self).__name__, method, sample_id, str(exc)) from exc

    @sample_operation
    def metadata(self, sample_id: SampleId) -> dict:
        return self._shadowed.metadata(sample_id)

    @sample_operation
    def load_csv(self, sample_id: SampleId) -> Any:
        return self._shadowed.load_csv(sample_id)

    @sample_operation
    def label(self, sample_id: SampleId) -> Any:
        return self._shadowed.label(sample_id)

    def raw_csv(self, sample_id: SampleId) -> Any:
        return self.load_csv(sample_id)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown capabilities to the shadowed dataset."""
        shadowed = self.__dict__.get("_shadowed")
        if shadowed is None:
            raise AttributeError(name)
        return getattr(shadowed, name)


class Chain:
    """Reusable sequence of new-architecture transforms."""

    def __init__(self, *transforms: Any) -> None:
        self.transforms = tuple(transforms)
        for transform in self.transforms:
            if not isinstance(transform, (Transform, Chain)):
                raise TypeError(
                    "Chain items must be Transform or Chain instances, "
                    f"got {type(transform).__name__}"
                )

    def bind(self, parent: Any):
        dataset = parent
        for transform in self.transforms:
            dataset = transform.bind(dataset)
        return dataset

    def __call__(self, parent: Any):
        return self.bind(parent)

    def __repr__(self) -> str:
        names = ", ".join(type(transform).__name__ for transform in self.transforms)
        return f"Chain({names})"


def _apply_layer(parent: Any, layer: Any):
    if not isinstance(layer, (Transform, Chain)):
        raise TypeError(
            "dataset >> layer expects a Transform or Chain, "
            f"got {type(layer).__name__}"
        )
    return layer.bind(parent)
