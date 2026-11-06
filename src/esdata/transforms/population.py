"""Eager capability population and invalid-sample filtering."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Optional

from tqdm.auto import tqdm

from esdata.layers import LayerError, SampleId, Transform


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PopulationFailure:
    """Lightweight description of one sample rejected during population."""

    sample_id: SampleId
    requested_method: str
    layer: Optional[str]
    method: Optional[str]
    error_type: str
    message: str

    @classmethod
    def from_exception(
        cls,
        sample_id: SampleId,
        requested_method: str,
        error: Exception,
    ) -> "PopulationFailure":
        if isinstance(error, LayerError):
            layer = error.layer
            method = error.method
            underlying_error = error.__cause__ or error
        else:
            layer = None
            method = None
            underlying_error = error

        return cls(
            sample_id=sample_id,
            requested_method=requested_method,
            layer=layer,
            method=method,
            error_type=type(underlying_error).__name__,
            message=str(error),
        )

    def as_dict(self) -> dict:
        """Return a serializable representation of this failure."""
        return asdict(self)


class Populate(Transform):
    """Eagerly evaluate capabilities and optionally exclude failed samples.

    Population runs when the transform is bound with ``dataset >> populate``.
    Every configured method is called for every parent sample, in method order.
    A sample is successful when all calls return without raising; return values,
    including ``None``, are not otherwise interpreted.

    ``on_error`` controls what happens after the first failed method for a
    sample:

    ``"exclude"``
        Record the failure and omit the sample from this layer's ``ids``.
    ``"keep"``
        Record the failure but retain the sample ID.
    ``"raise"``
        Immediately re-raise the original exception.

    Put this layer after :class:`~esdata.caching.CacheToRAM` to warm that cache
    as a side effect of validation.  Set ``progress=True`` to display an
    environment-aware tqdm progress bar while samples are evaluated.
    """

    _ERROR_POLICIES = frozenset(("exclude", "keep", "raise"))

    def __init__(
        self,
        methods: tuple,
        on_error: str = "exclude",
        progress: bool = False,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent=parent)
        if not isinstance(methods, tuple):
            raise TypeError("methods must be a tuple of method names")
        if any(not isinstance(method, str) for method in methods):
            raise TypeError("methods must contain only strings")
        if len(set(methods)) != len(methods):
            raise ValueError("methods must not contain duplicates")
        if not isinstance(on_error, str) or on_error not in self._ERROR_POLICIES:
            choices = ", ".join(repr(policy) for policy in sorted(self._ERROR_POLICIES))
            raise ValueError(f"on_error must be one of {choices}")
        if not isinstance(progress, bool):
            raise TypeError("progress must be a boolean")

        self.methods = tuple(methods)
        self.on_error = on_error
        self.progress = progress
        self.failures: tuple[PopulationFailure, ...] = ()
        self.excluded_ids: tuple[SampleId, ...] = ()
        self._population_total = 0

    def bind(self, parent: Any):
        bound = super().bind(parent)
        bound.failures = ()
        bound.excluded_ids = ()
        bound._population_total = 0
        return bound.populate()

    def _resolve_methods(self):
        resolved = []
        for name in self.methods:
            try:
                method = getattr(self.parent, name)
            except AttributeError as exc:
                raise AttributeError(
                    f"cannot populate missing capability {name!r}"
                ) from exc
            if not callable(method):
                raise TypeError(f"populated attribute {name!r} is not callable")
            resolved.append((name, method))
        return tuple(resolved)

    def populate(self):
        """Run population again and atomically replace IDs and diagnostics.

        Re-population always starts from the immediate parent's IDs, allowing
        transient failures to be retried.  The configured transform must first
        be bound to a dataset.
        """
        self._require_parent()
        resolved_methods = self._resolve_methods()
        parent_ids = tuple(self.parent.ids)
        kept_ids = []
        excluded_ids = []
        failures = []

        sample_ids = tqdm(parent_ids, desc="Populating dataset", unit="sample", disable=not self.progress)
        for sample_id in sample_ids:
            failure = None
            for requested_method, method in resolved_methods:
                try:
                    method(sample_id)
                except Exception as error:
                    failure = PopulationFailure.from_exception(
                        sample_id, requested_method, error
                    )
                    if self.on_error == "raise":
                        logger.error(
                            "Population failed for sample_id=%r in %s: %s",
                            sample_id,
                            requested_method,
                            error,
                        )
                        raise
                    logger.warning(
                        "Population failed for sample_id=%r in %s; %s sample: %s",
                        sample_id,
                        requested_method,
                        "excluding" if self.on_error == "exclude" else "keeping",
                        error,
                    )
                    break

            if failure is None or self.on_error == "keep":
                kept_ids.append(sample_id)
            else:
                excluded_ids.append(sample_id)
            if failure is not None:
                failures.append(failure)

        self._set_ids(kept_ids)
        self.failures = tuple(failures)
        self.excluded_ids = tuple(excluded_ids)
        self._population_total = len(parent_ids)
        logger.info(
            "Population complete: total=%d kept=%d excluded=%d failures=%d",
            self._population_total,
            len(self.ids),
            len(self.excluded_ids),
            len(self.failures),
        )
        return self

    @property
    def population_report(self) -> dict:
        """Return summary counts for the most recent population pass."""
        return {
            "total": self._population_total,
            "kept": len(self.ids),
            "excluded": len(self.excluded_ids),
            "failures": len(self.failures),
        }


__all__ = ["Populate", "PopulationFailure"]
