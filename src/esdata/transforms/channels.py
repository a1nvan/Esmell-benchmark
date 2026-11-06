"""Resistance-channel schema transforms."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from esdata.layers import Transform


DEFAULT_RESISTANCE_CHANNELS = tuple(f"R{i}" for i in range(1, 18))


def _channel_tuple(value: Sequence[str], argument: str) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{argument} must be a sequence of channel names, not a string")

    channels = tuple(value)
    if any(not isinstance(channel, str) for channel in channels):
        raise TypeError(f"{argument} must contain only channel names")
    if len(set(channels)) != len(channels):
        raise ValueError(f"{argument} contains duplicate channel names")
    return channels


class SelectRChannels(Transform):
    """Select active resistance channels by name.

    Selection changes the dataset's ``resistance_channels`` schema capability;
    it does not modify CSV payloads.  Consumers such as
    :class:`CanonicalTempAlign` use the capability to avoid processing channels
    that are not active.

    Included channels retain their order in the parent schema, independently of
    the order in which they are passed to ``include``.
    """

    def __init__(
        self,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        parent: Optional[Any] = None,
    ) -> None:
        if include is not None and exclude is not None:
            raise ValueError("include and exclude are mutually exclusive")

        self.include = _channel_tuple(include, "include") if include is not None else None
        self.exclude = _channel_tuple(exclude, "exclude") if exclude is not None else None
        self._resistance_channels: Optional[Tuple[str, ...]] = None
        super().__init__(parent=parent)

        if parent is not None:
            self._resistance_channels = self._resolve(parent)

    def _resolve(self, parent: Any) -> Tuple[str, ...]:
        try:
            available = _channel_tuple(parent.resistance_channels, "resistance_channels")
        except AttributeError as exc:
            raise AttributeError(
                "SelectRChannels requires its parent to expose resistance_channels"
            ) from exc

        requested = self.include if self.include is not None else self.exclude
        if requested is not None:
            unknown = tuple(channel for channel in requested if channel not in available)
            if unknown:
                raise ValueError(f"unknown resistance channels: {unknown}")

        if self.include is not None:
            included = set(self.include)
            selected = tuple(channel for channel in available if channel in included)
        elif self.exclude is not None:
            excluded = set(self.exclude)
            selected = tuple(channel for channel in available if channel not in excluded)
        else:
            selected = available

        if not selected:
            raise ValueError("channel selection must leave at least one resistance channel")
        return selected

    def bind(self, parent: Any):
        bound = super().bind(parent)
        bound._resistance_channels = bound._resolve(parent)
        return bound

    @property
    def resistance_channels(self) -> Tuple[str, ...]:
        if self._resistance_channels is None:
            self._require_parent()
            self._resistance_channels = self._resolve(self.parent)
        return self._resistance_channels


__all__ = ["DEFAULT_RESISTANCE_CHANNELS", "SelectRChannels"]
