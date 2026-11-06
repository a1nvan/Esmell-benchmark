"""Signal-alignment transforms for dataset samples."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from esdata.layers import SampleId, Transform, sample_operation
from esdata.signal import interpolate_data_by_canonical_temp_grid
from esdata.transforms.channels import DEFAULT_RESISTANCE_CHANNELS


class CanonicalTempAlign(Transform):
    """Align segment signals on a fixed canonical temperature grid."""

    def __init__(
        self,
        n_grid: int = 100,
        temp_range: Tuple[float, float] = (168, 395),
        temp_col: str = "T_mean",
        normalize: bool = True,
        conductance: bool = False,
        resistance_channels: Optional[Sequence[str]] = None,
        parent: Optional[Any] = None,
        *,
        r_cols: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(parent=parent)
        if normalize and conductance:
            raise ValueError("normalize and conductance are mutually exclusive")
        if resistance_channels is not None and r_cols is not None:
            raise ValueError("resistance_channels and deprecated r_cols are mutually exclusive")
        if r_cols is not None:
            warnings.warn(
                "r_cols is deprecated; use resistance_channels instead",
                DeprecationWarning,
                stacklevel=2,
            )
            resistance_channels = r_cols
        if isinstance(resistance_channels, (str, bytes)):
            raise TypeError("resistance_channels must be a sequence of channel names")
        if resistance_channels is not None:
            resistance_channels = tuple(resistance_channels)
            if not resistance_channels:
                raise ValueError("resistance_channels must not be empty")
            if any(not isinstance(channel, str) for channel in resistance_channels):
                raise TypeError("resistance_channels must contain only channel names")
            if len(set(resistance_channels)) != len(resistance_channels):
                raise ValueError("resistance_channels contains duplicate channel names")

        self.n_grid = n_grid
        self.temp_range = temp_range
        self.temp_col = temp_col
        self.normalize = normalize
        self.conductance = conductance
        self.configured_resistance_channels = resistance_channels
        self.canonical_temp_grid = np.linspace(temp_range[0], temp_range[1], n_grid)
        self._aligned_cache = None

    def bind(self, parent: Any):
        bound = super().bind(parent)
        # Transform.bind() uses a shallow copy, so never carry aligned data
        # from the configured transform or another parent into this binding.
        bound._aligned_cache = None
        return bound

    @property
    def resistance_channels(self) -> Tuple[str, ...]:
        """Return the ordered resistance-channel schema used for alignment."""
        if self.configured_resistance_channels is not None:
            return self.configured_resistance_channels
        self._require_parent()
        return tuple(getattr(self.parent, "resistance_channels", DEFAULT_RESISTANCE_CHANNELS))

    @sample_operation
    def _aligned(self, sample_id: SampleId):
        # features() and temperature_masks() are separate capabilities, but
        # CanonicalSubtractionFeatures requests them back-to-back for the
        # same sample. Keep only that latest result to avoid running the
        # expensive interpolation twice without retaining the whole dataset.
        if self._aligned_cache is not None:
            cached_sample_id, cached_aligned = self._aligned_cache
            if cached_sample_id == sample_id:
                return cached_aligned

        segments = self.parent.segments(sample_id)
        if not segments:
            raise ValueError("sample contains no segments to align")

        def _get_adc_max(seg) -> float:
            return float(seg.max().max())

        adc_min = 1.0
        adc_max = 10400000.0
        data_columns = list(self.resistance_channels)
        aligned = []

        adc_max = _get_adc_max(segments[0][data_columns])

        for segment in segments:
            _, ys, mask, _ = interpolate_data_by_canonical_temp_grid(
                data_columns,
                segment,
                # segment.iloc[::4, :],
                temp_col=self.temp_col,
                canonical_temp_grid=self.canonical_temp_grid,
            )

            if self.normalize:
                ys /= adc_max
            elif self.conductance:
                valid = mask.astype(bool)[:, np.newaxis]
                resistance = np.clip(ys, adc_min, adc_max)
                ys = np.divide(1.0, resistance, out=np.zeros_like(ys, dtype=float), where=valid)
                ys = np.clip(ys, None, 1.0)
                # log
                ys = np.clip(ys, 1e-12, 1.0)
                ys = np.log(ys)
                # ===
                ys = np.where(valid, ys, 0.0)

            aligned.append((mask, ys.T))

        if not aligned:
            raise ValueError("sample produced no aligned segments")
        self._aligned_cache = (sample_id, aligned)
        return aligned

    @sample_operation
    def features(self, sample_id: SampleId):
        return [ys for _, ys in self._aligned(sample_id)]

    @sample_operation
    def temperature_masks(self, sample_id: SampleId):
        return [mask for mask, _ in self._aligned(sample_id)]


__all__ = ["CanonicalTempAlign"]
