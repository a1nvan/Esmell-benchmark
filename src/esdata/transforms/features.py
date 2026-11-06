"""Feature-construction transforms."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Optional, Sequence

import numpy as np

from esdata.layers import SampleId, Transform, sample_operation


class CanonicalSubtractionFeatures(Transform):
    """Build mask-aware subtraction features from aligned signals."""

    def __init__(
        self,
        features: Sequence[Any] = ((0, 2),),
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent=parent)
        self._feature_operations = self._normalize_features(features)
        self.feature_spec = tuple(specification for _, specification in self._feature_operations)

    @staticmethod
    def _is_diff_spec(value: Any) -> bool:
        return (
            isinstance(value, (tuple, list))
            and len(value) in (2, 3)
            and all(isinstance(index, Integral) for index in value)
        )

    @classmethod
    def _normalize_features(cls, features: Sequence[Any]):
        """Normalize input into explicitly tagged diff/hysteresis operations."""
        if cls._is_diff_spec(features):
            items = (features,)
        elif isinstance(features, (tuple, list)):
            items = features
        else:
            raise ValueError("features must be a tuple or list")

        operations = []
        for item in items:
            if cls._is_diff_spec(item):
                operations.append(("diff", tuple(item)))
                continue

            if (
                isinstance(item, (tuple, list))
                and len(item) == 2
                and all(cls._is_diff_spec(spec) for spec in item)
            ):
                operations.append(("hysteresis", (tuple(item[0]), tuple(item[1]))))
                continue

            raise ValueError(f"invalid feature specification: {item!r}")

        if not operations:
            raise ValueError("no subtraction features were configured")
        return tuple(operations)

    @sample_operation
    def _subtracted(self, sample_id: SampleId):
        signals = self.parent.features(sample_id)
        temperature_masks = self.parent.temperature_masks(sample_id)

        def calculate_diff(feature):
            if len(feature) == 2:
                a_i, b_i = feature
                c_i = None
            elif len(feature) == 3:
                a_i, b_i, c_i = feature
            else:
                raise ValueError(f"feature specification must contain 2 or 3 indices: {feature!r}")

            reverse = b_i < 0
            b_i = abs(b_i)

            mask_a = np.asarray(temperature_masks[a_i]).astype(bool)
            mask_b = np.asarray(temperature_masks[b_i]).astype(bool)
            ys_a = np.asarray(signals[a_i]).astype(float, copy=False)
            ys_b = np.asarray(signals[b_i]).astype(float, copy=False)

            if (
                mask_a.ndim != 1
                or mask_b.ndim != 1
                or mask_a.shape != mask_b.shape
                or ys_a.ndim != 2
                or ys_b.ndim != 2
                or ys_a.shape != ys_b.shape
                or ys_a.shape[1] != mask_a.shape[0]
            ):
                raise ValueError(
                    f"incompatible feature shapes for feature specification {feature!r}"
                )

            common_mask = mask_a & mask_b
            ys_c = None
            if c_i is not None:
                mask_c = np.asarray(temperature_masks[c_i]).astype(bool)
                ys_c = np.asarray(signals[c_i]).astype(float, copy=False)
                if (
                    mask_c.ndim != 1
                    or mask_c.shape != mask_a.shape
                    or ys_c.ndim != 2
                    or ys_c.shape != ys_a.shape
                ):
                    raise ValueError(
                        f"incompatible third feature shapes for specification {feature!r}"
                    )
                common_mask = common_mask & mask_c

            diff = np.zeros_like(ys_a, dtype=float)
            if ys_c is None:
                diff[:, common_mask] = ys_a[:, common_mask] - ys_b[:, common_mask]
            else:
                diff[:, common_mask] = (
                    (ys_a[:, common_mask] + ys_c[:, common_mask]) / 2.0
                ) - ys_b[:, common_mask]

            if reverse:
                diff *= -1.0

            return diff, common_mask

        feature_blocks = []
        mask_blocks = []

        for operation, specification in self._feature_operations:
            if operation == "diff":
                diff, common_mask = calculate_diff(specification)
                feature_blocks.append(diff)
                mask_blocks.append(common_mask)
                continue

            first_diff, first_mask = calculate_diff(specification[0])
            second_diff, second_mask = calculate_diff(specification[1])
            common_mask = first_mask & second_mask
            hysteresis = np.zeros_like(first_diff, dtype=float)
            hysteresis[:, common_mask] = first_diff[:, common_mask] - second_diff[:, common_mask]
            feature_blocks.append(hysteresis)
            mask_blocks.append(common_mask)

        return np.concatenate(feature_blocks, axis=0), mask_blocks

    @sample_operation
    def features(self, sample_id: SampleId):
        return self._subtracted(sample_id)[0]

    @sample_operation
    def temperature_masks(self, sample_id: SampleId):
        return self._subtracted(sample_id)[1]


__all__ = ["CanonicalSubtractionFeatures"]
