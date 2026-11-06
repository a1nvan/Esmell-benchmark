"""Dataset transforms that split samples into labeled or thermal segments."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from esdata.layers import SampleId, Transform, sample_operation


class UserLabelCycleSegments(Transform):
    """Split a dataframe into segments by unique label/cycle combinations."""

    def __init__(
        self,
        columns: Union[str, Sequence[str]] = ("UserLabel", "Cycle"),
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent=parent)
        self.group_columns = [columns] if isinstance(columns, str) else list(columns)

    @sample_operation
    def segments(self, sample_id: SampleId):
        dataframe = self.parent.clean_csv(sample_id)

        if not self.group_columns:
            raise ValueError("at least one grouping column is required")

        missing_columns = [
            column for column in self.group_columns if column not in dataframe.columns
        ]
        if missing_columns:
            raise KeyError(f"missing grouping columns: {missing_columns}")

        dataframe_valid = dataframe.dropna(subset=self.group_columns)
        if dataframe_valid.empty:
            raise ValueError("sample contains no rows with valid segment labels")

        groupby_keys = (
            self.group_columns[0] if len(self.group_columns) == 1 else self.group_columns
        )
        segments = [
            segment.copy().reset_index(drop=True)
            for _, segment in dataframe_valid.groupby(groupby_keys, sort=False, dropna=True)
        ]
        if not segments:
            raise ValueError("sample produced no segments")
        return segments


def _split_segment_data_on_cooling_and_heating(
    dataframe: pd.DataFrame,
    temp_col: str = "T_mean",
    smooth_window: int = 11,
    eps: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return the longest cooling and heating runs from one segment."""
    work = dataframe.copy()

    temperature = work[temp_col].rolling(smooth_window, center=True, min_periods=1).mean()
    delta_temperature = temperature.diff()

    work["_state"] = np.where(
        delta_temperature > eps, 1, np.where(delta_temperature < -eps, -1, 0)
    )
    work["_segment"] = (work["_state"] != work["_state"].shift()).cumsum()

    segment_sizes = work.groupby(["_segment", "_state"]).size().reset_index(name="length")

    cooling_segment = (
        segment_sizes[segment_sizes["_state"] == -1]
        .sort_values("length", ascending=False)
        .iloc[0]["_segment"]
    )
    heating_segment = (
        segment_sizes[segment_sizes["_state"] == 1]
        .sort_values("length", ascending=False)
        .iloc[0]["_segment"]
    )

    cooling_dataframe = work[work["_segment"] == cooling_segment].copy()
    heating_dataframe = work[work["_segment"] == heating_segment].copy()

    return (
        cooling_dataframe.drop(columns=["_state", "_segment"]),
        heating_dataframe.drop(columns=["_state", "_segment"]),
    )


class CoolingHeatingSegments(Transform):
    """Split each cycle segment into its longest cooling and heating phases."""

    def __init__(
        self,
        temp_col: str = "T_mean",
        smooth_window: int = 1,
        eps: float = 0.2,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent=parent)
        self.temp_col = temp_col
        self.smooth_window = smooth_window
        self.eps = eps

    @sample_operation
    def segments(self, sample_id: SampleId):
        payload = self.parent.segments(sample_id)
        split_segments = []
        for segment in payload:
            cooling_dataframe, heating_dataframe = _split_segment_data_on_cooling_and_heating(
                segment,
                temp_col=self.temp_col,
                smooth_window=self.smooth_window,
                eps=self.eps,
            )
            # print(f"{len(cooling_dataframe) = }, {len(heating_dataframe) = }")
            split_segments.extend(
                [
                    cooling_dataframe.reset_index(drop=True),
                    heating_dataframe.reset_index(drop=True),
                ]
            )

        if not split_segments:
            raise ValueError("sample produced no cooling/heating segments")
        return split_segments


__all__ = [
    "CoolingHeatingSegments",
    "UserLabelCycleSegments",
]
