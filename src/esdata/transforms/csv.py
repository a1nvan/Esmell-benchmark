"""CSV/DataFrame-level dataset transforms."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from esdata.layers import SampleId, Transform, sample_operation
from esdata.transforms.channels import DEFAULT_RESISTANCE_CHANNELS


def _split_user_label_cycle(dataframe: pd.DataFrame) -> None:
    if "UserLabel" not in dataframe.columns:
        return

    user_label_parts = dataframe["UserLabel"].astype("string").str.split(",", n=1, expand=True)
    cycle = (
        user_label_parts[1]
        if 1 in user_label_parts.columns
        else pd.Series(pd.NA, index=dataframe.index, dtype="string")
    )

    dataframe["UserLabel"] = user_label_parts[0]
    dataframe["Cycle"] = cycle.astype("string").str.strip()
    dataframe.dropna(subset=["UserLabel", "Cycle"], inplace=True)


def _detect_max_omh(resistance_columns_df: pd.DataFrame) -> int:
    default_max_resistances = (10500000, 26500000)
    max_resistance = resistance_columns_df.max().max()

    max_omh = default_max_resistances[0]
    for r in default_max_resistances:
        if max_resistance < r:
            max_omh = r
            break

    return max_omh


def _preprocess_csv_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["Time"] = pd.to_datetime(dataframe["Timestamp"], unit="ms")

    _split_user_label_cycle(dataframe)

    for column in dataframe.columns:
        if dataframe[column].dtype == "object":
            dataframe[column] = (
                dataframe[column]
                .map({"OPEN": 1, "CLOSE": 0, "ON": 1, "OFF": 0})
                .fillna(dataframe[column])
            )

    resistance_columns = [f"R{i}" for i in range(1, 18)]
    existed_resistance_columns = list(set(resistance_columns) & set(dataframe.columns.to_list()))
    max_omh = _detect_max_omh(dataframe[existed_resistance_columns])
    for column in existed_resistance_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce").fillna(max_omh)

    if not dataframe.empty:
        dataframe.dropna(axis=1, how="all", inplace=True)

    dataframe["T_mean_adc"] = dataframe[["Chip_T(R1)", "Chip_T(R2)"]].mean(axis=1)
    if "Chip_T0_C" in dataframe.columns:
        dataframe["T_mean"] = dataframe[["Chip_T0_C", "Chip_T1_C"]].mean(axis=1)
    else:
        dataframe["T_mean"] = dataframe["T_mean_adc"]

    cleaned_columns = [
        "Timestamp",
        "Time",
        "Chip_temp_set",
        "Chip_grad_set",
        "Chip_function_set",
        "T_mean_adc",
        "T_mean",
        "P_analit",
        "T_analit",
        "h_analit",
        "Chip_T0_C",
        "Chip_T1_C",
        "UserLabel",
        "Cycle",
    ] + resistance_columns

    cleaned = dataframe[[column for column in cleaned_columns if column in dataframe.columns]]
    return cleaned.drop_duplicates(subset=["Timestamp"])


class DefaultCSVPreprocessing(Transform):
    """Expose cleaned CSV data as the node's primary sample payload."""

    resistance_channels = DEFAULT_RESISTANCE_CHANNELS

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent=parent)

    @sample_operation
    def clean_csv(self, sample_id: SampleId):
        raw_csv = self.parent.load_csv(sample_id)
        return _preprocess_csv_dataframe(raw_csv)


__all__ = ["DefaultCSVPreprocessing"]
