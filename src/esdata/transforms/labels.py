"""Label and metadata transforms."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from esdata.layers import SampleId, Transform, sample_operation


def _simplify_label(label: Any):
    """Simplify concentration/distance-specific labels locally."""
    return str(label).split("_d", 1)[0]


def _normalize_label_filter(labels: Union[str, Sequence[str], None]):
    if labels is None:
        return None
    if isinstance(labels, str):
        return {labels}
    return set(labels)


class SimplifyLabels(Transform):
    """Simplify labels in graph forwarding."""

    def __init__(self, meta_key: str = "classname", parent: Optional[Any] = None) -> None:
        super().__init__(parent=parent)
        self.meta_key = meta_key

    @sample_operation
    def label(self, sample_id: SampleId):
        label = self.parent.label(sample_id)
        return None if label is None else _simplify_label(label)


class MonikiLabels(Transform):
    """Replace participant codes with labels from a Moniki JSON file."""

    def __init__(
        self,
        json_path: Union[str, Path],
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent=parent)
        with Path(json_path).open(encoding="utf-8") as file:
            data = json.load(file)

        try:
            mapping = data["code_to_label_mapping"]
        except (TypeError, KeyError) as exc:
            raise ValueError("Moniki JSON must contain 'code_to_label_mapping'") from exc
        if not isinstance(mapping, dict):
            raise ValueError("'code_to_label_mapping' must be a JSON object")

        self.code_to_label_mapping = mapping
        self._normalized_mapping = {int(str(code[1:])): label for code, label in mapping.items()}
        if len(self._normalized_mapping) != len(mapping):
            raise ValueError(
                "'code_to_label_mapping' contains duplicate codes after normalization"
            )

    @sample_operation
    def label(self, sample_id: SampleId):
        code = self.parent.label(sample_id)
        code = code.replace("-", "_")
        clean_code = code.split("_")[0]
        label = self._normalized_mapping.get(int(str(clean_code[1:])), "Неизвестно")
        return label


class MonikiSimplifyLabels(Transform):
    """Replace participant codes with labels from a Moniki JSON file."""

    def __init__(
        self,
        json_path: Union[str, Path],
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent=parent)
        with Path(json_path).open(encoding="utf-8") as file:
            data = json.load(file)

        try:
            mapping = data["code_to_label_mapping"]
        except (TypeError, KeyError) as exc:
            raise ValueError("Moniki JSON must contain 'code_to_label_mapping'") from exc
        if not isinstance(mapping, dict):
            raise ValueError("'code_to_label_mapping' must be a JSON object")

        self.code_to_label_mapping = mapping
        self._normalized_mapping = {int(str(code[1:])): label for code, label in mapping.items()}
        if len(self._normalized_mapping) != len(mapping):
            raise ValueError(
                "'code_to_label_mapping' contains duplicate codes after normalization"
            )

    @sample_operation
    def label(self, sample_id: SampleId):
        code = self.parent.label(sample_id)
        code = code.replace("-", "_")
        clean_code = code.split("_")[0]
        label = self._normalized_mapping.get(int(str(clean_code[1:])), None)
        if label is None:
            return None
        LABELS = [
            "ХБП",
            "Гастрит",
            "ХСН",
            "Кетоз",
            "ХБП + Гастрит",
            "ХСН + Кетоз",
            "ХБП + ХСН",
            "ХБП + Кетоз",
        ]

        if label in LABELS:
            label = "Нозология"

        return label


class FilterLabels(Transform):
    """Expose only parent samples matching the requested labels."""

    def __init__(
        self,
        include: Union[str, Sequence[str], None] = None,
        exclude: Union[str, Sequence[str], None] = None,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent=parent)
        self.include_labels = _normalize_label_filter(include)
        self.exclude_labels = _normalize_label_filter(exclude)

    def bind(self, parent: Any):
        bound = super().bind(parent)
        kept_ids = []
        for sample_id in parent.ids:
            label = bound.label(sample_id)
            if bound._keeps_label(label):
                kept_ids.append(sample_id)
        bound._set_ids(kept_ids)
        return bound

    def _keeps_label(self, label: Any) -> bool:
        if self.include_labels is not None and label not in self.include_labels:
            return False
        if self.exclude_labels is not None and label in self.exclude_labels:
            return False
        return True


class FilterSamplesByDatetime(Transform):
    """Expose only parent samples matching the requested labels."""

    def __init__(
        self,
        begin: datetime | None = None,
        end: datetime | None = None,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent=parent)
        self.begin_datetime = begin
        self.end_datetime = end

        def _filter(begin: datetime | None, end: datetime | None):
            def default_func(t: datetime):
                return True

            _func = default_func

            if begin is None and end is not None:

                def func(t: datetime):
                    return t <= end

                _func = func

            elif begin is not None and end is None:

                def func(t: datetime):
                    return t >= begin

                _func = func

            elif begin is not None and end is not None:

                def func(t: datetime):
                    return t >= begin and t <= end

                _func = func

            return _func

        self._filter_by_time = _filter(self.begin_datetime, self.end_datetime)

    def bind(self, parent: Any):
        bound = super().bind(parent)
        kept_ids = []
        for sample_id in parent.ids:
            meta = bound.metadata(sample_id)
            if bound._keeps_label(meta):
                kept_ids.append(sample_id)
        bound._set_ids(kept_ids)
        return bound

    def _keeps_label(self, meta: Any) -> bool:
        return self._filter_by_time(meta["datetime"])


__all__ = ["FilterLabels", "MonikiLabels", "SimplifyLabels", "FilterSamplesByDatetime"]
