"""Data loading, signal transforms, and feature dataset utilities."""

from esdata.dataset import EsmellDataset
from esdata.caching import CacheToRAM
from esdata.layers import Chain, LayerError, Source, Transform, sample_operation
from esdata.transforms import (
    CanonicalSubtractionFeatures,
    CanonicalTempAlign,
    CoolingHeatingSegments,
    DefaultCSVPreprocessing,
    FilterLabels,
    FilterSamplesByDatetime,
    Populate,
    PopulationFailure,
    SelectRChannels,
    MonikiLabels,
    MonikiSimplifyLabels,
    SimplifyLabels,
    UserLabelCycleSegments,
)

__version__ = "0.2.0"

__all__ = [
    "DefaultCSVPreprocessing",
    "SelectRChannels",
    "SimplifyLabels",
    "MonikiLabels",
    "MonikiSimplifyLabels",
    "UserLabelCycleSegments",
    "FilterLabels",
    "FilterSamplesByDatetime",
    "CoolingHeatingSegments",
    "CanonicalTempAlign",
    "CanonicalSubtractionFeatures",
    "EsmellDataset",
    "Source",
    "Transform",
    "LayerError",
    "sample_operation",
    "Chain",
    "CacheToRAM",
    "Populate",
    "PopulationFailure",
]
