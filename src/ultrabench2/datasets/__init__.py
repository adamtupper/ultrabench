"""Loading the datasets of the UltraBench collection."""

from .load_dataset import load_dataset
from .registry import DatasetConfig, DatasetName, DatasetType

__all__ = [
    "DatasetConfig",
    "DatasetName",
    "DatasetType",
    "load_dataset",
]
