"""The registry of every dataset UltraBench can load.

Datasets come in three flavours. A processed dataset is published ready
to use on the Hugging Face Hub. A raw dataset publishes its original
files on the Hub and is preprocessed on first use. A local dataset
cannot be republished at all, so the user downloads it themselves and
supplies the path.

Adding a dataset means adding one entry here and, for a raw or local
dataset, one loader module.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

DatasetName = Literal[
    "AUL",
    "BUS-BRA",
    "CAMUS",
    "FetalPlanes",
    "MASLD",
    "PSFHS",
    "STMUS",
    "TN3K",
    "COVID-BLUES",
    "MMOTU",
    "GBCU",
    "OpenKidney",
]

IMAGE_SIZE = 256
"""The height and width the test split is resized to."""

DEFAULT_SPLITS = ("train", "test", f"test.{IMAGE_SIZE}")
"""The splits every dataset in the collection publishes."""


class DatasetType(Enum):
    """How a dataset reaches the user."""

    PROCESSED = auto()
    RAW = auto()
    LOCAL = auto()


@dataclass(frozen=True)
class DatasetConfig:
    """Everything `load_dataset` needs to validate and load a dataset.

    Args:
        type: How the dataset reaches the user.
        hub_id: The Hugging Face Hub repository ID. Set for processed
            and raw datasets, None for local ones.
        loader: The loader as a "module:function" string, resolved on
            dispatch so that importing the package does not import the
            preprocessing dependencies. Set for raw and local datasets,
            None for processed ones.
        subsets: The subset names, or an empty tuple for a dataset with
            a single subset.
        splits: The split names the dataset publishes.
    """

    type: DatasetType
    hub_id: str | None = None
    loader: str | None = None
    subsets: tuple[str, ...] = ()
    splits: tuple[str, ...] = DEFAULT_SPLITS


_REGISTRY: dict[str, DatasetConfig] = {
    # Processed -- preprocessed and ready on the Hugging Face Hub.
    "AUL": DatasetConfig(type=DatasetType.PROCESSED, hub_id="UltraBench/AUL"),
    "BUS-BRA": DatasetConfig(type=DatasetType.PROCESSED, hub_id="UltraBench/BUS-BRA"),
    "CAMUS": DatasetConfig(
        type=DatasetType.PROCESSED,
        hub_id="UltraBench/CAMUS",
        subsets=("classification", "segmentation"),
    ),
    "FetalPlanes": DatasetConfig(
        type=DatasetType.PROCESSED,
        hub_id="UltraBench/FetalPlanes",
        subsets=("fetal", "fetal_brain"),
    ),
    "MASLD": DatasetConfig(type=DatasetType.PROCESSED, hub_id="UltraBench/MASLD"),
    "PSFHS": DatasetConfig(type=DatasetType.PROCESSED, hub_id="UltraBench/PSFHS"),
    "STMUS": DatasetConfig(type=DatasetType.PROCESSED, hub_id="UltraBench/STMUS"),
    "TN3K": DatasetConfig(type=DatasetType.PROCESSED, hub_id="UltraBench/TN3K"),
    # Raw -- original files on the Hub, preprocessed locally on first use.
    "COVID-BLUES": DatasetConfig(
        type=DatasetType.RAW,
        hub_id="UltraBench/COVID-BLUES-raw",
        loader="ultrabench2.datasets.covid_blues:load_covid_blues",
    ),
    "MMOTU": DatasetConfig(
        type=DatasetType.RAW,
        hub_id="UltraBench/MMOTU-raw",
        loader="ultrabench2.datasets.mmotu:load_mmotu",
    ),
    # Local -- the user must supply the raw data path.
    "GBCU": DatasetConfig(
        type=DatasetType.LOCAL, loader="ultrabench2.datasets.gbcu:load_gbcu"
    ),
    "OpenKidney": DatasetConfig(
        type=DatasetType.LOCAL,
        loader="ultrabench2.datasets.open_kidney:load_open_kidney",
    ),
}
"""Every dataset name UltraBench accepts, mapped to its configuration."""
