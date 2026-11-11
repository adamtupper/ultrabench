"""Process and publish the datasets the package builds locally.

GBCU, MMOTU, Open Kidney and COVID-BLUES are built on the user's machine
by `ultrabench.load_dataset`, so their build code lives in the package
(`ultrabench.datasets.builders`). This module wraps those builders for
maintainers: it writes the dataset to disk and, optionally, publishes it
and its ground-truth CSVs to the Hugging Face Hub.
"""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import HfApi

from ultrabench.datasets.builders.utils import (
    DATASET_REVISION,
    BuildResult,
    save_version_info,
    write_test_ground_truth,
)

from .utils import ground_truth_commit_operations, verify_revision_unpublished


@dataclass(frozen=True)
class LocalDataset:
    """How to build and publish one locally processed dataset.

    Attributes:
        module: The builder module within `ultrabench.datasets.builders`.
        builder: The name of the module's build function.
        ground_truth: A mapping from the task name, as it appears in the
            published filename, to the CSV within each test split.
    """

    module: str
    builder: str
    ground_truth: dict[str, str]


LOCAL_DATASETS = {
    "gbcu": LocalDataset(
        "gbcu",
        "build_gbcu_dataset",
        {"classification": "ground_truth_classification.csv"},
    ),
    "mmotu": LocalDataset(
        "mmotu",
        "build_mmotu_dataset",
        {
            "classification": "ground_truth_classification.csv",
            "segmentation": "ground_truth_segmentation.csv",
        },
    ),
    "open-kidney": LocalDataset(
        "open_kidney",
        "build_open_kidney_dataset",
        {
            "segmentation-capsule": "ground_truth_segmentation_capsule.csv",
            "segmentation-regions": "ground_truth_segmentation_regions.csv",
        },
    ),
    "covid-blues": LocalDataset(
        "covid_blues",
        "build_covid_blues_dataset",
        {"classification": "ground_truth_classification.csv"},
    ),
}


def process_local(
    dataset: str, raw_data_dir: str, output_dir: str, size: int, publish: bool
) -> None:
    """Build a locally processed dataset, save it and optionally publish it.

    Args:
        dataset: The dataset's key in `LOCAL_DATASETS`.
        raw_data_dir: The path to the raw data directory.
        output_dir: The directory to save the processed dataset under.
        size: The height and width of the resized test images in pixels.
        publish: Whether to publish the dataset to the Hugging Face Hub.

    Raises:
        FileNotFoundError: If the raw data or output directory is missing.
        FileExistsError: If the processed dataset already exists.
    """
    spec = LOCAL_DATASETS[dataset]
    module = import_module(f"ultrabench.datasets.builders.{spec.module}")
    build: Callable[..., BuildResult] = getattr(module, spec.builder)
    repo_id: str = module.OUTPUT_NAME

    output_path = Path(output_dir) / repo_id
    if not Path(raw_data_dir).is_dir():
        raise FileNotFoundError(f"{raw_data_dir} is not an existing directory.")
    if not Path(output_dir).is_dir():
        raise FileNotFoundError(f"{output_dir} is not an existing directory.")
    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists.")
    if publish:
        verify_revision_unpublished(repo_id)

    result = build(raw_data_dir, size=size)
    result.dataset.save_to_disk(str(output_path))
    write_test_ground_truth(result, output_path, size, module.write_ground_truth)
    save_version_info(str(output_path), version("ultrabench"))
    print(f"Dataset saved to {output_path}.")

    if publish:
        result.dataset.push_to_hub(
            repo_id,
            commit_message=f"Update the dataset to revision {DATASET_REVISION}.",
            private=True,
        )
        HfApi().create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=ground_truth_commit_operations(
                output_path, spec.ground_truth, size
            ),
            commit_message=(
                f"Update ground truth CSV files for revision {DATASET_REVISION}."
            ),
        )
