"""Generation of the ground-truth CSVs used to score predictions.

Ground truth is built from the `test.256` split published for each
dataset rather than from the ground-truth CSVs published alongside it.
Those published segmentation CSVs encode full-resolution masks despite
their `test.256` name, so scoring 256x256 predictions against them
silently produces a near-zero Dice score.

The published classification CSVs are usable, but CAMUS publishes two of
them and only the subset-prefixed one carries the `sequence_id` column
needed to pool frames into sequences. Building every ground truth here
sidesteps both traps.
"""

import csv
import logging
from collections.abc import Sequence
from pathlib import Path

import datasets
import numpy as np

from ultrabench.core.aggregation import format_sequence_id
from ultrabench.core.rle import mask_to_rle
from ultrabench.evaluate.tasks import TaskSpec, get_task, resolve_tasks

logger = logging.getLogger(__name__)

TEST_SPLIT_TEMPLATE = "test.{image_size}"
"""The dataset split holding the resized test images and masks."""

FILENAME_KEY = "filename"
"""The dataset column holding an example's image identifier."""


def ground_truth_path(spec: TaskSpec, output_dir: Path) -> Path:
    """Locate the ground-truth CSV for a task.

    Args:
        spec: The task to locate the ground truth for.
        output_dir: The directory holding the generated ground truth.

    Returns:
        The path the task's ground-truth CSV is written to.
    """
    return Path(output_dir) / f"{spec.name}.csv"


def _load_test_split(spec: TaskSpec) -> datasets.Dataset:
    """Load the resized test split of a task's dataset.

    Args:
        spec: The task whose test split to load.

    Returns:
        The test split, restricted to the columns needed to build the
        ground truth so that the images are never decoded.
    """
    split = TEST_SPLIT_TEMPLATE.format(image_size=spec.image_size)
    data = datasets.load_dataset(
        spec.dataset, name=spec.subset, split=split, revision=spec.revision
    )

    columns = [FILENAME_KEY, spec.target_key, *(spec.sequence_by or ())]

    return data.select_columns(columns)


def _write_classification_ground_truth(
    spec: TaskSpec, data: datasets.Dataset, output_path: Path
) -> int:
    """Write the ground-truth CSV for a classification task.

    Args:
        spec: The task to write the ground truth for.
        data: The task's test split.
        output_path: Destination path for the CSV file.

    Returns:
        The number of rows written.
    """
    fieldnames = ["image_id", "label"]
    if spec.sequence_by is not None:
        fieldnames.append("sequence_id")

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in data:
            row = {
                "image_id": record[FILENAME_KEY],
                "label": record[spec.target_key],
            }
            if spec.sequence_by is not None:
                row["sequence_id"] = format_sequence_id(record, list(spec.sequence_by))
            writer.writerow(row)

    return len(data)


def _write_segmentation_ground_truth(
    spec: TaskSpec, data: datasets.Dataset, output_path: Path
) -> int:
    """Write the ground-truth CSV for a segmentation task.

    Writes one row per (image, foreground class) pair. Examples whose
    mask is missing are skipped.

    Args:
        spec: The task to write the ground truth for.
        data: The task's test split.
        output_path: Destination path for the CSV file.

    Returns:
        The number of rows written.

    Raises:
        ValueError: If a mask is not square at the task's image size,
            which would make it unscoreable against the predictions.
    """
    expected_shape = (spec.image_size, spec.image_size)
    rows_written = 0

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["image_id", "class_id", "encoded_pixels"]
        )
        writer.writeheader()

        for record in data:
            target = record[spec.target_key]
            if target is None:
                continue

            mask = np.asarray(target)
            if mask.shape != expected_shape:
                raise ValueError(
                    f"Mask for '{record[FILENAME_KEY]}' in task '{spec.name}' has "
                    f"shape {mask.shape}, expected {expected_shape}. The wrong "
                    f"dataset split may have been loaded."
                )

            for class_id in spec.foreground_classes:
                writer.writerow(
                    {
                        "image_id": record[FILENAME_KEY],
                        "class_id": class_id,
                        "encoded_pixels": mask_to_rle(
                            (mask == class_id).astype(np.uint8)
                        ),
                    }
                )
                rows_written += 1

    return rows_written


def build_ground_truth(
    task: str | TaskSpec, output_dir: Path, overwrite: bool = False
) -> Path:
    """Generate the ground-truth CSV for one task.

    Args:
        task: The task, or its name.
        output_dir: The directory to write the CSV to.
        overwrite: Whether to regenerate the CSV if it already exists.

    Returns:
        The path of the written ground-truth CSV.
    """
    spec = get_task(task) if isinstance(task, str) else task
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = ground_truth_path(spec, output_dir)

    if output_path.exists() and not overwrite:
        logger.info("Ground truth for '%s' already exists, skipping.", spec.name)
        return output_path

    data = _load_test_split(spec)

    if spec.task_type == "classification":
        rows = _write_classification_ground_truth(spec, data, output_path)
    else:
        rows = _write_segmentation_ground_truth(spec, data, output_path)

    logger.info("Wrote %d ground-truth rows for '%s'.", rows, spec.name)

    return output_path


def build_all_ground_truth(
    output_dir: Path,
    tasks: Sequence[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Generate the ground-truth CSVs for a set of tasks.

    Args:
        output_dir: The directory to write the CSVs to.
        tasks: The tasks to generate ground truth for, or None for every
            task in the benchmark.
        overwrite: Whether to regenerate CSVs that already exist.

    Returns:
        A dictionary mapping each task name to its ground-truth CSV.
    """
    paths = {}
    for name in resolve_tasks(tasks):
        paths[name] = build_ground_truth(name, output_dir, overwrite=overwrite)

    return paths
