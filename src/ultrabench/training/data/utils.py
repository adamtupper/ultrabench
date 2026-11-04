"""Data utilities."""

from functools import partial
from pathlib import Path

import albumentations as A
import numpy as np
from datasets import Dataset
from sklearn.model_selection import GroupShuffleSplit

from ultrabench.core.aggregation import join_sequence_values
from ultrabench.datasets import load_dataset
from ultrabench.training.data.transforms import albumentations_transforms

TEST_SPLIT = "test.256"
"""The test split downstream training and evaluation both use."""


def resolve_task_path(path: str | None, data_dir: str) -> str | None:
    """Resolve a task's raw data path against the run's data directory.

    A local dataset cannot be published, so its task config names the
    directory its raw data was staged into. A relative path is taken to
    be relative to the data directory; an absolute one is used as given,
    so that a command-line override still works.

    Args:
        path: The path from the task config, or None for a dataset that
            does not need one.
        data_dir: The run's data directory.

    Returns:
        The resolved path, or None if the task named none.
    """
    if path is None:
        return None

    return str(Path(data_dir) / path)


def train_test_split(
    data: Dataset,
    train_size: float = 0.25,
    stratify_by: str | None = None,
    group_by: str | None = None,
    seed: int | None = 42,
) -> tuple[Dataset, Dataset]:
    """Split data into training and test sets.

    Split the dataset into training and test sets, optionally
    stratifying or grouping by a column. Does not support both
    stratification and grouping at the same time.

    Args:
        data: The dataset to split.
        train_size: The proportion of the dataset to allocate to the
            training set. Defaults to 0.25.
        stratify_by: Optional column to stratify examples by.
        group_by: Optional column to group examples by.
        seed: The random seed to use for reproducibility. Defaults to
            42.

    Returns:
        A tuple containing the training and test datasets.
    """
    if stratify_by is not None and group_by is not None:
        raise ValueError("Cannot stratify and group at the same time.")

    if group_by is not None:
        gss = GroupShuffleSplit(train_size=train_size, n_splits=1, random_state=seed)
        groups = data[group_by]
        train_idx, test_idx = next(gss.split(np.arange(len(data)), groups=groups))
        return data.select(train_idx), data.select(test_idx)
    else:
        splits = data.train_test_split(
            train_size=train_size, stratify_by_column=stratify_by, seed=seed
        )

        return splits["train"], splits["test"]


def add_sequence_id(data: Dataset, sequence_by: list[str]) -> Dataset:
    """Add a "sequence_id" column built from the given columns.

    Reads only the grouping columns rather than mapping over the whole
    dataset, so image columns are not decoded and re-encoded.

    Args:
        data: The dataset to add the column to.
        sequence_by: The columns that jointly identify a sequence, for
            example ["patient", "view"].

    Returns:
        The dataset with an added "sequence_id" column.

    Raises:
        ValueError: If the dataset is missing one of the given columns.
    """
    missing = [column for column in sequence_by if column not in data.column_names]
    if missing:
        raise ValueError(
            f"Cannot build sequence IDs: dataset has no column(s) {missing}. "
            f"Available columns: {data.column_names}."
        )

    values = [data[column] for column in sequence_by]
    sequence_ids = [join_sequence_values(row) for row in zip(*values)]

    return data.add_column("sequence_id", sequence_ids)


def get_splits(
    name: str,
    input_key: str,
    target_key: str,
    train_transform: A.Compose,
    test_transform: A.Compose,
    train_size: float = 0.8,
    stratify_by: str | None = None,
    group_by: str | None = None,
    filter_by: str | None = None,
    sequence_by: list[str] | None = None,
    subset: str | None = None,
    revision: str | None = None,
    path: str | None = None,
    cache_dir: str | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Get the train, val and test splits for an UltraBench dataset.

    Args:
        name: The dataset to load, as registered in
            `ultrabench.datasets`.
        input_key: The column of the the dataset containing the input
            images.
        target_key: The column of the dataset containing the target
            labels or masks.
        train_transform: The Albumentations transformation to apply to
            the training data.
        test_transform: The Albumentations transformation to apply to
            the test data.
        train_size: The proportion of the dataset to allocate to the
            training set.
        stratify_by: Optional column to stratify by for train/val
            splitting.
        filter_by: Optional column to filter the training data by. If
            specified, only rows where this column is not null will be
            included.
        group_by: Optional column to to group by for train/val
            splitting.
        sequence_by: Optional columns that jointly identify the sequence
            an image belongs to. If given, a "sequence_id" column is
            added to every split so that test predictions can be pooled
            per sequence.
        subset: Optional subset/configuration of the dataset to load.
        revision: Optional Git revision (branch, tag or commit) of the
            dataset to load. Defaults to the latest revision.
        path: The directory a local dataset's raw data was staged into.
            Required for GBCU and Open Kidney, which are not published.
        cache_dir: The directory to cache processed datasets in.
            Defaults to $ULTRABENCH_CACHE, then `~/.ultrabench`.

    Returns:
        The train, val and test splits.
    """
    train_val_data = load_dataset(
        name,
        path=path,
        subset=subset,
        split="train",
        revision=revision,
        cache_dir=cache_dir,
    )

    if filter_by is not None:
        train_val_data = train_val_data.filter(lambda x: x[filter_by] is not None)

    train_data, val_data = train_test_split(
        train_val_data,
        train_size=train_size,
        stratify_by=stratify_by,
        group_by=group_by,
    )
    test_data = load_dataset(
        name,
        path=path,
        subset=subset,
        split=TEST_SPLIT,
        revision=revision,
        cache_dir=cache_dir,
    )

    train_data = train_data.rename_columns(
        {
            input_key: "image",
            target_key: "target",
        }
    )
    val_data = val_data.rename_columns(
        {
            input_key: "image",
            target_key: "target",
        }
    )
    test_data = test_data.rename_columns(
        {
            input_key: "image",
            target_key: "target",
        }
    )

    columns = ["image", "target", "scan_mask", "filename"]
    if sequence_by is not None:
        train_data = add_sequence_id(train_data, sequence_by)
        val_data = add_sequence_id(val_data, sequence_by)
        test_data = add_sequence_id(test_data, sequence_by)
        columns.append("sequence_id")

    train_data = train_data.with_transform(
        partial(albumentations_transforms, transform=train_transform), columns=columns
    )
    val_data = val_data.with_transform(
        partial(albumentations_transforms, transform=test_transform), columns=columns
    )
    test_data = test_data.with_transform(
        partial(albumentations_transforms, transform=test_transform), columns=columns
    )

    return train_data, val_data, test_data
