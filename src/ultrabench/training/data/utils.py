"""Data utilities."""

from functools import partial

import albumentations as A
import datasets
import numpy as np
from datasets import Dataset
from sklearn.model_selection import GroupShuffleSplit

from ultrabench.training.data.transforms import albumentations_transforms


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


def get_splits(
    path: str,
    input_key: str,
    target_key: str,
    train_transform: A.Compose,
    test_transform: A.Compose,
    train_size: float = 0.8,
    stratify_by: str | None = None,
    group_by: str | None = None,
    filter_by: str | None = None,
    subset: str | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """Get the train, val and test splits for a Hugging Face dataset.

    Args:
        path: The path to a local directory or dataset repository on the
            Hugging Face Hub.
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
        subset: Optional subset/configuration of the dataset to load.

    Returns:
        The train, val and test splits.
    """
    train_val_data = datasets.load_dataset(path, name=subset, split="train")

    if filter_by is not None:
        train_val_data = train_val_data.filter(lambda x: x[filter_by] is not None)

    train_data, val_data = train_test_split(
        train_val_data,
        train_size=train_size,
        stratify_by=stratify_by,
        group_by=group_by,
    )
    test_data = datasets.load_dataset(path, name=subset, split="test.256")

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
