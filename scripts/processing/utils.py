"""Helpers shared by the maintainer processing scripts.

These cover what only a maintainer needs: capping and subsampling the
training splits of the published datasets, and publishing to the
Hugging Face Hub. The helpers the package itself needs to build a
dataset live in `ultrabench.datasets.builders.utils`.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import RepositoryNotFoundError

from ultrabench.datasets.builders.utils import DATASET_REVISION

MAX_TRAIN_SAMPLES = 1000
"""The maximum number of examples in a published training split."""


def verify_revision_unpublished(repo_id: str, revision: str = DATASET_REVISION) -> None:
    """Verify that a dataset revision has not already been published.

    Datasets are versioned with Git tags on the Hugging Face Hub. A tag is
    immutable in practice, since experiments pin it, so publishing over an
    existing revision must be a deliberate act.

    Args:
        repo_id: The dataset repository ID on the Hugging Face Hub.
        revision: The dataset revision that is about to be published.

    Raises:
        ValueError: If the revision has already been published.
    """
    try:
        refs = HfApi().list_repo_refs(repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        return

    if revision in {tag.name for tag in refs.tags}:
        raise ValueError(
            f"Revision {revision} of {repo_id} has already been published. "
            f"To republish it, first delete the tag with `hf repos tag delete "
            f"{repo_id} {revision} --type dataset`."
        )


def cap_train_split_by_patient(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_samples: int = MAX_TRAIN_SAMPLES,
    *,
    patient_col: str = "patient",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cap the training split by patient, moving the overflow to the test set.

    Patients are considered in a seeded random order and kept while they fit
    within the cap, so no patient is ever split across the two splits. The
    overflow is appended after the existing test examples, which keeps the
    previous test split as a prefix of the returned one.

    Args:
        train_df: The training split metadata.
        test_df: The test split metadata.
        max_samples: The maximum number of training examples.
        patient_col: The column holding the patient IDs.
        seed: The random seed used to order the patients.

    Returns:
        The capped training split and the expanded test split.
    """
    if len(train_df) <= max_samples:
        return train_df, test_df

    counts = train_df[patient_col].value_counts()
    order = np.random.default_rng(seed).permutation(len(counts))

    kept_patients = []
    total = 0
    for index in order:
        n_examples = int(counts.iloc[index])
        if total + n_examples <= max_samples:
            kept_patients.append(counts.index[index])
            total += n_examples

    kept = train_df[patient_col].isin(kept_patients)

    return (
        train_df[kept].reset_index(drop=True),
        pd.concat([test_df, train_df[~kept]], ignore_index=True),
    )


def subsample_balanced_by_class(
    df: pd.DataFrame,
    max_samples: int = MAX_TRAIN_SAMPLES,
    *,
    label_col: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Subsample examples so that the class distribution becomes more uniform.

    Examples are removed from the largest classes first, so that classes
    smaller than the resulting per-class quota are kept whole. For example,
    classes of 800 and 400 examples capped at 1,000 become 600 and 400.

    Args:
        df: The metadata to subsample.
        max_samples: The maximum number of examples to keep.
        label_col: The column holding the class labels.
        seed: The random seed used to select examples within a class.

    Returns:
        The subsampled metadata, in the same order as the input.
    """
    if len(df) <= max_samples:
        return df

    df = df.reset_index(drop=True)
    quotas = _balanced_quotas(df[label_col].value_counts().to_dict(), max_samples)
    selected = [
        group.sample(n=quotas[label], random_state=seed)
        for label, group in df.groupby(label_col, sort=True)
    ]

    return pd.concat(selected).sort_index().reset_index(drop=True)


def _balanced_quotas(counts: dict, max_samples: int) -> dict:
    """Allocate a per-class quota that spreads examples as evenly as it can.

    Classes are visited from smallest to largest and each is offered an equal
    share of the examples that are still unallocated. A class smaller than its
    share is kept whole and the unused share is passed on to the larger
    classes.

    Args:
        counts: The number of available examples per class label.
        max_samples: The total number of examples to allocate.

    Returns:
        The number of examples to keep per class label.
    """
    quotas = {}
    remaining = max_samples
    ranked = sorted(counts.items(), key=lambda item: (item[1], item[0]))
    for position, (label, count) in enumerate(ranked):
        share = remaining // (len(ranked) - position)
        quotas[label] = min(count, share)
        remaining -= quotas[label]

    return quotas


def subsample_by_group(
    df: pd.DataFrame,
    max_samples: int = MAX_TRAIN_SAMPLES,
    *,
    group_cols: list[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Subsample examples uniformly at random from within each group.

    Every group is given an equal quota, with any remainder handed to a seeded
    random selection of groups. Groups holding fewer examples than their quota
    are kept whole.

    Args:
        df: The metadata to subsample.
        max_samples: The maximum number of examples to keep.
        group_cols: The columns that identify a group.
        seed: The random seed used to select examples within a group.

    Returns:
        The subsampled metadata, in the same order as the input.
    """
    if len(df) <= max_samples:
        return df

    df = df.reset_index(drop=True)
    groups = list(df.groupby(group_cols, sort=True))
    quota, remainder = divmod(max_samples, len(groups))
    extra = set(
        np.random.default_rng(seed).permutation(len(groups))[:remainder].tolist()
    )

    selected = [
        group.sample(
            n=min(len(group), quota + int(position in extra)), random_state=seed
        )
        for position, (_, group) in enumerate(groups)
    ]

    return pd.concat(selected).sort_index().reset_index(drop=True)


def ground_truth_path_in_repo(
    task: str, *, size: int | None = None, prefix: str | None = None
) -> str:
    """Build the name a ground-truth CSV is published under on the Hub.

    A Hugging Face dataset repository holds every ground-truth CSV in its
    root, so the name has to carry the split, and the subset for a
    dataset that has more than one. Building the names here keeps them
    consistent across datasets.

    Args:
        task: The task the CSV covers, for example "segmentation" or
            "classification-pathology".
        size: The height and width of the resized test images, or None
            for the original-resolution test split.
        prefix: The dataset subset the CSV belongs to, or None for a
            dataset with a single subset.

    Returns:
        The path to publish the CSV under, relative to the repository
        root.
    """
    split = "test" if size is None else f"test.{size}"
    name = f"{split}-ground-truth-{task}.csv"

    return f"{prefix}-{name}" if prefix else name


def ground_truth_commit_operations(
    output_path: Path,
    tasks: Mapping[str, str],
    size: int,
    *,
    prefix: str | None = None,
) -> list[CommitOperationAdd]:
    """Build the operations that publish a dataset's ground-truth CSVs.

    Both test splits carry a ground-truth CSV per task, so every task
    yields two operations: one for the original-resolution `test` split
    and one for the resized `test.{size}` split.

    Args:
        output_path: The directory the processed dataset was saved to,
            or the subset directory for a dataset with more than one
            subset.
        tasks: A mapping from the task name, as it appears in the
            published filename, to the name of the CSV within each split
            directory.
        size: The height and width of the resized test images.
        prefix: The dataset subset the CSVs belong to, or None for a
            dataset with a single subset.

    Returns:
        One operation per (test split, task) pair.
    """
    return [
        CommitOperationAdd(
            path_in_repo=ground_truth_path_in_repo(
                task, size=split_size, prefix=prefix
            ),
            path_or_fileobj=output_path / split / filename,
        )
        for split, split_size in (("test", None), (f"test.{size}", size))
        for task, filename in tasks.items()
    ]
