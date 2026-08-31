"""Utility functions for dataset processing."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io
import skimage.transform
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError
from tqdm.auto import tqdm

from ultrabench.core.aggregation import format_sequence_id
from ultrabench.core.rle import mask_to_rle

MAX_TRAIN_SAMPLES = 1000
"""The maximum number of examples in a published training split."""

DATASET_REVISION = "0.2"
"""The default revision of the published datasets that this code produces.

Individual dataset modules may define their own revision when they are
republished independently of the rest of the collection.
"""


def save_version_info(
    output_dir: str, version: str, revision: str = DATASET_REVISION
) -> None:
    """Create a file with the UltraBench version, revision and current date.

    Args:
        output_dir: The path to the output directory.
        version: The version string.
        revision: The dataset revision string.
    """
    with open(os.path.join(output_dir, "version.txt"), "w") as f:
        f.write(f"UltraBench Version: {version}\n")
        f.write(f"Dataset Revision: {revision}\n")
        f.write(f"Created: {os.popen('date').read().strip()}\n")  # ty:ignore[deprecated]


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


def to_3_channel(image: np.ndarray) -> np.ndarray:
    """Convert images to a 3-channel (RGB) array.

    Converts single-channel images (2D or HxWx1) and 4-channel
    (RGBA) images to 3-channel by replicating the single channel or
    dropping the alpha channel, respectively.

    Args:
        image: The input image array.

    Returns:
        A 3-channel array of shape (H, W, 3) with the same dtype as the input.

    Raises:
        ValueError: If the image has an unsupported number of channels.
    """
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    if image.ndim == 3 and image.shape[-1] == 1:
        return np.concatenate([image] * 3, axis=-1)
    if image.ndim == 3 and image.shape[-1] == 3:
        return image
    if image.ndim == 3 and image.shape[-1] == 4:
        return image[:, :, :3]
    raise ValueError(f"Expected a (1, 3, 4)-channel image, got shape {image.shape}.")


def generate_classification_ground_truth(
    examples: list[dict], output_path: Path
) -> None:
    """Create a CSV file with the ground-truth labels.

    Generate a CSV with columns "image_id" and "label", where each row
    corresponds to one example. The "image_id" column contains the image
    identifier (e.g., relative filepath) and the "label" column contains
    the class label.

    Args:
        examples: List of example dicts, each containing `image` and
            `label` keys.
        output_path: Destination path for the CSV file (including
            filename).
    """
    rows = [{"image_id": x["image"], "label": x["label"]} for x in examples]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def generate_segmentation_ground_truth(
    examples: list[dict],
    mask_key: str,
    classes: list[int],
    output_path: Path,
) -> None:
    """Create a segmentation ground truth CSV with RLE-encoded masks.

    Generate a CSV with columns "image_id", "class_id", and
    "encoded_pixels", where each row corresponds to one (image, class)
    pair. The "encoded_pixels" column contains the RLE-encoded binary
    mask for that class, or an empty string if the mask is all zeros.

    Rows are ordered by image: all classes for image 1, then image 2,
    etc.

    Args:
        examples: List of example dicts, each containing `image` and
            [mask_key] keys
        mask_key: The key in the example dicts that contains the path to
            the segmentation mask.
        classes: The list of foreground class IDs.
        output_path: Destination path for the CSV file (including
            the filename).
    """
    rows = []
    for example in examples:
        mask = skimage.io.imread(str(output_path.parent / example[mask_key]))
        for cls in classes:
            binary_mask = (mask == cls).astype(np.uint8)
            encoded = mask_to_rle(binary_mask)
            rows.append(
                {
                    "image_id": example["image"],
                    "class_id": cls,
                    "encoded_pixels": encoded,
                }
            )

    pd.DataFrame(rows).to_csv(output_path, index=False)


def generate_classification_ground_truth_from_records(
    records: list[dict],
    output_path: Path,
    sequence_keys: list[str] | None = None,
) -> None:
    """Create a classification ground-truth CSV from a list of records.

    Generate a CSV with columns "image_id" and "label", where each row
    corresponds to one example. Datasets whose labels apply to a whole
    sequence rather than a single image also get a "sequence_id" column,
    which the evaluation layer uses to pool per-image predictions.

    Args:
        records: List of record dicts, each containing "filename" and
            "label" keys, plus every key in sequence_keys.
        output_path: Destination path for the CSV file (including
            filename).
        sequence_keys: Optional record keys that jointly identify the
            sequence an image belongs to, for example
            ["patient", "view"]. If None, no "sequence_id" column is
            written and the task is evaluated per image.

    Raises:
        KeyError: If a record is missing one of the sequence keys.
    """
    rows = []
    for r in tqdm(records, desc="Generating classification ground truth CSV"):
        row = {"image_id": r["filename"], "label": r["label"]}
        if sequence_keys is not None:
            row["sequence_id"] = format_sequence_id(r, sequence_keys)
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)


def generate_segmentation_ground_truth_from_records(
    records: list[dict],
    mask_key: str,
    classes: list[int],
    output_path: Path,
) -> None:
    """Create a segmentation ground-truth CSV from a list of records.

    Generate a CSV with columns "image_id", "class_id", and
    "encoded_pixels", where each row corresponds to one (image, class)
    pair. "encoded_pixels" is a run-length encoded (RLE) string. Records
    where record[mask_key] is None are skipped.

    Args:
        records: List of record dicts, each containing "filename" and
            a key matching mask_key whose value is a numpy uint8 array
            or None.
        mask_key: The key in each record that contains the mask array.
        classes: The list of foreground class IDs.
        output_path: Destination path for the CSV file (including
            filename).
    """
    rows = []
    for r in tqdm(records, desc="Generating segmentation ground truth CSV"):
        mask = r[mask_key]
        if mask is None:
            continue
        for cls in classes:
            binary_mask = (mask == cls).astype(np.uint8)
            rows.append(
                {
                    "filename": r["filename"],
                    "class_id": cls,
                    "encoded_pixels": mask_to_rle(binary_mask),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def resize_and_pad(image: np.ndarray, size: int, order: int = 1) -> np.ndarray:
    """Resize and pad images to preserve aspect ratio.

    Preserves aspect ratio. Padding is zero (black). Uses
    skimage.transform.resize for the resize step.

    Args:
        image: The input image array of shape (H, W) or (H, W, C).
        size: The target height and width in pixels.
        order: Interpolation order passed to skimage.transform.resize.
            Use 1 (bilinear) for images and 0 (nearest-neighbor) for masks.

    Returns:
        An array of shape (size, size) or (size, size, C) with the same dtype
            as the input.
    """
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = round(h * scale), round(w * scale)
    resized = skimage.transform.resize(
        image,
        (new_h, new_w),
        order=order,
        preserve_range=True,
        anti_aliasing=(order > 0),
    ).astype(image.dtype)
    out_shape = (size, size, image.shape[2]) if image.ndim == 3 else (size, size)
    out = np.zeros(out_shape, dtype=image.dtype)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    out[top : top + new_h, left : left + new_w] = resized
    return out
