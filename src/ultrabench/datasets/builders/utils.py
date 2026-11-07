"""Helpers shared by the builders of the locally processed datasets."""

import csv
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.transform
from tqdm.auto import tqdm

from datasets import DatasetDict

from ultrabench.core.aggregation import format_sequence_id
from ultrabench.datasets.registry import IMAGE_SIZE  # noqa: F401
from ultrabench.core.rle import mask_to_rle

DATASET_REVISION = "0.4"


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

    Labels do not depend on the image resolution, so the same CSV is
    written into both test split directories.

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
    records: Iterable[dict],
    mask_key: str,
    classes: Sequence[int],
    output_path: Path,
    *,
    image_size: int | None,
) -> None:
    """Create a segmentation ground-truth CSV from a set of records.

    Generate a CSV with columns "image_id", "class_id", and
    "encoded_pixels", where each row corresponds to one (image, class)
    pair. "encoded_pixels" is a run-length encoded (RLE) string. Records
    where record[mask_key] is None are skipped.

    Rows are written as the records are consumed, so the records may be
    an iterator that generates its masks lazily.

    Args:
        records: Records, each containing "filename" and a key matching
            mask_key whose value is a numpy uint8 array or None.
        mask_key: The key in each record that contains the mask array.
        classes: The list of foreground class IDs.
        output_path: Destination path for the CSV file (including
            filename).
        image_size: The height and width the masks must have, or None
            for the original-resolution test split, whose masks vary in
            size. A mask encoded at the wrong resolution scores near
            zero against predictions rather than failing, so the
            resolution a CSV claims is checked rather than assumed.

    Raises:
        ValueError: If a mask is not square at image_size.
    """
    expected_shape = None if image_size is None else (image_size, image_size)

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_id", "class_id", "encoded_pixels"],
            lineterminator="\n",
        )
        writer.writeheader()

        for r in tqdm(records, desc="Generating segmentation ground truth CSV"):
            if r[mask_key] is None:
                continue

            mask = np.asarray(r[mask_key])
            if expected_shape is not None and mask.shape != expected_shape:
                raise ValueError(
                    f"Mask for '{r['filename']}' has shape {mask.shape}, expected "
                    f"{expected_shape}. The records of the wrong split may have "
                    f"been passed."
                )

            for cls in classes:
                writer.writerow(
                    {
                        "image_id": r["filename"],
                        "class_id": cls,
                        "encoded_pixels": mask_to_rle((mask == cls).astype(np.uint8)),
                    }
                )


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


@dataclass(frozen=True)
class BuildResult:
    """A processed dataset and the records its ground truth comes from.

    The ground-truth CSVs are built from the record dicts rather than
    from the dataset, because the masks have to be run-length encoded
    from the numpy arrays before the Image feature encodes them to PNG.

    Args:
        dataset: The processed splits.
        test_records: The records of the original-resolution test split.
        test_resized_records: The records of the resized test split.
    """

    dataset: DatasetDict
    test_records: list[dict]
    test_resized_records: list[dict]


def write_test_ground_truth(
    result: BuildResult,
    output_path: Path,
    size: int,
    writer: Callable[[list[dict], Path, int | None], None],
) -> None:
    """Write the ground-truth CSVs of both test splits.

    Args:
        result: The build result holding the test records.
        output_path: The directory the dataset was saved to.
        size: The height and width of the resized test images.
        writer: The dataset's ground-truth writer, called once per test
            split with the split's records, the split directory and the
            size its masks are encoded at, or None for the
            original-resolution split.
    """
    writer(result.test_records, output_path / "test", None)
    writer(result.test_resized_records, output_path / f"test.{size}", size)
