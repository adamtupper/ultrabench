"""Utility functions for dataset processing."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io
import skimage.transform
from tqdm.auto import tqdm

from ultrabench.core.rle import mask_to_rle


def save_version_info(output_dir: str, version: str) -> None:
    """Create a file with the UltraBench version and current date.

    Args:
        output_dir: The path to the output directory.
        version: The version string.
    """
    with open(os.path.join(output_dir, "version.txt"), "w") as f:
        f.write(f"UltraBench Version: {version}\n")
        f.write(f"Created: {os.popen('date').read().strip()}\n")


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
    raise ValueError(
        f"Expected a single-channel or 3-channel image, got shape {image.shape}."
    )


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
    records: list[dict], output_path: Path
) -> None:
    """Create a classification ground-truth CSV from a list of records.

    Generate a CSV with columns "filename" and "label", where each row
    corresponds to one example.

    Args:
        records: List of record dicts, each containing "filename" and
            "label" keys.
        output_path: Destination path for the CSV file (including
            filename).
    """
    rows = [
        {"image_id": r["filename"], "label": r["label"]}
        for r in tqdm(records, desc="Generating classification ground truth CSV")
    ]
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
    """Resize an image to fit within size×size, then center-pad to exactly size×size.

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
