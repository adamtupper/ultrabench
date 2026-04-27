"""Divide the Open Kidney dataset into training and test splits using an 8:2
split. For the annotations, use those created by Sonographer 1. Each image is
associated with a distinct patient so we do not need to worry about patient
overlap between the splits. However, we do need to remove 20 duplicate images.
The splits are stratified based on the view label (longitudinal, transverse,
other) to mitigate distribution shifts between the splits.

From inspecting the original code, the labels of the regions mask are as
follows:
    - 0: Background
    - 1: Central Echo Complex
    - 2: Medulla
    - 3: Cortex

Each example (a single image) is represented as an object in one of two JSON
array files (`train_val.json` or `test.json`). Each object has the following
key/value pairs:

    - image:        The path to the image file.
    - scan_mask:    The path to the scan mask file.
    - capsule_mask: The path to the capsule mask file.
    - regions_mask:  The path to the region mask file.
    - transplant:   True if the kidney is transplanted, False otherwise.
    - view:         The view of the kidney (longitudinal, transverse, other).
    - quality:      The quality of the image (unsatisfactory, poor, fair,
                    good).

Usage:
    ultrabench open_kidney RAW_DATA_DIR OUTPUT_DIR
"""

import ast
import json
import os
from importlib.metadata import version

import numpy as np
import pandas as pd
import skimage
import skimage.io as io
import typer
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated

from .utils import resize_and_pad, save_version_info, to_3_channel

__version__ = version("ultrabench")

OUTPUT_NAME = "open_kidney_v{}"
SONOGRAPHER = 1
CLINICAL_METADATA_FILE = f"labels/reviewed_labels_{SONOGRAPHER}.csv"


def generate_scan_mask(image: np.ndarray) -> np.ndarray:
    """Generate a scan mask using morphological operations.

    Args:
        image: The input 2D grayscale image array.

    Returns:
        A binary mask of the scan region.
    """
    mask = image > 0
    for i in range(5):
        mask = skimage.morphology.binary_erosion(mask)
    for i in range(5):
        mask = skimage.morphology.binary_dilation(mask)
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)

    # Keep only the largest connected component
    label = skimage.measure.label(mask)
    regions = skimage.measure.regionprops(label)
    largest_region = max(regions, key=lambda x: x.area)
    mask = label == largest_region.label

    # Reflect the larger half to fill gaps and make the mask symmetric
    _, width = mask.shape
    left_half = mask[:, : width // 2]
    right_half = mask[:, width // 2 :]
    if np.sum(left_half) > np.sum(right_half):
        right_half = np.fliplr(left_half)
    else:
        left_half = np.fliplr(right_half)
    mask = np.hstack((left_half, right_half))

    mask = skimage.morphology.convex_hull_image(mask)
    return mask.astype(np.uint8)


def verify_args(raw_data_dir: str, output_dir: str) -> None:
    """Verify the command line arguments.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
    """
    assert os.path.isdir(raw_data_dir), "raw_data_dir must be an existing directory"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(
        os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    ), "A matching version of the Open Kidney dataset already exists"


def process_open_kidney(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
    size: Annotated[
        int, typer.Option(help="Test image size (height and width in pixels)")
    ] = 256,
) -> None:
    """Prepare the Open Kidney dataset.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        size: The height and width of the test images in pixels.
    """
    verify_args(raw_data_dir, output_dir)

    dataset_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    test_dir_name = f"test_{size}"

    # Load metadata from CSV (no image I/O)
    metadata = pd.read_csv(os.path.join(raw_data_dir, CLINICAL_METADATA_FILE))
    metadata["quality"] = metadata["file_attributes"].apply(
        lambda x: ast.literal_eval(str(x)).get("Quality", None)
    )
    metadata["view"] = metadata["file_attributes"].apply(
        lambda x: ast.literal_eval(str(x)).get("View", None)
    )
    metadata["comments"] = metadata["file_attributes"].apply(
        lambda x: ast.literal_eval(str(x)).get("Comments", None)
    )
    metadata["transplant"] = metadata["comments"].apply(
        lambda x: "transplant" in x.lower()
    )
    metadata["file_id"] = metadata["filename"].map(lambda x: x.split("_")[1])
    metadata.drop(
        columns=["file_attributes", "region_attributes", "region_shape_attributes"],
        inplace=True,
    )
    metadata = metadata.drop_duplicates("file_id")

    print(f"Total # of images: {len(metadata)}")

    # Determine the train/test split from the view label before any file I/O
    train_val_examples, test_examples = train_test_split(
        metadata,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=metadata["view"],
    )

    test_filenames = set(test_examples["filename"])

    # Add relative path columns
    metadata["image"] = metadata["filename"].map(lambda x: os.path.join("images", x))
    metadata["scan_mask"] = metadata["filename"].map(
        lambda x: os.path.join("masks", "scan", x)
    )
    metadata["capsule_mask"] = metadata["filename"].map(
        lambda x: os.path.join("masks", "capsule", x)
    )
    metadata["regions_mask"] = metadata["filename"].map(
        lambda x: os.path.join("masks", "regions", x)
    )
    train_val_examples = metadata.loc[train_val_examples.index]
    test_examples = metadata.loc[test_examples.index]

    # Create output directories
    for split_dir_name in ["train_val", test_dir_name]:
        split_dir = os.path.join(dataset_dir, split_dir_name)
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "masks", "scan"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "masks", "capsule"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "masks", "regions"), exist_ok=True)

    # Process and save images and masks to the correct split directory
    for _, row in metadata.iterrows():
        is_test = row["filename"] in test_filenames
        split_dir = os.path.join(dataset_dir, test_dir_name if is_test else "train_val")

        raw_image = io.imread(os.path.join(raw_data_dir, "images", row["filename"]))
        gray = (
            np.mean(raw_image, axis=-1).astype(np.uint8)
            if raw_image.ndim == 3
            else raw_image
        )
        scan_mask = generate_scan_mask(gray)

        capsule_mask = io.imread(
            os.path.join(
                raw_data_dir,
                "labels",
                f"reviewed_masks_{SONOGRAPHER}",
                "capsule",
                row["filename"],
            )
        )
        regions_mask = io.imread(
            os.path.join(
                raw_data_dir,
                "labels",
                f"reviewed_masks_{SONOGRAPHER}",
                "regions",
                row["filename"],
            )
        )

        image = to_3_channel(raw_image)
        if is_test:
            image = resize_and_pad(image, size, order=1)
            scan_mask = resize_and_pad(scan_mask, size, order=0)
            capsule_mask = resize_and_pad(capsule_mask, size, order=0)
            regions_mask = resize_and_pad(regions_mask, size, order=0)

        io.imsave(os.path.join(split_dir, "images", row["filename"]), image)
        io.imsave(
            os.path.join(split_dir, "masks", "scan", row["filename"]),
            scan_mask,
            check_contrast=False,
        )
        io.imsave(
            os.path.join(split_dir, "masks", "capsule", row["filename"]),
            capsule_mask,
            check_contrast=False,
        )
        io.imsave(
            os.path.join(split_dir, "masks", "regions", row["filename"]),
            regions_mask,
            check_contrast=False,
        )

    print(f"# of training examples: {len(train_val_examples)}")
    print(f"# of test examples: {len(test_examples)}")

    # Save the metadata JSON files inside their split directories
    for split, subset, split_dir in [
        ("train_val", train_val_examples, os.path.join(dataset_dir, "train_val")),
        ("test", test_examples, os.path.join(dataset_dir, test_dir_name)),
    ]:
        with open(os.path.join(split_dir, f"{split}.json"), "w") as f:
            json.dump(subset.to_dict(orient="records"), f, indent=4)

    save_version_info(dataset_dir, __version__)
