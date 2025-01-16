"""Divide the Open Kidney dataset into training, validation, and test splits using a
7:1:2 split. For the annotations, use those created by Sonographer 1. Each image is
associated with a distinct patient so we do not need to worry about patient overlap
between the splits. However, we do need to remove 20 duplicate images. The splits are
stratified based on the view label (longitudinal, transverse, other) to mitigate
distribution shifts between the splits.

From inspecting the original code, the labels of the regions mask are as follows:
    - 0: Background
    - 1: Central Echo Complex
    - 2: Medulla
    - 3: Cortex

Each example (a single image) is represented as an object in one of three JSON array
files (`train.json`, `validation.json`, or `test.json`). Each object has the following
key/value pairs:

    - image:        The path to the image file.
    - scan_mask:    The path to the scan mask file.
    - capsule_mask: The path to the capsule mask file.
    - regions_mask:  The path to the region mask file.
    - transplant:   True if the kidney is transplanted, False otherwise.
    - view:         The view of the kidney (longitudinal, transverse, other).
    - quality:      The quality of the image (unsatisfactory, poor, fair, good).

Usage:
    ultrabench open_kidney RAW_DATA_DIR OUTPUT_DIR
"""

import ast
import json
import os
import shutil
from importlib.metadata import version

import numpy as np
import pandas as pd
import skimage
import typer
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated

from .utils import save_version_info

__version__ = version("ultrabench")

OUTPUT_NAME = "open_kidney_v{}"
SONOGRAPHER = 1
CLINICAL_METADATA_FILE = f"labels/reviewed_labels_{SONOGRAPHER}.csv"


def generate_scan_mask(image: np.ndarray):
    # Threshold the image
    mask = image > 0

    # Erode the mask
    for i in range(5):
        mask = skimage.morphology.binary_erosion(mask)

    # Dilate the mask
    for i in range(5):
        mask = skimage.morphology.binary_dilation(mask)

    # Fill small holes
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)

    # Label the connected components
    label = skimage.measure.label(mask)

    # Keep the only the largest connected component
    regions = skimage.measure.regionprops(label)
    largest_region = max(regions, key=lambda x: x.area)
    mask = label == largest_region.label

    # Reflect the larger half of the mask to fill larger gaps in the fan and make it
    # symmetric
    _, width = mask.shape
    left_half = mask[:, : width // 2]
    right_half = mask[:, width // 2 :]
    left_sum = np.sum(left_half)
    right_sum = np.sum(right_half)

    # Determine which half has the greater sum and reflect it
    if left_sum > right_sum:
        right_half = np.fliplr(left_half)
    else:
        left_half = np.fliplr(right_half)

    # Merge the two halves back together
    mask = np.hstack((left_half, right_half))

    # Extract convex hull of the mask
    mask = skimage.morphology.convex_hull_image(mask)

    return mask.astype(np.uint8)


def verify_args(raw_data_dir, output_dir):
    assert os.path.isdir(raw_data_dir), "raw_data_dir must be an existing directory"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(
        os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    ), "A matching version of the CAMUS dataset already exists"


def open_kidney(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
):
    """Prepare the Open Kidney dataset."""
    verify_args(raw_data_dir, output_dir)

    output_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))

    # Load the metadata and extract transplant labels from the comments
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

    # Keep only one entry per image (only the region annotation are different) and
    # remove the duplicates
    metadata = metadata.drop_duplicates("file_id")

    print(f"Total # of images: {len(metadata)}")

    # Copy the images and masks to the output directory
    image_dir = os.path.join(output_dir, "images")
    scan_mask_dir = os.path.join(output_dir, "masks", "scan")
    capsule_mask_dir = os.path.join(output_dir, "masks", "capsule")
    regions_mask_dir = os.path.join(output_dir, "masks", "regions")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(scan_mask_dir, exist_ok=True)
    os.makedirs(capsule_mask_dir, exist_ok=True)
    os.makedirs(regions_mask_dir, exist_ok=True)

    for _, row in metadata.iterrows():
        # Load, convert to single-channel, and save the image
        image = skimage.io.imread(os.path.join(raw_data_dir, "images", row["filename"]))
        if image.ndim == 3:
            image = np.mean(image, axis=-1).astype(np.uint8)
        skimage.io.imsave(os.path.join(image_dir, row["filename"]), image)

        # Generate and save the scan mask
        scan_mask = generate_scan_mask(image)
        skimage.io.imsave(
            os.path.join(scan_mask_dir, row["filename"]),
            scan_mask,
            check_contrast=False,
        )

        # Copy the capsule and regions masks
        for mask in ["capsule", "regions"]:
            mask_file = os.path.join(
                raw_data_dir,
                "labels",
                f"reviewed_masks_{SONOGRAPHER}",
                mask,
                row["filename"],
            )
            shutil.copy(
                mask_file, os.path.join(output_dir, "masks", mask, row["filename"])
            )

    # Add relative paths to the images and masks
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

    # Separate the test set
    train_val_examples, test_examples = train_test_split(
        metadata,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=metadata["view"],
    )

    # Separate the training and validation sets
    train_examples, val_examples = train_test_split(
        train_val_examples,
        test_size=0.1,
        random_state=42,
        shuffle=True,
        stratify=train_val_examples["view"],
    )

    print(f"# of training examples: {len(train_examples)}")
    print(f"# of validation examples: {len(val_examples)}")
    print(f"# of test examples: {len(test_examples)}")

    # Save the training, validation, and test indices to a JSON file
    for split, subset in [
        ("train", train_examples),
        ("validation", val_examples),
        ("test", test_examples),
    ]:
        subset = subset.to_dict(orient="records")

        with open(os.path.join(output_dir, f"{split}.json"), "w") as f:
            json.dump(subset, f, indent=4)

    save_version_info(output_dir, __version__)
