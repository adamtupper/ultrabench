"""Split the PSFHS dataset into training and test sets using an 8:2 split. The
dataset consists of 1358 images from 1124 patients. Unfortunately, there is no
metadata to provide additional information about the patients or the images.
Therefore, we cannot ensure that there is no patient overlap between the splits.

Each example (a single image) is represented as an object in one of two JSON
array files (`train_val.json` or `test.json`). Each object has the following
key/value pairs:

    - image:            The path to the image file.
    - psfh_mask:        The path to the PS-FH mask file.
    - scan_mask:        The path to the scan mask file.

Usage:
    ultrabench psfhs RAW_DATA_DIR OUTPUT_DIR
"""

import glob
import json
import os
from importlib.metadata import version
from typing import Annotated

import numpy as np
import pandas as pd
import SimpleITK as sitk
import skimage
import skimage.io as io
import typer
from sklearn.model_selection import train_test_split

from .utils import save_version_info

__version__ = version("ultrabench")

OUTPUT_NAME = "psfhs_v{}"


def generate_scan_mask(image: np.ndarray) -> np.ndarray:
    """Generate a scan mask using morphological operations.

    Args:
        image: The input image array.

    Returns:
        A binary mask of the scan region.
    """
    # Threshold the image
    mask = image > 1

    # Remove small objects
    mask = skimage.morphology.remove_small_objects(mask, min_size=64)

    # # Fill small holes
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)

    # Erode the mask
    for i in range(3):
        mask = skimage.morphology.binary_erosion(mask)

    # Dilate the mask
    for i in range(3):
        mask = skimage.morphology.binary_dilation(mask)

    # Take the union of the right half reflected onto the left half, and the left half
    # reflected onto the right half
    _, width = mask.shape
    left_half = mask[:, : width // 2]
    right_half = mask[:, width // 2 :]

    left_half = np.logical_or(left_half, np.fliplr(right_half))
    right_half = np.logical_or(right_half, np.fliplr(left_half))

    # Merge the two halves back together
    mask = np.hstack((left_half, right_half))

    # # Extract convex hull of the mask
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
    ), "A matching version of the PSFHS dataset already exists"


def psfhs(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
) -> None:
    """Prepare the training and test sets for the PSFHS dataset.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
    """
    verify_args(raw_data_dir, output_dir)

    output_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))

    # Create a dataframe of the examples and convert the images and masks to PNG files
    examples = []
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks", "scan"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks", "psfh"), exist_ok=True)
    for image_path in glob.glob(os.path.join(raw_data_dir, "image_mha", "*.mha")):
        # Convert the image to a single-channel PNG
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        image = image.mean(axis=0).astype(np.uint8) if image.ndim == 3 else image
        io.imsave(
            os.path.join(
                output_dir,
                "images",
                os.path.basename(image_path).replace(".mha", ".png"),
            ),
            image,
            check_contrast=False,
        )

        # Generate a scan mask
        mask = generate_scan_mask(image)
        io.imsave(
            os.path.join(
                output_dir,
                "masks",
                "scan",
                os.path.basename(image_path).replace(".mha", ".png"),
            ),
            mask,
            check_contrast=False,
        )

        # Convert the mask to a PNG file
        mask_path = image_path.replace("image_mha", "label_mha")
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        io.imsave(
            os.path.join(
                output_dir,
                "masks",
                "psfh",
                os.path.basename(mask_path).replace(".mha", ".png"),
            ),
            mask,
            check_contrast=False,
        )

        # Add the example to the list
        examples.append(
            {
                "image": os.path.join(
                    "images", os.path.basename(image_path).replace(".mha", ".png")
                ),
                "scan_mask": os.path.join(
                    "masks",
                    "scan",
                    os.path.basename(image_path).replace(".mha", ".png"),
                ),
                "psfh_mask": os.path.join(
                    "masks", "psfh", os.path.basename(mask_path).replace(".mha", ".png")
                ),
            }
        )

    examples = pd.DataFrame(examples)

    print(f"Number of examples: {len(examples)}")

    # Separate the test set
    train_val_examples, test_examples = train_test_split(
        examples, test_size=0.2, random_state=42, shuffle=True
    )

    # Save the training and test indices to a JSON file
    for split, subset in [
        ("train_val", train_val_examples),
        ("test", test_examples),
    ]:
        subset = subset.to_dict(orient="records")

        with open(os.path.join(output_dir, f"{split}.json"), "w") as f:
            json.dump(subset, f, indent=4)

    save_version_info(output_dir, __version__)
