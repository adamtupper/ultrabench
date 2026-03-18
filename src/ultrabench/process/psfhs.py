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

from .utils import resize_and_pad, save_version_info, to_3_channel

__version__ = version("ultrabench")

OUTPUT_NAME = "psfhs_v{}"


def generate_scan_mask(image: np.ndarray) -> np.ndarray:
    """Generate a scan mask using morphological operations.

    Args:
        image: The input 2D image array.

    Returns:
        A binary mask of the scan region.
    """
    mask = image > 1
    mask = skimage.morphology.remove_small_objects(mask, min_size=64)
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)
    for i in range(3):
        mask = skimage.morphology.binary_erosion(mask)
    for i in range(3):
        mask = skimage.morphology.binary_dilation(mask)

    # Take the union of each half reflected onto the other to make it symmetric
    _, width = mask.shape
    left_half = mask[:, : width // 2]
    right_half = mask[:, width // 2 :]
    left_half = np.logical_or(left_half, np.fliplr(right_half))
    right_half = np.logical_or(right_half, np.fliplr(left_half))
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
    ), "A matching version of the PSFHS dataset already exists"


def process_psfhs(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
    size: Annotated[
        int, typer.Option(help="Test image size (height and width in pixels)")
    ] = 256,
) -> None:
    """Prepare the training and test sets for the PSFHS dataset.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        size: The height and width of the test images in pixels.
    """
    verify_args(raw_data_dir, output_dir)

    dataset_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    test_dir_name = f"test_{size}"

    # Collect all image paths (no I/O) to determine the split before processing
    image_paths = glob.glob(os.path.join(raw_data_dir, "image_mha", "*.mha"))

    records = [
        {
            "image_path": p,
            "filename": os.path.basename(p).replace(".mha", ".png"),
            "image": f"images/{os.path.basename(p).replace('.mha', '.png')}",
            "scan_mask": f"masks/scan/{os.path.basename(p).replace('.mha', '.png')}",
            "psfh_mask": f"masks/psfh/{os.path.basename(p).replace('.mha', '.png')}",
        }
        for p in image_paths
    ]
    examples = pd.DataFrame(records)

    print(f"Number of examples: {len(examples)}")

    # Determine the train/test split before any image I/O
    train_val_examples, test_examples = train_test_split(
        examples, test_size=0.2, random_state=42, shuffle=True
    )
    test_filenames = set(test_examples["filename"])

    # Create output directories
    for split_dir_name in ["train_val", test_dir_name]:
        split_dir = os.path.join(dataset_dir, split_dir_name)
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "masks", "scan"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "masks", "psfh"), exist_ok=True)

    # Load, convert, and save each image and mask to the correct split directory
    for _, row in examples.iterrows():
        is_test = row["filename"] in test_filenames
        split_dir = os.path.join(dataset_dir, test_dir_name if is_test else "train_val")

        raw = sitk.GetArrayFromImage(sitk.ReadImage(row["image_path"]))
        gray = raw.mean(axis=0).astype(np.uint8) if raw.ndim == 3 else raw

        scan_mask = generate_scan_mask(gray)
        image = to_3_channel(gray)

        mask_path = row["image_path"].replace("image_mha", "label_mha")
        psfh_mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))

        if is_test:
            image = resize_and_pad(image, size, order=1)
            scan_mask = resize_and_pad(scan_mask, size, order=0)
            psfh_mask = resize_and_pad(psfh_mask, size, order=0)

        io.imsave(
            os.path.join(split_dir, "images", row["filename"]),
            image,
            check_contrast=False,
        )
        io.imsave(
            os.path.join(split_dir, "masks", "scan", row["filename"]),
            scan_mask,
            check_contrast=False,
        )
        io.imsave(
            os.path.join(split_dir, "masks", "psfh", row["filename"]),
            psfh_mask,
            check_contrast=False,
        )

    # Save metadata JSON files inside their split directories
    for split, subset, split_dir in [
        ("train_val", train_val_examples, os.path.join(dataset_dir, "train_val")),
        ("test", test_examples, os.path.join(dataset_dir, test_dir_name)),
    ]:
        cols_to_drop = ["image_path", "filename"]
        with open(os.path.join(split_dir, f"{split}.json"), "w") as f:
            json.dump(
                subset.drop(columns=cols_to_drop).to_dict(orient="records"), f, indent=4
            )
        print(f"Number of {split} examples: {len(subset)}")

    save_version_info(dataset_dir, __version__)
