"""Prepare the Butterfly dataset. The existing training and test splits
are preserved.

Each example (a single image) is represented as an object in one of two JSON
array files (`train_val.json` or `test.json`). Each object has the following
key/value pairs:

    - patient: The patient ID.
    - image: The path to the image file.
    - scan_mask: The path to the scan mask file.
    - class: The class name.
    - label: The integer label corresponding to the class.

Usage:
    ultrabench butterfly RAW_DATA_DIR OUTPUT_DIR
"""

import glob
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import skimage
import skimage.io as io
import typer
from typing_extensions import Annotated

from .utils import (
    generate_classification_ground_truth,
    resize_and_pad,
    save_version_info,
    to_3_channel,
)

__version__ = version("ultrabench")

OUTPUT_NAME = "butterfly_v{}"
CLASS_TO_LABEL = {
    "carotid": 0,
    "2ch": 1,
    "lungs": 2,
    "ivc": 3,
    "4ch": 4,
    "bladder": 5,
    "thyroid": 6,
    "plax": 7,
    "morisons_pouch": 8,
}


def generate_scan_mask(image: np.ndarray) -> np.ndarray:
    """Generate a scan mask for an image using morphological operations.

    Args:
        image: The input 2D grayscale image.

    Returns:
        A binary mask of the scan region.
    """
    mask = image > 0
    mask = skimage.morphology.convex_hull_image(mask)
    return mask.astype("uint8")


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
    ), "A matching version of the Butterfly dataset already exists"


def process_butterfly(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
    size: Annotated[
        int, typer.Option(help="Test image size (height and width in pixels)")
    ] = 256,
) -> None:
    """Prepare the training and test sets for the Butterfly dataset.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        size: The height and width of the test images in pixels.
    """
    # Verify arguments
    verify_args(raw_data_dir, output_dir)

    dataset_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    test_dir_name = f"test_{size}"

    # Parse the metadata for each example
    examples = []
    for path in glob.glob(f"{raw_data_dir}/*/*/*/*.png"):
        subpath = path.removeprefix(raw_data_dir).removeprefix("/")
        subset, patient, label, filename = subpath.split("/")
        new_filename = f"{patient}_{label}_{filename}"

        examples.append(
            {
                "subset": "train" if "training" in subset else "test",
                "patient": int(patient),
                "class": label,
                "label": CLASS_TO_LABEL[label],
                "filename": new_filename,
                "filepath": path,
                "image": f"images/{new_filename}",
                "scan_mask": f"masks/scan/{new_filename}",
            }
        )
    df = pd.DataFrame.from_records(examples)

    train_df = df[df["subset"] == "train"]
    test_df = df[df["subset"] == "test"]

    for split_name, subset_df, split_dir, is_test in [
        ("train_val", train_df, os.path.join(dataset_dir, "train_val"), False),
        ("test", test_df, os.path.join(dataset_dir, test_dir_name), True),
    ]:
        image_dir = os.path.join(split_dir, "images")
        mask_dir = os.path.join(split_dir, "masks", "scan")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        for row in subset_df.itertuples(index=False):
            raw_image = io.imread(str(row.filepath))
            # Use 2D image for scan mask generation
            gray = (
                raw_image.mean(axis=-1).astype("uint8")
                if raw_image.ndim == 3
                else raw_image
            )
            mask = generate_scan_mask(gray)

            image = to_3_channel(raw_image)
            if is_test:
                image = resize_and_pad(image, size, order=1)
                mask = resize_and_pad(mask, size, order=0)

            io.imsave(
                os.path.join(image_dir, str(row.filename)), image, check_contrast=False
            )
            io.imsave(
                os.path.join(mask_dir, str(row.filename)), mask, check_contrast=False
            )

        subset_records = subset_df.drop(
            ["filepath", "subset", "filename"], axis="columns"
        ).to_dict(orient="records")
        with open(os.path.join(split_dir, f"{split_name}.json"), "w") as f:
            json.dump(subset_records, f, indent=4)

        if is_test:
            generate_classification_ground_truth(
                subset_records, Path(split_dir) / "ground_truth_classification.csv"
            )

        print(f"Number of {split_name} examples: {len(subset_df)}")

    save_version_info(dataset_dir, __version__)
