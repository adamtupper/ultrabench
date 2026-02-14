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
import shutil
from importlib.metadata import version

import pandas as pd
import skimage
import typer
from typing_extensions import Annotated

from .utils import save_version_info

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


def generate_scan_mask(
    output_dir: str, rel_image_path: str, rel_mask_path: str
) -> None:
    """Generate a scan mask for an image using morphological operations.

    Args:
        output_dir: The output directory for the dataset.
        rel_image_path: The path to the image file.
        rel_mask_path: The path to save the scan mask file.
    """
    image = skimage.io.imread(rel_image_path)
    mask = image > 0  # Threshold the image
    mask = skimage.morphology.convex_hull_image(mask)  # Extract convex hull of the mask
    skimage.io.imsave(
        os.path.join(output_dir, rel_mask_path),
        mask.astype("uint8"),
        check_contrast=False,
    )


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


def butterfly(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
) -> None:
    """Prepare the training and test sets for the Butterfly dataset.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
    """
    # Verify arguments
    verify_args(raw_data_dir, output_dir)

    dataset_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))

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

    # Create output directories for images and scan masks
    mask_dir = os.path.join(dataset_dir, "masks", "scan")
    os.makedirs(mask_dir, exist_ok=True)

    image_dir = os.path.join(dataset_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    # Create scan masks and copy images
    for row in df.itertuples(index=False):
        generate_scan_mask(dataset_dir, str(row.filepath), str(row.scan_mask))
        shutil.copy(str(row.filepath), os.path.join(image_dir, str(row.filename)))

    # Save the metadata for the training and test sets to a JSON file
    for split, subset in [
        ("train_val", df[df["subset"] == "train"]),
        ("test", df[df["subset"] == "test"]),
    ]:
        subset = subset.drop(
            ["filepath", "subset", "filename"], axis="columns"
        ).to_dict(orient="records")

        with open(os.path.join(dataset_dir, f"{split}.json"), "w") as f:
            json.dump(subset, f, indent=4)

    save_version_info(dataset_dir, __version__)
