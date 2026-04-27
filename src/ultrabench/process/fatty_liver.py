"""Split the Fatty Liver dataset into training and test sets using an 8:2
split. The dataset consists of 550 images collected from 55 patients (10 images
per patient). The images are split by patient to ensure that there is no
patient overlap between the splits.

Each example (a single image) is represented as an object in one of two JSON
array files (`train_val.json` or `test.json`). Each object has the following
key/value pairs:

    - patient:      The patient ID.
    - image:        The path to the image file.
    - steatosis:    The percentage of hepatocytes with steatosis.
    - label:        Whether or not the liver is fatty (0 if <5% hepatocytes
                    with steatosis, 1 otherwise).
    - pathology:    0 = Normal or 1 = NFLD.

Usage:
    ultrabench fatty_liver RAW_DATA_PATH OUTPUT_DIR
"""

import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import skimage
import skimage.io as io
import typer
from sklearn.model_selection import GroupShuffleSplit
from typing_extensions import Annotated

from .utils import (
    generate_classification_ground_truth,
    resize_and_pad,
    save_version_info,
    to_3_channel,
)

__version__ = version("ultrabench")

OUTPUT_NAME = "fatty_liver_v{}"
LABEL_TO_CLASS = {0: "Normal", 1: "NFLD"}


def generate_scan_mask(image: np.ndarray) -> np.ndarray:
    """Generate a scan mask using morphological operations.

    Args:
        image: The input 2D image array.

    Returns:
        A binary mask of the scan region.
    """
    mask = image > 0
    mask = skimage.morphology.remove_small_objects(mask, min_size=2000)
    for i in range(5):
        mask = skimage.morphology.binary_erosion(mask)
    mask = skimage.morphology.remove_small_objects(mask, min_size=2000)
    for i in range(5):
        mask = skimage.morphology.binary_dilation(mask)
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)
    mask = skimage.morphology.convex_hull_image(mask)
    return mask.astype(np.uint8)


def verify_args(raw_data_path: str, output_dir: str) -> None:
    """Verify the command line arguments.

    Args:
        raw_data_path: The path to the raw data file.
        output_dir: The path to the output directory.
    """
    assert os.path.exists(raw_data_path), "raw_data_path must exist"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(
        os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    ), "A matching version of the Fatty Liver dataset already exists"


def process_fatty_liver(
    raw_data_path: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
    size: Annotated[
        int, typer.Option(help="Test image size (height and width in pixels)")
    ] = 256,
) -> None:
    """Prepare the training and test sets for the Fatty Liver dataset.

    Args:
        raw_data_path: The path to the raw data file.
        output_dir: The path to the output directory.
        size: The height and width of the test images in pixels.
    """
    verify_args(raw_data_path, output_dir)

    dataset_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    test_dir_name = f"test_{size}"

    # Load the data
    data = scipy.io.loadmat(raw_data_path)
    entries = data["data"][0]
    patient_ids = np.array([e[0][0][0] for e in entries])
    labels = np.array([e[1][0][0] for e in entries])
    steatosis_values = np.array([e[2][0][0] for e in entries])
    frame_sequences = [e[3] for e in entries]

    # Build a per-frame DataFrame for splitting (no file I/O yet)
    records = []
    for patient_id, label, steatosis, frames in zip(
        patient_ids, labels, steatosis_values, frame_sequences
    ):
        for i in range(frames.shape[0]):
            records.append(
                {
                    "patient": patient_id,
                    "image": f"images/{patient_id}_{i}.png",
                    "scan_mask": f"masks/scan/{patient_id}_{i}.png",
                    "steatosis": steatosis,
                    "label": label,
                    "pathology": LABEL_TO_CLASS[label],
                }
            )
    examples = pd.DataFrame.from_records(records)

    # Determine the train/test split from patient IDs before any file I/O
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_val_indices, test_indices = next(
        splitter.split(X=examples, groups=examples["patient"])
    )
    train_val_examples = examples.iloc[train_val_indices]
    test_examples = examples.iloc[test_indices]

    test_image_paths = set(test_examples["image"])

    # Save images and masks to the correct split directories
    for split_name, split_examples, split_dir, is_test in [
        (
            "train_val",
            train_val_examples,
            os.path.join(dataset_dir, "train_val"),
            False,
        ),
        ("test", test_examples, os.path.join(dataset_dir, test_dir_name), True),
    ]:
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "masks", "scan"), exist_ok=True)

    for patient_id, label, steatosis, frames in zip(
        patient_ids, labels, steatosis_values, frame_sequences
    ):
        for i in range(frames.shape[0]):
            rel_image_path = f"images/{patient_id}_{i}.png"
            is_test = rel_image_path in test_image_paths
            split_dir = os.path.join(
                dataset_dir, test_dir_name if is_test else "train_val"
            )

            frame = frames[i]
            mask = generate_scan_mask(frame)
            image = to_3_channel(frame)
            if is_test:
                image = resize_and_pad(image, size, order=1)
                mask = resize_and_pad(mask, size, order=0)

            io.imsave(
                os.path.join(split_dir, "images", f"{patient_id}_{i}.png"),
                image,
                check_contrast=False,
            )
            io.imsave(
                os.path.join(split_dir, "masks", "scan", f"{patient_id}_{i}.png"),
                mask,
                check_contrast=False,
            )

    # Save the metadata JSON files inside their split directories
    for split, subset, split_dir in [
        ("train_val", train_val_examples, os.path.join(dataset_dir, "train_val")),
        ("test", test_examples, os.path.join(dataset_dir, test_dir_name)),
    ]:
        with open(os.path.join(split_dir, f"{split}.json"), "w") as f:
            json.dump(subset.to_dict(orient="records"), f, indent=4)
        print(f"Number of {split} examples: {len(subset)}")

    generate_classification_ground_truth(
        test_examples.to_dict(orient="records"),
        Path(os.path.join(dataset_dir, test_dir_name))
        / "ground_truth_classification.csv",
    )

    save_version_info(dataset_dir, __version__)
