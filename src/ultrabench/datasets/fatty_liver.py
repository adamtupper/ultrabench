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

import numpy as np
import pandas as pd
import scipy
import skimage
import skimage.io as io
import typer
from sklearn.model_selection import GroupShuffleSplit
from typing_extensions import Annotated

from .utils import save_version_info

__version__ = version("ultrabench")

OUTPUT_NAME = "fatty_liver_v{}"
LABEL_TO_CLASS = {0: "Normal", 1: "NFLD"}


def generate_scan_mask(image: np.ndarray) -> np.ndarray:
    """Generate a scan mask using morphological operations.

    Args:
        image: The input image array.

    Returns:
        A binary mask of the scan region.
    """
    # Threshold the image
    mask = image > 0

    # Remove small objects
    mask = skimage.morphology.remove_small_objects(mask, min_size=2000)

    # Erode the mask
    for i in range(5):
        mask = skimage.morphology.binary_erosion(mask)

    # Remove remaining small objects
    mask = skimage.morphology.remove_small_objects(mask, min_size=2000)

    # Dilate the mask
    for i in range(5):
        mask = skimage.morphology.binary_dilation(mask)

    # Fill small holes
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)

    # Extract convex hull of the mask
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


def fatty_liver(
    raw_data_path: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
) -> None:
    """Prepare the training and test sets for the Fatty Liver dataset.

    Args:
        raw_data_path: The path to the raw data file.
        output_dir: The path to the output directory.
    """

    output_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))

    # Load the data
    data = scipy.io.loadmat(raw_data_path)
    examples = data["data"][0]
    patient_ids = np.array([examples[0][0][0] for examples in examples])
    labels = np.array([examples[1][0][0] for examples in examples])
    steatosis_values = np.array([examples[2][0][0] for examples in examples])
    frame_sequences = [examples[3] for examples in examples]

    # Save each frame as a PNG image
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    for id, frames in zip(patient_ids, frame_sequences):
        for i in range(frames.shape[0]):
            image_path = os.path.join(image_dir, f"{id}_{i}.png")
            io.imsave(image_path, frames[i])

    # Generate a scan mask for each image
    mask_dir = os.path.join(output_dir, "masks", "scan")
    os.makedirs(mask_dir, exist_ok=True)
    for id, frames in zip(patient_ids, frame_sequences):
        for i in range(frames.shape[0]):
            mask_path = os.path.join(mask_dir, f"{id}_{i}.png")
            mask = generate_scan_mask(frames[i])
            io.imsave(mask_path, mask, check_contrast=False)

    # Create a dataframe of examples
    examples = []
    for id, label, steatosis_value, frames in zip(
        patient_ids, labels, steatosis_values, frame_sequences
    ):
        for i in range(frames.shape[0]):
            examples.append(
                {
                    "patient": id,
                    "image": os.path.join("images", f"{id}_{i}.png"),
                    "scan_mask": os.path.join("masks", "scan", f"{id}_{i}.png"),
                    "steatosis": steatosis_value,
                    "label": label,
                    "pathology": LABEL_TO_CLASS[label],
                }
            )
    examples = pd.DataFrame.from_records(examples)

    # Split the dataset into training and test sets
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_val_indices, test_indices = next(
        splitter.split(X=examples, groups=examples["patient"])
    )
    train_val_examples = examples.iloc[train_val_indices]
    test_examples = examples.iloc[test_indices]

    # Save the training and test indices to a JSON file
    for split, subset in [
        ("train_val", train_val_examples),
        ("test", test_examples),
    ]:
        subset = subset.to_dict(orient="records")

        with open(os.path.join(output_dir, f"{split}.json"), "w") as f:
            json.dump(subset, f, indent=4)

    save_version_info(output_dir, __version__)
