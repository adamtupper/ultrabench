"""Prepare the training, validation, and test sets for the GBCU dataset. We use the
predefined training and test data split, and additionally separate 10% of the training
set to serve as a validation set.

While there is no patient overlap between the training and test splits, patient
information is not publicly available and therefore we cannot prevent patient overlap
between the training and validation sets.

Each example (a single image) is represented as an object in one of three JSON array
files (`train.json`, `validation.json`, or `test.json`). Each object has the following
key/value pairs:

    - image:        The path to the image file.
    - dimensions:   The dimensions of the image (width, height).
    - bbox_labels:  The bounding box labels for the region of interest (nml, abn), and
                    other pathologies (stn, malg, etc.) that may be present.
    - bboxes:       The bounding boxes for the region of interest (nml, abn), and other
                    pathologies (stn, malg, etc.) that maybe present.
    - pathology:    Normal, benign, or malignant.
    - label:        The pathology encoded as an integer (normal = 0, benign = 1,
                    malignant = 2).

Bounding boxes are stored in (x_min, y_min, x_max, y_max) format.

Usage:
    ultrabench gbcu RAW_DATA_DIR OUTPUT_DIR
"""

import csv
import json
import os
import shutil
from importlib.metadata import version

import numpy as np
import skimage
import typer
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated

from .utils import save_version_info

__version__ = version("ultrabench")

TRAIN_FILE = "train.txt"
TEST_FILE = "test.txt"
BBOX_FILE = "bbox_annot.json"
IMAGE_DIR = "imgs"
OUTPUT_NAME = "gbcu_v{}"


def collate_info(filename, label, bbox_annotations):
    """Collate the information for an image.

    Args:
        filename (str): The image filename.
        label (int): The pathology label.
        bbox_annotations (dict): The bounding box annotations.

    Returns:
        dict: The collated information for the image.
    """
    annotations = bbox_annotations[filename]
    dimensions, bboxes = annotations["dim"], annotations["bbs"]
    bbox_labels, bboxes = zip(*bboxes)

    pathology = "normal" if label == 0 else "benign" if label == 1 else "malignant"
    pathology_encoded = label

    return {
        "image": os.path.join("images", filename),
        "scan_mask": os.path.join("masks", "scan", filename),
        "dimensions": dimensions,
        "bbox_labels": bbox_labels,
        "bboxes": bboxes,
        "pathology": pathology,
        "label": pathology_encoded,
    }


def generate_scan_mask(image: np.ndarray):
    # Threshold the image
    mask = image > 0

    # Remove small objects
    mask = skimage.morphology.remove_small_objects(mask, min_size=2000)

    # Erode the mask
    for i in range(5):
        mask = skimage.morphology.binary_erosion(mask)

    # Dilate the mask
    for i in range(5):
        mask = skimage.morphology.binary_dilation(mask)

    # Extract convex hull of the mask
    mask = skimage.morphology.convex_hull_image(mask)

    return mask.astype(np.uint8)


def verify_args(raw_data_dir, output_dir):
    assert os.path.isdir(raw_data_dir), "raw_data_dir must be an existing directory"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(
        os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    ), "A matching version of the GBCU dataset already exists"


def gbcu(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
):
    """Prepare the training, validation, and test sets for the GBCU dataset."""
    verify_args(raw_data_dir, output_dir)

    output_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    os.makedirs(os.path.join(output_dir, "images"))
    os.makedirs(os.path.join(output_dir, "masks", "scan"))

    # Load the bounding box annotations
    with open(os.path.join(raw_data_dir, BBOX_FILE), "r") as f:
        bbox_annotations = json.load(f)

    # Process the test set
    with open(os.path.join(raw_data_dir, TEST_FILE), "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        test_data = [(image, int(label)) for image, label in csv_reader]

    test_set = []
    for filename, label in test_data:
        # Copy the image to the output directory
        shutil.copy(
            os.path.join(raw_data_dir, IMAGE_DIR, filename),
            os.path.join(output_dir, "images", filename),
        )

        # Generate the scan mask
        image = skimage.io.imread(os.path.join(output_dir, "images", filename))
        scan_mask = generate_scan_mask(image)
        skimage.io.imsave(
            os.path.join(output_dir, "masks", "scan", filename),
            scan_mask,
            check_contrast=False,
        )

        # Create metadata entry
        test_set.append(collate_info(filename, label, bbox_annotations))

    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(test_set, f, indent=4)

    # Process the training set into training and validation sets
    with open(os.path.join(raw_data_dir, TRAIN_FILE), "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        train_data = [(image, int(label)) for image, label in csv_reader]

    train_images, val_images = train_test_split(
        train_data,
        test_size=0.1,
        random_state=42,
        shuffle=True,
        stratify=[label for _, label in train_data],
    )

    for split, data in [("train", train_images), ("validation", val_images)]:
        split_set = []
        for filename, label in data:
            # Copy the image to the output directory
            shutil.copy(
                os.path.join(raw_data_dir, IMAGE_DIR, filename),
                os.path.join(output_dir, "images"),
            )

            # Generate the scan mask
            image = skimage.io.imread(os.path.join(output_dir, "images", filename))
            scan_mask = generate_scan_mask(image)
            skimage.io.imsave(
                os.path.join(output_dir, "masks", "scan", filename),
                scan_mask,
                check_contrast=False,
            )

            # Create metadata entry
            split_set.append(collate_info(filename, label, bbox_annotations))

        with open(os.path.join(output_dir, f"{split}.json"), "w") as f:
            json.dump(split_set, f, indent=4)

    save_version_info(output_dir, __version__)
