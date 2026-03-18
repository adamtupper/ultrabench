"""Prepare the training and test sets for the GBCU dataset. We use the
predefined training and test data split.

While patient information is not publicly available, there is no patient
overlap between the training and test splits.

Each example (a single image) is represented as an object in one of two JSON
array files (`train_val.json` or `test.json`). Each object has the following
key/value pairs:

    - image:        The path to the image file.
    - dimensions:   The dimensions of the image (width, height).
    - bbox_labels:  The bounding box labels for the region of interest (nml,
                    abn), and other pathologies (stn, malg, etc.) that may be
                    present.
    - bboxes:       The bounding boxes for the region of interest (nml, abn),
                    and other pathologies (stn, malg, etc.) that maybe present.
    - pathology:    Normal, benign, or malignant.
    - label:        The pathology encoded as an integer (normal = 0,
                    benign = 1, malignant = 2).

Bounding boxes are stored in (x_min, y_min, x_max, y_max) format.

Usage:
    ultrabench gbcu RAW_DATA_DIR OUTPUT_DIR
"""

import csv
import json
import os
from importlib.metadata import version

import numpy as np
import skimage
import skimage.io as io
import typer
from typing_extensions import Annotated

from .utils import resize_and_pad, save_version_info, to_3_channel

__version__ = version("ultrabench")

TRAIN_FILE = "train.txt"
TEST_FILE = "test.txt"
BBOX_FILE = "bbox_annot.json"
IMAGE_DIR = "imgs"
OUTPUT_NAME = "gbcu_v{}"


def collate_info(
    raw_filename: str, png_filename: str, label: int, bbox_annotations: dict
) -> dict:
    """Collate the information for an image.

    Args:
        raw_filename: The original image filename used to look up bbox annotations.
        png_filename: The output PNG filename used for image and mask paths.
        label: The pathology label.
        bbox_annotations: The bounding box annotations keyed by raw filename.

    Returns:
        The collated information for the image.
    """
    annotations = bbox_annotations[raw_filename]
    dimensions, bboxes = annotations["dim"], annotations["bbs"]
    bbox_labels, bboxes = zip(*bboxes)

    pathology = "normal" if label == 0 else "benign" if label == 1 else "malignant"

    return {
        "image": os.path.join("images", png_filename),
        "scan_mask": os.path.join("masks", "scan", png_filename),
        "dimensions": dimensions,
        "bbox_labels": bbox_labels,
        "bboxes": bboxes,
        "pathology": pathology,
        "label": label,
    }


def generate_scan_mask(image: np.ndarray) -> np.ndarray:
    """Generate a scan mask using morphological operations.

    Args:
        image: The input 2D grayscale image array.

    Returns:
        A binary mask of the scan region.
    """
    mask = image > 0
    mask = skimage.morphology.remove_small_objects(mask, min_size=2000)
    for i in range(5):
        mask = skimage.morphology.binary_erosion(mask)
    for i in range(5):
        mask = skimage.morphology.binary_dilation(mask)
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
    ), "A matching version of the GBCU dataset already exists"


def process_gbcu(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
    size: Annotated[
        int, typer.Option(help="Test image size (height and width in pixels)")
    ] = 256,
) -> None:
    """Prepare the training and test sets for the GBCU dataset.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        size: The height and width of the test images in pixels.
    """
    verify_args(raw_data_dir, output_dir)

    dataset_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    test_dir_name = f"test_{size}"
    train_dir = os.path.join(dataset_dir, "train_val")
    test_dir = os.path.join(dataset_dir, test_dir_name)

    for split_dir in [train_dir, test_dir]:
        os.makedirs(os.path.join(split_dir, "images"))
        os.makedirs(os.path.join(split_dir, "masks", "scan"))

    # Load the bounding box annotations
    with open(os.path.join(raw_data_dir, BBOX_FILE), "r") as f:
        bbox_annotations = json.load(f)

    # Process the test set
    with open(os.path.join(raw_data_dir, TEST_FILE), "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        test_data = [(image, int(label)) for image, label in csv_reader]

    test_set = []
    for filename, label in test_data:
        raw_image = io.imread(os.path.join(raw_data_dir, IMAGE_DIR, filename))
        gray = (
            raw_image.mean(axis=-1).astype("uint8")
            if raw_image.ndim == 3
            else raw_image
        )
        scan_mask = generate_scan_mask(gray)
        image = to_3_channel(raw_image)
        image = resize_and_pad(image, size, order=1)
        scan_mask = resize_and_pad(scan_mask, size, order=0)

        png_filename = os.path.splitext(filename)[0] + ".png"
        io.imsave(
            os.path.join(test_dir, "images", png_filename), image, check_contrast=False
        )
        io.imsave(
            os.path.join(test_dir, "masks", "scan", png_filename),
            scan_mask,
            check_contrast=False,
        )
        test_set.append(collate_info(filename, png_filename, label, bbox_annotations))

    with open(os.path.join(test_dir, "test.json"), "w") as f:
        json.dump(test_set, f, indent=4)

    print(f"Number of test examples: {len(test_set)}")

    # Process the training set
    with open(os.path.join(raw_data_dir, TRAIN_FILE), "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        train_val_data = [(image, int(label)) for image, label in csv_reader]

    train_val_set = []
    for filename, label in train_val_data:
        raw_image = io.imread(os.path.join(raw_data_dir, IMAGE_DIR, filename))
        gray = (
            raw_image.mean(axis=-1).astype("uint8")
            if raw_image.ndim == 3
            else raw_image
        )
        scan_mask = generate_scan_mask(gray)
        image = to_3_channel(raw_image)

        png_filename = os.path.splitext(filename)[0] + ".png"
        io.imsave(
            os.path.join(train_dir, "images", png_filename), image, check_contrast=False
        )
        io.imsave(
            os.path.join(train_dir, "masks", "scan", png_filename),
            scan_mask,
            check_contrast=False,
        )
        train_val_set.append(
            collate_info(filename, png_filename, label, bbox_annotations)
        )

    with open(os.path.join(train_dir, "train_val.json"), "w") as f:
        json.dump(train_val_set, f, indent=4)

    print(f"Number of train_val examples: {len(train_val_set)}")

    save_version_info(dataset_dir, __version__)
