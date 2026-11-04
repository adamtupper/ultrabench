"""Prepare the GBCU dataset as a Hugging Face DatasetDict. We use the
predefined training and test data split.

While patient information is not publicly available, there is no patient
overlap between the training and test splits. The training split is
subsampled to 1000 images, removing images from the largest
classes first so that the class distribution becomes more uniform. The
removed images are discarded rather than moved to the test split, since
the images lack patient identifiers and moving them could introduce
patient overlap between the splits.

The resulting dataset has three splits:

    - train: up to 1000 original-size images.
    - test: original-size images.
    - test.{size}: same examples as test, resized so the longest edge
      measures {size} px and then zero-padded to {size} x {size} px.

Each example has the following features:

    - filename: Original image filename (without the file extension).
    - label: ClassLabel encoding the tumour malignancy.
    - image: 3-channel ultrasound image.
    - scan_mask: Single-channel binary scan mask.

A ground-truth CSV file is also created in each test split directory:

    - ground_truth_classification.csv

Usage:
    ultrabench gbcu RAW_DATA_DIR OUTPUT_DIR
"""

import csv
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import skimage
import skimage.io as io
import typer
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    DatasetInfo,
    Features,
    Image,
    Value,
)
from huggingface_hub import HfApi
from tqdm.auto import tqdm
from typing_extensions import Annotated

from ultrabench.datasets.utils import partition_by_manifest

from .utils import (
    DATASET_REVISION,
    IMAGE_SIZE,
    BuildResult,
    generate_classification_ground_truth_from_records,
    ground_truth_commit_operations,
    resize_and_pad,
    save_version_info,
    to_3_channel,
    verify_revision_unpublished,
    write_test_ground_truth,
)

__version__ = version("ultrabench")

TRAIN_FILE = "train.txt"
TEST_FILE = "test.txt"
IMAGE_DIR = "imgs"
OUTPUT_NAME = "UltraBench/GBCU"
DATASET_INFO = DatasetInfo(
    description=("Abdominal ultrasound images for gallbladder tumour classification."),
    citation="""
    @inproceedings{basu2022gbcu,
        author    = {Basu, S. and Gupta, M. and Rana, P. and
                     Gupta, P. and Arora, C.},
        title     = {Surpassing the Human Accuracy: Detecting Gallbladder
                     Cancer from USG Images with Curriculum Learning},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer
                     Vision and Pattern Recognition},
        pages     = {20854--20864},
        year      = {2022},
        doi       = {10.1109/CVPR52688.2022.02022},
    }
    """,
    homepage="https://gbc-iitd.github.io/data/gbcu",
    features=Features(
        {
            "filename": Value("string"),
            "label": ClassLabel(names=["normal", "benign", "malignant"]),
            "image": Image(),
            "scan_mask": Image(),
        }
    ),
    license="other",
)


def generate_scan_mask(image: np.ndarray) -> np.ndarray:
    """Generate a scan mask using morphological operations.

    Args:
        image: The input 2D grayscale image array.

    Returns:
        A binary mask of the scan region.
    """
    mask = image > 0
    mask = skimage.morphology.remove_small_objects(mask, min_size=2000)
    for _ in range(5):
        mask = skimage.morphology.binary_erosion(mask)
    for _ in range(5):
        mask = skimage.morphology.binary_dilation(mask)
    mask = skimage.morphology.convex_hull_image(mask)
    return mask.astype(np.uint8)


def _process_gbcu_example(
    raw_filename: str,
    label: int,
    raw_data_dir: str,
    size: int | None = None,
) -> dict:
    """Process a single GBCU example into a record dict.

    Loads the raw image and generates a scan mask as numpy arrays.
    If size is provided, the image and mask are resized and padded to
    size x size.

    Args:
        raw_filename: The original image filename (with extension).
        label: The integer class label.
        raw_data_dir: Path to the raw data directory.
        size: If provided, resize and pad arrays to size x size pixels.

    Returns:
        A dict with keys filename, label, image, and scan_mask. Image
        values are numpy arrays.
    """
    raw_image = io.imread(os.path.join(raw_data_dir, IMAGE_DIR, raw_filename))
    gray = raw_image.mean(axis=-1).astype("uint8") if raw_image.ndim == 3 else raw_image
    scan_mask = generate_scan_mask(gray)
    image = to_3_channel(raw_image)
    if size is not None:
        image = resize_and_pad(image, size, order=1)
        scan_mask = resize_and_pad(scan_mask, size, order=0)

    return {
        "filename": Path(raw_filename).stem,
        "label": label,
        "image": image,
        "scan_mask": scan_mask,
    }


def verify_args(raw_data_dir: str, output_dir: str, publish: bool = False) -> None:
    """Verify the command line arguments.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        publish: Whether the dataset will be published to the Hugging Face Hub.
    """
    assert os.path.isdir(raw_data_dir), "raw_data_dir must be an existing directory"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(os.path.join(output_dir, OUTPUT_NAME)), (
        "A matching version of the GBCU dataset already exists"
    )

    if publish:
        verify_revision_unpublished(OUTPUT_NAME)


def write_ground_truth(
    records: list[dict], gt_dir: Path, image_size: int | None
) -> None:
    """Write the ground-truth CSV of one test split.

    Args:
        records: The records of the split.
        gt_dir: The split directory to write the CSV into.
        image_size: Unused. Classification labels do not depend on the
            image resolution, but every dataset's writer shares one
            signature.
    """
    generate_classification_ground_truth_from_records(
        records, gt_dir / "ground_truth_classification.csv"
    )


def build_gbcu_dataset(raw_data_dir: str | Path, size: int = IMAGE_SIZE) -> BuildResult:
    """Build the GBCU splits from the raw data.

    The splits come from the tracked manifests rather than from
    `subsample_balanced_by_class`, so that processing the raw data
    reproduces the published splits exactly. The surplus training
    images the manifests leave out are discarded rather than moved to
    the test split, because the images carry no patient identifiers and
    moving them could introduce patient overlap.

    Args:
        raw_data_dir: The path to the raw data directory.
        size: The height and width of the resized test images.

    Returns:
        The processed splits and the test records the ground truth is
        built from.

    Raises:
        ValueError: If the raw data is missing an example the manifests
            list.
    """
    raw_data_dir = Path(raw_data_dir)

    entries: list[tuple[str, int]] = []
    for split_file in (TRAIN_FILE, TEST_FILE):
        with open(raw_data_dir / split_file, "r") as f:
            entries.extend(
                (image, int(label)) for image, label in csv.reader(f, delimiter=",")
            )

    train_data, test_data = partition_by_manifest(
        entries, "GBCU", key=lambda entry: Path(entry[0]).stem
    )

    train_records = [
        _process_gbcu_example(filename, label, str(raw_data_dir))
        for filename, label in tqdm(train_data, desc="Processing train split")
    ]
    test_records = [
        _process_gbcu_example(filename, label, str(raw_data_dir))
        for filename, label in tqdm(test_data, desc="Processing test split")
    ]
    test_resized_records = [
        _process_gbcu_example(filename, label, str(raw_data_dir), size=size)
        for filename, label in tqdm(test_data, desc=f"Processing test.{size} split")
    ]

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_records, info=DATASET_INFO),
            "test": Dataset.from_list(test_records, info=DATASET_INFO),
            f"test.{size}": Dataset.from_list(test_resized_records, info=DATASET_INFO),
        }
    )

    return BuildResult(dataset, test_records, test_resized_records)


def process_gbcu(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
    size: Annotated[
        int, typer.Option(help="Test image size (height and width in pixels)")
    ] = 256,
    publish: Annotated[
        bool,
        typer.Option(help="Whether to publish the dataset to the Hugging Face Hub"),
    ] = False,
) -> None:
    """Prepare the training and test sets for the GBCU dataset.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        size: The height and width of the test images in pixels.
        publish: Whether to publish the dataset to the HF Hub.
    """
    verify_args(raw_data_dir, output_dir, publish)

    output_path = Path(output_dir) / OUTPUT_NAME
    result = build_gbcu_dataset(raw_data_dir, size=size)
    result.dataset.save_to_disk(str(output_path))
    write_test_ground_truth(result, output_path, size, write_ground_truth)

    save_version_info(str(output_path), __version__)

    if publish:
        result.dataset.push_to_hub(
            OUTPUT_NAME,
            commit_message=f"Update the dataset to revision {DATASET_REVISION}.",
            private=True,
        )

        api = HfApi()
        operations = ground_truth_commit_operations(
            output_path, {"classification": "ground_truth_classification.csv"}, size
        )
        api.create_commit(
            repo_id=OUTPUT_NAME,
            repo_type="dataset",
            operations=operations,
            commit_message=(
                f"Update ground truth CSV files for revision {DATASET_REVISION}."
            ),
        )
