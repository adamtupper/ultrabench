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

A ground-truth CSV file is also created for the test split:

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
from huggingface_hub import CommitOperationAdd, HfApi
from tqdm.auto import tqdm
from typing_extensions import Annotated

from .utils import (
    DATASET_REVISION,
    generate_classification_ground_truth_from_records,
    resize_and_pad,
    save_version_info,
    subsample_balanced_by_class,
    to_3_channel,
    verify_revision_unpublished,
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

    with open(os.path.join(raw_data_dir, TRAIN_FILE), "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        train_val_df = pd.DataFrame(
            [(image, int(label)) for image, label in csv_reader],
            columns=["raw_filename", "label"],
        )

    with open(os.path.join(raw_data_dir, TEST_FILE), "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        test_data = [(image, int(label)) for image, label in csv_reader]

    train_val_df = subsample_balanced_by_class(train_val_df, label_col="label")
    print(f"Train labels: {train_val_df['label'].value_counts().to_dict()}")

    train_records = [
        _process_gbcu_example(row["raw_filename"], int(row["label"]), raw_data_dir)
        for _, row in tqdm(
            train_val_df.iterrows(),
            total=len(train_val_df),
            desc="Processing train split",
        )
    ]
    test_records = [
        _process_gbcu_example(filename, label, raw_data_dir)
        for filename, label in tqdm(test_data, desc="Processing test split")
    ]
    test_resized_records = [
        _process_gbcu_example(filename, label, raw_data_dir, size=size)
        for filename, label in tqdm(test_data, desc=f"Processing test.{size} split")
    ]

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_records, info=DATASET_INFO),
            "test": Dataset.from_list(test_records, info=DATASET_INFO),
            f"test.{size}": Dataset.from_list(test_resized_records, info=DATASET_INFO),
        }
    )

    dataset.save_to_disk(str(output_path))

    generate_classification_ground_truth_from_records(
        test_records,
        output_path / f"test.{size}" / "ground_truth_classification.csv",
    )

    print(f"Number of train examples: {len(train_records)}")
    print(f"Number of test examples: {len(test_records)}")

    save_version_info(str(output_path), __version__)

    if publish:
        dataset.push_to_hub(
            OUTPUT_NAME,
            commit_message=f"Update the dataset to revision {DATASET_REVISION}.",
            private=True,
        )

        api = HfApi()
        operations = [
            CommitOperationAdd(
                path_in_repo="test-ground-truth-classification.csv",
                path_or_fileobj=output_path
                / f"test.{size}"
                / "ground_truth_classification.csv",
            ),
        ]
        api.create_commit(
            repo_id=OUTPUT_NAME,
            repo_type="dataset",
            operations=operations,
            commit_message=(
                f"Update ground truth CSV files for revision {DATASET_REVISION}."
            ),
        )
