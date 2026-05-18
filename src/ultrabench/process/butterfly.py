"""Prepare the Butterfly dataset as a Hugging Face DatasetDict. The
existing training and test splits are preserved.

The resulting dataset has three splits:

    - train: original-size images.
    - test: original-size images.
    - test.{size}: same examples as test, resized so the longest edge
      measures {size} px and then zero-padded to {size} x {size} px.

Each example has the following features:

    - filename: Composite image filename (without the file extension),
      prefixed by the patient ID and label.
    - patient: The patient ID.
    - label: ClassLabel encoding the organ class.
    - image: 3-channel ultrasound image.
    - scan_mask: Single-channel binary scan mask.

A ground-truth CSV file is also created for the test split:

    - ground_truth_classification.csv

Usage:
    ultrabench butterfly RAW_DATA_DIR OUTPUT_DIR
"""

import glob
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
    generate_classification_ground_truth_from_records,
    resize_and_pad,
    save_version_info,
    to_3_channel,
)

__version__ = version("ultrabench")

OUTPUT_NAME = "UltraBench/Butterfly"
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
DATASET_INFO = DatasetInfo(
    description=("A multi-region ultrasound image dataset for organ classification."),
    citation="""
    @misc{butterfly2018,
        author = {Butterfly Network},
        title  = {ButterflyNetwork/MITGrandHack2018},
        year   = {2018},
        url    = {https://github.com/ButterflyNetwork/MITGrandHack2018},
    }
    """,
    homepage="https://github.com/ButterflyNetwork/MITGrandHack2018",
    features=Features(
        {
            "filename": Value("string"),
            "patient": Value("int32"),
            "label": ClassLabel(
                names=[
                    "carotid",
                    "2ch",
                    "lungs",
                    "ivc",
                    "4ch",
                    "bladder",
                    "thyroid",
                    "plax",
                    "morisons_pouch",
                ]
            ),
            "image": Image(),
            "scan_mask": Image(),
        }
    ),
    license="other",
)


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


def _process_butterfly_example(row: pd.Series, size: int | None = None) -> dict:
    """Process a single DataFrame row into a record dict.

    Loads the raw image and generates a scan mask as numpy arrays.
    If size is provided, the image and mask are resized and padded to
    size x size.

    Args:
        row: A row from the examples DataFrame, with columns
            filepath, patient, label, and filename.
        size: If provided, resize and pad arrays to size x size pixels.

    Returns:
        A dict with keys filename, patient, label, image, and
        scan_mask. Image values are numpy arrays.
    """
    raw_image = io.imread(str(row["filepath"]))
    gray = raw_image.mean(axis=-1).astype("uint8") if raw_image.ndim == 3 else raw_image
    scan_mask = generate_scan_mask(gray)
    image = to_3_channel(raw_image)
    if size is not None:
        image = resize_and_pad(image, size, order=1)
        scan_mask = resize_and_pad(scan_mask, size, order=0)

    return {
        "filename": Path(row["filename"]).stem,
        "patient": int(row["patient"]),
        "label": int(row["label"]),
        "image": image,
        "scan_mask": scan_mask,
    }


def _make_train_generator(train_df: pd.DataFrame):
    """Return a generator function that yields train records one by one.

    Args:
        train_df: Training split metadata DataFrame.

    Returns:
        A zero-argument generator function.
    """

    def generator():
        for _, row in tqdm(
            train_df.iterrows(),
            total=len(train_df),
            desc="Processing train split",
        ):
            yield _process_butterfly_example(row)

    return generator


def verify_args(raw_data_dir: str, output_dir: str) -> None:
    """Verify the command line arguments.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
    """
    assert os.path.isdir(raw_data_dir), "raw_data_dir must be an existing directory"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(os.path.join(output_dir, OUTPUT_NAME)), (
        "A matching version of the Butterfly dataset already exists"
    )


def process_butterfly(
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
    """Prepare the training and test sets for the Butterfly dataset.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        size: The height and width of the test images in pixels.
        publish: Whether to publish the dataset to the HF Hub.
    """
    verify_args(raw_data_dir, output_dir)

    output_path = Path(output_dir) / OUTPUT_NAME

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
                "label": CLASS_TO_LABEL[label],
                "filename": new_filename,
                "filepath": path,
            }
        )
    df = pd.DataFrame.from_records(examples)

    train_df = df[df["subset"] == "train"]
    test_df = df[df["subset"] == "test"]

    test_records = [
        _process_butterfly_example(row)
        for _, row in tqdm(
            test_df.iterrows(),
            total=len(test_df),
            desc="Processing test split",
        )
    ]
    test_resized_records = [
        _process_butterfly_example(row, size=size)
        for _, row in tqdm(
            test_df.iterrows(),
            total=len(test_df),
            desc=f"Processing test.{size} split",
        )
    ]

    dataset = DatasetDict(
        {
            "train": Dataset.from_generator(
                _make_train_generator(train_df),
                features=DATASET_INFO.features,
            ),
            "test": Dataset.from_list(test_records, info=DATASET_INFO),
            f"test.{size}": Dataset.from_list(test_resized_records, info=DATASET_INFO),
        }
    )

    dataset.save_to_disk(str(output_path))

    generate_classification_ground_truth_from_records(
        test_records,
        output_path / f"test.{size}" / "ground_truth_classification.csv",
    )

    print(f"Number of train examples: {len(train_df)}")
    print(f"Number of test examples: {len(test_records)}")

    save_version_info(str(output_path), __version__)

    if publish:
        dataset.push_to_hub(
            OUTPUT_NAME,
            commit_message=f"Update the dataset to Version {__version__}.",
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
            commit_message=f"Update ground truth CSV files for Version {__version__}.",
        )
