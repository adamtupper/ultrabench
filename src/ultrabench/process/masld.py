"""Split the MASLD (Metabolic dysfunction-associated steatotic liver
disease) dataset into training and test sets using an 8:2 split. The
dataset consists of 550 images collected from 55 patients (10 images
per patient). The images are split by patient to ensure that there is
no patient overlap between the splits.

The resulting dataset has three splits:

    - train: original-size images.
    - test: original-size images.
    - test.{size}: same examples as test, resized so the longest edge
      measures {size} px and then zero-padded to {size} x {size} px.

Each example has the following features:

    - patient: Patient ID.
    - frame: Frame index.
    - image: 3-channel ultrasound image.
    - scan_mask: Single-channel binary scan mask.
    - percentage: Percentage of hepatocytes with steatosis.
    - label: ClassLabel encoding the diagnosis (negative=0, positive=1).

One ground-truth CSV file is also created for the test split for use
by the evaluation layer:

    - ground_truth_classification.csv

Usage:
    ultrabench process masld RAW_DATA_PATH OUTPUT_DIR
"""

import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import skimage
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
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm
from typing_extensions import Annotated

from .utils import (
    generate_classification_ground_truth_from_records,
    resize_and_pad,
    save_version_info,
    to_3_channel,
)

__version__ = version("ultrabench")

OUTPUT_NAME = "UltraBench/MASLD"
DATASET_INFO = DatasetInfo(
    description=(
        "A dataset of liver ultrasound images labeled for "
        "Metabolic dysfunction-associated steatotic liver "
        "disease (MASLD) classification."
    ),
    citation="""
    @article{byra2018transfer,
        author  = {Byra, Michal and Styczynski, Grzegorz and
                   Szmigielski, Cezary and Kalinowski, Piotr and
                   Michałowski, Łukasz and Paluszkiewicz, Rafał and
                   Ziarkiewicz-Wróblewska, Bogna and
                   Zieniewicz, Krzysztof and Sobieraj, Piotr and
                   Nowicki, Andrzej},
        title   = {Transfer learning with deep convolutional neural
                   network for liver steatosis assessment in
                   ultrasound images},
        journal = {International Journal of Computer Assisted
                   Radiology and Surgery},
        year    = {2018},
        doi     = {10.1007/s11548-018-1843-2},
    }
    """,
    homepage="https://zenodo.org/records/3577273",
    license="cc-by-4.0",
    features=Features(
        {
            "filename": Value("string"),
            "patient": Value("int32"),
            "frame": Value("int32"),
            "image": Image(),
            "scan_mask": Image(),
            "percentage": Value("float32"),
            "label": ClassLabel(names=["negative", "positive"]),
        }
    ),
)


def generate_scan_mask(image: np.ndarray) -> np.ndarray:
    """Generate a scan mask using morphological operations.

    Args:
        image: The input 2D image array.

    Returns:
        A binary mask of the scan region.
    """
    mask = image > 0
    mask = skimage.morphology.remove_small_objects(mask, min_size=2000)
    for _ in range(5):
        mask = skimage.morphology.binary_erosion(mask)
    mask = skimage.morphology.remove_small_objects(mask, min_size=2000)
    for _ in range(5):
        mask = skimage.morphology.binary_dilation(mask)
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)
    mask = skimage.morphology.convex_hull_image(mask)
    return mask.astype(np.uint8)


def _process_masld_example(
    patient_id: int,
    frame_idx: int,
    frame: np.ndarray,
    label: int,
    steatosis: float,
    size: int | None = None,
) -> dict:
    """Process a single frame into a record dict.

    Args:
        patient_id: The patient ID.
        frame_idx: The frame index (0-based).
        frame: The raw 2D frame array.
        label: The class label (0=negative, 1=positive).
        steatosis: The percentage of hepatocytes with steatosis.
        size: If provided, resize and pad all arrays to size x size.

    Returns:
        A dict with keys patient, frame, image, scan_mask, percentage,
        label, and filename (used only for ground truth CSV generation).
    """
    mask = generate_scan_mask(frame)
    image = to_3_channel(frame)
    if size is not None:
        image = resize_and_pad(image, size, order=1)
        mask = resize_and_pad(mask, size, order=0)

    return {
        "filename": f"{patient_id}_{frame_idx}",
        "patient": int(patient_id),
        "frame": int(frame_idx),
        "image": image,
        "scan_mask": mask,
        "percentage": float(steatosis),
        "label": int(label),
    }


def verify_args(raw_data_path: str, output_dir: str) -> None:
    """Verify the command line arguments.

    Args:
        raw_data_path: The path to the raw data file.
        output_dir: The path to the output directory.
    """
    assert os.path.exists(raw_data_path), "raw_data_path must exist"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(os.path.join(output_dir, OUTPUT_NAME)), (
        "A version of the MASLD dataset already exists"
    )


def process_masld(
    raw_data_path: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed dataset")
    ],
    size: Annotated[
        int, typer.Option(help="Test image size (height and width in pixels)")
    ] = 256,
    publish: Annotated[
        bool,
        typer.Option(help="Whether to publish the dataset to the Hugging Face Hub"),
    ] = False,
) -> None:
    """Prepare the MASLD dataset as a Hugging Face DatasetDict.

    Args:
        raw_data_path: The path to the raw data file.
        output_dir: The path to the output directory.
        size: The height and width of the resized test images in pixels.
        publish: Whether to publish the dataset to the Hugging Face Hub.
    """
    verify_args(raw_data_path, output_dir)

    data = scipy.io.loadmat(raw_data_path)
    entries = data["data"][0]
    patient_ids = np.array([e[0][0][0] for e in entries])
    labels = np.array([e[1][0][0] for e in entries])
    steatosis_values = np.array([e[2][0][0] for e in entries])
    frame_sequences = [e[3] for e in entries]

    # Build a per-frame DataFrame for patient-based splitting
    records = []
    for patient_id, label, steatosis, frames in zip(
        patient_ids, labels, steatosis_values, frame_sequences
    ):
        for i in range(frames.shape[0]):
            records.append({"patient": patient_id, "frame_idx": i})
    examples = pd.DataFrame.from_records(records)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_val_indices, test_indices = next(
        splitter.split(X=examples, groups=examples["patient"])
    )

    # Build a lookup from (patient_id, frame_idx) → split
    test_keys = set(
        zip(
            examples.iloc[test_indices]["patient"],
            examples.iloc[test_indices]["frame_idx"],
        )
    )

    train_records = []
    test_records = []
    test_resized_records = []

    for patient_id, label, steatosis, frames in tqdm(
        zip(patient_ids, labels, steatosis_values, frame_sequences),
        total=len(patient_ids),
        desc="Processing examples",
    ):
        for i in range(frames.shape[0]):
            frame = frames[i]
            is_test = (patient_id, i) in test_keys
            record = _process_masld_example(patient_id, i, frame, label, steatosis)
            if is_test:
                test_records.append(record)
                test_resized_records.append(
                    _process_masld_example(patient_id, i, frame, label, steatosis, size)
                )
            else:
                train_records.append(record)

    print(train_records[0].keys())
    print(test_records[0].keys())

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_records, info=DATASET_INFO),
            "test": Dataset.from_list(test_records, info=DATASET_INFO),
            f"test.{size}": Dataset.from_list(test_resized_records, info=DATASET_INFO),
        }
    )

    output_path = Path(output_dir) / OUTPUT_NAME
    dataset.save_to_disk(str(output_path))

    generate_classification_ground_truth_from_records(
        test_records,
        output_path / f"test.{size}" / "ground_truth_classification.csv",
    )

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
