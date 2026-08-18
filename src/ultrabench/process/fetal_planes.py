"""Prepare the Fetal Planes dataset as Hugging Face DatasetDicts.

The dataset contains 12,400 maternal-fetal ultrasound images and is organized
into two subsets:

    - fetal: all images, labelled by fetal plane (6 classes).
    - fetal_brain: brain images only, labelled by brain plane (4 classes).

The original train/test split is preserved for both subsets. Each
training split is then capped at a maximum of 1000 examples by moving
whole patients into the corresponding test split. The two subsets are
capped independently, so a patient may be in the training split of one
subset and the test split of the other. Each subset remains split by
patient internally.

Each subset produces three splits:

    - train: 1000 original-size images.
    - test: the remaining patients, original-size images.
    - test.{size}: same examples as test, resized so the longest edge measures
      {size} px and then zero-padded to {size} x {size} px.

Features for the ``fetal`` subset:

    - filename: Original image filename (without the file extension).
    - patient: Patient ID.
    - plane: ClassLabel for the fetal plane (6 classes).
    - brain_plane: Fetal brain plane as a string (e.g. "Not A Brain").
    - operator: Operator ID string.
    - machine: Ultrasound machine string.
    - image: 3-channel ultrasound image.
    - scan_mask: Single-channel binary ultrasound scan mask.

Features for the ``fetal_brain`` subset:

    - filename: Original image filename (without the file extension).
    - patient: Patient ID.
    - brain_plane: ClassLabel for the fetal brain plane (4 classes).
    - operator: Operator ID string.
    - machine: Ultrasound machine string.
    - image: 3-channel ultrasound image.
    - scan_mask: Single-channel binary ultrasound scan mask.

A classification ground-truth CSV is generated for each subset:

    - fetal/test.{size}/ground_truth_classification.csv
    - fetal_brain/test.{size}/ground_truth_classification.csv

Usage:
    ultrabench process fetal-planes RAW_DATA_DIR OUTPUT_DIR
"""

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
    cap_train_split_by_patient,
    generate_classification_ground_truth_from_records,
    resize_and_pad,
    save_version_info,
    to_3_channel,
    verify_revision_unpublished,
)

__version__ = version("ultrabench")

CSV_FILE = "FETAL_PLANES_DB_data.csv"
IMAGE_DIR = "Images"
OUTPUT_NAME = "UltraBench/FetalPlanes"
PLANE_NAMES = [
    "Fetal abdomen",
    "Fetal brain",
    "Fetal femur",
    "Fetal thorax",
    "Maternal cervix",
    "Other",
]
BRAIN_PLANE_NAMES = [
    "Trans-thalamic",
    "Trans-cerebellum",
    "Trans-ventricular",
    "Other",
]

_SHARED_INFO = dict(
    description=(
        "Maternal-fetal screening ultrasound images for fetal plane "
        "and fetal brain plane classification."
    ),
    citation="""
    @article{burgos2020_fetalplanes,
        author  = {Burgos-Artizzu, X.P. and
                   Coronado-Guti{\\'{e}}rrez, D. and
                   Valenzuela-Alcaraz, B. and others},
        title   = {Evaluation of deep convolutional neural networks for
                   automatic classification of common maternal fetal
                   ultrasound planes},
        journal = {Scientific Reports},
        volume  = {10},
        pages   = {10200},
        year    = {2020},
        doi     = {10.1038/s41598-020-67076-5},
    }
    """,
    homepage="https://zenodo.org/record/3904280",
    license="cc-by-4.0",
)

FETAL_DATASET_INFO = DatasetInfo(
    **_SHARED_INFO,  # type: ignore
    features=Features(
        {
            "filename": Value("string"),
            "patient": Value("int32"),
            "plane": ClassLabel(names=PLANE_NAMES),
            "brain_plane": Value("string"),
            "operator": Value("string"),
            "machine": Value("string"),
            "image": Image(),
            "scan_mask": Image(),
        }
    ),
)

FETAL_BRAIN_DATASET_INFO = DatasetInfo(
    **_SHARED_INFO,  # type: ignore
    features=Features(
        {
            "filename": Value("string"),
            "patient": Value("int32"),
            "brain_plane": ClassLabel(names=BRAIN_PLANE_NAMES),
            "operator": Value("string"),
            "machine": Value("string"),
            "image": Image(),
            "scan_mask": Image(),
        }
    ),
)


def generate_scan_mask(image: np.ndarray) -> np.ndarray:
    """Generate a binary scan mask using morphological operations.

    Args:
        image: A 2D grayscale image array (uint8).

    Returns:
        A binary uint8 mask of the ultrasound scan region.
    """
    mask = image > 0
    mask = skimage.morphology.remove_small_objects(mask, min_size=5000)
    for _ in range(5):
        mask = skimage.morphology.binary_erosion(mask)
    for _ in range(5):
        mask = skimage.morphology.binary_dilation(mask)
    mask = skimage.morphology.convex_hull_image(mask)

    return mask.astype(np.uint8)


def load_metadata(raw_data_dir: str) -> pd.DataFrame:
    """Load and clean the dataset metadata CSV.

    Args:
        raw_data_dir: Path to the raw dataset directory containing
            ``FETAL_PLANES_DB_data.csv``.

    Returns:
        A DataFrame with columns ``Image_name``, ``Patient_num``,
        ``Plane``, ``Brain_plane``, ``Operator``, ``US_Machine``,
        and ``train`` (renamed from ``Train ``).
    """
    csv_path = os.path.join(raw_data_dir, CSV_FILE)
    df = pd.read_csv(csv_path, sep=";")
    df.rename(columns={"Train ": "train"}, inplace=True)
    return df


def process_example(
    row: pd.Series,
    raw_data_dir: str,
    size: int | None = None,
) -> dict:
    """Process a single example into a base record dict.

    Args:
        row: A row from the metadata DataFrame.
        raw_data_dir: Path to the raw dataset directory.
        size: If provided, resize and pad to ``size x size`` pixels.

    Returns:
        A dict with keys ``filename``, ``patient``, ``plane_str``,
        ``brain_plane_str``, ``operator``, ``machine``, ``image``,
        and ``scan_mask``. Image values are numpy arrays.
    """
    filename = row["Image_name"]
    image_path = os.path.join(raw_data_dir, IMAGE_DIR, f"{filename}.png")
    raw_image = io.imread(image_path)
    gray = raw_image.mean(axis=-1).astype("uint8") if raw_image.ndim == 3 else raw_image
    scan_mask = generate_scan_mask(gray)
    image = to_3_channel(raw_image)
    if size is not None:
        image = resize_and_pad(image, size, order=1)
        scan_mask = resize_and_pad(scan_mask, size, order=0)

    return {
        "filename": filename,
        "patient": int(row["Patient_num"]),
        "plane_str": row["Plane"],
        "brain_plane_str": row["Brain_plane"],
        "operator": row["Operator"],
        "machine": row["US_Machine"],
        "image": image,
        "scan_mask": scan_mask,
    }


def to_fetal_record(record: dict) -> dict:
    """Convert a base record to a ``fetal`` subset record.

    Args:
        record: A base record from ``process_example``.

    Returns:
        A dict matching ``FETAL_DATASET_INFO`` features, with ``plane`` as an
        integer class index and ``brain_plane`` as a raw string.
    """
    return {
        "filename": record["filename"],
        "patient": record["patient"],
        "plane": PLANE_NAMES.index(record["plane_str"]),
        "brain_plane": record["brain_plane_str"],
        "operator": record["operator"],
        "machine": record["machine"],
        "image": record["image"],
        "scan_mask": record["scan_mask"],
    }


def to_fetal_brain_record(record: dict) -> dict:
    """Convert a base record to a ``fetal_brain`` subset record.

    Args:
        record: A base record from ``process_example`` where
            ``plane_str == "Fetal brain"``.

    Returns:
        A dict matching ``FETAL_BRAIN_DATASET_INFO`` features, with
        ``brain_plane`` as an integer class index.
    """
    return {
        "filename": record["filename"],
        "patient": record["patient"],
        "brain_plane": BRAIN_PLANE_NAMES.index(record["brain_plane_str"]),
        "operator": record["operator"],
        "machine": record["machine"],
        "image": record["image"],
        "scan_mask": record["scan_mask"],
    }


def _make_generator(
    df: pd.DataFrame,
    raw_data_dir: str,
    to_record_fn: callable,  # ty:ignore[invalid-type-form]
    size: int | None = None,
    desc: str = "Processing",
):
    """Return a zero-argument generator function for one split.

    Args:
        df: Metadata DataFrame for the split.
        raw_data_dir: Path to the raw dataset directory.
        to_record_fn: Converts a base record to subset-specific format.
        size: If provided, resize and pad to size x size pixels.
        desc: tqdm progress bar label.

    Returns:
        A zero-argument generator function.
    """

    def generator():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            yield to_record_fn(process_example(row, raw_data_dir, size=size))

    return generator


def verify_args(raw_data_dir: str, output_dir: str, publish: bool = False) -> None:
    """Verify command-line arguments before processing begins.

    Args:
        raw_data_dir: Path to the raw dataset directory.
        output_dir: Path to the output directory.
        publish: Whether the dataset will be published to the Hugging
            Face Hub.

    Raises:
        AssertionError: If any argument is invalid or outputs already exist.
    """
    assert os.path.isdir(raw_data_dir), "raw_data_dir must be an existing directory"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(os.path.join(output_dir, OUTPUT_NAME, "fetal")), (
        "A fetal subset already exists in output_dir"
    )
    assert not os.path.exists(os.path.join(output_dir, OUTPUT_NAME, "fetal_brain")), (
        "A fetal_brain subset already exists in output_dir"
    )

    if publish:
        verify_revision_unpublished(OUTPUT_NAME)


def process_fetal_planes(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
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
    """Prepare the Fetal Planes dataset as Hugging Face DatasetDicts.

    Args:
        raw_data_dir: Path to the raw data directory.
        output_dir: Path to the output directory.
        size: Height and width of resized test images in pixels.
        publish: Whether to publish the dataset to the HF Hub.
    """
    verify_args(raw_data_dir, output_dir, publish)

    output_path = Path(output_dir) / OUTPUT_NAME
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_metadata(raw_data_dir)
    train_df = df[df["train"] == 1].reset_index(drop=True)
    test_df = df[df["train"] == 0].reset_index(drop=True)
    brain_train_df = train_df[train_df["Plane"] == "Fetal brain"].reset_index(drop=True)
    brain_test_df = test_df[test_df["Plane"] == "Fetal brain"].reset_index(drop=True)

    # The two subsets are capped independently so that each gets the full
    # training budget. Both remain split by patient internally.
    train_df, test_df = cap_train_split_by_patient(
        train_df, test_df, patient_col="Patient_num"
    )
    brain_train_df, brain_test_df = cap_train_split_by_patient(
        brain_train_df, brain_test_df, patient_col="Patient_num"
    )

    for name, split_df in [
        ("fetal train", train_df),
        ("fetal test", test_df),
        ("fetal_brain train", brain_train_df),
        ("fetal_brain test", brain_test_df),
    ]:
        print(
            f"{name}: {len(split_df)} examples from "
            f"{split_df['Patient_num'].nunique()} patients"
        )

    fetal_dataset = DatasetDict(
        {
            "train": Dataset.from_generator(
                _make_generator(
                    train_df, raw_data_dir, to_fetal_record, desc="fetal train"
                ),
                features=FETAL_DATASET_INFO.features,
            ),
            "test": Dataset.from_generator(
                _make_generator(
                    test_df, raw_data_dir, to_fetal_record, desc="fetal test"
                ),
                features=FETAL_DATASET_INFO.features,
            ),
            f"test.{size}": Dataset.from_generator(
                _make_generator(
                    test_df,
                    raw_data_dir,
                    to_fetal_record,
                    size=size,
                    desc=f"fetal test.{size}",
                ),
                features=FETAL_DATASET_INFO.features,
            ),
        }
    )
    fetal_brain_dataset = DatasetDict(
        {
            "train": Dataset.from_generator(
                _make_generator(
                    brain_train_df,
                    raw_data_dir,
                    to_fetal_brain_record,
                    desc="fetal_brain train",
                ),
                features=FETAL_BRAIN_DATASET_INFO.features,
            ),
            "test": Dataset.from_generator(
                _make_generator(
                    brain_test_df,
                    raw_data_dir,
                    to_fetal_brain_record,
                    desc="fetal_brain test",
                ),
                features=FETAL_BRAIN_DATASET_INFO.features,
            ),
            f"test.{size}": Dataset.from_generator(
                _make_generator(
                    brain_test_df,
                    raw_data_dir,
                    to_fetal_brain_record,
                    size=size,
                    desc=f"fetal_brain test.{size}",
                ),
                features=FETAL_BRAIN_DATASET_INFO.features,
            ),
        }
    )

    fetal_dataset.save_to_disk(str(output_path / "fetal"))
    fetal_brain_dataset.save_to_disk(str(output_path / "fetal_brain"))

    fetal_gt_path = (
        output_path / "fetal" / f"test.{size}" / "ground_truth_classification.csv"
    )
    fetal_brain_gt_path = (
        output_path / "fetal_brain" / f"test.{size}" / "ground_truth_classification.csv"
    )

    generate_classification_ground_truth_from_records(
        [
            {"filename": row["Image_name"], "label": PLANE_NAMES.index(row["Plane"])}
            for _, row in test_df.iterrows()
        ],
        fetal_gt_path,
    )
    generate_classification_ground_truth_from_records(
        [
            {
                "filename": row["Image_name"],
                "label": BRAIN_PLANE_NAMES.index(row["Brain_plane"]),
            }
            for _, row in brain_test_df.iterrows()
        ],
        fetal_brain_gt_path,
    )

    save_version_info(str(output_path), __version__)

    if publish:
        fetal_dataset.push_to_hub(
            OUTPUT_NAME,
            config_name="fetal",
            commit_message=(f"Update the fetal subset to revision {DATASET_REVISION}."),
            private=True,
        )
        fetal_brain_dataset.push_to_hub(
            OUTPUT_NAME,
            config_name="fetal_brain",
            commit_message=(
                f"Update the fetal brain subset to revision {DATASET_REVISION}."
            ),
            private=True,
        )

        api = HfApi()
        operations = [
            CommitOperationAdd(
                path_in_repo="fetal-test-ground-truth-classification.csv",
                path_or_fileobj=fetal_gt_path,
            ),
            CommitOperationAdd(
                path_in_repo="fetal-brain-test-ground-truth-classification.csv",
                path_or_fileobj=fetal_brain_gt_path,
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
