"""Prepare the training and test sets for the BUS-BRA dataset.

The dataset is split by patient into training and test data to ensure there is
no patient overlap between the splits.

The resulting dataset has three splits:

    - train: 1000 examples, original-size images.
    - test: the remaining patients, original-size images.
    - test.{size}: same examples as test, resized so the longest edge measures
      {size} px and then zero-padded to {size} x {size} px.

Each example has the following features:

    - filename: Original filename without the file extension.
    - patient: Zero-padded 4-digit patient case number.
    - side: The breast that was imaged ("left" or "right").
    - histology: Histopathology of the breast lesion (text).
    - histology_label: ClassLabel for the 8 most common histologies plus
      "other" (9 classes total).
    - pathology: ClassLabel for the pathology (benign or malignant).
    - birads: BI-RADS assessment score.
    - device: Ultrasound device used.
    - image: 3-channel ultrasound image.
    - tumor_mask: Single-channel binary tumor mask.

Three ground-truth CSV files are also created in each test split
directory for use by the evaluation layer, with the segmentation masks
encoded at the resolution of that split:

    - ground_truth_classification_pathology.csv
    - ground_truth_classification_histology.csv
    - ground_truth_segmentation.csv

Usage:

    uv run python scripts/process_dataset.py busbra RAW_DATA_DIR OUTPUT_DIR
"""

import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as io
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
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

from ultrabench.datasets.builders.utils import (
    DATASET_REVISION,
    generate_classification_ground_truth_from_records,
    generate_segmentation_ground_truth_from_records,
    resize_and_pad,
    save_version_info,
    to_3_channel,
)

from .utils import (
    cap_train_split_by_patient,
    ground_truth_commit_operations,
    verify_revision_unpublished,
)

__version__ = version("ultrabench")

OUTPUT_NAME = "UltraBench/BUS-BRA"

KNOWN_HISTOLOGIES = [
    "fibroadenoma",
    "invasive ductal carcinoma",
    "cyst",
    "fibrocystic changes",
    "invasive lobular carcinoma",
    "intraductal papilloma",
    "sclerosing adenosis",
    "hyperplasia",
]

DATASET_INFO = DatasetInfo(
    description=(
        "A dataset of breast ultrasound images for breast tumor "
        "segmentation and classification."
    ),
    citation="""
    @article{gomezflores2024_busbra,
        author = {Gómez-Flores, Wilfrido and
                  Gregorio-Calas, Miguel J. and
                  Coelho de Albuquerque Pereira, Wagner},
        title = {BUS-BRA: A breast ultrasound dataset for assessing
                 computer-aided diagnosis systems},
        journal = {Medical Physics},
        volume = {51},
        pages = {3110--3123},
        year = {2024},
        doi = {10.1002/mp.16812},
    }
    """,
    homepage="https://zenodo.org/records/8231412",
    license="cc-by-4.0",
    features=Features(
        {
            "filename": Value("string"),
            "patient": Value("string"),
            "side": Value("string"),
            "histology": Value("string"),
            "histology_label": ClassLabel(names=KNOWN_HISTOLOGIES + ["other"]),
            "pathology": ClassLabel(names=["benign", "malignant"]),
            "birads": Value("int32"),
            "device": Value("string"),
            "image": Image(),
            "tumor_mask": Image(),
            "scan_mask": Image(),
        }
    ),
)


def collect_metadata(dataset_dir: str) -> pd.DataFrame:
    """Load and prepare the metadata for all images in the dataset.

    Args:
        dataset_dir: The path to the raw BUS-BRA dataset directory.

    Returns:
        A DataFrame with one row per image containing preprocessed metadata
        fields.

    Raises:
        FileNotFoundError: If the metadata CSV is not found.
    """
    csv_path = os.path.join(dataset_dir, "bus_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    df["filename"] = df["ID"]
    df["patient"] = df["Case"].apply(lambda x: f"{x:04d}")
    df["side"] = df["Side"].str.lower()
    df["histology"] = df["Histology"]
    df["histology_label"] = df["Histology"].apply(
        lambda x: KNOWN_HISTOLOGIES.index(x) if x in KNOWN_HISTOLOGIES else 8
    )
    df["pathology"] = df["Pathology"].apply(lambda x: 0 if x == "benign" else 1)
    df["birads"] = df["BIRADS"].astype(np.int32)
    df["device"] = df["Device"]

    return df[
        [
            "filename",
            "patient",
            "side",
            "histology",
            "histology_label",
            "pathology",
            "birads",
            "device",
        ]
    ]


def process_examples(
    dataset_dir: str, df: pd.DataFrame, size: int | None = None
) -> list[dict]:
    """Process a subset of examples into record dicts.

    Args:
        dataset_dir: The path to the raw BUS-BRA dataset directory.
        df: DataFrame of examples to process (from collect_metadata).
        size: If provided, resize and pad images and masks to size x size
            pixels.

    Returns:
        A list of example dictionaries with numpy array values for image and
        tumor_mask fields.
    """
    images_dir = os.path.join(dataset_dir, "Images")
    masks_dir = os.path.join(dataset_dir, "Masks")

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing examples"):
        filename = row["filename"]
        image_path = os.path.join(images_dir, f"{filename}.png")
        mask_path = os.path.join(masks_dir, f"mask_{filename.removeprefix('bus_')}.png")

        raw_image = io.imread(image_path)
        image = to_3_channel(raw_image)
        tumor_mask = io.imread(mask_path).astype(np.uint8)
        scan_mask = np.ones(tumor_mask.shape, dtype=np.uint8)

        if size is not None:
            image = resize_and_pad(image, size, order=1)
            tumor_mask = resize_and_pad(tumor_mask, size, order=0)
            scan_mask = np.ones((size, size), dtype=np.uint8)

        records.append(
            {
                "filename": filename,
                "patient": row["patient"],
                "side": row["side"],
                "histology": row["histology"],
                "histology_label": int(row["histology_label"]),
                "pathology": int(row["pathology"]),
                "birads": int(row["birads"]),
                "device": row["device"],
                "image": image,
                "tumor_mask": tumor_mask,
                "scan_mask": scan_mask,
            }
        )

    return records


def write_ground_truth(
    records: list[dict], gt_dir: Path, image_size: int | None
) -> None:
    """Write the ground-truth CSVs of one test split.

    Args:
        records: The records of the split.
        gt_dir: The split directory to write the CSVs into.
        image_size: The height and width of the split's images, or None
            for the original-resolution split.
    """
    generate_classification_ground_truth_from_records(
        [{"filename": r["filename"], "label": r["pathology"]} for r in records],
        gt_dir / "ground_truth_classification_pathology.csv",
    )
    generate_classification_ground_truth_from_records(
        [{"filename": r["filename"], "label": r["histology_label"]} for r in records],
        gt_dir / "ground_truth_classification_histology.csv",
    )
    generate_segmentation_ground_truth_from_records(
        records,
        mask_key="tumor_mask",
        classes=[1],
        output_path=gt_dir / "ground_truth_segmentation.csv",
        image_size=image_size,
    )


def verify_args(raw_data_dir: str, output_dir: str, publish: bool = False) -> None:
    """Verify the command line arguments.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        publish: Whether the dataset will be published to the Hugging Face Hub.

    Raises:
        AssertionError: If any argument is invalid.
    """
    assert os.path.isdir(raw_data_dir), "raw_data_dir must be an existing directory"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(os.path.join(output_dir, OUTPUT_NAME)), (
        "A version of the BUS-BRA dataset already exists in output_dir"
    )

    if publish:
        verify_revision_unpublished(OUTPUT_NAME)


def process_busbra(
    raw_data_dir: str,
    output_dir: str,
    size: int = 256,
    publish: bool = False,
) -> None:
    """Prepare the BUS-BRA dataset as a Hugging Face DatasetDict.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        size: The height and width of the resized test images in pixels.
        publish: Whether to publish the dataset to the Hugging Face Hub.
    """
    verify_args(raw_data_dir, output_dir, publish)
    raw_data_dir = os.path.join(raw_data_dir, "BUSBRA")

    df = collect_metadata(raw_data_dir)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["patient"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_df, test_df = cap_train_split_by_patient(train_df, test_df)

    print(
        f"Train examples: {len(train_df)} from {train_df['patient'].nunique()} "
        f"patients, Test examples: {len(test_df)} from "
        f"{test_df['patient'].nunique()} patients"
    )

    train_records = process_examples(raw_data_dir, train_df)
    test_records = process_examples(raw_data_dir, test_df)
    test_resized_records = process_examples(raw_data_dir, test_df, size=size)

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_records, info=DATASET_INFO),
            "test": Dataset.from_list(test_records, info=DATASET_INFO),
            f"test.{size}": Dataset.from_list(test_resized_records, info=DATASET_INFO),
        }
    )

    output_path = Path(output_dir) / OUTPUT_NAME
    dataset.save_to_disk(str(output_path))

    write_ground_truth(test_records, output_path / "test", None)
    write_ground_truth(test_resized_records, output_path / f"test.{size}", size)

    save_version_info(str(output_path), __version__)

    if publish:
        dataset.push_to_hub(
            OUTPUT_NAME,
            commit_message=f"Update the dataset to revision {DATASET_REVISION}.",
            private=True,
        )

        api = HfApi()
        operations = ground_truth_commit_operations(
            output_path,
            {
                "classification-pathology": (
                    "ground_truth_classification_pathology.csv"
                ),
                "classification-histology": (
                    "ground_truth_classification_histology.csv"
                ),
                "segmentation": "ground_truth_segmentation.csv",
            },
            size,
        )
        api.create_commit(
            repo_id=OUTPUT_NAME,
            repo_type="dataset",
            operations=operations,
            commit_message=(
                f"Update ground truth CSV files for revision {DATASET_REVISION}."
            ),
        )
