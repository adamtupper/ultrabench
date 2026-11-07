"""Prepare the training and test sets for the TN3K dataset.

The dataset uses the original train and test splits provided with the
dataset (2,879 training and 614 test images). The training split is then
subsampled to a maximum of 1000 images. The images are removed from the
largest classes first, which makes the class distribution more uniform
and preserves the rarer malignant nodules. The removed images are
discarded rather than moved to the test split, since the images lack
patient identifiers and moving them could introduce patient overlap
between the splits.

The resulting dataset has three splits:

    - train: up to 1000 images in their original sizes.
    - test: 614 images in their original sizes.
    - test.{size}: same examples as test, resized so the longest edge
      measures {size} px and then zero-padded to {size} x {size} px.

Each example has the following features:

    - filename: Original filename without the file extension.
    - image: 3-channel ultrasound image.
    - nodule_mask: Single-channel binary nodule mask (background=0, nodule=1).
    - label: ClassLabel for the pathology (benign=0, malignant=1).

Two ground-truth CSV files are also created in each test split
directory for use by the evaluation layer, with the segmentation masks
encoded at the resolution of that split:

    - ground_truth_classification.csv
    - ground_truth_segmentation.csv

Usage:

    uv run python scripts/process_dataset.py tn3k RAW_DATA_DIR OUTPUT_DIR
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
    ground_truth_commit_operations,
    subsample_balanced_by_class,
    verify_revision_unpublished,
)

__version__ = version("ultrabench")

OUTPUT_NAME = "UltraBench/TN3K"

DATASET_INFO = DatasetInfo(
    description=(
        "A dataset of thyroid ultrasound images for thyroid nodule "
        "segmentation and classification."
    ),
    citation="""
    @article{gong2023thyroid,
        author = {Gong, Haifan and
                  Chen, Jiaxin and
                  Chen, Guanqi and
                  Li, Haofeng and
                  Li, Guanbin and
                  Chen, Fei},
        title = {Thyroid region prior guided attention for ultrasound
                 segmentation of thyroid nodules},
        journal = {Computers in Biology and Medicine},
        volume = {155},
        pages = {106389},
        year = {2023},
        doi = {10.1016/j.compbiomed.2022.106389},
    }
    """,
    license="mit",
    homepage=("https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation"),
    features=Features(
        {
            "filename": Value("string"),
            "image": Image(),
            "nodule_mask": Image(),
            "scan_mask": Image(),
            "label": ClassLabel(names=["benign", "malignant"]),
        }
    ),
)


def collect_examples(dataset_dir: str, split: str) -> pd.DataFrame:
    """Load label metadata for a given split.

    Args:
        dataset_dir: Path to the raw TN3K dataset directory (the
            ``tn3k`` subdirectory of the downloaded archive).
        split: Either ``"trainval"`` or ``"test"``.

    Returns:
        A DataFrame with columns ``filename`` (stem, no extension)
        and ``label`` (int, 0=benign / 1=malignant).

    Raises:
        FileNotFoundError: If the label CSV is not found.
    """
    csv_path = os.path.join(dataset_dir, f"label4{split}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Label CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, names=["image", "label"])
    df["filename"] = df["image"].apply(lambda x: Path(x).stem)
    df["label"] = df["label"].astype(int)

    return df[["filename", "label"]]


def process_examples(
    dataset_dir: str,
    split: str,
    df: pd.DataFrame,
    size: int | None = None,
) -> list[dict]:
    """Process a subset of examples into record dicts.

    Args:
        dataset_dir: Path to the raw TN3K dataset directory.
        split: Either ``"trainval"`` or ``"test"``.
        df: DataFrame of examples from ``collect_examples``.
        size: If provided, resize and pad images and masks to ``size x size``
            pixels.

    Returns:
        A list of example dictionaries with numpy array values for the
        ``image`` and ``nodule_mask`` fields.
    """
    images_dir = os.path.join(dataset_dir, f"{split}-image")
    masks_dir = os.path.join(dataset_dir, f"{split}-mask")

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing examples"):
        filename = row["filename"]
        image_path = os.path.join(images_dir, f"{filename}.jpg")
        mask_path = os.path.join(masks_dir, f"{filename}.jpg")

        raw_image = io.imread(image_path)
        image = to_3_channel(raw_image)
        nodule_mask = (io.imread(mask_path) >= 128).astype(np.uint8)
        scan_mask = np.ones(nodule_mask.shape, dtype=np.uint8)

        if size is not None:
            image = resize_and_pad(image, size, order=1)
            nodule_mask = resize_and_pad(nodule_mask, size, order=0)
            scan_mask = np.ones((size, size), dtype=np.uint8)

        records.append(
            {
                "filename": filename,
                "image": image,
                "nodule_mask": nodule_mask,
                "scan_mask": scan_mask,
                "label": int(row["label"]),
            }
        )

    return records


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
        "A version of the TN3K dataset already exists in output_dir"
    )

    if publish:
        verify_revision_unpublished(OUTPUT_NAME)


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
        [{"filename": r["filename"], "label": r["label"]} for r in records],
        gt_dir / "ground_truth_classification.csv",
    )
    generate_segmentation_ground_truth_from_records(
        records,
        mask_key="nodule_mask",
        classes=[1],
        output_path=gt_dir / "ground_truth_segmentation.csv",
        image_size=image_size,
    )


def process_tn3k(
    raw_data_dir: str,
    output_dir: str,
    size: int = 256,
    publish: bool = False,
) -> None:
    """Prepare the TN3K dataset as a Hugging Face DatasetDict.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        size: The height and width of the resized test images in pixels.
        publish: Whether to publish the dataset to the Hugging Face Hub.
    """
    verify_args(raw_data_dir, output_dir, publish)
    dataset_dir = os.path.join(raw_data_dir, "tn3k")

    trainval_df = collect_examples(dataset_dir, "trainval")
    test_df = collect_examples(dataset_dir, "test")

    trainval_df = subsample_balanced_by_class(trainval_df, label_col="label")

    print(f"Train examples: {len(trainval_df)}, Test examples: {len(test_df)}")
    print(f"Train labels: {trainval_df['label'].value_counts().to_dict()}")

    train_records = process_examples(dataset_dir, "trainval", trainval_df)
    test_records = process_examples(dataset_dir, "test", test_df)
    test_resized_records = process_examples(dataset_dir, "test", test_df, size=size)

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
                "classification": "ground_truth_classification.csv",
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
