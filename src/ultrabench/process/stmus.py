"""Prepare the training and test sets for the STMUS dataset.

The dataset is split by patient into training and test data (80:20)
to ensure there is no patient overlap between the splits.

The resulting dataset has three splits:

    - train: 80% of patients, original-size images.
    - test: 20% of patients, original-size images.
    - test.{size}: same examples as test, resized so the longest edge
      measures {size} px and then zero-padded to {size} x {size} px.

Each example has the following features:

    - filename: Original image filename (without file extension) prefixed
      by muscle (bb, gm, or ta) and echogenicity (healthy or pathological).
    - patient: Patient ID string.
    - muscle: Muscle name (biceps brachii, gastrocnemius medialis, or
      tibialis anterior).
    - echogenicity: Echogenicity level (healthy or pathological).
    - image: 3-channel ultrasound image.
    - mask: Single-channel binary muscle segmentation mask
      (background=0, muscle=1).

A segmentation ground-truth CSV is also created for the test split
for use by the evaluation layer:

    - ground_truth_segmentation.csv

Usage:

    ultrabench process stmus RAW_DATA_DIR OUTPUT_DIR
"""

import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io as io
import typer
from datasets import (
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
    generate_segmentation_ground_truth_from_records,
    resize_and_pad,
    save_version_info,
    to_3_channel,
)

__version__ = version("ultrabench")

OUTPUT_NAME = "UltraBench/STMUS"

MUSCLE_NAMES = {
    "BB": "biceps brachii",
    "GM": "gastrocnemius medialis",
    "TA": "tibialis anterior",
}

DATASET_INFO = DatasetInfo(
    description=(
        "A dataset of transverse musculoskeletal ultrasound images for "
        "binary muscle segmentation."
    ),
    citation="""
    @article{marzola2021_stmus,
        author = {Marzola, Francesco and
                  van Alfen, Nens and
                  Doorduin, Jonne and
                  Meiburger, Kristen M.},
        title = {Deep learning segmentation of transverse musculoskeletal
                 ultrasound images for neuromuscular disease assessment},
        journal = {Computers in Biology and Medicine},
        volume = {135},
        pages = {104623},
        year = {2021},
        doi = {10.1016/j.compbiomed.2021.104623},
    }
    """,
    homepage="https://data.mendeley.com/datasets/3jykz7wz8d/1",
    license="cc-by-4.0",
    features=Features(
        {
            "filename": Value("string"),
            "patient": Value("string"),
            "muscle": Value("string"),
            "echogenicity": Value("string"),
            "image": Image(),
            "mask": Image(),
            "scan_mask": Image(),
        }
    ),
)


def collect_examples(dataset_dir: str) -> pd.DataFrame:
    """Collect metadata for all images in the STMUS dataset.

    Scans the raw dataset directory for image files and parses their
    paths to extract per-example metadata.

    Args:
        dataset_dir: Path to the Polito-Radboud-DeepLearningUS
            directory inside the raw download.

    Returns:
        A DataFrame with one row per image containing the columns
        ``filename``, ``patient``, ``muscle``, ``echogenicity``,
        ``image_path``, and ``mask_path``.
    """
    image_paths = sorted(Path(dataset_dir).glob("**/Images/*.png"))

    rows = []
    for path in image_paths:
        parts = path.parts
        muscle_code = parts[-4]
        echogenicity = parts[-3]
        stem = path.stem

        subject_id = int(stem.split("_")[1])

        rows.append(
            {
                "filename": (f"{muscle_code.lower()}_{echogenicity.lower()}_{stem}"),
                "patient": str(subject_id),
                "muscle": MUSCLE_NAMES[muscle_code],
                "echogenicity": echogenicity.lower(),
                "image_path": str(path),
                "mask_path": str(path.parent.parent / "Masks" / path.name),
            }
        )

    return pd.DataFrame(rows)


def _process_row(row: pd.Series, size: int | None = None) -> dict:
    """Process a single example into a record dict.

    Args:
        row: A row from the DataFrame returned by
            ``collect_examples``.
        size: If provided, resize and pad the image and mask to
            ``size x size`` pixels.

    Returns:
        A dict with keys ``filename``, ``patient``, ``muscle``,
        ``echogenicity``, ``image`` (numpy array, uint8, HxWx3),
        and ``mask`` (numpy array, uint8, HxW, values in {0, 1}).
    """
    raw_image = io.imread(row["image_path"])
    image = to_3_channel(raw_image)
    mask = (io.imread(row["mask_path"]) >= 128).astype(np.uint8)
    scan_mask = np.ones(mask.shape, dtype=np.uint8)

    if size is not None:
        image = resize_and_pad(image, size, order=1)
        mask = resize_and_pad(mask, size, order=0)
        scan_mask = np.ones((size, size), dtype=np.uint8)

    return {
        "filename": row["filename"],
        "patient": row["patient"],
        "muscle": row["muscle"],
        "echogenicity": row["echogenicity"],
        "image": image,
        "mask": mask,
        "scan_mask": scan_mask,
    }


def _make_generator(
    df: pd.DataFrame,
    size: int | None = None,
    desc: str = "Processing",
):
    """Return a zero-argument generator function for one split.

    Args:
        df: Metadata DataFrame for the split.
        size: If provided, resize and pad to ``size x size`` pixels.
        desc: tqdm progress bar label.

    Returns:
        A zero-argument generator function that yields one record
        dict per example.
    """

    def generator():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            yield _process_row(row, size=size)

    return generator


def verify_args(raw_data_dir: str, output_dir: str) -> None:
    """Verify command-line arguments before processing begins.

    Args:
        raw_data_dir: Path to the raw STMUS download directory.
        output_dir: Path to the output directory.

    Raises:
        AssertionError: If any argument is invalid or the output
            dataset already exists.
    """
    assert os.path.isdir(raw_data_dir), "raw_data_dir must be an existing directory"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(os.path.join(output_dir, OUTPUT_NAME)), (
        "A version of the STMUS dataset already exists in output_dir"
    )


def process_stmus(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str,
        typer.Argument(help="The output directory for the processed dataset"),
    ],
    size: Annotated[
        int,
        typer.Option(help="Test image size (height and width in pixels)"),
    ] = 256,
    publish: Annotated[
        bool,
        typer.Option(help="Whether to publish the dataset to the Hugging Face Hub"),
    ] = False,
) -> None:
    """Prepare the STMUS dataset as a Hugging Face DatasetDict.

    Args:
        raw_data_dir: Path to the raw data directory (the
            ``stmus_raw_YYYYMMDD`` folder created by the download
            command).
        output_dir: Path to the output directory.
        size: Height and width of resized test images in pixels.
        publish: Whether to publish the dataset to the HF Hub.
    """
    verify_args(raw_data_dir, output_dir)

    dataset_dir = os.path.join(raw_data_dir, "Polito-Radboud-DeepLearningUS")
    df = collect_examples(dataset_dir)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["patient"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"Train examples: {len(train_df)}, Test examples: {len(test_df)}")

    dataset = DatasetDict(
        {
            "train": Dataset.from_generator(
                _make_generator(train_df, desc="train"),
                features=DATASET_INFO.features,
            ),
            "test": Dataset.from_generator(
                _make_generator(test_df, desc="test"),
                features=DATASET_INFO.features,
            ),
            f"test.{size}": Dataset.from_generator(
                _make_generator(test_df, size=size, desc=f"test.{size}"),
                features=DATASET_INFO.features,
            ),
        }
    )

    output_path = Path(output_dir) / OUTPUT_NAME
    dataset.save_to_disk(str(output_path))

    gt_dir = output_path / f"test.{size}"

    gt_records = [
        {
            "filename": row["filename"],
            "mask": (io.imread(row["mask_path"]) >= 128).astype(np.uint8),
        }
        for _, row in tqdm(
            test_df.iterrows(),
            total=len(test_df),
            desc="Generating ground truth",
        )
    ]
    generate_segmentation_ground_truth_from_records(
        gt_records,
        mask_key="mask",
        classes=[1],
        output_path=gt_dir / "ground_truth_segmentation.csv",
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
                path_in_repo=f"test.{size}-ground-truth-segmentation.csv",
                path_or_fileobj=gt_dir / "ground_truth_segmentation.csv",
            ),
        ]
        api.create_commit(
            repo_id=OUTPUT_NAME,
            repo_type="dataset",
            operations=operations,
            commit_message=(
                f"Update ground truth CSV files for Version {__version__}."
            ),
        )
