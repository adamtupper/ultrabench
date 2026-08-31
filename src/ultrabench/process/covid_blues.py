"""Prepare the COVID-BLUES dataset as a Hugging Face DatasetDict by
extracting frames from videos and computing per-video scan masks.

The dataset is split by patients into training and test sets, assigning 43
patients to the training set and 20 patients to the test set.

The resulting dataset has three splits:

    - train: original-size frames (350 x 350 px) for 80% of patients.
    - test: original-size frames for the remaining 20% of patients.
    - test.{size}: same examples as test, resized so the longest edge
      measures {size} px and then zero-padded to {size} x {size} px.

Each example has the following features:

    - filename: Video stem suffixed by the frame index (0-based).
    - video: Original video filename stem.
    - frame: 0-based frame index within the video.
    - patient: Integer patient ID.
    - blue_point: BLUE protocol scan point (L1, L2, L3, R1, R2, R3).
    - covid19: ClassLabel encoding the COVID-19 PCR test result
      (negative=0, positive=1).
    - image: 3-channel ultrasound frame.
    - scan_mask: Single-channel binary ultrasound scan mask generated
      from per-pixel variance across all frames of the video.

A ground-truth CSV file is also created for the test split:

    - ground_truth_classification.csv

Usage:
    ultrabench process covid-blues RAW_DATA_DIR OUTPUT_DIR
"""

import os
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import imageio
import numpy as np
import pandas as pd
import skimage.morphology
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

from .utils import (
    generate_classification_ground_truth_from_records,
    resize_and_pad,
    save_version_info,
    verify_revision_unpublished,
)

__version__ = version("ultrabench")

OUTPUT_NAME = "UltraBench/COVID-BLUES"
DATASET_REVISION = "0.3"
DATASET_INFO = DatasetInfo(
    description=(
        "Lung ultrasound videos from suspected COVID-19 patients "
        "for COVID-19 classification."
    ),
    citation="""
    @article{wiedemann2025covidblues,
        author={Wiedemann, N. and Boer, D. de K. and Richter, M. and
                van de Weijer, S. and Buhre, C. and Eggert, F. A. M.
                and Aarnoudse, S. and Grevendonk, L. and Röber, S. and
                Remie, C. M. E. and Buhre, W. and Henry, R. and
                Born, J.},
        title={COVID-BLUeS - a Prospective Study on the Value of AI
               in Lung Ultrasound Analysis},
        journal={IEEE Journal of Biomedical and Health Informatics},
        volume={29},
        number={9},
        pages={6301--6310},
        year={2025},
        doi={10.1109/JBHI.2025.3543686},
    }
    """,
    homepage="https://github.com/NinaWie/COVID-BLUES",
    license="cc-by-nc-nd-4.0",
    features=Features(
        {
            "filename": Value("string"),
            "video": Value("string"),
            "frame": Value("int32"),
            "patient": Value("int32"),
            "blue_point": Value("string"),
            "covid19": ClassLabel(names=["negative", "positive"]),
            "image": Image(),
            "scan_mask": Image(),
        }
    ),
)


def parse_video_filename(path: Path) -> dict:
    """Extract patient ID, BLUE point, and video stem from a video path.

    Expects filenames of the form
    ``patient_{patient_id}_{blue_point}.mp4``.

    Args:
        path: Path to the video file.

    Returns:
        A dict with keys ``patient`` (int), ``blue_point`` (str), and
        ``video`` (str, the filename stem).
    """
    parts = path.stem.split("_")
    return {
        "patient": int(parts[1]),
        "blue_point": parts[2],
        "video": path.stem,
    }


def load_clinical_data(raw_data_dir: str) -> pd.DataFrame:
    """Load and return the patient-level clinical variables.

    Reads ``clinical_variables.csv`` and returns a DataFrame with only
    the ``patient_id`` and ``cov_test`` columns.

    Args:
        raw_data_dir: Path to the ``covid_blues`` directory containing
            ``clinical_variables.csv``.

    Returns:
        A DataFrame with columns ``patient_id`` (int) and
        ``cov_test`` (int, 0 or 1).
    """
    csv_path = os.path.join(raw_data_dir, "clinical_variables.csv")
    df = pd.read_csv(csv_path, usecols=["patient_id", "cov_test"])  # ty:ignore[no-matching-overload]
    df["patient_id"] = df["patient_id"].astype(int)
    df["cov_test"] = df["cov_test"].astype(int)
    return df


def extract_frames(video_path: str) -> np.ndarray:
    """Read all frames from a video file into a NumPy array.

    Args:
        video_path: Path to the MP4 video file.

    Returns:
        An array of shape ``(N, H, W, 3)`` containing all N frames.
    """
    reader = imageio.get_reader(video_path)
    frames = [frame for frame in reader]  # type: ignore[union-attr]  # ty:ignore[not-iterable]
    return np.array(frames)


def generate_scan_mask(frames: np.ndarray) -> np.ndarray:
    """Create a binary scan mask from per-pixel variance across frames.

    Pixels that never change across frames (variance = 0) are outside
    the scan region. A convex hull is applied to fill concavities.

    Args:
        frames: Array of shape ``(N, H, W, 3)`` with all video frames.

    Returns:
        A binary ``uint8`` array of shape ``(H, W)`` where 1 indicates
        the ultrasound scan region and 0 indicates background.
    """
    grayscale = np.mean(frames, axis=-1)  # (N, H, W)
    variance = np.var(grayscale, axis=0)  # (H, W)
    mask = variance > 0
    mask = skimage.morphology.convex_hull_image(mask)
    return mask.astype(np.uint8)


def _records_from_video(
    video_path: str,
    patient: int,
    blue_point: str,
    covid19: int,
    size: int | None = None,
) -> list[dict]:
    """Build per-frame records for one video.

    Extracts all frames, computes a single scan mask from their
    variance, and constructs one record per frame.

    Args:
        video_path: Path to the MP4 video file.
        patient: Integer patient ID.
        blue_point: BLUE protocol scan point (e.g., ``"L1"``).
        covid19: COVID-19 test result (0 = negative, 1 = positive).
        size: If provided, resize and pad image and mask to
            ``size × size`` pixels.

    Returns:
        A list of record dicts, one per frame.
    """
    video_stem = Path(video_path).stem
    frames = extract_frames(video_path)
    scan_mask = generate_scan_mask(frames)

    records = []
    for i, frame in enumerate(frames):
        image = frame  # already (H, W, 3) uint8 RGB
        mask = scan_mask.copy()

        if size is not None:
            image = resize_and_pad(image, size, order=1)
            mask = resize_and_pad(mask, size, order=0)

        records.append(
            {
                "filename": f"{video_stem}_{i}",
                "video": video_stem,
                "frame": i,
                "patient": patient,
                "blue_point": blue_point,
                "covid19": covid19,
                "image": image,
                "scan_mask": mask,
            }
        )

    return records


def _make_train_generator(train_df: pd.DataFrame, raw_data_dir: str):
    """Return a generator that yields train records one frame at a time.

    Processes one video at a time to keep peak memory usage low.

    Args:
        train_df: DataFrame with columns ``video``, ``patient``,
            ``blue_point``, ``covid19``, and ``video_path``.
        raw_data_dir: Path to the ``covid_blues`` directory.

    Returns:
        A zero-argument generator function.
    """
    rows = list(train_df.itertuples(index=False))

    def generator():
        for row in tqdm(rows, desc="Processing train split"):
            for record in _records_from_video(
                str(row.video_path),
                row.patient,
                row.blue_point,
                row.covid19,
            ):
                yield record

    return generator


def verify_args(raw_data_dir: str, output_dir: str, publish: bool = False) -> None:
    """Verify the command-line arguments before processing begins.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        publish: Whether the dataset will be published to the Hugging Face Hub.
    """
    assert os.path.isdir(raw_data_dir), "raw_data_dir must be an existing directory"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(os.path.join(output_dir, OUTPUT_NAME)), (
        "A version of the COVID-BLUES dataset already exists"
    )

    if publish:
        verify_revision_unpublished(OUTPUT_NAME, DATASET_REVISION)


def process_covid_blues(
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
    """Prepare the COVID-BLUES dataset as a Hugging Face DatasetDict.

    Args:
        raw_data_dir: Path to the raw data directory (the ``covid_blues``
            folder produced by ``ultrabench download covid-blues``).
        output_dir: Path to the directory where the processed dataset
            will be saved.
        size: Height and width in pixels for resized test images.
        publish: Whether to push the dataset to the Hugging Face Hub.
    """
    verify_args(raw_data_dir, output_dir, publish)

    raw_data_dir = os.path.join(raw_data_dir, "covid_blues")
    output_path = Path(output_dir) / OUTPUT_NAME

    # Build per-video metadata DataFrame
    video_paths = list(Path(raw_data_dir).glob("lus_videos/*.mp4"))
    video_rows = [parse_video_filename(p) for p in video_paths]
    video_df = pd.DataFrame(video_rows)
    video_df["video_path"] = [str(p) for p in video_paths]

    # Merge with clinical data to attach the covid19 label
    clinical_df = load_clinical_data(raw_data_dir)
    video_df = video_df.merge(
        clinical_df.rename(columns={"patient_id": "patient", "cov_test": "covid19"}),
        on="patient",
        how="left",
    )

    # Reserve 20 patients for testing
    splitter = GroupShuffleSplit(n_splits=1, test_size=20, random_state=42)
    train_idx, test_idx = next(splitter.split(X=video_df, groups=video_df["patient"]))
    train_df = video_df.iloc[train_idx].reset_index(drop=True)
    test_df = video_df.iloc[test_idx].reset_index(drop=True)

    typer.echo(
        f"Split: {train_df['patient'].nunique()} train patients, "
        f"{test_df['patient'].nunique()} test patients."
    )

    # Process test splits eagerly (20 patients, fits in memory)
    test_records: list[dict] = []
    test_resized_records: list[dict] = []
    for row in tqdm(
        test_df.itertuples(index=False),
        desc="Processing test split",
        total=len(test_df),
    ):
        test_records.extend(
            _records_from_video(
                str(row.video_path), row.patient, row.blue_point, row.covid19
            )
        )
        test_resized_records.extend(
            _records_from_video(
                str(row.video_path),
                row.patient,
                row.blue_point,
                row.covid19,
                size=size,
            )
        )

    typer.echo(
        f"Examples — train: (streaming), "
        f"test: {len(test_records)}, "
        f"test.{size}: {len(test_resized_records)}"
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_generator(
                _make_train_generator(train_df, raw_data_dir),
                features=DATASET_INFO.features,
            ),
            "test": Dataset.from_list(test_records, info=DATASET_INFO),
            f"test.{size}": Dataset.from_list(test_resized_records, info=DATASET_INFO),
        }
    )

    dataset.save_to_disk(str(output_path))

    generate_classification_ground_truth_from_records(
        [
            {
                "filename": r["filename"],
                "label": r["covid19"],
                "patient": r["patient"],
            }
            for r in test_records
        ],
        output_path / f"test.{size}" / "ground_truth_classification.csv",
        sequence_keys=["patient"],
    )

    save_version_info(str(output_path), __version__, revision=DATASET_REVISION)

    typer.echo(f"Dataset saved to {output_path}.")

    if publish:
        dataset.push_to_hub(
            OUTPUT_NAME,
            commit_message=f"Update the dataset to revision {DATASET_REVISION}.",
            private=True,
        )

        api = HfApi()
        operations = [
            CommitOperationAdd(
                path_in_repo=f"test.{size}-ground-truth-classification.csv",
                path_or_fileobj=(
                    output_path / f"test.{size}" / "ground_truth_classification.csv"
                ),
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
