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

A ground-truth CSV file is also created in each test split directory:

    - ground_truth_classification.csv

The dataset is built by `ultrabench.load_dataset` on first use.
"""

import os
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import skimage.morphology
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    DatasetInfo,
    Features,
    Image,
    Value,
)
from tqdm.auto import tqdm

from ultrabench.datasets.utils import partition_by_manifest

from .utils import (
    IMAGE_SIZE,
    BuildResult,
    generate_classification_ground_truth_from_records,
    resize_and_pad,
)

OUTPUT_NAME = "UltraBench/COVID-BLUES"
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


def _make_train_generator(rows: list):
    """Return a generator that yields train records one frame at a time.

    Processes one video at a time to keep peak memory usage low.

    Args:
        rows: The video rows of the train split, each carrying
            ``patient``, ``blue_point``, ``covid19`` and
            ``video_path``.

    Returns:
        A zero-argument generator function.
    """

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


def write_ground_truth(
    records: list[dict], gt_dir: Path, image_size: int | None
) -> None:
    """Write the ground-truth CSV of one test split.

    The label applies to a whole video rather than a single frame, so
    the CSV carries a sequence ID that lets predictions be pooled per
    patient.

    Args:
        records: The records of the split.
        gt_dir: The split directory to write the CSV into.
        image_size: Unused. Classification labels do not depend on the
            image resolution, but every dataset's writer shares one
            signature.
    """
    generate_classification_ground_truth_from_records(
        [
            {
                "filename": r["filename"],
                "label": r["covid19"],
                "patient": r["patient"],
            }
            for r in records
        ],
        gt_dir / "ground_truth_classification.csv",
        sequence_keys=["patient"],
    )


def build_covid_blues_dataset(
    raw_data_dir: str | Path, size: int = IMAGE_SIZE
) -> BuildResult:
    """Build the COVID-BLUES splits from the raw data.

    The splits come from the tracked manifests rather than from a
    seeded `GroupShuffleSplit`, so that processing the raw data
    reproduces the published splits exactly. The manifests list videos
    rather than frames, so a video belonging to neither split is never
    decoded.

    Args:
        raw_data_dir: The directory holding the `covid_blues` folder
            that `scripts/download_dataset.py covid-blues` produces.
        size: The height and width of the resized test images.

    Returns:
        The processed splits and the test records the ground truth is
        built from.

    Raises:
        ValueError: If the raw data is missing a video the manifests
            list.
    """
    raw_data_dir = os.path.join(str(raw_data_dir), "covid_blues")

    video_paths = list(Path(raw_data_dir).glob("lus_videos/*.mp4"))
    video_rows = [parse_video_filename(p) for p in video_paths]
    video_df = pd.DataFrame(video_rows)
    video_df["video_path"] = [str(p) for p in video_paths]

    clinical_df = load_clinical_data(raw_data_dir)
    video_df = video_df.merge(
        clinical_df.rename(columns={"patient_id": "patient", "cov_test": "covid19"}),
        on="patient",
        how="left",
    )

    rows = list(video_df.itertuples(index=False))
    train_rows, test_rows = partition_by_manifest(
        rows, "COVID-BLUES", key=lambda row: Path(row.video_path).stem
    )

    print(f"Split: {len(train_rows)} train videos, {len(test_rows)} test videos.")

    # The test splits are held in memory: 20 patients fit comfortably,
    # and the ground-truth CSVs are built from their records.
    test_records: list[dict] = []
    test_resized_records: list[dict] = []
    for row in tqdm(test_rows, desc="Processing test split"):
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

    print(
        f"Examples -- train: (streaming), "
        f"test: {len(test_records)}, "
        f"test.{size}: {len(test_resized_records)}"
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_generator(
                _make_train_generator(train_rows),
                features=DATASET_INFO.features,
            ),
            "test": Dataset.from_list(test_records, info=DATASET_INFO),
            f"test.{size}": Dataset.from_list(test_resized_records, info=DATASET_INFO),
        }
    )

    return BuildResult(dataset, test_records, test_resized_records)
