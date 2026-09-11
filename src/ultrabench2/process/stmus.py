"""Prepare the training and test sets for the STMUS dataset.

The dataset is split by patient into training and test data (80:20)
to ensure there is no patient overlap between the splits. The training
split is then capped at a maximum of 1000 examples by moving whole
patients into the test split.

The subject numbers in the filenames are local to each muscle and
echogenicity directory. The same number denotes different people in
different directories, and one person is numbered differently in each
directory they appear in, so the numbers cannot be used to group images
by patient. Patients are instead recovered by joining every image to the
per-muscle ``characteristics.xlsx`` spreadsheets on (muscle,
echogenicity, number) and grouping on the demographics recorded there,
which are identical for a patient across directories. Two patients share
demographics and are treated as one, so that their images cannot be
split across the training and test data. This yields 1223 patients for
the 8169 images.

The resulting dataset has three splits:

    - train: up to 1000 examples, original-size images.
    - test: the remaining patients, original-size images.
    - test.{size}: same examples as test, resized so the longest edge
      measures {size} px and then zero-padded to {size} x {size} px.

Each example has the following features:

    - filename: Original image filename (without file extension) prefixed
      by muscle (bb, gm, or ta) and echogenicity (healthy or pathological).
    - patient: Patient ID string.
    - muscle: Muscle name (biceps brachii, gastrocnemius medialis, or
      tibialis anterior).
    - echogenicity: Echogenicity level (healthy or pathological).
    - age: Patient age in years.
    - height: Patient height in cm.
    - weight: Patient weight in kg.
    - sex: Patient sex (M or F).
    - image: 3-channel ultrasound image.
    - mask: Single-channel binary muscle segmentation mask
      (background=0, muscle=1).
    - scan_mask: Single-channel binary ultrasound scan mask (all ones).

A segmentation ground-truth CSV is also created in each test split
directory for use by the evaluation layer, with the masks encoded at
the resolution of that split:

    - ground_truth_segmentation.csv

Usage:

    ultrabench2 process stmus RAW_DATA_DIR OUTPUT_DIR
"""

import os
from collections.abc import Iterator
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
from huggingface_hub import HfApi
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm
from typing_extensions import Annotated

from .utils import (
    DATASET_REVISION,
    cap_train_split_by_patient,
    generate_segmentation_ground_truth_from_records,
    ground_truth_commit_operations,
    resize_and_pad,
    save_version_info,
    to_3_channel,
    verify_revision_unpublished,
)

__version__ = version("ultrabench2")

OUTPUT_NAME = "UltraBench/STMUS"

CHARACTERISTICS_FILENAME = "characteristics.xlsx"

DEMOGRAPHICS = ["age", "height", "weight", "sex"]

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
            "age": Value("float32"),
            "height": Value("float32"),
            "weight": Value("float32"),
            "sex": Value("string"),
            "image": Image(),
            "mask": Image(),
            "scan_mask": Image(),
        }
    ),
)


def load_characteristics(dataset_dir: str) -> pd.DataFrame:
    """Load the subject characteristics recorded for each muscle.

    Every muscle directory holds a spreadsheet with two side-by-side
    blocks, healthy subjects in the first five columns and pathological
    subjects in columns seven to eleven, each laid out as (Number, Age,
    Length, Weight, Sex) beneath a two-row header. The blocks have
    different lengths, so the shorter one is padded, and both are
    followed by summary rows holding means, standard deviations and
    totals. Rows whose subject number is not numeric are discarded,
    which removes the padding and the summary rows alike.

    Args:
        dataset_dir: Path to the Polito-Radboud-DeepLearningUS
            directory inside the raw download.

    Returns:
        A DataFrame with one row per subject entry containing the
        columns ``muscle_code``, ``echogenicity``, ``number``, ``age``,
        ``height``, ``weight``, and ``sex``.

    Raises:
        FileNotFoundError: If a muscle is missing its spreadsheet.
    """
    blocks = []
    for muscle_code in MUSCLE_NAMES:
        path = Path(dataset_dir) / muscle_code / CHARACTERISTICS_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"Missing subject characteristics: {path}")

        sheet = pd.read_excel(path, header=None)
        for echogenicity, first_column in [("healthy", 0), ("pathological", 6)]:
            block = sheet.iloc[2:, first_column : first_column + 5].copy()
            block.columns = ["number", *DEMOGRAPHICS]
            block = block[pd.to_numeric(block["number"], errors="coerce").notna()]
            blocks.append(
                block.assign(muscle_code=muscle_code, echogenicity=echogenicity)
            )

    characteristics = pd.concat(blocks, ignore_index=True)

    return characteristics.astype(
        {"number": int, "age": float, "height": float, "weight": float, "sex": str}
    )


def assign_patient_ids(characteristics: pd.DataFrame) -> pd.DataFrame:
    """Assign a patient ID to every subject entry.

    The subject numbers are local to each muscle and echogenicity
    directory, so patients are identified by their demographics, which
    are recorded identically wherever a patient appears. Two patients
    share demographics and are therefore treated as a single patient,
    which errs towards keeping images together rather than risking a
    patient being split across the training and test data.

    Args:
        characteristics: The subject entries to label, as returned by
            ``load_characteristics``.

    Returns:
        ``characteristics`` with an added ``patient`` column.

    Raises:
        ValueError: If any entry could not be assigned a patient ID.
    """
    patients = (
        characteristics[DEMOGRAPHICS]
        .drop_duplicates()
        .sort_values(DEMOGRAPHICS)
        .reset_index(drop=True)
    )
    patients["patient"] = [f"p{i:04d}" for i in range(1, len(patients) + 1)]

    labelled = characteristics.merge(patients, on=DEMOGRAPHICS, how="left")
    if labelled["patient"].isna().any():
        raise ValueError("Failed to assign a patient ID to every subject entry")

    return labelled


def collect_examples(dataset_dir: str) -> pd.DataFrame:
    """Collect metadata for all images in the STMUS dataset.

    Scans the raw dataset directory for image files, parses their paths
    to extract per-example metadata, and joins the subject
    characteristics to recover the patient each image belongs to.

    Args:
        dataset_dir: Path to the Polito-Radboud-DeepLearningUS
            directory inside the raw download.

    Returns:
        A DataFrame with one row per image containing the columns
        ``filename``, ``patient``, ``muscle``, ``echogenicity``,
        ``age``, ``height``, ``weight``, ``sex``, ``image_path``, and
        ``mask_path``.

    Raises:
        ValueError: If an image has no matching subject entry.
    """
    image_paths = sorted(Path(dataset_dir).glob("**/Images/*.png"))

    rows = []
    for path in image_paths:
        parts = path.parts
        muscle_code = parts[-4]
        echogenicity = parts[-3].lower()
        stem = path.stem

        rows.append(
            {
                "filename": f"{muscle_code.lower()}_{echogenicity}_{stem}",
                "muscle_code": muscle_code,
                "echogenicity": echogenicity,
                "number": int(stem.split("_")[1]),
                "image_path": str(path),
                "mask_path": str(path.parent.parent / "Masks" / path.name),
            }
        )

    df = pd.DataFrame(rows)

    patients = assign_patient_ids(load_characteristics(dataset_dir))
    examples = df.merge(
        patients[["muscle_code", "echogenicity", "number", "patient", *DEMOGRAPHICS]],
        on=["muscle_code", "echogenicity", "number"],
        how="left",
        validate="many_to_one",
    )

    unmatched = examples[examples["patient"].isna()]
    if not unmatched.empty:
        missing = unmatched[["muscle_code", "echogenicity", "number"]]
        raise ValueError(
            "No subject characteristics for: "
            f"{missing.drop_duplicates().to_dict('records')}"
        )

    examples["muscle"] = examples["muscle_code"].map(MUSCLE_NAMES)

    return examples[
        [
            "filename",
            "patient",
            "muscle",
            "echogenicity",
            *DEMOGRAPHICS,
            "image_path",
            "mask_path",
        ]
    ]


def _process_row(row: pd.Series, size: int | None = None) -> dict:
    """Process a single example into a record dict.

    Args:
        row: A row from the DataFrame returned by
            ``collect_examples``.
        size: If provided, resize and pad the image and mask to
            ``size x size`` pixels.

    Returns:
        A dict with keys ``filename``, ``patient``, ``muscle``,
        ``echogenicity``, ``age``, ``height``, ``weight``, ``sex``,
        ``image`` (numpy array, uint8, HxWx3), ``mask`` (numpy array,
        uint8, HxW, values in {0, 1}), and ``scan_mask``.
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
        "age": float(row["age"]),
        "height": float(row["height"]),
        "weight": float(row["weight"]),
        "sex": str(row["sex"]),
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


def _mask_records(
    df: pd.DataFrame, size: int | None
) -> Iterator[dict[str, str | np.ndarray]]:
    """Generate the records the ground-truth CSV of one split is built from.

    The splits are streamed rather than held in memory, so the masks are
    read again here. They go through the same resize as ``_process_row``
    does, which keeps them identical to the masks of the split they are
    published with.

    Args:
        df: Metadata DataFrame for the split.
        size: The height and width to resize and pad the masks to, or
            None to keep their original size.

    Yields:
        A dict with the ``filename`` and ``mask`` of one example.
    """
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating ground truth"):
        mask = (io.imread(row["mask_path"]) >= 128).astype(np.uint8)
        if size is not None:
            mask = resize_and_pad(mask, size, order=0)

        yield {"filename": row["filename"], "mask": mask}


def verify_args(raw_data_dir: str, output_dir: str, publish: bool = False) -> None:
    """Verify command-line arguments before processing begins.

    Args:
        raw_data_dir: Path to the raw STMUS download directory.
        output_dir: Path to the output directory.
        publish: Whether the dataset will be published to the Hugging
            Face Hub.

    Raises:
        AssertionError: If any argument is invalid or the output
            dataset already exists.
    """
    assert os.path.isdir(raw_data_dir), "raw_data_dir must be an existing directory"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(os.path.join(output_dir, OUTPUT_NAME)), (
        "A version of the STMUS dataset already exists in output_dir"
    )

    if publish:
        verify_revision_unpublished(OUTPUT_NAME)


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
    verify_args(raw_data_dir, output_dir, publish)

    dataset_dir = os.path.join(raw_data_dir, "Polito-Radboud-DeepLearningUS")
    df = collect_examples(dataset_dir)

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

    for split, image_size in (("test", None), (f"test.{size}", size)):
        generate_segmentation_ground_truth_from_records(
            _mask_records(test_df, image_size),
            mask_key="mask",
            classes=[1],
            output_path=output_path / split / "ground_truth_segmentation.csv",
            image_size=image_size,
        )

    save_version_info(str(output_path), __version__)

    if publish:
        dataset.push_to_hub(
            OUTPUT_NAME,
            commit_message=f"Update the dataset to revision {DATASET_REVISION}.",
            private=True,
        )

        api = HfApi()
        operations = ground_truth_commit_operations(
            output_path, {"segmentation": "ground_truth_segmentation.csv"}, size
        )
        api.create_commit(
            repo_id=OUTPUT_NAME,
            repo_type="dataset",
            operations=operations,
            commit_message=(
                f"Update ground truth CSV files for revision {DATASET_REVISION}."
            ),
        )
