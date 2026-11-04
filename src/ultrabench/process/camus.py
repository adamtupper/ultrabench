"""Split the CAMUS dataset using the original train, validation, and
test splits provided with the dataset.

The images and masks are extracted from the NIfTI files. The
corresponding metadata for each sequence is in the ".cfg" files.

The dataset is organized into two subsets:

    - classification: all frames. Image quality is a sequence-level
      label and the training split holds fewer than 1000 sequences, so
      the frames are not subsampled.
    - segmentation: the training split is subsampled to a maximum of
      1000 frames, drawn uniformly at random from within each (patient,
      view) sequence. Every training patient and both views are
      retained, and the discarded frames are not moved to the test
      split.

Both subsets share the same test splits. Each subset has three splits:

    - train: original-size images (original train + validation patients).
    - test: original-size images (original test patients).
    - test.{size}: same examples as test, resized so the longest edge
      measures {size} px and then zero-padded to {size} x {size} px.

Each example has the following features:

    - filename: Unique image identifier.
    - patient: Patient ID.
    - view: View of the sequence (2CH or 4CH).
    - frame: Frame number of the image in the sequence.
    - sex: Sex of the patient (M or F).
    - age: Age of the patient.
    - image_quality: ClassLabel encoding the image quality
      (poor=0, medium=1, good=2).
    - ejection_fraction: Ejection fraction.
    - end_systole: End-Systole flag.
    - end_diastole: End-Diastole flag.
    - frame_rate: Frame rate of the sequence.
    - image: 3-channel ultrasound image.
    - mask: Single-channel multi-class cardiac structure mask.
    - scan_mask: Single-channel binary scan mask.

A ground-truth CSV file is also created for each test split of each
subset for use by the evaluation layer, with the segmentation masks
encoded at the resolution of that split:

    - classification/test/ground_truth_classification.csv
    - classification/test.{size}/ground_truth_classification.csv
    - segmentation/test/ground_truth_segmentation.csv
    - segmentation/test.{size}/ground_truth_segmentation.csv

The classification CSVs carry a "sequence_id" column, since image
quality is graded per (patient, view) acquisition rather than per
frame.

Usage:
    ultrabench process camus RAW_DATA_DIR OUTPUT_DIR
"""

import glob
import os
from importlib.metadata import version
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import skimage
import typer
import yaml
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    DatasetInfo,
    Features,
    Image,
    Value,
)
from huggingface_hub import CommitOperationDelete, HfApi
from tqdm.auto import tqdm
from typing_extensions import Annotated

from .utils import (
    DATASET_REVISION,
    generate_classification_ground_truth_from_records,
    generate_segmentation_ground_truth_from_records,
    ground_truth_commit_operations,
    ground_truth_path_in_repo,
    resize_and_pad,
    save_version_info,
    subsample_by_group,
    to_3_channel,
    verify_revision_unpublished,
)

__version__ = version("ultrabench")

OUTPUT_NAME = "UltraBench/CAMUS"
CLASS_TO_LABEL = {
    "Poor": 0,
    "Medium": 1,
    "Good": 2,
}
DATASET_INFO = DatasetInfo(
    description=(
        "A dataset of apical four-chamber and two-chamber cardiac "
        "ultrasound images for cardiac structure segmentation and "
        "image quality assessment."
    ),
    citation="""
    @article{leclerc2019_camus,
        author={Leclerc, Sarah and
                Smistad, Erik and
                Pedrosa, João and
                Østvik, Andreas and
                Cervenansky, Frederic and
                Espinosa, Florian and
                Espeland, Torvald and
                Berg, Erik Andreas Rye and
                Jodoin, Pierre-Marc and
                Grenier, Thomas and
                Lartizien, Carole and
                D’hooge, Jan and
                Lovstakken, Lasse and
                Bernard, Olivier},
        journal={IEEE Transactions on Medical Imaging},
        title={Deep Learning for Segmentation Using an Open Large-Scale
               Dataset in 2D Echocardiography},
        year={2019},
        volume={38},
        number={9},
        pages={2198-2210},
        doi={10.1109/TMI.2019.2900516}}
    }
    """,
    homepage="https://www.creatis.insa-lyon.fr/Challenge/camus/",
    license="cc-by-nc-sa-4.0",
    features=Features(
        {
            "filename": Value("string"),
            "patient": Value("string"),
            "view": Value("string"),
            "frame": Value("int32"),
            "sex": Value("string"),
            "age": Value("int32"),
            "image_quality": ClassLabel(names=["poor", "medium", "good"]),
            "ejection_fraction": Value("float32"),
            "end_systole": Value("bool"),
            "end_diastole": Value("bool"),
            "frame_rate": Value("float32"),
            "image": Image(),
            "mask": Image(),
            "scan_mask": Image(),
        }
    ),
)


def segment_fan(image: np.ndarray) -> np.ndarray:
    """Generate a fan-shaped scan mask using morphological operations.

    Args:
        image: The input 2D image array.

    Returns:
        A binary mask of the fan-shaped region.
    """
    mask = image > 0
    mask = skimage.morphology.convex_hull_image(mask)
    return mask.astype(np.uint8)


def extract_metadata(dataset_dir: str) -> pd.DataFrame:
    """Extract the metadata from the ".cfg" files.

    Args:
        dataset_dir: The path to the dataset directory.

    Returns:
        A DataFrame containing the metadata for each example.
    """
    metadata = []
    for file_path in glob.glob(
        os.path.join(dataset_dir, "database_nifti", "**", "*.cfg")
    ):
        patient_dir, filename = file_path.split("/")[-2:]
        patient_id = int(patient_dir.removeprefix("patient"))
        view = filename.removesuffix(".cfg").split("_")[-1]

        with open(file_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        for i in range(config["NbFrame"]):
            frame = i + 1
            row = {
                "patient": patient_id,
                "view": view,
                "frame": frame,
                "sex": config["Sex"],
                "age": config["Age"],
                "image_quality": config["ImageQuality"],
                "ejection_fraction": config["EF"],
                "frame_rate": config["FrameRate"],
                "end_diastole": config["ED"] == frame,
                "end_systole": config["ES"] == frame,
            }
            metadata.append(row)

    return pd.DataFrame(metadata)


def load_split_patients(file_path: str) -> set[int]:
    """Load patient IDs from a dataset split text file.

    Each line of the file contains a patient string of the form
    "patient{id:04d}".

    Args:
        file_path: Path to the split text file.

    Returns:
        A set of integer patient IDs.
    """
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
    return {int(line.removeprefix("patient")) for line in lines if line}


def _records_from_nifti(
    patient_id: int,
    view: str,
    rows: pd.DataFrame,
    raw_data_dir: str,
    size: int | None = None,
) -> list[dict]:
    """Process all frames from one (patient, view) NIfTI file.

    Loads the NIfTI image and ground-truth volumes once, then extracts
    each requested frame as a record dict.

    Args:
        patient_id: Integer patient ID.
        view: View string (e.g. "2CH" or "4CH").
        rows: Subset of the metadata DataFrame for this (patient, view).
        raw_data_dir: The path to the raw dataset directory.
        size: If provided, resize and pad arrays to size x size pixels.

    Returns:
        A list of record dicts, one per frame.
    """
    patient_str = f"patient{patient_id:04d}"
    nifti_dir = os.path.join(raw_data_dir, "database_nifti", patient_str)
    image_path = os.path.join(nifti_dir, f"{patient_str}_{view}_half_sequence.nii.gz")
    mask_path = os.path.join(nifti_dir, f"{patient_str}_{view}_half_sequence_gt.nii.gz")

    image_vol = nib.load(image_path).get_fdata().astype(np.uint8)  # type: ignore
    mask_vol = nib.load(mask_path).get_fdata().astype(np.uint8)  # type: ignore

    records = []
    for _, row in rows.iterrows():
        frame_idx = row["frame"] - 1
        image = np.rot90(image_vol[:, :, frame_idx], axes=(1, 0)).astype(np.uint8)
        mask = np.rot90(mask_vol[:, :, frame_idx], axes=(1, 0)).astype(np.uint8)
        scan_mask = segment_fan(image)
        image_3ch = to_3_channel(image)

        if size is not None:
            image_3ch = resize_and_pad(image_3ch, size, order=1)
            mask = resize_and_pad(mask, size, order=0)
            scan_mask = resize_and_pad(scan_mask, size, order=0)

        records.append(
            {
                "filename": f"{patient_str}_{view}_half_sequence_{row['frame']}",
                "patient": row["patient"],
                "view": row["view"],
                "frame": int(row["frame"]),
                "sex": row["sex"],
                "age": int(row["age"]),
                "image_quality": CLASS_TO_LABEL[row["image_quality"]],
                "ejection_fraction": float(row["ejection_fraction"]),
                "end_systole": bool(row["end_systole"]),
                "end_diastole": bool(row["end_diastole"]),
                "frame_rate": float(row["frame_rate"]),
                "image": image_3ch,
                "mask": mask,
                "scan_mask": scan_mask,
            }
        )

    return records


def _make_train_generator(train_df: pd.DataFrame, raw_data_dir: str):
    """Return a generator function that yields train records one by one.

    Groups the metadata by (patient, view) so each NIfTI file is
    loaded only once.

    Args:
        train_df: Training split metadata DataFrame.
        raw_data_dir: The path to the raw dataset directory.

    Returns:
        A zero-argument generator function.
    """
    groups = list(train_df.groupby(["patient", "view"]))

    def generator():
        for (patient_id, view), group in tqdm(
            groups,
            desc="Processing train split",
        ):
            for record in _records_from_nifti(patient_id, view, group, raw_data_dir):
                yield record

    return generator


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
        "A version of the CAMUS dataset already exists"
    )

    if publish:
        verify_revision_unpublished(OUTPUT_NAME)


def _superseded_ground_truth_deletions(
    api: HfApi, size: int
) -> list[CommitOperationDelete]:
    """Build the operations that remove ground-truth CSVs of earlier revisions.

    Both CAMUS subsets prefix their ground-truth CSVs with the subset
    name, so the un-prefixed CSVs published before the dataset was split
    into subsets are never overwritten. The classification one predates
    the "sequence_id" column and scores image quality per frame instead
    of per acquisition, so it is removed rather than left to be picked
    up by mistake. Deleting a file on the main branch does not alter the
    revisions that are already tagged.

    Args:
        api: The Hugging Face Hub API client.
        size: The height and width of the resized test images.

    Returns:
        One delete operation per superseded CSV that is still published.
    """
    published = set(api.list_repo_files(OUTPUT_NAME, repo_type="dataset"))
    superseded = [
        ground_truth_path_in_repo(task, size=split_size)
        for task in ("classification", "segmentation")
        for split_size in (None, size)
    ]

    return [
        CommitOperationDelete(path_in_repo=name)
        for name in superseded
        if name in published
    ]


def process_camus(
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
    """Prepare the CAMUS dataset as a Hugging Face DatasetDict.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        size: The height and width of the resized test images in pixels.
        publish: Whether to publish the dataset to the Hugging Face Hub.
    """
    verify_args(raw_data_dir, output_dir, publish)
    raw_data_dir = os.path.join(raw_data_dir, "CAMUS_public")

    metadata = extract_metadata(raw_data_dir)

    split_dir = os.path.join(raw_data_dir, "database_split")
    train_patients = load_split_patients(
        os.path.join(split_dir, "subgroup_training.txt")
    ) | load_split_patients(os.path.join(split_dir, "subgroup_validation.txt"))
    test_patients = load_split_patients(os.path.join(split_dir, "subgroup_testing.txt"))
    train_df = metadata[metadata["patient"].isin(train_patients)]
    test_df = metadata[metadata["patient"].isin(test_patients)]

    # Image quality is a sequence-level label and there are fewer than 1000
    # training sequences, so only the segmentation subset is subsampled.
    segmentation_train_df = subsample_by_group(train_df, group_cols=["patient", "view"])

    print(
        f"classification train: {len(train_df)} frames from "
        f"{train_df['patient'].nunique()} patients"
    )
    print(
        f"segmentation train: {len(segmentation_train_df)} frames from "
        f"{segmentation_train_df['patient'].nunique()} patients"
    )
    print(f"test: {len(test_df)} frames from {test_df['patient'].nunique()} patients")

    # Process test splits into lists (≈20% of data, fits in memory)
    test_records = []
    test_resized_records = []
    test_groups = list(test_df.groupby(["patient", "view"]))
    for (patient_id, view), group in tqdm(test_groups, desc="Processing test split"):
        test_records.extend(_records_from_nifti(patient_id, view, group, raw_data_dir))
        test_resized_records.extend(
            _records_from_nifti(patient_id, view, group, raw_data_dir, size=size)
        )

    # The two subsets share the same test splits.
    test_dataset = Dataset.from_list(test_records, info=DATASET_INFO)
    test_resized_dataset = Dataset.from_list(test_resized_records, info=DATASET_INFO)

    classification_dataset = DatasetDict(
        {
            "train": Dataset.from_generator(
                _make_train_generator(train_df, raw_data_dir),
                features=DATASET_INFO.features,
            ),
            "test": test_dataset,
            f"test.{size}": test_resized_dataset,
        }
    )
    segmentation_dataset = DatasetDict(
        {
            "train": Dataset.from_generator(
                _make_train_generator(segmentation_train_df, raw_data_dir),
                features=DATASET_INFO.features,
            ),
            "test": test_dataset,
            f"test.{size}": test_resized_dataset,
        }
    )

    output_path = Path(output_dir) / OUTPUT_NAME
    output_path.mkdir(parents=True, exist_ok=True)
    classification_dataset.save_to_disk(str(output_path / "classification"))
    segmentation_dataset.save_to_disk(str(output_path / "segmentation"))

    for records, split, image_size in (
        (test_records, "test", None),
        (test_resized_records, f"test.{size}", size),
    ):
        generate_classification_ground_truth_from_records(
            [
                {
                    "filename": r["filename"],
                    "label": r["image_quality"],
                    "patient": r["patient"],
                    "view": r["view"],
                }
                for r in records
            ],
            output_path / "classification" / split / "ground_truth_classification.csv",
            sequence_keys=["patient", "view"],
        )
        generate_segmentation_ground_truth_from_records(
            records,
            mask_key="mask",
            classes=[1, 2, 3],
            output_path=(
                output_path / "segmentation" / split / "ground_truth_segmentation.csv"
            ),
            image_size=image_size,
        )

    save_version_info(str(output_path), __version__)

    if publish:
        classification_dataset.push_to_hub(
            OUTPUT_NAME,
            config_name="classification",
            commit_message=(
                f"Update the classification subset to revision {DATASET_REVISION}."
            ),
            private=True,
        )
        segmentation_dataset.push_to_hub(
            OUTPUT_NAME,
            config_name="segmentation",
            commit_message=(
                f"Update the segmentation subset to revision {DATASET_REVISION}."
            ),
            private=True,
        )

        api = HfApi()
        operations = [
            *ground_truth_commit_operations(
                output_path / "classification",
                {"classification": "ground_truth_classification.csv"},
                size,
                prefix="classification",
            ),
            *ground_truth_commit_operations(
                output_path / "segmentation",
                {"segmentation": "ground_truth_segmentation.csv"},
                size,
                prefix="segmentation",
            ),
            *_superseded_ground_truth_deletions(api, size),
        ]
        api.create_commit(
            repo_id=OUTPUT_NAME,
            repo_type="dataset",
            operations=operations,
            commit_message=(
                f"Update ground truth CSV files for revision {DATASET_REVISION}."
            ),
        )
