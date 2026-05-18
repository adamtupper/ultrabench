"""Prepare the Stanford Thyroid Ultrasound Cine-clip dataset as a
Hugging Face DatasetDict. Images are split into training and test sets
using an 8:2 patient-level split.

The images and masks are extracted from the `dataset.hdf5` file. Mask
values are mapped from {0, 255} to {0, 1}.

The resulting dataset has three splits:

    - train: original-size images.
    - test: original-size images.
    - test.{size}: same examples as test, resized so the longest edge
      measures {size} px and then zero-padded to {size} x {size} px.

Each example has the following features:

    - filename: Image filename stem (patient ID + HDF5 index).
    - patient: The patient ID.
    - image: 3-channel ultrasound image.
    - scan_mask: Single-channel binary scan mask.
    - tumor_mask: Single-channel binary nodule/tumour mask.
    - pathology: Lesion pathology string (benign or malignant).
    - label: ClassLabel encoding the lesion pathology.
    - age: Patient age.
    - sex: Patient sex.
    - location: Lesion location.
    - lesion_size_x/y/z: Lesion dimensions.
    - tirads_level: TIRADS level.
    - tirads_composition/echogenicity/shape/margin/echogenicfoci:
      TIRADS descriptors.
    - histopath_diagnosis: Histopathological diagnosis.

Two ground-truth CSV files are also created for the test split:

    - ground_truth_classification.csv
    - ground_truth_segmentation.csv

Usage:
    ultrabench stanford_thyroid RAW_DATA_DIR OUTPUT_DIR
"""

import os
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pandas as pd
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

from .utils import (
    generate_classification_ground_truth_from_records,
    generate_segmentation_ground_truth_from_records,
    resize_and_pad,
    save_version_info,
    to_3_channel,
)

__version__ = version("ultrabench")
DATASET_FILE = "dataset.hdf5"
METADATA_FILE = "metadata.csv"
OUTPUT_NAME = "UltraBench/StanfordThyroid"
DATASET_INFO = DatasetInfo(
    description=(
        "Thyroid nodule ultrasound cine-clip images for nodule "
        "segmentation and malignancy classification."
    ),
    citation="""
    @misc{stanford2021thyroid,
        author = {{Stanford AIMI Center}},
        title  = {Thyroid Ultrasound Cine-clip Dataset},
        month  = oct,
        year   = {2021},
        url    = {https://stanfordaimi.azurewebsites.net/datasets/
                  a72f2b02-7b53-4c5d-963c-d7253220bfd5},
    }
    """,
    homepage=(
        "https://stanfordaimi.azurewebsites.net/datasets/"
        "a72f2b02-7b53-4c5d-963c-d7253220bfd5"
    ),
    features=Features(
        {
            "filename": Value("string"),
            "patient": Value("string"),
            "image": Image(),
            "scan_mask": Image(),
            "tumor_mask": Image(),
            "pathology": Value("string"),
            "label": ClassLabel(names=["benign", "malignant"]),
            "age": Value("int32"),
            "sex": Value("string"),
            "location": Value("string"),
            "lesion_size_x": Value("float32"),
            "lesion_size_y": Value("float32"),
            "lesion_size_z": Value("float32"),
            "tirads_level": Value("int32"),
            "tirads_composition": Value("string"),
            "tirads_echogenicity": Value("string"),
            "tirads_shape": Value("string"),
            "tirads_margin": Value("string"),
            "tirads_echogenicfoci": Value("string"),
            "histopath_diagnosis": Value("string"),
        }
    ),
    license="other",
)

PATIENTS = [
    103,
    104,
    106,
    108,
    109,
    10,
    121,
    117,
    114,
    111,
    110,
    122,
    123,
    124,
    125,
    128,
    129,
    130,
    140,
    138,
    137,
    134,
    143,
    145,
    146,
    149,
    14,
    162,
    161,
    160,
    159,
    157,
    155,
    154,
    152,
    169,
    170,
    172,
    181,
    180,
    178,
    177,
    176,
    175,
    174,
    182,
    183,
    184,
    185,
    186,
    189,
    18,
    191,
    199,
    198,
    197,
    196,
    195,
    194,
    192,
    202,
    206,
    20,
    210,
    27,
    24,
    23,
    211,
    28,
    34,
    35,
    36,
    37,
    49,
    45,
    42,
    41,
    40,
    3,
    4,
    53,
    55,
    57,
    60,
    6,
    68,
    67,
    65,
    63,
    62,
    61,
    72,
    74,
    75,
    76,
    78,
    79,
    7,
    87,
    83,
    82,
    81,
    80,
    8,
    90,
]


def generate_scan_mask(image: np.ndarray) -> np.ndarray:
    """Generate a scan mask using morphological operations.

    Args:
        image: The input 2D image array.

    Returns:
        A binary mask of the scan region.
    """
    mask = image > 1
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=64)
    for _ in range(30):
        mask = skimage.morphology.binary_erosion(mask)
    for _ in range(30):
        mask = skimage.morphology.binary_dilation(mask)

    # Logical OR of each half reflected onto the other to mitigate shadows
    _, width = mask.shape
    left_half = mask[:, : width // 2]
    right_half = mask[:, width // 2 :]
    left_half = np.logical_or(left_half, np.fliplr(right_half))
    right_half = np.logical_or(right_half, np.fliplr(left_half))
    mask = np.hstack((left_half, right_half))

    mask = skimage.morphology.convex_hull_image(mask)
    return mask.astype(np.uint8)


def _nan_to_none(value):
    """Convert a pandas NA/NaN value to None.

    Args:
        value: A scalar value, possibly NaN or pd.NA.

    Returns:
        None if the value is NA/NaN, otherwise the original value.
    """
    if pd.isna(value):
        return None
    return value


def _process_stanford_thyroid_example(
    i: int,
    patient_id: str,
    image: np.ndarray,
    mask: np.ndarray,
    patient_metadata: pd.DataFrame,
    size: int | None = None,
) -> dict:
    """Process a single HDF5 entry into a record dict.

    Args:
        i: The HDF5 index, used to build the filename stem.
        patient_id: The raw patient ID string from the HDF5 file.
        image: The raw grayscale image array from HDF5.
        mask: The raw mask array from HDF5 (values 0 or 255).
        patient_metadata: Single-row DataFrame slice for this patient.
        size: If provided, resize and pad arrays to size x size pixels.

    Returns:
        A dict matching the DATASET_INFO features schema. Image values
        are numpy arrays.
    """
    tumor_mask = (mask == 255).astype("uint8")
    scan_mask = generate_scan_mask(image)
    image_3ch = to_3_channel(image)

    if size is not None:
        image_3ch = resize_and_pad(image_3ch, size, order=1)
        tumor_mask = resize_and_pad(tumor_mask, size, order=0)
        scan_mask = resize_and_pad(scan_mask, size, order=0)

    label = int((patient_metadata["ti-rads_level"] > 3).astype(int).item())
    stem = f"{patient_id}{i}"

    return {
        "filename": stem,
        "patient": patient_id.removesuffix("_"),
        "image": image_3ch,
        "scan_mask": scan_mask,
        "tumor_mask": tumor_mask,
        "pathology": "benign" if label == 0 else "malignant",
        "label": label,
        "age": int(patient_metadata["age"].item()),
        "sex": _nan_to_none(patient_metadata["sex"].item()),
        "location": _nan_to_none(patient_metadata["location"].item()),
        "lesion_size_x": float(patient_metadata["size_x"].item()),
        "lesion_size_y": float(patient_metadata["size_y"].item()),
        "lesion_size_z": float(patient_metadata["size_z"].item()),
        "tirads_level": int(patient_metadata["ti-rads_level"].item()),
        "tirads_composition": _nan_to_none(
            patient_metadata["ti-rads_composition"].item()
        ),
        "tirads_echogenicity": _nan_to_none(
            patient_metadata["ti-rads_echogenicity"].item()
        ),
        "tirads_shape": _nan_to_none(patient_metadata["ti-rads_shape"].item()),
        "tirads_margin": _nan_to_none(patient_metadata["ti-rads_margin"].item()),
        "tirads_echogenicfoci": _nan_to_none(
            patient_metadata["ti-rads_echogenicfoci"].item()
        ),
        "histopath_diagnosis": _nan_to_none(
            patient_metadata["histopath_diagnosis"].item()
        ),
    }


def _make_train_generator(
    train_index_df: pd.DataFrame,
    annot_ids: np.ndarray,
    metadata: pd.DataFrame,
    raw_data_dir: str,
):
    """Return a generator function that yields train records one by one.

    Args:
        train_index_df: DataFrame of train-split rows with an
            hdf5_index column.
        annot_ids: Array of annotation IDs from the HDF5 file.
        metadata: Patient metadata DataFrame.
        raw_data_dir: Path to the raw dataset directory.

    Returns:
        A zero-argument generator function.
    """

    def generator():
        with h5py.File(os.path.join(raw_data_dir, DATASET_FILE), "r") as ds:
            for row in tqdm(
                train_index_df.itertuples(index=False),
                total=len(train_index_df),
                desc="Processing train split",
            ):
                i = row.hdf5_index
                patient_id = annot_ids[i]
                image = ds["image"][i]
                mask = ds["mask"][i]
                patient_metadata = metadata[metadata["annot_id"] == patient_id]
                yield _process_stanford_thyroid_example(
                    i, patient_id, image, mask, patient_metadata
                )

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
        "A matching version of the Stanford Thyroid dataset already exists"
    )


def process_stanford_thyroid(
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
    """Preprocess the Stanford Thyroid Ultrasound Cine-clip dataset.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        size: The height and width of the test images in pixels.
        publish: Whether to publish the dataset to the HF Hub.
    """
    verify_args(raw_data_dir, output_dir)

    output_path = Path(output_dir) / OUTPUT_NAME

    metadata = pd.read_csv(os.path.join(raw_data_dir, METADATA_FILE))
    included_patients = set(PATIENTS)

    # First pass: collect patient IDs for all included examples
    with h5py.File(os.path.join(raw_data_dir, DATASET_FILE), "r") as dataset:
        annot_ids = dataset["annot_id"].asstr()[:]

    index_records = []
    for i, patient_id in enumerate(annot_ids):
        if int(patient_id.removesuffix("_")) not in included_patients:
            continue
        index_records.append({"hdf5_index": i, "patient": patient_id.removesuffix("_")})

    index_df = pd.DataFrame(index_records)

    # Determine the train/test split from patient groups
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_indices, test_indices = next(
        splitter.split(X=index_df, groups=index_df["patient"])
    )
    train_index_df = index_df.iloc[train_val_indices].reset_index(drop=True)
    test_index_df = index_df.iloc[test_indices].reset_index(drop=True)

    actual = len(index_df["patient"].unique())
    expected = len(included_patients)
    assert actual == expected, (
        f"The actual ({actual}) and expected ({expected}) number of "
        f"patients does not match!"
    )

    test_records: list[dict] = []
    test_resized_records: list[dict] = []

    with h5py.File(os.path.join(raw_data_dir, DATASET_FILE), "r") as ds:
        for row in tqdm(
            test_index_df.itertuples(index=False),
            total=len(test_index_df),
            desc="Processing test split",
        ):
            i = row.hdf5_index
            patient_id = annot_ids[i]
            image = ds["image"][i]
            mask = ds["mask"][i]
            patient_metadata = metadata[metadata["annot_id"] == patient_id]
            test_records.append(
                _process_stanford_thyroid_example(
                    i, patient_id, image, mask, patient_metadata
                )
            )
            test_resized_records.append(
                _process_stanford_thyroid_example(
                    i, patient_id, image, mask, patient_metadata, size=size
                )
            )

    dataset = DatasetDict(
        {
            "train": Dataset.from_generator(
                _make_train_generator(
                    train_index_df, annot_ids, metadata, raw_data_dir
                ),
                features=DATASET_INFO.features,
            ),
            "test": Dataset.from_list(test_records, info=DATASET_INFO),
            f"test.{size}": Dataset.from_list(test_resized_records, info=DATASET_INFO),
        }
    )

    dataset.save_to_disk(str(output_path))

    test_size_dir = output_path / f"test.{size}"
    generate_classification_ground_truth_from_records(
        test_records,
        test_size_dir / "ground_truth_classification.csv",
    )
    generate_segmentation_ground_truth_from_records(
        test_records,
        mask_key="tumor_mask",
        classes=[1],
        output_path=test_size_dir / "ground_truth_segmentation.csv",
    )

    print(f"Training examples: {len(train_index_df)}")
    print(f"Test examples: {len(test_records)}")

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
                path_or_fileobj=test_size_dir / "ground_truth_classification.csv",
            ),
            CommitOperationAdd(
                path_in_repo=f"test.{size}-ground-truth-segmentation.csv",
                path_or_fileobj=test_size_dir / "ground_truth_segmentation.csv",
            ),
        ]
        api.create_commit(
            repo_id=OUTPUT_NAME,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Update ground truth CSV files for Version {__version__}.",
        )
