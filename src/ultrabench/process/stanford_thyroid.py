"""Split the Stanford Thyroid Ultrasound Cine-clip dataset into training and
test data using an 8:2 split, ensuring that there is no patient overlap between
the splits.

The images and masks are extracted from the `dataset.hdf5` file. The images and
masks are saved as single-channel uint8 PNG files. Mask values are mapped from
the values {0, 255} to {0, 1}.

Each example (image) is represented as an object in one of two JSON array files
(`train_val.json` or `test.json`). Each object has the following key/value
pairs:

    - patient:              The patient ID.
    - image:                The path to the image file.
    - tumor_mask:           The path to the tumor mask file.
    - scan_mask:            The path to the scan mask file.
    - pathology:            The pathology of the lesion (benign or
                            malignant = 1).
    - label:                The pathology of the lesion encoded as an integer
                            (benign = 0, malignant = 1). Lesions with TIRADS
                            levels between 1-3 are considered benign and levels
                            4-5 are considered malignant.
    - age:                  The age of the patient.
    - sex:                  The sex of the patient.
    - location:             The location of the lesion.
    - lesion_size_x:        The size of the lesion in the x-direction.
    - lesion_size_y:        The size of the lesion in the y-direction.
    - lesion_size_z:        The size of the lesion in the z-direction.
    - tirads_level:         The TIRADS level of the lesion.
    - tirads_composition:   No description provided.
    - tirads_echogenicity:  No description provided.
    - tirads_shape:         No description provided.
    - tirads_margin:        No description provided.
    - tirads_echogenicfoci: No description provided.
    - histopath_diagnosis:  No description provided.

Usage:
    ultrabench stanford_thyroid RAW_DATA_DIR OUTPUT_DIR
"""

import json
import os
from importlib.metadata import version
from typing import Annotated

import h5py
import numpy as np
import pandas as pd
import skimage
import skimage.io as io
import typer
from sklearn.model_selection import GroupShuffleSplit

from .utils import resize_and_pad, save_version_info, to_3_channel

__version__ = version("ultrabench")
DATASET_FILE = "dataset.hdf5"
METADATA_FILE = "metadata.csv"
OUTPUT_NAME = "stanford_thyroid_v{}"

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


def verify_args(raw_data_dir: str, output_dir: str) -> None:
    """Verify the command line arguments.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
    """
    assert os.path.isdir(raw_data_dir), "raw_data_dir must be an existing directory"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(
        os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    ), "A matching version of the Stanford Thyroid dataset already exists"


def process_stanford_thyroid(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
    size: Annotated[
        int, typer.Option(help="Test image size (height and width in pixels)")
    ] = 256,
) -> None:
    """Preprocess the Stanford Thyroid Ultrasound Cine-clip dataset.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        size: The height and width of the test images in pixels.
    """
    verify_args(raw_data_dir, output_dir)

    dataset_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    test_dir_name = f"test_{size}"

    metadata = pd.read_csv(os.path.join(raw_data_dir, METADATA_FILE))
    included_patients = set(PATIENTS)

    # First pass: collect patient IDs for all included examples (no image loading)
    with h5py.File(os.path.join(raw_data_dir, DATASET_FILE), "r") as dataset:
        annot_ids = dataset["annot_id"].asstr()[:]

    index_records = []
    for i, patient_id in enumerate(annot_ids):
        if int(patient_id.removesuffix("_")) not in included_patients:
            continue
        index_records.append({"hdf5_index": i, "patient": patient_id.removesuffix("_")})

    index_df = pd.DataFrame(index_records)

    # Determine the train/test split from patient groups before image I/O
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_indices, test_indices = next(
        splitter.split(X=index_df, groups=index_df["patient"])
    )
    test_hdf5_indices = set(index_df.iloc[test_indices]["hdf5_index"])

    # Confirm the number of patients
    actual = len(index_df["patient"].unique())
    expected = len(included_patients)
    assert actual == expected, (
        f"The actual ({actual}) and expected ({expected}) number of patients does not match!"  # noqa: E501
    )

    # Create output directories
    for split_dir_name in ["train_val", test_dir_name]:
        split_dir = os.path.join(dataset_dir, split_dir_name)
        os.makedirs(os.path.join(split_dir, "images"))
        os.makedirs(os.path.join(split_dir, "masks", "tumor"))
        os.makedirs(os.path.join(split_dir, "masks", "scan"))

    # Second pass: load images and masks, save to correct split directory
    train_examples = []
    test_examples = []

    with h5py.File(os.path.join(raw_data_dir, DATASET_FILE), "r") as dataset:
        for row in index_df.itertuples(index=False):
            i = row.hdf5_index
            patient_id = annot_ids[i]
            image = dataset["image"][i]
            mask = dataset["mask"][i]
            patient_metadata = metadata[metadata["annot_id"] == patient_id]

            is_test = i in test_hdf5_indices
            split_dir = os.path.join(
                dataset_dir, test_dir_name if is_test else "train_val"
            )

            # Map mask values from {0, 255} to {0, 1}
            tumor_mask = (mask == 255).astype("uint8")
            scan_mask = generate_scan_mask(image)
            image_3ch = to_3_channel(image)

            if is_test:
                image_3ch = resize_and_pad(image_3ch, size, order=1)
                tumor_mask = resize_and_pad(tumor_mask, size, order=0)
                scan_mask = resize_and_pad(scan_mask, size, order=0)

            stem = f"{patient_id}{i}"
            io.imsave(
                os.path.join(split_dir, "images", f"{stem}.png"),
                image_3ch,
                check_contrast=False,
            )
            io.imsave(
                os.path.join(split_dir, "masks", "tumor", f"{stem}.png"),
                tumor_mask,
                check_contrast=False,
            )
            io.imsave(
                os.path.join(split_dir, "masks", "scan", f"{stem}.png"),
                scan_mask,
                check_contrast=False,
            )

            example = {
                "patient": patient_id.removesuffix("_"),
                "image": f"images/{stem}.png",
                "tumor_mask": f"masks/tumor/{stem}.png",
                "scan_mask": f"masks/scan/{stem}.png",
                "label": (patient_metadata["ti-rads_level"] > 3).astype(int).item(),
                "age": patient_metadata["age"].item(),
                "sex": patient_metadata["sex"].item(),
                "location": patient_metadata["location"].item(),
                "lesion_size_x": patient_metadata["size_x"].item(),
                "lesion_size_y": patient_metadata["size_y"].item(),
                "lesion_size_z": patient_metadata["size_z"].item(),
                "tirads_level": patient_metadata["ti-rads_level"].item(),
                "tirads_composition": patient_metadata["ti-rads_composition"].item(),
                "tirads_echogenicity": patient_metadata["ti-rads_echogenicity"].item(),
                "tirads_shape": patient_metadata["ti-rads_shape"].item(),
                "tirads_margin": patient_metadata["ti-rads_margin"].item(),
                "tirads_echogenicfoci": patient_metadata[
                    "ti-rads_echogenicfoci"
                ].item(),
                "histopath_diagnosis": patient_metadata["histopath_diagnosis"].item(),
            }
            example["pathology"] = "benign" if example["label"] == 0 else "malignant"
            (test_examples if is_test else train_examples).append(example)

    # Save the metadata JSON files inside their split directories
    train_dir = os.path.join(dataset_dir, "train_val")
    test_dir = os.path.join(dataset_dir, test_dir_name)
    with open(os.path.join(train_dir, "train_val.json"), "w") as f:
        json.dump(train_examples, f, indent=4)
    with open(os.path.join(test_dir, "test.json"), "w") as f:
        json.dump(test_examples, f, indent=4)

    print(f"Training examples: {len(train_examples)}")
    print(f"Test examples: {len(test_examples)}")

    save_version_info(dataset_dir, __version__)
