"""Split the Stanford Thyroid Ultrasound Cine-clip dataset into training, validation,
and test data using a 7:1:2 split, ensuring that there is no patient overlap between the
splits.

The images and masks are extracted from the `dataset.hdf5` file. The images and masks
are saved as single-channel uint8 PNG files. Mask values are mapped from the values
{0, 255} to {0, 1}.

Each example (image) is represented as an object in one of three JSON array files
(`train.json`, `validation.json`, or `test.json`). Each object has the following
key/value pairs:

    - patient:              The patient ID.
    - image:                The path to the image file.
    - tumor_mask:           The path to the tumor mask file.
    - scan_mask:            The path to the scan mask file.
    - pathology:            The pathology of the lesion (benign or malignant = 1).
    - label:                The pathology of the lesion encoded as an integer
                            (benign = 0, malignant = 1). Lesions with TIRADS levels
                            between 1-3 are considered benign and levels 4-5 are
                            considered malignant.
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
    ultrabenck stanford_thyroid RAW_DATA_DIR OUTPUT_DIR
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

from .utils import save_version_info

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


def generate_scan_mask(image: np.ndarray):
    # Threshold the image
    mask = image > 1

    # Fill small holes
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=64)

    # Erode the mask
    for _ in range(30):
        mask = skimage.morphology.binary_erosion(mask)

    # Dilate the mask
    for _ in range(30):
        mask = skimage.morphology.binary_dilation(mask)

    # Logical OR of left and right halves to mitigate shadows
    _, width = mask.shape
    left_half = mask[:, : width // 2]
    right_half = mask[:, width // 2 :]

    left_half = np.logical_or(left_half, np.fliplr(right_half))
    right_half = np.logical_or(right_half, np.fliplr(left_half))

    # Merge the two halves back together
    mask = np.hstack((left_half, right_half))

    # Extract convex hull of the mask
    mask = skimage.morphology.convex_hull_image(mask)

    return mask.astype(np.uint8)


def verify_args(raw_data_dir, output_dir):
    assert os.path.isdir(raw_data_dir), "raw_data_dir must be an existing directory"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(
        os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    ), "A matching version of the Stanford Thyroid dataset already exists"


def stanford_thyroid(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
):
    """Preprocess the Stanford Thyroid Ultrasound Cine-clip dataset."""
    verify_args(raw_data_dir, output_dir)

    output_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))

    # Load the image and mask data, and the metadata
    dataset = h5py.File(os.path.join(raw_data_dir, DATASET_FILE), "r")
    metadata = pd.read_csv(os.path.join(raw_data_dir, METADATA_FILE))

    # Load the list of patients to include (those whose scans were acquired using a
    # convex probe)
    included_patients = set(PATIENTS)

    # Create the output directories
    images_dir = os.path.join(output_dir, "images")
    tumor_masks_dir = os.path.join(output_dir, "masks", "tumor")
    scan_masks_dir = os.path.join(output_dir, "masks", "scan")
    os.makedirs(images_dir)
    os.makedirs(tumor_masks_dir)
    os.makedirs(scan_masks_dir)

    # For each image, create a dictionary containing the associated metadata, and save
    # the image and mask as PNG files
    examples = []
    for i in range(len(dataset["image"])):
        patient_id, image, mask = (
            dataset["annot_id"].asstr()[i],
            dataset["image"][i],
            dataset["mask"][i],
        )
        patient_metadata = metadata[metadata["annot_id"] == patient_id]
        assert (
            len(patient_metadata) == 1
        ), f"metadata for patient {patient_id} not found"

        if int(patient_id.removesuffix("_")) not in included_patients:
            continue

        # Map the mask values from {0, 255} to {0, 1}
        mask = (mask == 255).astype("uint8")

        # Save the image and tumor mask as PNG files
        image_path = f"images/{patient_id}{i}.png"
        tumor_mask_path = f"masks/tumor/{patient_id}{i}.png"
        io.imsave(os.path.join(output_dir, image_path), image, check_contrast=False)
        io.imsave(os.path.join(output_dir, tumor_mask_path), mask, check_contrast=False)

        # Generate the scan mask
        scan_mask = generate_scan_mask(image)
        scan_mask_path = f"masks/scan/{patient_id}{i}.png"
        io.imsave(
            os.path.join(output_dir, scan_mask_path), scan_mask, check_contrast=False
        )

        # Create the metadata dictionary
        examples.append(
            {
                "patient": patient_id.removesuffix("_"),
                "image": image_path,
                "tumor_mask": tumor_mask_path,
                "scan_mask": scan_mask_path,
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
        )

    dataset.close()

    # Add pathology labels
    examples = pd.DataFrame.from_records(examples)
    examples["pathology"] = examples["label"].map({0: "benign", 1: "malignant"})

    # Confirm the number of patients
    actual = len(examples["patient"].unique())
    expected = len(included_patients)
    assert actual == expected, (
        f"The actual ({actual}) and expected ({expected}) number of patients does not match!"  # noqa: E501
    )

    # Split the examples into training, validation, and test sets
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_indices, test_indices = next(
        splitter.split(X=examples, groups=examples["patient"])
    )
    train_val_examples = examples.iloc[train_val_indices]
    test_examples = examples.iloc[test_indices]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_indices, val_indices = next(
        splitter.split(X=train_val_examples, groups=train_val_examples["patient"])
    )
    train_examples = train_val_examples.iloc[train_indices]
    val_examples = train_val_examples.iloc[val_indices]

    # Save the training, validation, and test examples as JSON files
    for split, examples in [
        ("train", train_examples),
        ("validation", val_examples),
        ("test", test_examples),
    ]:
        file_path = os.path.join(output_dir, f"{split}.json")
        with open(file_path, "w") as f:
            json.dump(examples.to_dict(orient="records"), f, indent=4)

    save_version_info(output_dir, __version__)
