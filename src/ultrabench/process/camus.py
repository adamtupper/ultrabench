"""Split the CAMUS dataset into training and test sets using a 8:2 split,
ensuring that there is no patient overlap between the splits.

The images and masks are extracted from the NIfTI files and saved as 8-bit PNG
files. The corresponding metadata for each sequence is the ".cfg" files.

Each example (a single image) is represented as an object in one of two JSON
array files (`train_val.json` or `test.json`). Each object has
the following key/value pairs:

    - patient:          The patient ID.
    - view:             The view of the sequence (2CH or 4CH).
    - frame:            The frame number of the image in the sequence.
    - sex:              The sex of the patient.
    - age:              The age of the patient.
    - image_quality:    The image quality of the image.
    - EF:               The ejection fraction.
    - frame_rate:       The frame rate of the sequence.
    - image:            The path to the image file.
    - mask:             The path to the mask file.
    - label:            The quality of the image encoded as an integer
                        (poor = 0, medium = 1, good = 2).

Usage:
    ultrabench camus RAW_DATA_DIR OUTPUT_DIR
"""

import glob
import json
import os
from importlib.metadata import version

import nibabel as nib
import numpy as np
import pandas as pd
import skimage
import typer
import yaml
from sklearn.model_selection import GroupShuffleSplit
from typing_extensions import Annotated

from .utils import resize_and_pad, save_version_info, to_3_channel

__version__ = version("ultrabench")

OUTPUT_NAME = "camus_v{}"
CLASS_TO_LABEL = {
    "Poor": 0,
    "Medium": 1,
    "Good": 2,
}


def segment_fan(image: np.ndarray) -> np.ndarray:
    """Generate a fan-shaped mask using morphological operations.

    Args:
        image: The input 2D image array.

    Returns:
        A binary mask of the fan-shaped region.
    """
    mask = image > 0
    mask = skimage.morphology.convex_hull_image(mask)
    return mask.astype(np.uint8)


def extract_images_and_masks(
    dataset_dir: str,
    train_dir: str,
    test_dir: str,
    test_filenames: set[str],
    size: int,
) -> None:
    """Extract images and masks from NIfTI files and save them as PNG files.

    Images are routed to the train or test split directory based on their
    filename. Test images are resized and padded to size×size pixels.

    Args:
        dataset_dir: The path to the raw dataset directory.
        train_dir: The path to the training split output directory.
        test_dir: The path to the test split output directory.
        test_filenames: The set of filenames belonging to the test split.
        size: The target height and width for test images in pixels.
    """
    for split_dir in [train_dir, test_dir]:
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "masks", "scan"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "masks", "structures"), exist_ok=True)

    for nifti_file in glob.glob(
        os.path.join(dataset_dir, "database_nifti", "**", "*_half_sequence*.nii.gz")
    ):
        images = nib.load(nifti_file).get_fdata().astype(np.uint8)  # type: ignore
        patient, view = nifti_file.split("/")[-1].split("_")[:2]
        for i in range(images.shape[2]):
            image = np.rot90(images[:, :, i], axes=(1, 0)).astype(np.uint8)
            output_filename = f"{patient}_view{view}_frame{i + 1}.png"

            is_test = output_filename in test_filenames
            split_dir = test_dir if is_test else train_dir

            if "_gt" in nifti_file:
                mask = image
                if is_test:
                    mask = resize_and_pad(mask, size, order=0)
                skimage.io.imsave(
                    os.path.join(split_dir, "masks", "structures", output_filename),
                    mask,
                    check_contrast=False,
                )
            else:
                scan_mask = segment_fan(image)
                image_3ch = to_3_channel(image)
                if is_test:
                    scan_mask = resize_and_pad(scan_mask, size, order=0)
                    image_3ch = resize_and_pad(image_3ch, size, order=1)
                skimage.io.imsave(
                    os.path.join(split_dir, "masks", "scan", output_filename),
                    scan_mask,
                    check_contrast=False,
                )
                skimage.io.imsave(
                    os.path.join(split_dir, "images", output_filename),
                    image_3ch,
                    check_contrast=False,
                )


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
        patient_id, filename = file_path.split("/")[-2:]
        patient_id = int(patient_id.removeprefix("patient"))
        view = filename.removesuffix(".cfg").split("_")[-1]

        with open(file_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        for i in range(config["NbFrame"]):
            frame = i + 1
            filename = f"patient{patient_id:04d}_view{view}_frame{frame}.png"
            row = {
                "patient": patient_id,
                "view": view,
                "frame": frame,
                "sex": config["Sex"],
                "age": config["Age"],
                "image_quality": config["ImageQuality"],
                "EF": config["EF"],
                "frame_rate": config["FrameRate"],
                "image": f"images/{filename}",
                "mask": f"masks/structures/{filename}",
                "scan_mask": f"masks/scan/{filename}",
                "label": CLASS_TO_LABEL[config["ImageQuality"]],
            }
            row["ED"] = True if config["ED"] == frame else False
            row["ES"] = True if config["ES"] == frame else False
            metadata.append(row)

    return pd.DataFrame(metadata)


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
    ), "A matching version of the CAMUS dataset already exists"


def process_camus(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
    size: Annotated[
        int, typer.Option(help="Test image size (height and width in pixels)")
    ] = 256,
) -> None:
    """Divide the CAMUS dataset into training and test splits.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        size: The height and width of the test images in pixels.
    """
    verify_args(raw_data_dir, output_dir)

    dataset_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    test_dir_name = f"test_{size}"
    train_dir = os.path.join(dataset_dir, "train_val")
    test_dir = os.path.join(dataset_dir, test_dir_name)

    # Extract clinical metadata (reads only .cfg files, no images loaded)
    metadata = extract_metadata(raw_data_dir)

    # Split the dataset into training and test sets
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_indices, test_indices = next(
        splitter.split(X=metadata, groups=metadata["patient"])
    )
    train_val_examples = metadata.iloc[train_val_indices]
    test_examples = metadata.iloc[test_indices]

    # Build the set of test filenames for routing during image extraction
    test_filenames = {os.path.basename(p) for p in test_examples["image"]}

    # Extract and save images and masks, routing to correct split directory
    extract_images_and_masks(raw_data_dir, train_dir, test_dir, test_filenames, size)

    # Save the training and test examples as JSON files inside their split dirs
    for split, examples, split_dir in [
        ("train_val", train_val_examples, train_dir),
        ("test", test_examples, test_dir),
    ]:
        with open(os.path.join(split_dir, f"{split}.json"), "w") as f:
            json.dump(examples.to_dict(orient="records"), f, indent=4)
        print(f"Number of {split} examples: {len(examples)}")

    save_version_info(dataset_dir, __version__)
