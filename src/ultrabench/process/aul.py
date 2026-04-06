"""Split the Annotated Ultrasound Liver (AUL) dataset into training and test
sets using a 8:2 split. The dataset consists of 735 images from different
patients.

Two versions of the dataset are created, one for liver segmentation and one for
mass classification/segmentation. This is because one of the examples
(image 374.jpg) is missing a liver segmentation mask and is therefore excluded
from the liver segmentation task. For images 229.jpg and 306.jpg, the scan
annotations actually correspond to the liver (they are missing the scan
annotations).

Each example (a single image) is represented as an object in one of two JSON
array files (`train_val.json` or `test.json`). Each object has the following
key/value pairs:

    - image: The path to the image file.
    - scan_mask:  The path to the scan mask file.
    - liver_mask: The path to the liver mask file.
    - mass_mask: The path to the mass mask file.
    - pathology: The pathology of the mass (normal/no mass, benign or
        malignant).
    - label: The pathology of the mass encoded as an integer (normal/no
        mass = 0, benign = 1, malignant = 2).

Usage:
    ultrabench aul RAW_DATA_DIR OUTPUT_DIR
"""

import glob
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import skimage
import skimage.io as io
import typer
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated

from .utils import (
    generate_classification_ground_truth,
    generate_segmentation_ground_truth,
    resize_and_pad,
    save_version_info,
    to_3_channel,
)

__version__ = version("ultrabench")

OUTPUT_NAME_MASS = "aul_mass_v{}"
OUTPUT_NAME_LIVER = "aul_liver_v{}"
CLASS_TO_LABEL = {
    "Normal": 0,
    "Benign": 1,
    "Malignant": 2,
}


def collect_examples(dataset_dir: str) -> list[dict[str, Any]]:
    """Collect all examples from the dataset directory.

    Args:
        dataset_dir: The path to the dataset directory.

    Returns:
        A list of examples, where each example is represented as a
            dictionary.
    """
    examples = []
    for class_name in CLASS_TO_LABEL.keys():
        images = glob.glob(f"{dataset_dir}/{class_name}/image/*.jpg")
        for image_path in images:
            filename = os.path.basename(image_path).removesuffix(".jpg")
            scan_polygon_path = os.path.join(
                dataset_dir, class_name, "segmentation/outline", f"{filename}.json"
            )
            liver_polygon_path = os.path.join(
                dataset_dir, class_name, "segmentation/liver", f"{filename}.json"
            )
            mass_polygon_path = os.path.join(
                dataset_dir, class_name, "segmentation/mass", f"{filename}.json"
            )

            has_liver_polygon = os.path.exists(liver_polygon_path)
            has_mass_polygon = os.path.exists(mass_polygon_path)

            new_filename = f"{class_name.lower()}_{filename}"

            example = {
                "filename": new_filename,
                "image_path": image_path,
                "scan_polygon_path": scan_polygon_path,
                "liver_polygon_path": liver_polygon_path,
                "mass_polygon_path": mass_polygon_path,
                "pathology": class_name,
                "label": CLASS_TO_LABEL[class_name],
                "image": f"images/{new_filename}.png",
                "scan_mask": f"masks/scan/{new_filename}.png",
                "liver_mask": f"masks/liver/{new_filename}.png"
                if has_liver_polygon
                else None,
                "mass_mask": f"masks/mass/{new_filename}.png"
                if (has_mass_polygon or CLASS_TO_LABEL[class_name] == 0)
                else None,
            }

            # Correct the scan/liver annotations for images 229.jpg and 306.jpg
            if filename in ["229", "306"]:
                example["liver_polygon_path"] = example["scan_polygon_path"]
                example["liver_mask"] = f"masks/liver/{new_filename}.png"
                example["scan_polygon_path"] = None

            examples.append(example)

    return examples


def generate_scan_mask(image_path: str) -> np.ndarray:
    """Generate a scan mask for the given image using morphological operations.

    Args:
        image_path: The path to the image file.

    Returns:
        A binary mask of the scan region.
    """
    image = io.imread(image_path)
    if image.ndim == 3:
        image = image.mean(axis=-1).astype("uint8")

    # Threshold the image
    mask = image > 1

    # Morphological operations
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)
    mask = skimage.morphology.remove_small_objects(mask, min_size=1000)

    # Extract convex hull of the mask
    mask = skimage.morphology.convex_hull_image(mask)

    return mask.astype("uint8")


def generate_mask(image_shape: tuple[int, int], polygon_file: str) -> np.ndarray:
    """Generate a pixel mask for the polygon coordinates in the given file.

    Args:
        image_shape: The shape of the image (height, width).
        polygon_file: The path to the JSON file containing the polygon
            coordinates.

    Returns:
        A binary mask of the polygon region.
    """
    with open(polygon_file) as f:
        points = json.load(f)

    points = [[y, x] for x, y in points]
    mask = skimage.draw.polygon2mask(image_shape, points).astype("uint8")

    return mask


def copy_images(split_dir: str, df: pd.DataFrame, size: int | None = None) -> None:
    """Save the images as 3-channel PNG files in the split directory.

    Args:
        split_dir: The path to the split directory (train/ or test_{size}/).
        df: The DataFrame containing the examples.
        size: If provided, resize and pad images to size×size pixels.
    """
    image_dir = os.path.join(split_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    for _, example in df.iterrows():
        dest_filename = f"{example['filename']}.png"

        image = io.imread(example["image_path"])
        image = to_3_channel(image)
        if size is not None:
            image = resize_and_pad(image, size, order=1)

        io.imsave(os.path.join(image_dir, dest_filename), image, check_contrast=False)


def generate_masks(split_dir: str, df: pd.DataFrame, size: int | None = None) -> None:
    """Generate and save the scan, liver, and mass masks for the examples.

    Args:
        split_dir: The path to the split directory (train/ or test_{size}/).
        df: The DataFrame containing the examples.
        size: If provided, resize and pad masks to size×size pixels using
            nearest-neighbor interpolation.
    """
    scan_mask_dir = os.path.join(split_dir, "masks", "scan")
    liver_mask_dir = os.path.join(split_dir, "masks", "liver")
    mass_mask_dir = os.path.join(split_dir, "masks", "mass")
    os.makedirs(scan_mask_dir, exist_ok=True)
    os.makedirs(liver_mask_dir, exist_ok=True)
    os.makedirs(mass_mask_dir, exist_ok=True)

    for _, example in df.iterrows():
        image = io.imread(example["image_path"])

        if example["scan_polygon_path"]:
            scan_mask = generate_mask(image.shape[:2], example["scan_polygon_path"])
        else:
            scan_mask = generate_scan_mask(example["image_path"])
        if size is not None:
            scan_mask = resize_and_pad(scan_mask, size, order=0)
        io.imsave(
            os.path.join(scan_mask_dir, f"{example['filename']}.png"),
            scan_mask,
            check_contrast=False,
        )

        if example["liver_mask"]:
            liver_mask = generate_mask(image.shape[:2], example["liver_polygon_path"])
            if size is not None:
                liver_mask = resize_and_pad(liver_mask, size, order=0)
            io.imsave(
                os.path.join(liver_mask_dir, f"{example['filename']}.png"),
                liver_mask,
                check_contrast=False,
            )

        if example["mass_mask"]:
            if os.path.exists(example["mass_polygon_path"]):
                mass_mask = generate_mask(image.shape[:2], example["mass_polygon_path"])
            else:
                # Create an empty mask for images with no mass
                mass_mask = np.zeros(image.shape[:2], dtype="uint8")
            if size is not None:
                mass_mask = resize_and_pad(mass_mask, size, order=0)
            io.imsave(
                os.path.join(mass_mask_dir, f"{example['filename']}.png"),
                mass_mask,
                check_contrast=False,
            )


def save_examples(split_dir: str, df: pd.DataFrame, split: str) -> None:
    """Save the examples to a JSON file inside the split directory.

    Args:
        split_dir: The path to the split directory (train/ or test_{size}/).
        df: The DataFrame containing the examples.
        split: The dataset split (train_val, or test).
    """
    columns_to_drop = [
        "filename",
        "image_path",
        "scan_polygon_path",
        "liver_polygon_path",
        "mass_polygon_path",
    ]
    df = df.drop(columns=columns_to_drop)
    examples = df.to_dict(orient="records")
    with open(os.path.join(split_dir, f"{split}.json"), "w") as f:
        json.dump(examples, f, indent=4)


def verify_args(raw_data_dir: str, output_dir: str) -> None:
    """Verify the command line arguments.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
    """
    assert os.path.isdir(raw_data_dir), "raw_data_dir must be an existing directory"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(
        os.path.join(output_dir, OUTPUT_NAME_MASS.format(__version__))
    ), "A matching version of the AUL Mass dataset already exists"
    assert not os.path.exists(
        os.path.join(output_dir, OUTPUT_NAME_LIVER.format(__version__))
    ), "A matching version of the AUL Liver dataset already exists"


def process_aul(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
    size: Annotated[
        int, typer.Option(help="Test image size (height and width in pixels)")
    ] = 256,
) -> None:
    """Prepare the training and test sets for the mass classification,
    mass segmentation and liver segmentation tasks.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        size: The height and width of the test images in pixels.
    """
    # Verify arguments
    verify_args(raw_data_dir, output_dir)

    # Collect examples
    examples = collect_examples(raw_data_dir)
    df = pd.DataFrame.from_records(examples)

    print(f"Total number of examples: {len(df)}")

    # Separate the test set
    train_val_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True, stratify=df["label"]
    )

    test_dir_name = f"test_{size}"

    # Create mass classification/segmentation dataset
    mass_dataset_dir = os.path.join(output_dir, OUTPUT_NAME_MASS.format(__version__))
    mass_train_dir = os.path.join(mass_dataset_dir, "train_val")
    mass_test_dir = os.path.join(mass_dataset_dir, test_dir_name)

    copy_images(mass_train_dir, train_val_df)
    generate_masks(mass_train_dir, train_val_df)
    save_examples(mass_train_dir, train_val_df, "train_val")

    copy_images(mass_test_dir, test_df, size=size)
    generate_masks(mass_test_dir, test_df, size=size)
    save_examples(mass_test_dir, test_df, "test")

    mass_test_examples = test_df.drop(
        columns=[
            "filename",
            "image_path",
            "scan_polygon_path",
            "liver_polygon_path",
            "mass_polygon_path",
        ]
    ).to_dict(orient="records")
    generate_classification_ground_truth(
        mass_test_examples, Path(mass_test_dir) / "ground_truth_classification.csv"
    )
    generate_segmentation_ground_truth(
        mass_test_examples,
        mask_key="mass_mask",
        classes=[1],
        output_path=Path(mass_test_dir) / "ground_truth_segmentation.csv",
    )

    save_version_info(mass_dataset_dir, __version__)

    print(f"Mass dataset - Number of train_val examples: {len(train_val_df)}")
    print(f"Mass dataset - Number of test examples: {len(test_df)}")

    # Create liver segmentation dataset
    liver_train_val_df = train_val_df.dropna(subset=["liver_mask"])
    liver_test_df = test_df.dropna(subset=["liver_mask"])

    liver_dataset_dir = os.path.join(output_dir, OUTPUT_NAME_LIVER.format(__version__))
    liver_train_dir = os.path.join(liver_dataset_dir, "train_val")
    liver_test_dir = os.path.join(liver_dataset_dir, test_dir_name)

    copy_images(liver_train_dir, liver_train_val_df)
    generate_masks(liver_train_dir, liver_train_val_df)
    save_examples(liver_train_dir, liver_train_val_df, "train_val")

    copy_images(liver_test_dir, liver_test_df, size=size)
    generate_masks(liver_test_dir, liver_test_df, size=size)
    save_examples(liver_test_dir, liver_test_df, "test")

    save_version_info(liver_dataset_dir, __version__)

    print(f"Liver dataset - Number of train_val examples: {len(liver_train_val_df)}")
    print(f"Liver dataset - Number of test examples: {len(liver_test_df)}")
