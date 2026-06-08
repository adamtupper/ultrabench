"""Split the Annotated Ultrasound Liver (AUL) dataset into training and
test sets using a stratified (by label) 8:2 split. Save the resulting
dataset as a Hugging Face dataset.

Corrections:

    - For images 229.jpg and 306.jpg the scan annotations actually
      correspond to the liver, so the scan masks are generated via
      morphological operations instead.
    - Image 374.jpg is missing a liver segmentation mask, so its
      liver_mask feature is None.

The resulting dataset has three splits:

    - train: original-size images.
    - test: original-size images.
    - test.{size}: same examples as test, resized so the longest edge
      measures {size} px and then zero-padded to {size} x {size} px.

Each example has the following features:

    - filename: Original filename without the file extension.
    - image: 3-channel ultrasound image.
    - scan_mask: Single-channel binary scan mask.
    - liver_mask: Single-channel binary liver mask.
    - mass_mask: Single-channel binary mass mask.
    - label: ClassLabel encoding the mass pathology (normal=0, benign=1,
      malignant=2).

Three ground-truth CSV files are also created for each test split for
use by the evaluation layer:

    - ground_truth_classification.csv
    - ground_truth_segmentation_mass.csv
    - ground_truth_segmentation_liver.csv

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
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from typing_extensions import Annotated

from .utils import (
    generate_classification_ground_truth_from_records,
    generate_segmentation_ground_truth_from_records,
    resize_and_pad,
    save_version_info,
    to_3_channel,
)

__version__ = version("ultrabench")

OUTPUT_NAME = "UltraBench/AUL-test"
CLASS_TO_LABEL = {
    "Normal": 0,
    "Benign": 1,
    "Malignant": 2,
}
DATASET_INFO = DatasetInfo(
    description=(
        "A dataset of liver ultrasound images for liver and mass "
        "segmentation, and liver mass classification."
    ),
    citation="""
    @dataset{yiming2022_aul,
        author       = {Xu Yiming and
                        Zheng Bowen and
                        Liu Xiaohong and
                        Wu Tao and
                        Ju Jinxiu and
                        Wang Shijie and
                        Lian Yufan and
                        Zhang Hongjun and
                        Liang Tong and
                        Sang Ye and
                        Jiang Rui and
                        Wang Guangyu and
                        Ren Jie and
                        Chen Ting},
        title        = {Annotated Ultrasound Liver images},
        month        = nov,
        year         = 2022,
        publisher    = {Zenodo},
        doi          = {10.5281/zenodo.7272660},
        url          = {https://doi.org/10.5281/zenodo.7272660},
    }
    """,
    homepage="https://zenodo.org/records/7272660",
    license="cc-by-4.0",
    features=Features(
        {
            "filename": Value("string"),
            "image": Image(),
            "scan_mask": Image(),
            "liver_mask": Image(),
            "mass_mask": Image(),
            "label": ClassLabel(names=["normal", "benign", "malignant"]),
        }
    ),
)


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
                "has_liver_mask": has_liver_polygon,
                "has_mass_mask": has_mass_polygon or CLASS_TO_LABEL[class_name] == 0,
            }

            # Correct the scan/liver annotations for images 229.jpg and 306.jpg
            if filename in ["229", "306"]:
                example["liver_polygon_path"] = example["scan_polygon_path"]
                example["scan_polygon_path"] = None  # ty:ignore[invalid-assignment]
                example["has_liver_mask"] = True

            examples.append(example)

    return examples


def generate_scan_mask(image_path: str) -> np.ndarray:
    """Generate a scan mask for the given image using morphological
    operations.

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


def _process_example(row: pd.Series, size: int | None = None) -> dict:
    """Process a single DataFrame row into a record dict.

    Loads the raw image and generates all masks as numpy arrays. If size
    is provided, the image and masks are resized and padded to size x
    size.

    Args:
        row: A row from the examples DataFrame.
        size: If provided, resize and pad all arrays to size x size
            pixels.

    Returns:
        A dict with keys filename, image, scan_mask, liver_mask,
        mass_mask, and label. Image values are numpy arrays (or None if
        a mask is absent for this example).
    """
    # Preprocess image
    image = io.imread(row["image_path"])
    original_shape = image.shape[:2]
    image = to_3_channel(image)
    if size is not None:
        image = resize_and_pad(image, size, order=1)

    # Generate scan mask
    if not pd.isna(row["scan_polygon_path"]):
        scan_mask = generate_mask(original_shape, row["scan_polygon_path"])
    else:
        scan_mask = generate_scan_mask(row["image_path"])
    if size is not None:
        scan_mask = resize_and_pad(scan_mask, size, order=0)

    # Generate liver (if it exists)
    liver_mask = None
    if row["has_liver_mask"]:
        liver_mask = generate_mask(original_shape, row["liver_polygon_path"])
        if size is not None:
            liver_mask = resize_and_pad(liver_mask, size, order=0)

    # Generate mass (if it exists)
    mass_mask = None
    if row["has_mass_mask"]:
        if os.path.exists(row["mass_polygon_path"]):
            mass_mask = generate_mask(original_shape, row["mass_polygon_path"])
        else:
            mass_mask = np.zeros(original_shape, dtype="uint8")
        if size is not None:
            mass_mask = resize_and_pad(mass_mask, size, order=0)

    return {
        "filename": row["filename"],
        "image": image,
        "scan_mask": scan_mask,
        "liver_mask": liver_mask,
        "mass_mask": mass_mask,
        "label": row["label"],
    }


def verify_args(raw_data_dir: str, output_dir: str) -> None:
    """Verify the command line arguments.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
    """
    assert os.path.isdir(raw_data_dir), "raw_data_dir must be an existing directory"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(os.path.join(output_dir, OUTPUT_NAME)), (
        "A version of the AUL dataset already exists"
    )


def process_aul(
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
    """Prepare the AUL dataset as a Hugging Face DatasetDict and save to
    disk.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        size: The height and width of the resized test images in pixels.
    """
    verify_args(raw_data_dir, output_dir)

    examples = collect_examples(raw_data_dir)
    df = pd.DataFrame.from_records(examples)

    train_val_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True, stratify=df["label"]
    )

    train_records = [
        _process_example(row)
        for _, row in tqdm(
            train_val_df.iterrows(),
            total=len(train_val_df),
            desc="Processing train split",
        )
    ]
    test_records = [
        _process_example(row)
        for _, row in tqdm(
            test_df.iterrows(), total=len(test_df), desc="Processing test split"
        )
    ]
    test_resized_records = [
        _process_example(row, size=size)
        for _, row in tqdm(
            test_df.iterrows(),
            total=len(test_df),
            desc=f"Processing test.{size} split",
        )
    ]

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_records, info=DATASET_INFO),
            "test": Dataset.from_list(test_records, info=DATASET_INFO),
            f"test.{size}": Dataset.from_list(test_resized_records, info=DATASET_INFO),
        }
    )

    output_path = Path(output_dir) / OUTPUT_NAME
    dataset.save_to_disk(str(output_path))

    generate_classification_ground_truth_from_records(
        test_records, output_path / f"test.{size}" / "ground_truth_classification.csv"
    )
    generate_segmentation_ground_truth_from_records(
        test_records,
        mask_key="mass_mask",
        classes=[1],
        output_path=output_path / f"test.{size}" / "ground_truth_segmentation_mass.csv",
    )
    liver_test_records = [r for r in test_records if r["liver_mask"] is not None]
    generate_segmentation_ground_truth_from_records(
        liver_test_records,
        mask_key="liver_mask",
        classes=[1],
        output_path=output_path
        / f"test.{size}"
        / "ground_truth_segmentation_liver.csv",
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
                path_in_repo="test-ground-truth-classification.csv",
                path_or_fileobj=output_path
                / f"test.{size}"
                / "ground_truth_classification.csv",
            ),
            CommitOperationAdd(
                path_in_repo=f"test.{size}-ground-truth-segmentation-liver.csv",
                path_or_fileobj=output_path
                / f"test.{size}"
                / "ground_truth_segmentation_liver.csv",
            ),
            CommitOperationAdd(
                path_in_repo=f"test.{size}-ground-truth-segmentation-mass.csv",
                path_or_fileobj=output_path
                / f"test.{size}"
                / "ground_truth_segmentation_mass.csv",
            ),
        ]
        api.create_commit(
            repo_id=OUTPUT_NAME,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Update ground truth CSV files for Version {__version__}.",
        )
