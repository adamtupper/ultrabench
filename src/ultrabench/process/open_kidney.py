"""Prepare the Open Kidney dataset as a Hugging Face DatasetDict.
Images are divided into training and test splits using an 8:2 split.
For the annotations, use those created by Sonographer 1. Each image is
associated with a distinct patient so we do not need to worry about
patient overlap between the splits. However, we do need to remove 20
duplicate images. The splits are stratified based on the view label
(longitudinal, transverse, other) to mitigate distribution shifts.

From inspecting the original code, the labels of the regions mask are:
    - 0: Background
    - 1: Central Echo Complex
    - 2: Medulla
    - 3: Cortex

The resulting dataset has three splits:

    - train: original-size images.
    - test: original-size images.
    - test.{size}: same examples as test, resized so the longest edge
      measures {size} px and then zero-padded to {size} x {size} px.

Each example has the following features:

    - filename: Original image filename (without the file extension).
    - transplant: Whether the kidney is transplanted.
    - view: ClassLabel encoding the kidney view.
    - quality: Image quality rating string.
    - image: 3-channel ultrasound image.
    - capsule_mask: Single-channel binary kidney capsule mask.
    - regions_mask: Single-channel multi-class kidney regions mask.
    - scan_mask: Single-channel binary scan mask.

Two ground-truth CSV files are also created in each test split
directory, with the masks encoded at the resolution of that split:

    - ground_truth_segmentation_capsule.csv
    - ground_truth_segmentation_regions.csv

Usage:
    ultrabench open_kidney RAW_DATA_DIR OUTPUT_DIR
"""

import ast
import os
from importlib.metadata import version
from pathlib import Path

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
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from typing_extensions import Annotated

from .utils import (
    DATASET_REVISION,
    generate_segmentation_ground_truth_from_records,
    ground_truth_commit_operations,
    resize_and_pad,
    save_version_info,
    to_3_channel,
    verify_revision_unpublished,
)

__version__ = version("ultrabench")

OUTPUT_NAME = "UltraBench/OpenKidney"
SONOGRAPHER = 1
CLINICAL_METADATA_FILE = f"labels/reviewed_labels_{SONOGRAPHER}.csv"
VIEW_TO_LABEL = {"longitudinal": 0, "transverse": 1, "other": 2}
DATASET_INFO = DatasetInfo(
    description=(
        "B-mode kidney ultrasound images for kidney capsule and region segmentation."
    ),
    citation="""
    @inproceedings{singla2023openkidney,
        author    = {Singla, R. and others},
        title     = {The Open Kidney Ultrasound Data Set},
        booktitle = {Simplifying Medical Ultrasound. ASMUS 2023.
                     Lecture Notes in Computer Science, vol 14337},
        publisher = {Springer, Cham},
        year      = {2023},
        doi       = {10.1007/978-3-031-44521-7_15},
    }
    """,
    homepage="https://doi.org/10.1007/978-3-031-44521-7_15",
    features=Features(
        {
            "filename": Value("string"),
            "transplant": Value("bool"),
            "view": ClassLabel(names=["longitudinal", "transverse", "other"]),
            "quality": Value("string"),
            "image": Image(),
            "capsule_mask": Image(),
            "regions_mask": Image(),
            "scan_mask": Image(),
        }
    ),
    license="other",
)


def generate_scan_mask(image: np.ndarray) -> np.ndarray:
    """Generate a scan mask using morphological operations.

    Args:
        image: The input 2D grayscale image array.

    Returns:
        A binary mask of the scan region.
    """
    mask = image > 0
    for _ in range(5):
        mask = skimage.morphology.binary_erosion(mask)
    for _ in range(5):
        mask = skimage.morphology.binary_dilation(mask)
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)

    # Keep only the largest connected component
    label = skimage.measure.label(mask)
    regions = skimage.measure.regionprops(label)
    largest_region = max(regions, key=lambda x: x.area)
    mask = label == largest_region.label

    # Reflect the larger half to fill gaps and make the mask symmetric
    _, width = mask.shape
    left_half = mask[:, : width // 2]
    right_half = mask[:, width // 2 :]
    if np.sum(left_half) > np.sum(right_half):
        right_half = np.fliplr(left_half)
    else:
        left_half = np.fliplr(right_half)
    mask = np.hstack((left_half, right_half))

    mask = skimage.morphology.convex_hull_image(mask)
    return mask.astype(np.uint8)


def _process_open_kidney_example(
    row: pd.Series, raw_data_dir: str, size: int | None = None
) -> dict:
    """Process a single DataFrame row into a record dict.

    Loads the raw image and all masks as numpy arrays. If size is
    provided, the image and masks are resized and padded to size x size.

    Args:
        row: A row from the metadata DataFrame.
        raw_data_dir: Path to the raw data directory.
        size: If provided, resize and pad arrays to size x size pixels.

    Returns:
        A dict with keys filename, transplant, view, quality, image,
        capsule_mask, regions_mask, and scan_mask. Image values are
        numpy arrays.
    """
    raw_image = io.imread(os.path.join(raw_data_dir, "images", row["filename"]))
    gray = (
        np.mean(raw_image, axis=-1).astype(np.uint8)
        if raw_image.ndim == 3
        else raw_image
    )
    scan_mask = generate_scan_mask(gray)

    capsule_mask = io.imread(
        os.path.join(
            raw_data_dir,
            "labels",
            f"reviewed_masks_{SONOGRAPHER}",
            "capsule",
            row["filename"],
        )
    )
    regions_mask = io.imread(
        os.path.join(
            raw_data_dir,
            "labels",
            f"reviewed_masks_{SONOGRAPHER}",
            "regions",
            row["filename"],
        )
    )

    image = to_3_channel(raw_image)
    if size is not None:
        image = resize_and_pad(image, size, order=1)
        scan_mask = resize_and_pad(scan_mask, size, order=0)
        capsule_mask = resize_and_pad(capsule_mask, size, order=0)
        regions_mask = resize_and_pad(regions_mask, size, order=0)

    return {
        "filename": Path(row["filename"]).stem,
        "transplant": bool(row["transplant"]),
        "view": VIEW_TO_LABEL[row["view"].lower()],
        "quality": str(row["quality"]),
        "image": image,
        "capsule_mask": capsule_mask,
        "regions_mask": regions_mask,
        "scan_mask": scan_mask,
    }


def write_ground_truth(
    records: list[dict], gt_dir: Path, image_size: int | None
) -> None:
    """Write the ground-truth CSVs of one test split.

    Args:
        records: The records of the split.
        gt_dir: The split directory to write the CSVs into.
        image_size: The height and width of the split's images, or None
            for the original-resolution split.
    """
    generate_segmentation_ground_truth_from_records(
        records,
        mask_key="capsule_mask",
        classes=[1],
        output_path=gt_dir / "ground_truth_segmentation_capsule.csv",
        image_size=image_size,
    )
    generate_segmentation_ground_truth_from_records(
        records,
        mask_key="regions_mask",
        classes=[1, 2, 3],
        output_path=gt_dir / "ground_truth_segmentation_regions.csv",
        image_size=image_size,
    )


def verify_args(raw_data_dir: str, output_dir: str, publish: bool = False) -> None:
    """Verify the command line arguments.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        publish: Whether the dataset will be published to the Hugging
            Face Hub.
    """
    assert os.path.isdir(raw_data_dir), "raw_data_dir must be an existing directory"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(os.path.join(output_dir, OUTPUT_NAME)), (
        "A matching version of the Open Kidney dataset already exists"
    )

    if publish:
        verify_revision_unpublished(OUTPUT_NAME)


def process_open_kidney(
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
    """Prepare the Open Kidney dataset.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        size: The height and width of the test images in pixels.
        publish: Whether to publish the dataset to the HF Hub.
    """
    verify_args(raw_data_dir, output_dir, publish)

    output_path = Path(output_dir) / OUTPUT_NAME

    # Load metadata from CSV
    metadata = pd.read_csv(os.path.join(raw_data_dir, CLINICAL_METADATA_FILE))
    metadata["quality"] = metadata["file_attributes"].apply(
        lambda x: ast.literal_eval(str(x)).get("Quality", None)
    )
    metadata["view"] = metadata["file_attributes"].apply(
        lambda x: ast.literal_eval(str(x)).get("View", None)
    )
    metadata["comments"] = metadata["file_attributes"].apply(
        lambda x: ast.literal_eval(str(x)).get("Comments", None)
    )
    metadata["transplant"] = metadata["comments"].apply(
        lambda x: "transplant" in x.lower()
    )
    metadata["file_id"] = metadata["filename"].map(lambda x: x.split("_")[1])
    metadata.drop(
        columns=["file_attributes", "region_attributes", "region_shape_attributes"],
        inplace=True,
    )
    metadata = metadata.drop_duplicates("file_id")

    print(f"Total # of images: {len(metadata)}")

    train_val_df, test_df = train_test_split(
        metadata,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=metadata["view"],
    )

    train_records = [
        _process_open_kidney_example(row, raw_data_dir)
        for _, row in tqdm(
            train_val_df.iterrows(),
            total=len(train_val_df),
            desc="Processing train split",
        )
    ]
    test_records = [
        _process_open_kidney_example(row, raw_data_dir)
        for _, row in tqdm(
            test_df.iterrows(),
            total=len(test_df),
            desc="Processing test split",
        )
    ]
    test_resized_records = [
        _process_open_kidney_example(row, raw_data_dir, size=size)
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

    dataset.save_to_disk(str(output_path))

    write_ground_truth(test_records, output_path / "test", None)
    write_ground_truth(test_resized_records, output_path / f"test.{size}", size)

    print(f"# of training examples: {len(train_records)}")
    print(f"# of test examples: {len(test_records)}")

    save_version_info(str(output_path), __version__)

    if publish:
        dataset.push_to_hub(
            OUTPUT_NAME,
            commit_message=f"Update the dataset to revision {DATASET_REVISION}.",
            private=True,
        )

        api = HfApi()
        operations = ground_truth_commit_operations(
            output_path,
            {
                "segmentation-capsule": "ground_truth_segmentation_capsule.csv",
                "segmentation-regions": "ground_truth_segmentation_regions.csv",
            },
            size,
        )
        api.create_commit(
            repo_id=OUTPUT_NAME,
            repo_type="dataset",
            operations=operations,
            commit_message=(
                f"Update ground truth CSV files for revision {DATASET_REVISION}."
            ),
        )
