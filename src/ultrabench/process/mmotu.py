"""Prepare the training and test sets for the MMOTU dataset. The dataset is
already pre-split into training and test data, ensuring that there is no
patient overlap between these splits.

Each example (a single image) is represented as an object in one of two JSON
array files (`train_val.json` or `test.json`). Each object has the following
key/value pairs:

    - image:                    The path to the image file.
    - tumor_mask_binary:        The path to the binary mask file (tumor
                                segmentation).
    - tumor_mask_multiclass:    The path to the multi-class mask file (tumor
                                segmentation by type).
    - scan_mask:                The path to the scan mask file (scan
                                segmentation).
    - label:                    The class label of the image.


Usage:
    ultrabench mmotu RAW_DATA_DIR OUTPUT_DIR
"""

import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import skimage
import skimage.io as io
import typer
from typing_extensions import Annotated

from .utils import (
    generate_classification_ground_truth,
    generate_segmentation_ground_truth,
    resize_and_pad,
    save_version_info,
    to_3_channel,
)

__version__ = version("ultrabench")

OUTPUT_NAME = "mmotu_v{}"
PIXEL_TO_CLASS = {
    (0, 0, 0): 0,  # Background
    (64, 0, 0): 1,  # Chocolate cyst
    (0, 64, 0): 2,  # Serious cystadenoma
    (0, 0, 64): 3,  # Teratoma
    (64, 0, 64): 4,  # Theca cell tumor
    (64, 64, 0): 5,  # Simple cyst
    (64, 64, 64): 6,  # Normal ovary
    (0, 128, 0): 7,  # Mucinous cystadenoma
    (0, 0, 128): 8,  # High grade serous
}


def generate_scan_mask(image: np.ndarray) -> np.ndarray:
    """Generate a scan mask using morphological operations.

    Args:
        image: The input 2D grayscale image array.

    Returns:
        A binary mask of the scan region.
    """
    mask = image > 0
    for i in range(10):
        mask = skimage.morphology.binary_erosion(mask)
    mask = skimage.morphology.remove_small_objects(mask, min_size=5000)
    for i in range(10):
        mask = skimage.morphology.binary_dilation(mask)
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)

    # Reflect the smaller half to remove annotations and make the scan symmetric
    _, width = mask.shape
    left_half = mask[:, : width // 2]
    right_half = mask[:, width // 2 :]
    if np.sum(left_half) < np.sum(right_half):
        right_half = np.fliplr(left_half)
    else:
        left_half = np.fliplr(right_half)
    mask = np.hstack((left_half, right_half))

    if mask.shape[1] < image.shape[1]:
        mask = np.hstack((mask, np.zeros((mask.shape[0], 1))))
    elif mask.shape[1] > image.shape[1]:
        mask = mask[:, :-1]

    mask = skimage.morphology.convex_hull_image(mask)
    return mask.astype(np.uint8)


def process_examples(
    dataset_dir: str, split_dir: str, split: str, size: int | None = None
) -> list[dict]:
    """Process the examples for a given split.

    Args:
        dataset_dir: The path to the dataset directory.
        split_dir: The path to the split output directory (train/ or test_{size}/).
        split: The dataset split (train or test).
        size: If provided, resize and pad images and masks to size×size pixels.

    Returns:
        A list of example dictionaries.
    """
    prefix = "train" if split == "train" else "val"

    with open(os.path.join(dataset_dir, "OTU_2d", f"{prefix}.txt"), "r") as file:
        filenames = file.read().splitlines()

    label_df = pd.read_csv(
        os.path.join(dataset_dir, "OTU_2d", f"{prefix}_cls.txt"),
        delimiter="  ",
        names=["image", "label"],
        engine="python",
    )
    label_df["image"] = label_df["image"].str.removesuffix(".JPG")

    examples = []
    for filename in filenames:
        image_path = os.path.join(dataset_dir, "OTU_2d", "images", f"{filename}.JPG")
        binary_mask_path = os.path.join(
            dataset_dir, "OTU_2d", "annotations", f"{filename}_binary.PNG"
        )
        multiclass_mask_path = os.path.join(
            dataset_dir, "OTU_2d", "annotations", f"{filename}.PNG"
        )
        label = int(label_df.loc[label_df["image"] == filename, "label"].values[0])

        raw_image = io.imread(image_path)
        # Use 2D grayscale for scan mask generation
        gray = (
            raw_image.mean(axis=-1).astype(np.uint8)
            if raw_image.ndim == 3
            else raw_image
        )
        scan_mask = generate_scan_mask(gray)

        # Load and convert the binary and multiclass masks
        binary_mask = io.imread(binary_mask_path)
        multiclass_mask = io.imread(multiclass_mask_path)
        multiclass_mask = np.apply_along_axis(
            lambda x: PIXEL_TO_CLASS[tuple(x)], axis=-1, arr=multiclass_mask
        ).astype(np.uint8)

        image = to_3_channel(raw_image)
        if size is not None:
            image = resize_and_pad(image, size, order=1)
            binary_mask = resize_and_pad(binary_mask, size, order=0)
            multiclass_mask = resize_and_pad(multiclass_mask, size, order=0)
            scan_mask = resize_and_pad(scan_mask, size, order=0)

        io.imsave(
            os.path.join(split_dir, "images", f"{filename}.png"),
            image,
            check_contrast=False,
        )
        io.imsave(
            os.path.join(split_dir, "masks", "tumor", f"{filename}_binary.png"),
            binary_mask,
            check_contrast=False,
        )
        io.imsave(
            os.path.join(split_dir, "masks", "tumor", f"{filename}_multiclass.png"),
            multiclass_mask,
            check_contrast=False,
        )
        io.imsave(
            os.path.join(split_dir, "masks", "scan", f"{filename}.png"),
            scan_mask,
            check_contrast=False,
        )

        examples.append(
            {
                "image": f"images/{filename}.png",
                "tumor_mask_binary": f"masks/tumor/{filename}_binary.png",
                "tumor_mask_multiclass": f"masks/tumor/{filename}_multiclass.png",
                "scan_mask": f"masks/scan/{filename}.png",
                "label": label,
            }
        )

    return examples


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
    ), "A matching version of the MMOTU dataset already exists"


def process_mmotu(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
    size: Annotated[
        int, typer.Option(help="Test image size (height and width in pixels)")
    ] = 256,
) -> None:
    """Prepare the training and test sets for the MMOTU dataset.

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

    for split_dir in [train_dir, test_dir]:
        os.makedirs(os.path.join(split_dir, "images"))
        os.makedirs(os.path.join(split_dir, "masks", "tumor"))
        os.makedirs(os.path.join(split_dir, "masks", "scan"))

    test_examples = process_examples(raw_data_dir, test_dir, "test", size=size)
    with open(os.path.join(test_dir, "test.json"), "w") as file:
        json.dump(test_examples, file, indent=4)

    generate_classification_ground_truth(
        test_examples, Path(test_dir) / "ground_truth_classification.csv"
    )
    generate_segmentation_ground_truth(
        test_examples,
        mask_key="tumor_mask_binary",
        classes=[1],
        output_path=Path(test_dir) / "ground_truth_segmentation_binary.csv",
    )
    generate_segmentation_ground_truth(
        test_examples,
        mask_key="tumor_mask_multiclass",
        classes=[1, 2, 3, 4, 5, 6, 7, 8],
        output_path=Path(test_dir) / "ground_truth_segmentation_multiclass.csv",
    )

    print(f"Number of test examples: {len(test_examples)}")

    train_val_examples = process_examples(raw_data_dir, train_dir, "train")
    with open(os.path.join(train_dir, "train_val.json"), "w") as file:
        json.dump(train_val_examples, file, indent=4)
    print(f"Number of train_val examples: {len(train_val_examples)}")

    save_version_info(dataset_dir, __version__)
