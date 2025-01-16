"""Prepare the training, validation, and test sets for the MMOTU dataset. The dataset is
already pre-split into training and test data, ensuring that there is no patient overlap
between these splits. However, no patient data is retained and therefore we cannot
ensure that there is not any patient overlap when splitting the training set into
training and validation data. We use a 8:2 split to separate the training and validation
sets.

Each example (a single image) is represented as an object in one of three JSON array
files (`train.json`, `validation.json`, or `test.json`). Each object has the following
key/value pairs:

    - image:                    The path to the image file.
    - tumor_mask_binary:        The path to the binary mask file (tumor segmentation).
    - tumor_mask_multiclass:    The path to the multi-class mask file (tumor
                                segmentation by type).
    - scan_mask:                The path to the scan mask file (scan segmentation).
    - label:                    The class label of the image.


Usage:
    ultrabench mmotu RAW_DATA_DIR OUTPUT_DIR
"""

import json
import os
import shutil
from importlib.metadata import version

import numpy as np
import pandas as pd
import skimage
import skimage.io as io
import typer
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated

from .utils import save_version_info

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


def generate_scan_mask(image: np.ndarray):
    # Threshold the image
    mask = image > 0

    # Erode the mask
    for i in range(10):
        mask = skimage.morphology.binary_erosion(mask)

    # Remove small objects
    mask = skimage.morphology.remove_small_objects(mask, min_size=5000)

    # Dilate the mask
    for i in range(10):
        mask = skimage.morphology.binary_dilation(mask)

    # Fill small holes
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)

    # Reflect the smaller half of the mask to remove annoations fill and make the scan
    # symmetric
    _, width = mask.shape
    left_half = mask[:, : width // 2]
    right_half = mask[:, width // 2 :]
    left_sum = np.sum(left_half)
    right_sum = np.sum(right_half)

    # Determine which half has the smaller sum and reflect it
    if left_sum < right_sum:
        right_half = np.fliplr(left_half)
    else:
        left_half = np.fliplr(right_half)

    # Merge the two halves back together
    mask = np.hstack((left_half, right_half))

    if mask.shape[1] < image.shape[1]:
        # Pad the mask by adding a single column of zeros to the right
        mask = np.hstack((mask, np.zeros((mask.shape[0], 1))))
    elif mask.shape[1] > image.shape[1]:
        # Crop the mask by removing the rightmost column
        mask = mask[:, :-1]

    # Extract convex hull of the mask
    mask = skimage.morphology.convex_hull_image(mask)

    return mask.astype(np.uint8)


def process_examples(dataset_dir, output_dir, split):
    # Read the text file containing the image file names
    prefix = "train" if split == "train" else "val"

    # Load the filenames
    with open(os.path.join(dataset_dir, "OTU_2d", f"{prefix}.txt"), "r") as file:
        filenames = file.read().splitlines()

    # Load labels
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

        # Copy the binary masks to the output directory
        shutil.copyfile(
            binary_mask_path,
            os.path.join(output_dir, "masks", "tumor", f"{filename}_binary.png"),
        )

        # Load the image
        image = io.imread(image_path)

        # Convert the image to a single channel
        if image.ndim == 3:
            image = image.mean(axis=-1).astype(np.uint8)

        # Save the image as a PNG file
        io.imsave(os.path.join(output_dir, "images", f"{filename}.png"), image)

        # Convert the multi-class masks Save the multi-class masks as PNG files
        multiclass_mask = io.imread(multiclass_mask_path)
        multiclass_mask = np.apply_along_axis(
            lambda x: PIXEL_TO_CLASS[tuple(x)], axis=-1, arr=multiclass_mask
        ).astype(np.uint8)
        io.imsave(
            os.path.join(output_dir, "masks", "tumor", f"{filename}_multiclass.png"),
            multiclass_mask,
            check_contrast=False,
        )

        # Generate the scan mask
        scan_mask = generate_scan_mask(image)
        io.imsave(
            os.path.join(output_dir, "masks", "scan", f"{filename}.png"),
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


def verify_args(raw_data_dir, output_dir):
    assert os.path.isdir(raw_data_dir), "raw_data_dir must be an existing directory"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(
        os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    ), "A matching version of the MMOTU dataset already exists"


def mmotu(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
):
    """Prepare the training and test sets for the MMOTU dataset."""
    verify_args(raw_data_dir, output_dir)

    output_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    os.makedirs(os.path.join(output_dir, "images"))
    os.makedirs(os.path.join(output_dir, "masks", "tumor"))
    os.makedirs(os.path.join(output_dir, "masks", "scan"))

    # Process the test set
    test_examples = process_examples(raw_data_dir, output_dir, "test")
    with open(os.path.join(output_dir, "test.json"), "w") as file:
        json.dump(test_examples, file, indent=4)

    # Divide the training set into training and validation splits
    examples = process_examples(raw_data_dir, output_dir, "train")
    train_examples, val_examples = train_test_split(
        examples,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=[x["label"] for x in examples],
    )

    for split, examples in zip(["train", "validation"], [train_examples, val_examples]):
        with open(os.path.join(output_dir, f"{split}.json"), "w") as file:
            json.dump(examples, file, indent=4)

    save_version_info(output_dir, __version__)
