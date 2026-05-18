"""Split the PSFHS dataset into training and test sets using an 8:2
split. The dataset consists of 1358 images from 1124 patients.
Unfortunately, there is no metadata to provide additional information
about the patients or the images. Therefore, we cannot ensure that there
is no patient overlap between the splits.

The resulting dataset has two splits:

    - train: images in their original 256 x 256 px resolution.
    - test: images in their original 256 x 256 px resolution.

Each example has the following features:

    - filename: Original filename without the file extension.
    - image: 3-channel ultrasound image.
    - scan_mask: Single-channel binary scan mask.
    - psfh_mask: Single-channel multi-class PS-FH segmentation mask
      (background=0, pubic symphysis=1, fetal head=2).

One ground-truth CSV file is also created for the test split for use
by the evaluation layer:

    - ground_truth_segmentation.csv

Usage:
    ultrabench process psfhs RAW_DATA_DIR OUTPUT_DIR
"""

import glob
import os
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import SimpleITK as sitk
import skimage
import typer
from datasets import (
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

from .utils import (
    generate_segmentation_ground_truth_from_records,
    save_version_info,
    to_3_channel,
)

__version__ = version("ultrabench")

OUTPUT_NAME = "UltraBench/PSFHS"
DATASET_INFO = DatasetInfo(
    description=(
        "A dataset of intrapartum ultrasound images with pixel-level "
        "segmentation masks for the pubic symphysis and fetal head."
    ),
    citation="""
    @article{chen2024_psfhs,
        title = {{{PSFHS}}: {{Intrapartum Ultrasound Image Dataset}} for
                {{AI-Based Segmentation}} of {{Pubic Symphysis}} and
                {{Fetal Head}}},
        author = {Chen, Gaowen and
                  Bai, Jieyun and
                  Ou, Zhanhong and
                  Lu, Yaosheng and
                  Wang, Huijin},
        year = 2024,
        month = may,
        journal = {Scientific Data},
        volume = {11},
        number = {1},
        pages = {436},
        publisher = {Nature Publishing Group},
        issn = {2052-4463},
        doi = {10.1038/s41597-024-03266-4}
    }
    """,
    homepage="https://zenodo.org/records/10969427",
    license="cc-by-4.0",
    features=Features(
        {
            "filename": Value("string"),
            "image": Image(),
            "scan_mask": Image(),
            "psfh_mask": Image(),
        }
    ),
)


def generate_scan_mask(image: np.ndarray) -> np.ndarray:
    """Generate a scan mask using morphological operations.

    Args:
        image: The input 2D image array.

    Returns:
        A binary mask of the scan region.
    """
    mask = image > 1
    mask = skimage.morphology.remove_small_objects(mask, min_size=64)
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)
    for _ in range(3):
        mask = skimage.morphology.binary_erosion(mask)
    for _ in range(3):
        mask = skimage.morphology.binary_dilation(mask)

    # Take the union of each half reflected onto the other to make it symmetric
    _, width = mask.shape
    left_half = mask[:, : width // 2]
    right_half = mask[:, width // 2 :]
    left_half = np.logical_or(left_half, np.fliplr(right_half))
    right_half = np.logical_or(right_half, np.fliplr(left_half))
    mask = np.hstack((left_half, right_half))

    mask = skimage.morphology.convex_hull_image(mask)
    return mask.astype(np.uint8)


def _process_psfhs_example(image_path: str) -> dict:
    """Process a single image into a record dict.

    Args:
        image_path: The path to the .mha image file.

    Returns:
        A dict with keys filename, image, scan_mask, and psfh_mask,
        with numpy array values.
    """
    filename = os.path.basename(image_path).replace(".mha", "")
    raw = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    gray = raw.mean(axis=0).astype(np.uint8) if raw.ndim == 3 else raw

    scan_mask = generate_scan_mask(gray)
    image = to_3_channel(gray)

    mask_path = image_path.replace("image_mha", "label_mha")
    psfh_mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))

    return {
        "filename": filename,
        "image": image,
        "scan_mask": scan_mask,
        "psfh_mask": psfh_mask,
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
        "A version of the PSFHS dataset already exists"
    )


def process_psfhs(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed dataset")
    ],
    publish: Annotated[
        bool,
        typer.Option(help="Whether to publish the dataset to the Hugging Face Hub"),
    ] = False,
) -> None:
    """Prepare the PSFHS dataset as a Hugging Face DatasetDict.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        publish: Whether to publish the dataset to the Hugging Face Hub.
    """
    verify_args(raw_data_dir, output_dir)

    psfhs_dir = os.path.join(raw_data_dir, "PSFHS")
    image_paths = glob.glob(os.path.join(psfhs_dir, "image_mha", "*.mha"))
    filenames = [os.path.basename(p).replace(".mha", "") for p in image_paths]

    examples = pd.DataFrame({"image_path": image_paths, "filename": filenames})

    train_val_examples, test_examples = train_test_split(
        examples, test_size=0.2, random_state=42, shuffle=True
    )

    train_records = [
        _process_psfhs_example(row["image_path"])
        for _, row in tqdm(
            train_val_examples.iterrows(),
            total=len(train_val_examples),
            desc="Processing train split",
        )
    ]
    test_records = [
        _process_psfhs_example(row["image_path"])
        for _, row in tqdm(
            test_examples.iterrows(),
            total=len(test_examples),
            desc="Processing test split",
        )
    ]

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_records, info=DATASET_INFO),
            "test": Dataset.from_list(test_records, info=DATASET_INFO),
        }
    )

    output_path = Path(output_dir) / OUTPUT_NAME
    dataset.save_to_disk(str(output_path))

    generate_segmentation_ground_truth_from_records(
        test_records,
        mask_key="psfh_mask",
        classes=[1, 2],
        output_path=output_path / "test" / "ground_truth_segmentation.csv",
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
                path_in_repo="test-ground-truth-segmentation.csv",
                path_or_fileobj=output_path / "test" / "ground_truth_segmentation.csv",
            ),
        ]
        api.create_commit(
            repo_id=OUTPUT_NAME,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Update ground truth CSV files for Version {__version__}.",
        )
