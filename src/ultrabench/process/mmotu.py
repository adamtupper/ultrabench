"""Prepare the training and test sets for the MMOTU-2D dataset. The
dataset is already pre-split into training and test data, ensuring
that there is no patient overlap between the splits.

The resulting dataset has three splits:

    - train: original-size images.
    - test: original-size images.
    - test.{size}: same examples as test, resized so the longest edge
      measures {size} px and then zero-padded to {size} x {size} px.

Each example has the following features:

    - filename: Original filename without the file extension.
    - image: 3-channel ultrasound image.
    - scan_mask: Single-channel binary scan mask.
    - tumor_mask: Single-channel binary tumor mask.
    - tumor_type: ClassLabel encoding the tumor type.

Two ground-truth CSV files are also created in each test split
directory for use by the evaluation layer, with the segmentation masks
encoded at the resolution of that split:

    - ground_truth_classification.csv
    - ground_truth_segmentation.csv

Usage:
    ultrabench process mmotu RAW_DATA_DIR OUTPUT_DIR
"""

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
from tqdm.auto import tqdm
from typing_extensions import Annotated

from ultrabench.datasets.utils import partition_by_manifest

from .utils import (
    DATASET_REVISION,
    IMAGE_SIZE,
    BuildResult,
    generate_classification_ground_truth_from_records,
    generate_segmentation_ground_truth_from_records,
    ground_truth_commit_operations,
    resize_and_pad,
    save_version_info,
    to_3_channel,
    verify_revision_unpublished,
    write_test_ground_truth,
)

__version__ = version("ultrabench")

OUTPUT_NAME = "UltraBench/MMOTU-2D"
DATASET_INFO = DatasetInfo(
    description=(
        "A dataset of ovarian tumor ultrasound images for ovarian "
        "tumor segmentation and classification."
    ),
    citation="""
    @article{lyu2026_mmotu,
        title = {Unsupervised cross-domain semantic segmentation on
                 multi-modality ovarian tumor ultrasound data},
        journal = {Pattern Recognition},
        volume = {171},
        pages = {112311},
        year = {2026},
        issn = {0031-3203},
        doi = {https://doi.org/10.1016/j.patcog.2025.112311},
        url = {https://www.sciencedirect.com/science/article/pii/S0031320325009720},
        author = {Shuchang Lyu and
                  Qi Zhao and
                  Wenpei Bai and
                  Linghan Cai and
                  Guangliang Cheng and
                  Guangxia Cui and
                  Min Yang and
                  Lijiang Chen and
                  Huiyu Zhou}
    }
    """,
    homepage="https://github.com/cv516Buaa/MMOTU_DS2Net",
    license="unknown",
    features=Features(
        {
            "filename": Value("string"),
            "image": Image(),
            "scan_mask": Image(),
            "tumor_mask": Image(),
            "tumor_type": ClassLabel(
                names=[
                    "chocolate_cyst",
                    "serous_cystadenoma",
                    "teratoma",
                    "theca_cell_tumour",
                    "simple_cyst",
                    "normal_ovary",
                    "mucinous_cystadenoma",
                    "high_grade_serous_cystadenocarcinoma",
                ]
            ),
        }
    ),
)


def generate_scan_mask(image: np.ndarray) -> np.ndarray:
    """Generate a scan mask using morphological operations.

    Args:
        image: The input 2D grayscale image array.

    Returns:
        A binary mask of the scan region.
    """
    mask = image > 0
    for _ in range(10):
        mask = skimage.morphology.binary_erosion(mask)
    mask = skimage.morphology.remove_small_objects(mask, min_size=5000)
    for _ in range(10):
        mask = skimage.morphology.binary_dilation(mask)
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=1000)

    # Reflect the smaller half to remove annotations and make scan symmetric
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


def read_split_names(dataset_dir: str) -> list[str]:
    """Read every filename the raw dataset ships, across both splits.

    Args:
        dataset_dir: The path to the dataset directory.

    Returns:
        The filenames, without their extension.
    """
    names = []
    for prefix in ("train", "val"):
        with open(os.path.join(dataset_dir, "OTU_2d", f"{prefix}.txt"), "r") as file:
            names.extend(file.read().splitlines())

    return names


def process_examples(
    dataset_dir: str,
    filenames: list[str],
    desc: str,
    size: int | None = None,
) -> list[dict]:
    """Process the named examples into record dicts.

    Args:
        dataset_dir: The path to the dataset directory.
        filenames: The examples to process, without their extension.
        desc: The label to show on the progress bar.
        size: If provided, resize and pad images and masks to
            size x size pixels.

    Returns:
        A list of example dictionaries with numpy array values.
    """
    label_frames = [
        pd.read_csv(
            os.path.join(dataset_dir, "OTU_2d", f"{prefix}_cls.txt"),
            delimiter="  ",
            names=["image", "label"],
            engine="python",
        )
        for prefix in ("train", "val")
    ]
    label_df = pd.concat(label_frames, ignore_index=True)
    label_df["image"] = label_df["image"].str.removesuffix(".JPG")

    examples = []
    for filename in tqdm(filenames, desc=desc):
        image_path = os.path.join(dataset_dir, "OTU_2d", "images", f"{filename}.JPG")
        binary_mask_path = os.path.join(
            dataset_dir, "OTU_2d", "annotations", f"{filename}_binary.PNG"
        )
        tumor_type = int(label_df.loc[label_df["image"] == filename, "label"].values[0])

        raw_image = io.imread(image_path)
        gray = (
            raw_image.mean(axis=-1).astype(np.uint8)
            if raw_image.ndim == 3
            else raw_image
        )
        scan_mask = generate_scan_mask(gray)
        tumor_mask = io.imread(binary_mask_path)
        image = to_3_channel(raw_image)

        if size is not None:
            image = resize_and_pad(image, size, order=1)
            tumor_mask = resize_and_pad(tumor_mask, size, order=0)
            scan_mask = resize_and_pad(scan_mask, size, order=0)

        examples.append(
            {
                "filename": filename,
                "image": image,
                "scan_mask": scan_mask,
                "tumor_mask": tumor_mask,
                "tumor_type": tumor_type,
            }
        )

    return examples


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
    generate_classification_ground_truth_from_records(
        [{"filename": r["filename"], "label": r["tumor_type"]} for r in records],
        gt_dir / "ground_truth_classification.csv",
    )
    generate_segmentation_ground_truth_from_records(
        records,
        mask_key="tumor_mask",
        classes=[1],
        output_path=gt_dir / "ground_truth_segmentation.csv",
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
        "A version of the MMOTU dataset already exists"
    )

    if publish:
        verify_revision_unpublished(OUTPUT_NAME)


def build_mmotu_dataset(
    raw_data_dir: str | Path, size: int = IMAGE_SIZE
) -> BuildResult:
    """Build the MMOTU-2D splits from the raw data.

    The splits come from the tracked manifests rather than from the
    dataset's own split files, so that processing the raw data
    reproduces the published splits exactly even if the raw files
    change.

    Args:
        raw_data_dir: The path to the raw data directory.
        size: The height and width of the resized test images.

    Returns:
        The processed splits and the test records the ground truth is
        built from.

    Raises:
        ValueError: If the raw data is missing an example the manifests
            list.
    """
    raw_data_dir = str(raw_data_dir)

    train_names, test_names = partition_by_manifest(
        read_split_names(raw_data_dir), "MMOTU", key=lambda name: name
    )

    train_records = process_examples(
        raw_data_dir, train_names, "Processing train split"
    )
    test_records = process_examples(raw_data_dir, test_names, "Processing test split")
    test_resized_records = process_examples(
        raw_data_dir, test_names, f"Processing test.{size} split", size=size
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_records, info=DATASET_INFO),
            "test": Dataset.from_list(test_records, info=DATASET_INFO),
            f"test.{size}": Dataset.from_list(test_resized_records, info=DATASET_INFO),
        }
    )

    return BuildResult(dataset, test_records, test_resized_records)


def process_mmotu(
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
    """Prepare the MMOTU-2D dataset as a Hugging Face DatasetDict.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        size: The height and width of the resized test images in pixels.
        publish: Whether to publish the dataset to the Hugging Face Hub.
    """
    verify_args(raw_data_dir, output_dir, publish)

    output_path = Path(output_dir) / OUTPUT_NAME
    result = build_mmotu_dataset(raw_data_dir, size=size)
    result.dataset.save_to_disk(str(output_path))
    write_test_ground_truth(result, output_path, size, write_ground_truth)

    save_version_info(str(output_path), __version__)

    if publish:
        result.dataset.push_to_hub(
            OUTPUT_NAME,
            commit_message=f"Update the dataset to revision {DATASET_REVISION}.",
            private=True,
        )

        api = HfApi()
        operations = ground_truth_commit_operations(
            output_path,
            {
                "classification": "ground_truth_classification.csv",
                "segmentation": "ground_truth_segmentation.csv",
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
