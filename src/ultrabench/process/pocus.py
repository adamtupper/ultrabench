"""Prepare the POCUS dataset as a Hugging Face DatasetDict by extracting
frames from videos and copying still images. The dataset is split into
training and test sets using an 8:2 split.

Notes:
    - Following the same procedure as the authors of the original paper,
      we sample the videos at a rate of 3 Hz up to a maximum of 30
      frames.
    - We include only convex probe ultrasound images in the dataset.
    - Following the same procedure as the authors of the original paper,
      we do not include the viral pneumonia class in the dataset due to
      the lack of examples.
    - The frames extracted from videos are grouped by the video source
      when splitting the dataset into training and test sets to prevent
      data leakage.

The resulting dataset has three splits:

    - train: original-size images.
    - test: original-size images.
    - test.{size}: same examples as test, resized so the longest edge
      measures {size} px and then zero-padded to {size} x {size} px.

Each example has the following features:

    - filename: Image filename stem (or video stem with frame index).
    - label: ClassLabel encoding the lung pathology.
    - image: 3-channel ultrasound image.
    - scan_mask: Single-channel binary scan mask.

A ground-truth CSV file is also created for the test split:

    - ground_truth_classification.csv

Usage:
    ultrabench pocus RAW_DATA_DIR OUTPUT_DIR
"""

import os
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import cv2
import imageio
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
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

from .utils import (
    generate_classification_ground_truth_from_records,
    resize_and_pad,
    save_version_info,
    to_3_channel,
)

__version__ = version("ultrabench")

OUTPUT_NAME = "UltraBench/POCUS"
ABBR_TO_LABEL = {
    "reg": "normal",
    "pne": "pneumonia",
    "cov": "covid",
}
LABEL_TO_CLASS = {
    "normal": 0,
    "pneumonia": 1,
    "covid": 2,
}
INPUT_IMAGE_DIR = "covid19_ultrasound/data/pocus_images/convex"
INPUT_VIDEO_DIR = "covid19_ultrasound/data/pocus_videos/convex"
FRAME_RATE = 3
MAX_FRAMES = 30
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".JPG", ".PNG")
VIDEO_EXTENSIONS = (".mpeg", ".gif", ".mp4", ".m4v", ".avi", ".mov")
VIDEO_INFO_TEMPLATE = "--> {}: {} frames at {:.2f} Hz, {} dimensions"
DATASET_INFO = DatasetInfo(
    description=("Convex probe lung ultrasound images for COVID-19 classification."),
    citation="""
    @article{born2021pocus,
        author  = {Born, J. and Wiedemann, N. and Cossio, M. and
                   Buhre, C. and Br{\\"{a}}ndle, G. and Leidermann, K.
                   and Goulet, J. and Aujayeb, A. and Moor, M. and
                   Rieck, B. and Borgwardt, K.},
        title   = {Accelerating Detection of Lung Pathologies with
                   Explainable Ultrasound Image Analysis},
        journal = {Applied Sciences},
        volume  = {11},
        number  = {2},
        pages   = {672},
        year    = {2021},
        doi     = {10.3390/app11020672},
    }
    """,
    homepage="https://github.com/jannisborn/covid19_ultrasound",
    features=Features(
        {
            "filename": Value("string"),
            "source": Value("string"),
            "label": ClassLabel(names=["normal", "pneumonia", "covid"]),
            "image": Image(),
            "scan_mask": Image(),
        }
    ),
    license="cc",
)


def generate_scan_mask(image: np.ndarray) -> np.ndarray:
    """Generate a scan mask using morphological operations.

    Args:
        image: The input 2D grayscale image array.

    Returns:
        A binary mask of the scan region.
    """
    threshold = skimage.filters.threshold_local(image, block_size=3)
    mask = image > threshold
    mask = mask > 0
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=500)
    mask = skimage.morphology.remove_small_objects(mask, min_size=32)

    # Reflect the larger half to compensate for shadows
    _, width = mask.shape
    left_half = mask[:, : width // 2]
    right_half = mask[:, width // 2 :]
    if np.sum(left_half) > np.sum(right_half):
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


def extract_frames_ffmpeg(video_path: str) -> tuple[np.ndarray, float]:
    """Extract the frames from the video using ffmpeg.

    Args:
        video_path: The path to the video file.

    Returns:
        A tuple containing the frames array and the frame rate.
    """
    video = imageio.get_reader(video_path, "ffmpeg")
    frame_rate = video.get_meta_data()["fps"]

    frames = []
    frame = video.get_next_data()
    while frame is not None:
        frames.append(frame)
        try:
            frame = video.get_next_data()
        except IndexError:
            frame = None
    frames = np.array(frames)

    print(
        VIDEO_INFO_TEMPLATE.format(
            os.path.basename(video_path), len(frames), frame_rate, frames[0].shape
        )
    )

    return frames, frame_rate


def extract_frames_opencv(video_path: str) -> tuple[np.ndarray, float]:
    """Extract the frames from the video using OpenCV.

    Args:
        video_path: The path to the video file.

    Returns:
        A tuple containing the frames array and the frame rate.
    """
    video = cv2.VideoCapture(video_path)
    frame_rate = video.get(cv2.CAP_PROP_FPS)

    frames = []
    ret, frame = video.read()
    while ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        ret, frame = video.read()
    frames = np.array(frames)

    print(
        VIDEO_INFO_TEMPLATE.format(
            os.path.basename(video_path), len(frames), frame_rate, frames[0].shape
        )
    )

    video.release()

    return frames, frame_rate


def _process_pocus_image(
    raw_image: np.ndarray, size: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Process a single raw image into an image/mask pair.

    Args:
        raw_image: The raw image array.
        size: If provided, resize and pad arrays to size x size pixels.

    Returns:
        A tuple of (image, scan_mask) numpy arrays.
    """
    gray = (
        raw_image.mean(axis=-1).astype(np.uint8) if raw_image.ndim == 3 else raw_image
    )
    mask = generate_scan_mask(gray)
    image = to_3_channel(raw_image)
    if size is not None:
        image = resize_and_pad(image, size, order=1)
        mask = resize_and_pad(mask, size, order=0)
    return image, mask


def verify_args(raw_data_dir: str, output_dir: str) -> None:
    """Verify the command line arguments.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
    """
    assert os.path.isdir(raw_data_dir), "raw_data_dir must be an existing directory"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(os.path.join(output_dir, OUTPUT_NAME)), (
        "A matching version of the POCUS dataset already exists"
    )


def process_pocus(
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
    """Prepare the training and test sets for the POCUS dataset.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        size: The height and width of the test images in pixels.
        publish: Whether to publish the dataset to the HF Hub.
    """
    verify_args(raw_data_dir, output_dir)

    output_path = Path(output_dir) / OUTPUT_NAME

    # Collect source-level metadata (no file I/O) to determine the split
    image_dir = os.path.join(raw_data_dir, INPUT_IMAGE_DIR)
    video_dir = os.path.join(raw_data_dir, INPUT_VIDEO_DIR)

    source_records = []
    for abbr, label in ABBR_TO_LABEL.items():
        for file in os.listdir(image_dir):
            if file.endswith(IMAGE_EXTENSIONS) and file.lower().startswith(abbr):
                source_records.append({"source": file, "label": label})
        for file in os.listdir(video_dir):
            if file.endswith(VIDEO_EXTENSIONS) and file.lower().startswith(abbr):
                source_records.append({"source": file, "label": label})

    sources_df = pd.DataFrame(source_records)

    # Split by source to avoid data leakage between train and test
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(
        splitter.split(X=sources_df, groups=sources_df["source"])
    )
    test_sources = set(sources_df.iloc[test_indices]["source"])

    train_records: list[dict] = []
    test_records: list[dict] = []
    test_resized_records: list[dict] = []

    # Process image files
    for abbr, label in tqdm(ABBR_TO_LABEL.items(), desc="Processing image files"):
        files = [
            p
            for p in os.listdir(image_dir)
            if p.endswith(IMAGE_EXTENSIONS) and p.lower().startswith(abbr)
        ]
        for file in files:
            is_test = file in test_sources
            raw_image = io.imread(os.path.join(image_dir, file))
            image, mask = _process_pocus_image(raw_image)
            stem = Path(file).stem

            record = {
                "filename": stem,
                "source": file,
                "label": LABEL_TO_CLASS[label],
                "image": image,
                "scan_mask": mask,
            }
            if is_test:
                test_records.append(record)
                image_r, mask_r = _process_pocus_image(raw_image, size=size)
                test_resized_records.append(
                    {
                        "filename": stem,
                        "source": file,
                        "label": LABEL_TO_CLASS[label],
                        "image": image_r,
                        "scan_mask": mask_r,
                    }
                )
            else:
                train_records.append(record)

    # Process video files
    for abbr, label in tqdm(ABBR_TO_LABEL.items(), desc="Processing video files"):
        files = [
            p
            for p in os.listdir(video_dir)
            if p.endswith(VIDEO_EXTENSIONS) and p.lower().startswith(abbr)
        ]
        for file in files:
            is_test = file in test_sources

            if file.endswith(".gif"):
                frames, frame_rate = extract_frames_opencv(
                    os.path.join(video_dir, file)
                )
            else:
                frames, frame_rate = extract_frames_ffmpeg(
                    os.path.join(video_dir, file)
                )

            every_nth_frame = int(frame_rate / FRAME_RATE)
            frames = frames[1::every_nth_frame][:MAX_FRAMES]

            for i, frame in enumerate(frames):
                image, mask = _process_pocus_image(frame)
                stem = f"{Path(file).stem}_{i}"

                record = {
                    "filename": stem,
                    "source": file,
                    "label": LABEL_TO_CLASS[label],
                    "image": image,
                    "scan_mask": mask,
                }
                if is_test:
                    test_records.append(record)
                    image_r, mask_r = _process_pocus_image(frame, size=size)
                    test_resized_records.append(
                        {
                            "filename": stem,
                            "source": file,
                            "label": LABEL_TO_CLASS[label],
                            "image": image_r,
                            "scan_mask": mask_r,
                        }
                    )
                else:
                    train_records.append(record)

    print(f"Number of examples: {len(train_records) + len(test_records)}")
    print(f"Training examples: {len(train_records)}")
    print(f"Test examples: {len(test_records)}")

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_records, info=DATASET_INFO),
            "test": Dataset.from_list(test_records, info=DATASET_INFO),
            f"test.{size}": Dataset.from_list(test_resized_records, info=DATASET_INFO),
        }
    )

    dataset.save_to_disk(str(output_path))

    generate_classification_ground_truth_from_records(
        test_records,
        output_path / f"test.{size}" / "ground_truth_classification.csv",
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
        ]
        api.create_commit(
            repo_id=OUTPUT_NAME,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Update ground truth CSV files for Version {__version__}.",
        )
