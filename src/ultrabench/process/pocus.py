"""Build the POCUS image dataset by extracting the frames from the videos and
copying the images. The dataset is split into training and test sets using an
8:2 split.

Notes:
    - Following the same procedure as the authors of the original paper, we
      sample the videos at a rate of 3 Hz upto a maximum of 30 frames.
    - We include only convex probe ultrasound images in the dataset.
    - Following the same procedure as the authors of the original paper, we do
      not include the viral pneumonia class in the dataset due to the lack of
      examples.
    - The frames extracted from videos are grouped by the video source when
      splitting the dataset into training and test sets to prevent data
      leakage.
    - In total, there are 2102 examples split into 1645 training and 457 test
      examples.

Each example (a single image) is represented as an object in one of two JSON
array files (`train_val.json` or `test.json`). Each object has the following
key/value pairs:

    - source:       The name of the source video or image file.
    - image:        The path to the image file.
    - scan_mask:    The path to the scan mask file.
    - pathology:    The pathology of the mass (regular, pneumonia, or covid).
    - label:        The pathology of the mass encoded as an integer
                    (regular = 0, pneumonia = 1, covid = 2).

Usage:
    ultrabench pocus RAW_DATA_DIR OUTPUT_DIR
"""

import json
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
from sklearn.model_selection import GroupShuffleSplit

from .utils import resize_and_pad, save_version_info, to_3_channel

__version__ = version("ultrabench")

OUTPUT_NAME = "pocus_v{}"
ABBR_TO_LABEL = {
    "reg": "regular",
    "pne": "pneumonia",
    "cov": "covid",
}
LABEL_TO_CLASS = {
    "regular": 0,
    "pneumonia": 1,
    "covid": 2,
}
INPUT_IMAGE_DIR = "data/pocus_images/convex"
INPUT_VIDEO_DIR = "data/pocus_videos/convex"
FRAME_RATE = 3
MAX_FRAMES = 30
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".JPG", ".PNG")
VIDEO_EXTENSIONS = (".mpeg", ".gif", ".mp4", ".m4v", ".avi", ".mov")
VIDEO_INFO_TEMPLATE = "--> {}: {} frames at {:.2f} Hz, {} dimensions"


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
    ), "A matching version of the POCUS dataset already exists"


def process_pocus(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
    size: Annotated[
        int, typer.Option(help="Test image size (height and width in pixels)")
    ] = 256,
) -> None:
    """Prepare the training and test sets for the POCUS dataset.

    Args:
        raw_data_dir: The path to the raw data directory.
        output_dir: The path to the output directory.
        size: The height and width of the test images in pixels.
    """
    verify_args(raw_data_dir, output_dir)

    dataset_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    test_dir_name = f"test_{size}"

    # Collect source-level metadata (no file I/O) to determine the split
    image_dir = os.path.join(raw_data_dir, INPUT_IMAGE_DIR)
    video_dir = os.path.join(raw_data_dir, INPUT_VIDEO_DIR)

    source_records = []
    for abbr, label in ABBR_TO_LABEL.items():
        for file in os.listdir(image_dir):
            if file.endswith(IMAGE_EXTENSIONS) and file.lower().startswith(abbr):
                source_records.append({"source": file, "pathology": label})
        for file in os.listdir(video_dir):
            if file.endswith(VIDEO_EXTENSIONS) and file.lower().startswith(abbr):
                source_records.append({"source": file, "pathology": label})

    sources_df = pd.DataFrame(source_records)

    # Split by source to avoid data leakage between train and test
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(
        splitter.split(X=sources_df, groups=sources_df["source"])
    )
    test_sources = set(sources_df.iloc[test_indices]["source"])

    # Create output directories
    for split_dir_name in ["train_val", test_dir_name]:
        split_dir = os.path.join(dataset_dir, split_dir_name)
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "masks", "scan"), exist_ok=True)

    train_examples = []
    test_examples = []

    # Process image files
    for abbr, label in ABBR_TO_LABEL.items():
        files = [
            p
            for p in os.listdir(image_dir)
            if p.endswith(IMAGE_EXTENSIONS) and p.lower().startswith(abbr)
        ]
        for file in files:
            is_test = file in test_sources
            split_dir = os.path.join(
                dataset_dir, test_dir_name if is_test else "train_val"
            )

            raw_image = io.imread(os.path.join(image_dir, file))
            gray = (
                raw_image.mean(axis=-1).astype(np.uint8)
                if raw_image.ndim == 3
                else raw_image
            )
            mask = generate_scan_mask(gray)
            image = to_3_channel(raw_image)
            if is_test:
                image = resize_and_pad(image, size, order=1)
                mask = resize_and_pad(mask, size, order=0)

            stem = Path(file).stem
            io.imsave(os.path.join(split_dir, "images", f"{stem}.png"), image)
            io.imsave(
                os.path.join(split_dir, "masks", "scan", f"{stem}.png"),
                mask,
                check_contrast=False,
            )

            example = {
                "source": file,
                "image": f"images/{stem}.png",
                "scan_mask": f"masks/scan/{stem}.png",
                "pathology": label,
                "label": LABEL_TO_CLASS[label],
            }
            (test_examples if is_test else train_examples).append(example)

    # Process video files
    for abbr, label in ABBR_TO_LABEL.items():
        files = [
            p
            for p in os.listdir(video_dir)
            if p.endswith(VIDEO_EXTENSIONS) and p.lower().startswith(abbr)
        ]
        for file in files:
            is_test = file in test_sources
            split_dir = os.path.join(
                dataset_dir, test_dir_name if is_test else "train_val"
            )

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
                gray = (
                    frame.mean(axis=-1).astype(np.uint8) if frame.ndim == 3 else frame
                )
                mask = generate_scan_mask(gray)
                image = to_3_channel(frame)
                if is_test:
                    image = resize_and_pad(image, size, order=1)
                    mask = resize_and_pad(mask, size, order=0)

                stem = f"{Path(file).stem}_{i}"
                io.imsave(os.path.join(split_dir, "images", f"{stem}.png"), image)
                io.imsave(
                    os.path.join(split_dir, "masks", "scan", f"{stem}.png"),
                    mask,
                    check_contrast=False,
                )

                example = {
                    "source": file,
                    "image": f"images/{stem}.png",
                    "scan_mask": f"masks/scan/{stem}.png",
                    "pathology": label,
                    "label": LABEL_TO_CLASS[label],
                }
                (test_examples if is_test else train_examples).append(example)

    print(f"Number of examples: {len(train_examples) + len(test_examples)}")
    print(f"Training examples: {len(train_examples)}")
    print(f"Test examples: {len(test_examples)}")

    # Save the metadata JSON files inside their split directories
    train_dir = os.path.join(dataset_dir, "train_val")
    test_dir = os.path.join(dataset_dir, test_dir_name)
    with open(os.path.join(train_dir, "train_val.json"), "w") as f:
        json.dump(train_examples, f, indent=4)
    with open(os.path.join(test_dir, "test.json"), "w") as f:
        json.dump(test_examples, f, indent=4)

    save_version_info(dataset_dir, __version__)
