"""Build the POCUS image dataset by extracting the frames from the videos and copying
the images. The dataset is split into training, validation, and test sets using a 7:1:2
split.

Notes:
    - Following the same procedure as the authors of the original paper, we sample the
      videos at a rate of 3 Hz upto a maximum of 30 frames.
    - We include only convex probe ultrasound images in the dataset.
    - Following the same procedure as the authors of the original paper, we do not
      include the viral pneumonia class in the dataset due to the lack of examples.
    - The frames extracted from videos are grouped by the video source when splitting
      the dataset into training, validation, and test sets to prevent data leakage.
    - In total, there are 2102 examples split into 1468 training, 177 validation, and
      457 test examples.

Each example (a single image) is represented as an object in one of three JSON array
files (`train.json`, `validation.json`, or `test.json`). Each object has the following
key/value pairs:

    - source:       The name of the source video or image file.
    - image:        The path to the image file.
    - scan_mask:    The path to the scan mask file.
    - pathology:    The pathology of the mass (regular, pneumonia, or covid).
    - label:        The pathology of the mass encoded as an integer (regular = 0,
                    pneumonia = 1, covid = 2).

Usage:
    ultrabench pocus RAW_DATA_DIR OUTPUT_DIR
"""

import argparse
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
import typer
from sklearn.model_selection import GroupShuffleSplit

from .utils import save_version_info

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


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(prog="prepare_pocus.py")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="The path to the original dataset directory",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory in which to save the processed dataset",
        required=True,
    )
    parser.add_argument(
        "--version",
        type=int,
        help="The version number to assign the processed dataset",
        required=True,
    )

    args = parser.parse_args()
    assert os.path.isdir(args.dataset_dir), "dataset_dir must be an existing directory"
    assert os.path.exists(args.output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(
        os.path.join(args.output_dir, OUTPUT_NAME.format(args.version))
    ), "a version of the dataset with this version number already exists"

    return args


def generate_scan_mask(image: np.ndarray):
    # Convert the image to single-channel grayscale
    image = image.mean(axis=-1).astype(np.uint8) if image.ndim == 3 else image

    # Dynamic/adaptive thresholding to separate foreground
    threshold = skimage.filters.threshold_local(image, block_size=3)
    mask = image > threshold
    mask = mask > 0

    # Fill small holes
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=500)

    # Remove small objects
    mask = skimage.morphology.remove_small_objects(mask, min_size=32)

    # Reflect the larger half of the mask to compensate for shadows
    _, width = mask.shape
    left_half = mask[:, : width // 2]
    right_half = mask[:, width // 2 :]
    left_sum = np.sum(left_half)
    right_sum = np.sum(right_half)

    # Determine which half has the smaller sum and reflect it
    if left_sum > right_sum:
        right_half = np.fliplr(left_half)
    else:
        left_half = np.fliplr(right_half)

    # Merge the two halves back together (pad/crop if image width is odd)
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


def image_filter(path, prefix):
    """Filter the images in the directory."""
    return path.endswith(IMAGE_EXTENSIONS) and path.startswith(prefix)


def extract_frames_ffmpeg(video_path):
    """Extract the frames from the video using ffmpeg."""
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


def extract_frames_opencv(video_path):
    """Extract the frames from the video using OpenCV."""
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


def verify_args(raw_data_dir, output_dir):
    assert os.path.isdir(raw_data_dir), "raw_data_dir must be an existing directory"
    assert os.path.isdir(output_dir), "output_dir must be an existing directory"
    assert not os.path.exists(
        os.path.join(output_dir, OUTPUT_NAME.format(__version__))
    ), "A matching version of the POCUS dataset already exists"


def pocus(
    raw_data_dir: Annotated[str, typer.Argument(help="The path to the raw data")],
    output_dir: Annotated[
        str, typer.Argument(help="The output directory for the processed datasets")
    ],
):
    verify_args(raw_data_dir, output_dir)

    output_dir = os.path.join(output_dir, OUTPUT_NAME.format(__version__))

    # Copy the images and add them to a list of examples
    examples = []
    image_dir = os.path.join(raw_data_dir, INPUT_IMAGE_DIR)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks", "scan"), exist_ok=True)
    for abbr, label in ABBR_TO_LABEL.items():
        files = [
            p
            for p in os.listdir(image_dir)
            if p.endswith(IMAGE_EXTENSIONS) and p.lower().startswith(abbr)
        ]
        for file in files:
            # Convert any 1-channel (greyscale) or 4-channel (RGBA) images to 3-channel
            # images
            image = skimage.io.imread(os.path.join(image_dir, file))
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[-1] == 4:
                image = skimage.color.rgba2rgb(image)
                image = (image * 255).astype(np.uint8)
            skimage.io.imsave(
                os.path.join(output_dir, "images", f"{Path(file).stem}.png"), image
            )

            # Generate a scan mask for the image
            mask = generate_scan_mask(image)
            skimage.io.imsave(
                os.path.join(output_dir, "masks", "scan", f"{Path(file).stem}.png"),
                mask,
                check_contrast=False,
            )

            examples.append(
                {
                    "source": file,
                    "image": os.path.join("images", f"{Path(file).stem}.png"),
                    "scan_mask": os.path.join(
                        "masks", "scan", f"{Path(file).stem}.png"
                    ),
                    "pathology": label,
                    "label": LABEL_TO_CLASS[label],
                }
            )

    # Extract the frames from the videos and add them to the list of examples
    video_dir = os.path.join(raw_data_dir, INPUT_VIDEO_DIR)
    for abbr, label in ABBR_TO_LABEL.items():
        files = [
            p
            for p in os.listdir(video_dir)
            if p.endswith(VIDEO_EXTENSIONS) and p.lower().startswith(abbr)
        ]
        for file in files:
            if file.endswith(".gif"):
                frames, frame_rate = extract_frames_opencv(
                    os.path.join(video_dir, file)
                )
            else:
                # Process with imageio/ffmpeg
                frames, frame_rate = extract_frames_ffmpeg(
                    os.path.join(video_dir, file)
                )

            every_nth_frame = int(frame_rate / FRAME_RATE)
            frames = frames[1::every_nth_frame][:MAX_FRAMES]

            for i, frame in enumerate(frames):
                # Save the frame as a 3-channel image
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                elif frame.shape[-1] == 4:
                    frame = skimage.color.rgba2rgb(frame)
                    image = (image * 255).astype(np.uint8)
                skimage.io.imsave(
                    os.path.join(output_dir, "images", f"{Path(file).stem}_{i}.png"),
                    frame,
                )

                # Generate a scan mask for the frame
                mask = generate_scan_mask(frame)
                skimage.io.imsave(
                    os.path.join(
                        output_dir, "masks", "scan", f"{Path(file).stem}_{i}.png"
                    ),
                    mask,
                    check_contrast=False,
                )

                examples.append(
                    {
                        "source": file,
                        "image": os.path.join("images", f"{Path(file).stem}_{i}.png"),
                        "scan_mask": os.path.join(
                            "masks", "scan", f"{Path(file).stem}_{i}.png"
                        ),
                        "pathology": label,
                        "label": LABEL_TO_CLASS[label],
                    }
                )

    # Create a DataFrame with the examples
    examples = pd.DataFrame(examples)

    print(f"Number of examples: {len(examples)}")

    # Split the dataset into training, validation, and test sets
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_indices, test_indices = next(
        splitter.split(X=examples, groups=examples["source"])
    )
    train_val_examples = examples.iloc[train_val_indices]
    test_examples = examples.iloc[test_indices]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_indices, val_indices = next(
        splitter.split(X=train_val_examples, groups=train_val_examples["source"])
    )
    train_examples = train_val_examples.iloc[train_indices]
    val_examples = train_val_examples.iloc[val_indices]

    # Save the training, validation, and test examples as JSON files
    for split, examples in [
        ("train", train_examples),
        ("validation", val_examples),
        ("test", test_examples),
    ]:
        file_path = os.path.join(output_dir, f"{split}.json")
        with open(file_path, "w") as f:
            json.dump(examples.to_dict(orient="records"), f, indent=4)

    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Test examples: {len(test_examples)}")

    save_version_info(output_dir, __version__)
