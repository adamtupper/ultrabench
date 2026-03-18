"""Utility functions for dataset processing."""

import os

import numpy as np
import skimage.transform


def save_version_info(output_dir: str, version: str) -> None:
    """Create a file with the UltraBench version and current date.

    Args:
        output_dir: The path to the output directory.
        version: The version string.
    """
    with open(os.path.join(output_dir, "version.txt"), "w") as f:
        f.write(f"UltraBench Version: {version}\n")
        f.write(f"Created: {os.popen('date').read().strip()}\n")


def to_3_channel(image: np.ndarray) -> np.ndarray:
    """Convert a single-channel image to a 3-channel (RGB) array.

    Replicates the single channel across all three channels. Raises a
    ValueError for inputs that are not single-channel (2D, HxWx1) or
    already 3-channel (HxWx3).

    Args:
        image: The input image array.

    Returns:
        A 3-channel array of shape (H, W, 3) with the same dtype as the input.

    Raises:
        ValueError: If the image has an unsupported number of channels.
    """
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    if image.ndim == 3 and image.shape[-1] == 1:
        return np.concatenate([image] * 3, axis=-1)
    if image.ndim == 3 and image.shape[-1] == 3:
        return image
    raise ValueError(
        f"Expected a single-channel or 3-channel image, got shape {image.shape}."
    )


def resize_and_pad(image: np.ndarray, size: int, order: int = 1) -> np.ndarray:
    """Resize an image to fit within size×size, then center-pad to exactly size×size.

    Preserves aspect ratio. Padding is zero (black). Uses
    skimage.transform.resize for the resize step.

    Args:
        image: The input image array of shape (H, W) or (H, W, C).
        size: The target height and width in pixels.
        order: Interpolation order passed to skimage.transform.resize.
            Use 1 (bilinear) for images and 0 (nearest-neighbor) for masks.

    Returns:
        An array of shape (size, size) or (size, size, C) with the same dtype
            as the input.
    """
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = round(h * scale), round(w * scale)
    resized = skimage.transform.resize(
        image,
        (new_h, new_w),
        order=order,
        preserve_range=True,
        anti_aliasing=(order > 0),
    ).astype(image.dtype)
    out_shape = (size, size, image.shape[2]) if image.ndim == 3 else (size, size)
    out = np.zeros(out_shape, dtype=image.dtype)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    out[top : top + new_h, left : left + new_w] = resized
    return out
