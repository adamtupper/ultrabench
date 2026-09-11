"""Run-length encoding (RLE) utilities for encoding and decoding
segmentation masks.

Examples:
    Binary segmentation (single foreground class)::

        mask = np.array([[0, 1, 1], [0, 1, 0]], dtype=np.uint8)
        rle = mask_to_rle(mask)
        reconstructed = rle_to_mask(rle, height=2, width=3)

    Multi-class segmentation (one RLE string per class)::

        # mask has pixel values 0 (background), 1, 2, ...
        mask = np.array([[0, 1, 1], [2, 1, 0]], dtype=np.uint8)

        # Encode each class as a separate binary mask
        rle_class1 = mask_to_rle((mask == 1).astype(np.uint8))
        rle_class2 = mask_to_rle((mask == 2).astype(np.uint8))

        # Decode and reconstruct the multi-class mask
        reconstructed = np.zeros_like(mask)
        reconstructed[rle_to_mask(rle_class1, height=2, width=3) == 1] = 1
        reconstructed[rle_to_mask(rle_class2, height=2, width=3) == 1] = 2

    In a CSV, store one row per (image, class) pair. The background class
    (0) is omitted since it is implied by the absence of other classes.
"""

import numpy as np


def mask_to_rle(mask: np.ndarray) -> str:
    """Encode a binary mask using Kaggle-style run-length encoding.

    Creates 1-indexed, column-major order RLE string.

    Args:
        mask: A 2D binary uint8 array of shape (H, W).

    Returns:
        A space-separated RLE string, or an empty string for an
        all-zeros mask.
    """
    flat = mask.flatten(order="F")
    if flat.sum() == 0:
        return ""

    # Pad with zeros on both ends to detect transitions
    padded = np.concatenate([[0], flat, [0]])
    transitions = np.diff(padded.astype(np.int8))
    starts = np.where(transitions == 1)[0] + 1  # 1-indexed
    ends = np.where(transitions == -1)[0] + 1  # 1-indexed
    lengths = ends - starts

    pairs = np.stack([starts, lengths], axis=1).flatten()

    return " ".join(str(p) for p in pairs)


def rle_to_mask(rle: str, height: int, width: int) -> np.ndarray:
    """Decode a Kaggle-style RLE string back to a binary mask.

    Expects a 1-indexed, column-major order RLE string.

    Args:
        rle: A space-separated RLE string (1-indexed, column-major
            order). An empty string produces an all-zeros mask.
        height: The height of the output mask.
        width: The width of the output mask.

    Returns:
        A binary uint8 array of shape (height, width).
    """
    mask = np.zeros(height * width, dtype=np.uint8)
    if rle == "":
        return mask.reshape(height, width, order="F")

    values = list(map(int, rle.split()))
    for i in range(0, len(values), 2):
        start = values[i] - 1  # convert to 0-indexed
        length = values[i + 1]
        mask[start : start + length] = 1

    return mask.reshape(height, width, order="F")
