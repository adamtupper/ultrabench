"""Unit tests for RLE encode/decode utilities."""

import numpy as np

from ultrabench.core.rle import mask_to_rle, rle_to_mask


def test_round_trip_random_mask():
    """Test that encoding/decoding a random mask returns the original."""
    rng = np.random.default_rng(0)
    mask = rng.integers(0, 2, size=(8, 8), dtype=np.uint8)
    assert np.array_equal(rle_to_mask(mask_to_rle(mask), 8, 8), mask)


def test_all_zeros_encodes_to_empty_string():
    """Test that an all-zeros mask encodes to an empty string."""
    mask = np.zeros((4, 4), dtype=np.uint8)
    assert mask_to_rle(mask) == ""


def test_all_zeros_decodes_from_empty_string():
    """Test that decoding an empty string produces an all-zeros mask."""
    result = rle_to_mask("", 4, 4)
    assert np.array_equal(result, np.zeros((4, 4), dtype=np.uint8))


def test_all_ones_round_trip():
    """Test that an all-ones mask round-trips correctly."""
    mask = np.ones((4, 4), dtype=np.uint8)
    assert np.array_equal(rle_to_mask(mask_to_rle(mask), 4, 4), mask)


def test_known_pattern():
    """Test encoding and decoding of a known pattern.

    1) Input:
        [[1, 0],
         [0, 1]]

    2) Flattened (column-major order) array: [1, 0, 0, 1]

    3) Runs:
        - Start at index 1 (1-indexed), length 1 (the first '1')
        - Start at index 4 (1-indexed), length 1 (the second '1')

    4) Expected RLE: "1 1 4 1"
    """
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    assert mask_to_rle(mask) == "1 1 4 1"
    assert np.array_equal(rle_to_mask("1 1 4 1", 2, 2), mask)
