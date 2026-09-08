"""Unit tests for ultrabench.training.data.transforms."""

import cv2
import numpy as np
import pytest

from ultrabench.training.data.transforms import (
    AdaptiveLongestMaxSize,
    get_train_test_transforms,
)

IMAGE_SIZE = 256


def make_image(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Build a random 3-channel uint8 image.

    Args:
        height: The image height in pixels.
        width: The image width in pixels.
        seed: The seed for the random number generator.

    Returns:
        An array of shape (height, width, 3).
    """
    rng = np.random.default_rng(seed)

    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def make_mask(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Build a random multi-class uint8 mask with labels {0, 1, 2, 3}.

    Args:
        height: The mask height in pixels.
        width: The mask width in pixels.
        seed: The seed for the random number generator.

    Returns:
        An array of shape (height, width).
    """
    rng = np.random.default_rng(seed)

    return rng.integers(0, 4, (height, width)).astype(np.uint8)


def as_hwc(image: np.ndarray) -> np.ndarray:
    """Convert a ToTensorV2 CHW tensor back to an HWC array.

    Args:
        image: The transformed image tensor of shape (C, H, W).

    Returns:
        An array of shape (H, W, C).
    """
    return image.numpy().transpose(1, 2, 0)


class TestAdaptiveLongestMaxSize:
    """Tests for AdaptiveLongestMaxSize."""

    def test_downscale_uses_inter_area(self):
        """Shrinking an image averages over the source footprint."""
        # 500×300 → scale = 0.512 → new_h=256, new_w=154. The scale is
        # deliberately not 0.5: at exactly one half, OpenCV's INTER_AREA and
        # INTER_LINEAR agree, which would hide the difference.
        image = make_image(500, 300)
        transform = AdaptiveLongestMaxSize(
            max_size=IMAGE_SIZE, interpolation=cv2.INTER_AREA
        )
        result = transform(image=image)["image"]
        np.testing.assert_array_equal(
            result, cv2.resize(image, (154, 256), interpolation=cv2.INTER_AREA)
        )
        assert not np.array_equal(
            result, cv2.resize(image, (154, 256), interpolation=cv2.INTER_LINEAR)
        )

    def test_upscale_falls_back_to_inter_cubic(self):
        """INTER_AREA is swapped for INTER_CUBIC when enlarging."""
        # 100×50 → scale = 2.56 → new_h=256, new_w=128.
        image = make_image(100, 50)
        transform = AdaptiveLongestMaxSize(
            max_size=IMAGE_SIZE, interpolation=cv2.INTER_AREA
        )
        result = transform(image=image)["image"]
        np.testing.assert_array_equal(
            result, cv2.resize(image, (128, 256), interpolation=cv2.INTER_CUBIC)
        )
        assert not np.array_equal(
            result, cv2.resize(image, (128, 256), interpolation=cv2.INTER_AREA)
        )

    @pytest.mark.parametrize("shape", [(500, 300), (100, 50)])
    def test_mask_stays_nearest(self, shape):
        """Masks use INTER_NEAREST in both resize directions."""
        mask = make_mask(*shape)
        transform = AdaptiveLongestMaxSize(
            max_size=IMAGE_SIZE,
            interpolation=cv2.INTER_AREA,
            mask_interpolation=cv2.INTER_NEAREST,
        )
        result = transform(image=make_image(*shape), mask=mask)["mask"]
        height, width = shape
        scale = IMAGE_SIZE / max(height, width)
        expected_size = (round(width * scale), round(height * scale))
        np.testing.assert_array_equal(
            result,
            cv2.resize(mask, expected_size, interpolation=cv2.INTER_NEAREST),
        )
        assert set(np.unique(result).tolist()) == {0, 1, 2, 3}


class TestGetTrainTestTransforms:
    """Tests for get_train_test_transforms."""

    def test_pipelines_use_the_adaptive_resize(self):
        """Both pipelines resize with the direction-aware transform."""
        transform_train, transform_test = get_train_test_transforms(IMAGE_SIZE)
        for pipeline in (transform_train, transform_test):
            resize = pipeline.transforms[0]
            assert isinstance(resize, AdaptiveLongestMaxSize)
            assert resize.interpolation == cv2.INTER_AREA
            assert resize.mask_interpolation == cv2.INTER_NEAREST

    @pytest.mark.parametrize("shape", [(500, 300), (300, 500), (100, 50)])
    def test_output_is_padded_to_the_target_size(self, shape):
        """Images, masks and scan masks all come out at the target size."""
        _, transform_test = get_train_test_transforms(IMAGE_SIZE)
        result = transform_test(
            image=make_image(*shape),
            mask=make_mask(*shape),
            scan_mask=np.ones(shape, dtype=np.uint8),
        )
        assert as_hwc(result["image"]).shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        assert result["mask"].shape == (IMAGE_SIZE, IMAGE_SIZE)
        assert result["scan_mask"].shape == (IMAGE_SIZE, IMAGE_SIZE)

    def test_scan_mask_is_resized_as_a_mask(self):
        """The scan_mask additional target stays binary, not blended."""
        _, transform_test = get_train_test_transforms(IMAGE_SIZE)
        scan_mask = make_mask(500, 300)
        result = transform_test(
            image=make_image(500, 300),
            scan_mask=scan_mask,
        )
        assert set(np.unique(result["scan_mask"]).tolist()) <= {0, 1, 2, 3}
