"""Unit tests for ultrabench.process.utils."""

import numpy as np
import pytest

from ultrabench.process.utils import resize_and_pad, to_3_channel


class TestTo3Channel:
    """Tests for to_3_channel."""

    def test_2d_input(self):
        """A 2D (H, W) array is stacked into (H, W, 3)."""
        image = np.arange(12, dtype=np.uint8).reshape(3, 4)
        result = to_3_channel(image)
        assert result.shape == (3, 4, 3)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result[:, :, 0], image)
        np.testing.assert_array_equal(result[:, :, 1], image)
        np.testing.assert_array_equal(result[:, :, 2], image)

    def test_hwx1_input(self):
        """An (H, W, 1) array is concatenated into (H, W, 3)."""
        image = np.arange(12, dtype=np.uint8).reshape(3, 4, 1)
        result = to_3_channel(image)
        assert result.shape == (3, 4, 3)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result[:, :, 0], image[:, :, 0])
        np.testing.assert_array_equal(result[:, :, 1], image[:, :, 0])
        np.testing.assert_array_equal(result[:, :, 2], image[:, :, 0])

    def test_hwx3_input_passthrough(self):
        """An (H, W, 3) array is returned unchanged."""
        image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        result = to_3_channel(image)
        assert result.shape == (3, 4, 3)
        assert result is image

    def test_hwx4_input(self):
        """The alpha channel of a (H, W, 4) array is dropped."""
        image = np.arange(48, dtype=np.uint8).reshape(3, 4, 4)
        result = to_3_channel(image)
        assert result.shape == (3, 4, 3)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result[:, :, 0], image[:, :, 0])
        np.testing.assert_array_equal(result[:, :, 1], image[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 2], image[:, :, 2])

    def test_4_channel_raises_value_error(self):
        """An (H, W, 4) array raises ValueError."""
        image = np.zeros((3, 4, 5), dtype=np.uint8)
        with pytest.raises(ValueError):
            to_3_channel(image)

    def test_dtype_preserved_float32(self):
        """Output dtype matches input dtype for float32."""
        image = np.ones((4, 4), dtype=np.float32)
        result = to_3_channel(image)
        assert result.dtype == np.float32

    def test_dtype_preserved_uint16(self):
        """Output dtype matches input dtype for uint16."""
        image = np.ones((4, 4), dtype=np.uint16)
        result = to_3_channel(image)
        assert result.dtype == np.uint16


class TestResizeAndPad:
    """Tests for resize_and_pad."""

    def test_square_image_exact_size(self):
        """A square image resized to the same size is unchanged."""
        image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        result = resize_and_pad(image, 256, order=1)
        assert result.shape == (256, 256, 3)

    def test_tall_image_3channel(self):
        """A tall (H > W) 3-channel image fits within the target square."""
        image = np.ones((400, 200, 3), dtype=np.uint8) * 100
        result = resize_and_pad(image, 256, order=1)
        assert result.shape == (256, 256, 3)

    def test_wide_image_3channel(self):
        """A wide (W > H) 3-channel image fits within the target square."""
        image = np.ones((200, 400, 3), dtype=np.uint8) * 100
        result = resize_and_pad(image, 256, order=1)
        assert result.shape == (256, 256, 3)

    def test_2d_mask(self):
        """A 2D mask is padded to the exact target size."""
        mask = np.ones((300, 200), dtype=np.uint8)
        result = resize_and_pad(mask, 256, order=0)
        assert result.shape == (256, 256)

    def test_tall_image_centering(self):
        """Padding is centered: left and right pads are equal (or differ by 1)."""
        # 100×50 → scale = 256/100 = 2.56 → new_h=256, new_w=128
        # left pad = (256 - 128) // 2 = 64
        image = np.ones((100, 50, 3), dtype=np.uint8) * 200
        result = resize_and_pad(image, 256, order=1)
        assert result.shape == (256, 256, 3)
        # Padding columns should be zero
        np.testing.assert_array_equal(result[:, :64, :], 0)
        np.testing.assert_array_equal(result[:, 192:, :], 0)

    def test_wide_image_centering(self):
        """Padding is centered for wide images: top and bottom pads are equal."""
        # 50×100 → scale = 256/100 = 2.56 → new_h=128, new_w=256
        # top pad = (256 - 128) // 2 = 64
        image = np.ones((50, 100, 3), dtype=np.uint8) * 200
        result = resize_and_pad(image, 256, order=1)
        assert result.shape == (256, 256, 3)
        np.testing.assert_array_equal(result[:64, :, :], 0)
        np.testing.assert_array_equal(result[192:, :, :], 0)

    def test_dtype_preserved_uint8(self):
        """Output dtype matches uint8 input."""
        image = np.ones((300, 200, 3), dtype=np.uint8)
        result = resize_and_pad(image, 256, order=1)
        assert result.dtype == np.uint8

    def test_mask_order0_binary_values(self):
        """Nearest-neighbor mask resize preserves binary {0, 1} values."""
        mask = np.zeros((300, 200), dtype=np.uint8)
        mask[50:250, 50:150] = 1
        result = resize_and_pad(mask, 256, order=0)
        unique_values = np.unique(result)
        assert set(unique_values).issubset({0, 1})
