"""Unit tests for ultrabench.datasets.builders.utils."""

import numpy as np
import pandas as pd
import pytest

from ultrabench.core.rle import mask_to_rle, rle_to_mask
from ultrabench.evaluate.segmentation import SegmentationEvaluator
from ultrabench.datasets.builders.utils import (
    generate_classification_ground_truth_from_records,
    generate_segmentation_ground_truth_from_records,
    resize_and_pad,
    to_3_channel,
)


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
        """Check left and right pads are equal for tall images."""
        # 100×50 → scale = 256/100 = 2.56 → new_h=256, new_w=128
        # left pad = (256 - 128) // 2 = 64
        image = np.ones((100, 50, 3), dtype=np.uint8) * 200
        result = resize_and_pad(image, 256, order=1)
        assert result.shape == (256, 256, 3)
        # Padding columns should be zero
        np.testing.assert_array_equal(result[:, :64, :], 0)
        np.testing.assert_array_equal(result[:, 192:, :], 0)

    def test_wide_image_centering(self):
        """Check top and bottom pads are equal for wide images."""
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


class TestGenerateClassificationGroundTruth:
    """Tests for generate_classification_ground_truth_from_records."""

    records = [
        {"filename": "p1_0", "label": 0, "patient": 1, "view": "2CH"},
        {"filename": "p1_1", "label": 0, "patient": 1, "view": "2CH"},
        {"filename": "p2_0", "label": 1, "patient": 2, "view": "4CH"},
    ]

    def test_image_level_csv_has_no_sequence_column(self, tmp_path):
        """Without sequence keys, only image_id and label are written."""
        output_path = tmp_path / "gt.csv"
        generate_classification_ground_truth_from_records(self.records, output_path)

        written = pd.read_csv(output_path)
        assert list(written.columns) == ["image_id", "label"]
        assert written["image_id"].tolist() == ["p1_0", "p1_1", "p2_0"]

    def test_sequence_keys_add_sequence_id(self, tmp_path):
        """Sequence keys are joined with underscores into sequence_id."""
        output_path = tmp_path / "gt.csv"
        generate_classification_ground_truth_from_records(
            self.records, output_path, sequence_keys=["patient", "view"]
        )

        written = pd.read_csv(output_path)
        assert list(written.columns) == ["image_id", "label", "sequence_id"]
        assert written["sequence_id"].tolist() == ["1_2CH", "1_2CH", "2_4CH"]

    def test_missing_sequence_key_raises(self, tmp_path):
        """A record missing a sequence key is rejected."""
        with pytest.raises(KeyError, match="video"):
            generate_classification_ground_truth_from_records(
                self.records, tmp_path / "gt.csv", sequence_keys=["video"]
            )


def make_mask(size: int, value: int = 1) -> np.ndarray:
    """Build a mask with a square of one class in its top-left corner.

    Args:
        size: The height and width of the mask.
        value: The class ID to fill the square with.

    Returns:
        A uint8 array of shape (size, size).
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[: size // 2, : size // 2] = value

    return mask


class TestGenerateSegmentationGroundTruth:
    """Tests for generate_segmentation_ground_truth_from_records."""

    records = [
        {"filename": "a", "mask": make_mask(8, value=1)},
        {"filename": "b", "mask": make_mask(8, value=2)},
    ]

    def test_csv_columns_and_row_order(self, tmp_path):
        """One row per (image, class), images in order, classes within them."""
        output_path = tmp_path / "gt.csv"
        generate_segmentation_ground_truth_from_records(
            self.records, "mask", [1, 2], output_path, image_size=8
        )

        written = pd.read_csv(output_path)
        assert list(written.columns) == ["image_id", "class_id", "encoded_pixels"]
        assert written["image_id"].tolist() == ["a", "a", "b", "b"]
        assert written["class_id"].tolist() == [1, 2, 1, 2]

    def test_masks_round_trip(self, tmp_path):
        """Each encoded mask decodes back to that class of the record."""
        output_path = tmp_path / "gt.csv"
        generate_segmentation_ground_truth_from_records(
            self.records, "mask", [1, 2], output_path, image_size=8
        )

        written = pd.read_csv(output_path, converters={"encoded_pixels": str})
        for row in written.to_dict("records"):
            record = next(r for r in self.records if r["filename"] == row["image_id"])
            expected = np.asarray(record["mask"] == row["class_id"], dtype=np.uint8)
            decoded = rle_to_mask(row["encoded_pixels"], height=8, width=8)
            np.testing.assert_array_equal(decoded, expected)

    def test_empty_class_is_an_empty_string(self, tmp_path):
        """A class absent from an image encodes to an empty string."""
        output_path = tmp_path / "gt.csv"
        generate_segmentation_ground_truth_from_records(
            self.records, "mask", [1], output_path, image_size=8
        )

        written = pd.read_csv(output_path, converters={"encoded_pixels": str})
        assert written.loc[written["image_id"] == "b", "encoded_pixels"].item() == ""

    def test_missing_mask_is_skipped(self, tmp_path):
        """A record without a mask contributes no rows."""
        output_path = tmp_path / "gt.csv"
        records = [*self.records, {"filename": "c", "mask": None}]
        generate_segmentation_ground_truth_from_records(
            records, "mask", [1], output_path, image_size=8
        )

        written = pd.read_csv(output_path)
        assert written["image_id"].tolist() == ["a", "b"]

    def test_wrong_size_mask_raises(self, tmp_path):
        """A mask that is not at the split's resolution is rejected."""
        records = [{"filename": "a", "mask": make_mask(4)}]
        with pytest.raises(ValueError, match=r"expected \(8, 8\)"):
            generate_segmentation_ground_truth_from_records(
                records, "mask", [1], tmp_path / "gt.csv", image_size=8
            )

    def test_non_square_mask_raises(self, tmp_path):
        """A mask that was not padded to a square is rejected."""
        records = [{"filename": "a", "mask": np.zeros((4, 8), dtype=np.uint8)}]
        with pytest.raises(ValueError, match="has shape"):
            generate_segmentation_ground_truth_from_records(
                records, "mask", [1], tmp_path / "gt.csv", image_size=8
            )

    def test_none_image_size_accepts_mixed_shapes(self, tmp_path):
        """The original-resolution split holds masks of differing sizes."""
        output_path = tmp_path / "gt.csv"
        records = [
            {"filename": "a", "mask": make_mask(4)},
            {"filename": "b", "mask": make_mask(6)},
        ]
        generate_segmentation_ground_truth_from_records(
            records, "mask", [1], output_path, image_size=None
        )

        written = pd.read_csv(output_path)
        assert written["image_id"].tolist() == ["a", "b"]

    def test_accepts_an_iterator(self, tmp_path):
        """Records may be generated lazily rather than held in memory."""
        output_path = tmp_path / "gt.csv"
        generate_segmentation_ground_truth_from_records(
            (r for r in self.records), "mask", [1], output_path, image_size=8
        )

        written = pd.read_csv(output_path)
        assert written["image_id"].tolist() == ["a", "b"]

    def test_evaluator_scores_the_written_csv(self, tmp_path):
        """The evaluation layer can score what the dataset layer writes.

        The ground truth and the predictions are generated from the same
        masks here, so a perfect score is the only correct answer. It
        fails if the columns, the class IDs or the mask resolution of
        the two layers disagree.
        """
        ground_truth_path = tmp_path / "gt.csv"
        generate_segmentation_ground_truth_from_records(
            self.records, "mask", [1, 2], ground_truth_path, image_size=8
        )

        predictions_path = tmp_path / "predictions.csv"
        pd.DataFrame(
            [
                {
                    "image_id": r["filename"],
                    "class_id": class_id,
                    "encoded_pixels": mask_to_rle(
                        np.asarray(r["mask"] == class_id, dtype=np.uint8)
                    ),
                }
                for r in self.records
                for class_id in (1, 2)
            ]
        ).to_csv(predictions_path, index=False)

        metrics = SegmentationEvaluator(
            ground_truth_path,
            predictions_path,
            tmp_path / "results",
            image_height=8,
            image_width=8,
        ).evaluate()

        assert metrics["positive_dice"] == pytest.approx(1.0)

    def test_evaluator_penalises_a_wrong_prediction(self, tmp_path):
        """A prediction that misses the mask scores below a perfect one."""
        ground_truth_path = tmp_path / "gt.csv"
        generate_segmentation_ground_truth_from_records(
            self.records, "mask", [1], ground_truth_path, image_size=8
        )

        predictions_path = tmp_path / "predictions.csv"
        pd.DataFrame(
            [
                {
                    "image_id": r["filename"],
                    "class_id": 1,
                    "encoded_pixels": mask_to_rle(np.zeros((8, 8), dtype=np.uint8)),
                }
                for r in self.records
            ]
        ).to_csv(predictions_path, index=False)

        metrics = SegmentationEvaluator(
            ground_truth_path,
            predictions_path,
            tmp_path / "results",
            image_height=8,
            image_width=8,
        ).evaluate()

        assert metrics["positive_dice"] < 1.0
