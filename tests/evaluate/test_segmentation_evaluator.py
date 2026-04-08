"""Unit tests for SegmentationEvaluator."""

import logging

import numpy as np
import pandas as pd
import pytest

from ultrabench.core.rle import mask_to_rle
from ultrabench.evaluate.segmentation import SegmentationEvaluator

from .utils import _write_csv

HEIGHT, WIDTH = 4, 4


def _make_mask(ones_flat_indices: list[int]) -> np.ndarray:
    """Create a binary mask of shape (HEIGHT, WIDTH) with 1s at the
    specified flat indices and 0s elsewhere.

    Args:
        ones_flat_indices: List of flat indices where the mask should
            have 1s. The flat index is computed as `row * WIDTH + col`.

    Returns:
        A binary mask of shape (HEIGHT, WIDTH) with 1s at the specified
        indices and 0s elsewhere.
    """
    mask = np.zeros(HEIGHT * WIDTH, dtype=np.uint8)
    mask[ones_flat_indices] = 1
    return mask.reshape(HEIGHT, WIDTH)


def _make_row(image_id: str, class_id: int, mask: np.ndarray) -> dict:
    """Create a single row of the ground-truth or predictions CSV.

    Args:
        image_id: The ID of the image (e.g., filename).
        class_id: The class ID for the mask.
        mask: A binary mask of shape where 1s indicate the presence of
            the class and 0s indicate absence.

    Returns:
        A dictionary containing the row values.
    """
    return {
        "image_id": image_id,
        "class_id": class_id,
        "encoded_pixels": mask_to_rle(mask),
    }


def test_perfect_predictions(tmp_path):
    """Test that perfect predictions yield a macro Dice score of 1.0 and
    per-class Dice scores of 1.0.
    """
    mask = _make_mask([0, 1, 2])
    gt_rows = [_make_row("img1.png", 1, mask)]
    pred_rows = [_make_row("img1.png", 1, mask)]
    _write_csv(tmp_path / "gt.csv", gt_rows)
    _write_csv(tmp_path / "pred.csv", pred_rows)

    ev = SegmentationEvaluator(
        tmp_path / "gt.csv", tmp_path / "pred.csv", tmp_path, HEIGHT, WIDTH
    )
    metrics = ev.evaluate()
    assert metrics["macro_dice"] == pytest.approx(1.0)
    assert metrics["dice_class_1"] == pytest.approx(1.0)


def test_all_wrong_predictions(tmp_path):
    """Test that completely incorrect predictions yield a macro Dice score of
    0.0 and per-class Dice scores of 0.0.
    """
    gt_mask = _make_mask([0, 1, 2, 3])
    pred_mask = _make_mask([4, 5, 6, 7])
    gt_rows = [_make_row("img1.png", 1, gt_mask)]
    pred_rows = [_make_row("img1.png", 1, pred_mask)]
    _write_csv(tmp_path / "gt.csv", gt_rows)
    _write_csv(tmp_path / "pred.csv", pred_rows)

    ev = SegmentationEvaluator(
        tmp_path / "gt.csv", tmp_path / "pred.csv", tmp_path, HEIGHT, WIDTH
    )
    metrics = ev.evaluate()
    assert metrics["macro_dice"] == pytest.approx(0.0)


def test_missing_prediction_treated_as_empty_mask(tmp_path):
    """Test that if a GT row has no corresponding prediction, it is
    treated as an empty mask (all 0s), which should yield a Dice score
    of 0.0 for that class.
    """
    gt_mask = _make_mask([0, 1, 2, 3])
    gt_rows = [_make_row("img1.png", 1, gt_mask)]

    _write_csv(tmp_path / "gt.csv", gt_rows)
    _write_csv(
        tmp_path / "pred.csv", [], columns=["image_id", "class_id", "encoded_pixels"]
    )

    ev = SegmentationEvaluator(
        tmp_path / "gt.csv", tmp_path / "pred.csv", tmp_path, HEIGHT, WIDTH
    )
    metrics = ev.evaluate()
    assert metrics["macro_dice"] == pytest.approx(0.0)


def test_background_class_excluded_from_macro(tmp_path):
    """Test that the background class (Class 0) is excluded from the
    macro Dice score and per-class Dice scores.
    """
    mask = _make_mask([0, 1])
    gt_rows = [
        _make_row("img1.png", 0, mask),  # background — should be excluded
        _make_row("img1.png", 1, mask),
    ]
    pred_rows = [
        _make_row("img1.png", 0, np.zeros((HEIGHT, WIDTH), dtype=np.uint8)),  # wrong
        _make_row("img1.png", 1, mask),  # correct
    ]
    print(pred_rows)
    print(gt_rows)
    _write_csv(tmp_path / "gt.csv", gt_rows)
    _write_csv(tmp_path / "pred.csv", pred_rows)

    ev = SegmentationEvaluator(
        tmp_path / "gt.csv", tmp_path / "pred.csv", tmp_path, HEIGHT, WIDTH
    )
    metrics = ev.evaluate()
    # macro_dice should only reflect class 1, not class 0
    assert metrics["macro_dice"] == pytest.approx(1.0)
    assert "dice_class_0" not in metrics


def test_missing_image_in_predictions_warns(tmp_path, caplog):
    """Check that a warning is logged if an image in the ground-truth
    CSV is missing from the predictions CSV.
    """
    mask = _make_mask([0])
    gt_rows = [_make_row("img1.png", 1, mask), _make_row("img2.png", 1, mask)]
    pred_rows = [_make_row("img1.png", 1, mask)]
    _write_csv(tmp_path / "gt.csv", gt_rows)
    _write_csv(tmp_path / "pred.csv", pred_rows)

    ev = SegmentationEvaluator(
        tmp_path / "gt.csv", tmp_path / "pred.csv", tmp_path, HEIGHT, WIDTH
    )
    with caplog.at_level(logging.WARNING):
        ev.evaluate()
    assert any("img2.png" in m for m in caplog.messages)


def test_extra_image_in_predictions_warns(tmp_path, caplog):
    """Check that a warning is logged if an image in the predictions CSV
    is missing from the ground-truth CSV.
    """
    mask = _make_mask([0])
    gt_rows = [_make_row("img1.png", 1, mask)]
    pred_rows = [
        _make_row("img1.png", 1, mask),
        _make_row("img_extra.png", 1, mask),
    ]
    _write_csv(tmp_path / "gt.csv", gt_rows)
    _write_csv(tmp_path / "pred.csv", pred_rows)

    ev = SegmentationEvaluator(
        tmp_path / "gt.csv", tmp_path / "pred.csv", tmp_path, HEIGHT, WIDTH
    )
    with caplog.at_level(logging.WARNING):
        ev.evaluate()
    assert any("img_extra.png" in m for m in caplog.messages)


def test_output_csvs_written(tmp_path):
    """Check that the per-image metrics CSV and summary CSV are written
    to the output directory.
    """
    mask = _make_mask([0, 1])
    gt_rows = [_make_row("img1.png", 1, mask)]
    pred_rows = [_make_row("img1.png", 1, mask)]
    _write_csv(tmp_path / "gt.csv", gt_rows)
    _write_csv(tmp_path / "pred.csv", pred_rows)

    out_dir = tmp_path / "out"
    ev = SegmentationEvaluator(
        tmp_path / "gt.csv", tmp_path / "pred.csv", out_dir, HEIGHT, WIDTH
    )
    ev.evaluate()

    assert (out_dir / "per_image_metrics.csv").exists()
    assert (out_dir / "summary.csv").exists()

    per_image = pd.read_csv(out_dir / "per_image_metrics.csv")
    assert set(per_image.columns) == {"image_id", "class_id", "dice"}

    summary = pd.read_csv(out_dir / "summary.csv")
    assert "macro_dice" in summary["metric"].values
