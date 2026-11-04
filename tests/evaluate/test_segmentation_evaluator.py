"""Unit tests for SegmentationEvaluator."""

import logging
import math

import numpy as np
import pandas as pd
import pytest
import torch
from torchmetrics.segmentation import DiceScore

from ultrabench.core.rle import mask_to_rle
from ultrabench.evaluate.segmentation import SegmentationEvaluator
from ultrabench.evaluate.tasks import get_task

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
    """Test that perfect predictions yield a positive Dice score of 1.0
    and per-class Dice scores of 1.0.
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
    assert metrics["positive_dice"] == pytest.approx(1.0)
    assert metrics["dice_class_1"] == pytest.approx(1.0)


def test_all_wrong_predictions(tmp_path):
    """Test that completely incorrect predictions yield a positive Dice
    score of 0.0 and per-class Dice scores of 0.0.
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
    assert metrics["positive_dice"] == pytest.approx(0.0)


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
    assert metrics["positive_dice"] == pytest.approx(0.0)


def test_background_class_excluded(tmp_path):
    """Test that the background class (Class 0) is excluded from the
    positive Dice score and per-class Dice scores.
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
    # positive_dice should only reflect class 1, not class 0
    assert metrics["positive_dice"] == pytest.approx(1.0)
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
    assert "positive_dice" in summary["metric"].values


def test_class_empty_in_both_is_undefined(tmp_path):
    """Check that a class empty in both the GT and the prediction is
    excluded from the score rather than counted as a perfect match.
    """
    empty = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    _write_csv(tmp_path / "gt.csv", [_make_row("img1.png", 1, empty)])
    _write_csv(tmp_path / "pred.csv", [_make_row("img1.png", 1, empty)])

    ev = SegmentationEvaluator(
        tmp_path / "gt.csv", tmp_path / "pred.csv", tmp_path, HEIGHT, WIDTH
    )
    metrics = ev.evaluate()

    assert math.isnan(metrics["positive_dice"])
    per_image = pd.read_csv(tmp_path / "per_image_metrics.csv")
    assert per_image["dice"].isna().all()


def test_empty_ground_truth_with_prediction_scores_zero(tmp_path):
    """Check that predicting a class absent from the GT scores 0.0 and is
    counted, unlike the case where neither mask has the class.
    """
    empty = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    _write_csv(tmp_path / "gt.csv", [_make_row("img1.png", 1, empty)])
    _write_csv(tmp_path / "pred.csv", [_make_row("img1.png", 1, _make_mask([0, 1]))])

    ev = SegmentationEvaluator(
        tmp_path / "gt.csv", tmp_path / "pred.csv", tmp_path, HEIGHT, WIDTH
    )
    metrics = ev.evaluate()

    assert metrics["positive_dice"] == pytest.approx(0.0)


def test_undefined_classes_are_averaged_sample_major(tmp_path):
    """Check that the classes of an image are averaged before the images.

    The fixture leaves class 2 of the first image undefined, so the two
    per-class groups hold a different number of defined scores. Averaging
    the images first gives (1.0 + (0.0 + 0.5) / 2) / 2 = 0.625, whereas
    averaging the classes first would give (0.5 + 0.5) / 2 = 0.5.
    """
    empty = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask = _make_mask([0, 1])
    gt_rows = [
        _make_row("img1.png", 1, mask),
        _make_row("img1.png", 2, empty),
        _make_row("img2.png", 1, mask),
        _make_row("img2.png", 2, mask),
    ]
    pred_rows = [
        _make_row("img1.png", 1, mask),  # 1.0
        _make_row("img1.png", 2, empty),  # undefined
        _make_row("img2.png", 1, _make_mask([2, 3])),  # 0.0
        _make_row("img2.png", 2, _make_mask([1, 2])),  # 0.5
    ]
    _write_csv(tmp_path / "gt.csv", gt_rows)
    _write_csv(tmp_path / "pred.csv", pred_rows)

    ev = SegmentationEvaluator(
        tmp_path / "gt.csv", tmp_path / "pred.csv", tmp_path, HEIGHT, WIDTH
    )
    metrics = ev.evaluate()

    assert metrics["positive_dice"] == pytest.approx(0.625)
    assert metrics["dice_class_1"] == pytest.approx(0.5)
    assert metrics["dice_class_2"] == pytest.approx(0.5)


def test_every_class_undefined_warns(tmp_path, caplog):
    """Check that a score of NaN is reported and warned about when no
    (image, class) pair is defined.
    """
    empty = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    rows = [_make_row("img1.png", 1, empty), _make_row("img2.png", 1, empty)]
    _write_csv(tmp_path / "gt.csv", rows)
    _write_csv(tmp_path / "pred.csv", rows)

    ev = SegmentationEvaluator(
        tmp_path / "gt.csv", tmp_path / "pred.csv", tmp_path, HEIGHT, WIDTH
    )
    with caplog.at_level(logging.WARNING):
        metrics = ev.evaluate()

    assert math.isnan(metrics["positive_dice"])
    assert any("No (image, class) pair is defined" in m for m in caplog.messages)


def test_summary_reports_the_task_primary_metric(tmp_path):
    """Check that the reported key is the one the task registry names, so
    that the evaluator and `TaskSpec.primary_metric` cannot drift apart.
    """
    mask = _make_mask([0, 1])
    _write_csv(tmp_path / "gt.csv", [_make_row("img1.png", 1, mask)])
    _write_csv(tmp_path / "pred.csv", [_make_row("img1.png", 1, mask)])

    ev = SegmentationEvaluator(
        tmp_path / "gt.csv", tmp_path / "pred.csv", tmp_path, HEIGHT, WIDTH
    )
    metrics = ev.evaluate()

    assert get_task("tn3k_segmentation").primary_metric in metrics


def _index_maps() -> tuple[np.ndarray, np.ndarray]:
    """Build a ground-truth and a predicted label map per test image.

    The fixture covers every case the averaging has to handle: classes
    present in both masks, a class predicted where the GT has none, a
    class absent from both masks, and an image with no foreground at all.

    Returns:
        The ground-truth and predicted label maps, both of shape
        (4, 8, 8) and holding class indices 0, 1 or 2.
    """
    size = 8
    gt = np.zeros((4, size, size), dtype=np.int64)
    pred = np.zeros((4, size, size), dtype=np.int64)

    gt[0, 0:4, 0:4] = 1
    gt[0, 4:8, 4:8] = 2
    pred[0, 0:4, 0:2] = 1
    pred[0, 4:8, 4:8] = 2

    gt[1, 0:2, :] = 1  # class 2 is empty in both masks
    pred[1, 0:3, :] = 1

    gt[2, 0:4, 0:4] = 1
    pred[2, 0:4, 0:4] = 1
    pred[2, 5:7, 5:7] = 2  # class 2 is predicted but absent from the GT

    return gt, pred  # the fourth image has no foreground in either mask


def test_matches_torchmetrics_dice_score(tmp_path):
    """Check that the score reproduces the metric the training layer
    reports with `torchmetrics.segmentation.DiceScore`.
    """
    gt, pred = _index_maps()
    num_classes = 3
    size = gt.shape[-1]

    for name, maps in [("gt", gt), ("pred", pred)]:
        _write_csv(
            tmp_path / f"{name}.csv",
            [
                _make_row(
                    f"img{index}.png",
                    class_id,
                    (label_map == class_id).astype(np.uint8),
                )
                for index, label_map in enumerate(maps)
                for class_id in range(1, num_classes)
            ],
        )

    ev = SegmentationEvaluator(
        tmp_path / "gt.csv", tmp_path / "pred.csv", tmp_path, size, size
    )
    metrics = ev.evaluate()

    reference = DiceScore(
        num_classes=num_classes, include_background=False, input_format="index"
    )(torch.from_numpy(pred), torch.from_numpy(gt))

    assert metrics["positive_dice"] == pytest.approx(float(reference), abs=1e-6)

    # Guard the fixture itself: the parity is only meaningful if it
    # exercises the undefined pairs and the all-undefined image.
    per_image = pd.read_csv(tmp_path / "per_image_metrics.csv")
    assert per_image["dice"].isna().sum() == 3
    assert per_image.groupby("image_id")["dice"].mean().isna().sum() == 1
