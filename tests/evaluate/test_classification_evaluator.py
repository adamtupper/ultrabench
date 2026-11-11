"""Unit tests for ClassificationEvaluator."""

import logging

import pandas as pd
import pytest

from ultrabench.evaluate.classification import ClassificationEvaluator

from .utils import _write_csv


def test_perfect_predictions(tmp_path):
    """Test that perfect predictions yield a balanced accuracy of 1.0."""
    gt = [{"image_id": f"img{i}.png", "label": i % 3} for i in range(6)]
    _write_csv(tmp_path / "gt.csv", gt)
    _write_csv(tmp_path / "pred.csv", gt)

    ev = ClassificationEvaluator(tmp_path / "gt.csv", tmp_path / "pred.csv", tmp_path)
    metrics = ev.evaluate()
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)


def test_all_wrong_predictions(tmp_path):
    """Test that all incorrect predictions yield a balanced accuracy of
    0.0.
    """
    gt = [{"image_id": "a.png", "label": 0}, {"image_id": "b.png", "label": 1}]
    pred = [{"image_id": "a.png", "label": 1}, {"image_id": "b.png", "label": 0}]
    _write_csv(tmp_path / "gt.csv", gt)
    _write_csv(tmp_path / "pred.csv", pred)

    ev = ClassificationEvaluator(tmp_path / "gt.csv", tmp_path / "pred.csv", tmp_path)
    metrics = ev.evaluate()
    assert metrics["balanced_accuracy"] == pytest.approx(0.0)


def test_missing_image_in_predictions_warns(tmp_path, caplog):
    """Check that a warning is logged if an image in the ground-truth
    CSV is missing from the predictions CSV.
    """
    gt = [{"image_id": "a.png", "label": 0}, {"image_id": "b.png", "label": 1}]
    pred = [{"image_id": "a.png", "label": 0}]
    _write_csv(tmp_path / "gt.csv", gt)
    _write_csv(tmp_path / "pred.csv", pred)

    ev = ClassificationEvaluator(tmp_path / "gt.csv", tmp_path / "pred.csv", tmp_path)
    with caplog.at_level(logging.WARNING):
        ev.evaluate()
    assert any("b.png" in m for m in caplog.messages)


def test_extra_image_in_predictions_warns(tmp_path, caplog):
    """Check that a warning is logged if an image in the predictions CSV
    is missing from the ground-truth CSV.
    """
    gt = [{"image_id": "a.png", "label": 0}]
    pred = [{"image_id": "a.png", "label": 0}, {"image_id": "extra.png", "label": 1}]
    _write_csv(tmp_path / "gt.csv", gt)
    _write_csv(tmp_path / "pred.csv", pred)

    ev = ClassificationEvaluator(tmp_path / "gt.csv", tmp_path / "pred.csv", tmp_path)
    with caplog.at_level(logging.WARNING):
        ev.evaluate()
    assert any("extra.png" in m for m in caplog.messages)


def test_output_csv_written(tmp_path):
    """Check that the summary CSV with the computed metrics is written
    to the output directory.
    """
    gt = [{"image_id": "a.png", "label": 0}]
    _write_csv(tmp_path / "gt.csv", gt)
    _write_csv(tmp_path / "pred.csv", gt)

    out_dir = tmp_path / "out"
    ev = ClassificationEvaluator(tmp_path / "gt.csv", tmp_path / "pred.csv", out_dir)
    ev.evaluate()

    assert (out_dir / "summary.csv").exists()
    summary = pd.read_csv(out_dir / "summary.csv")
    assert "balanced_accuracy" in summary["metric"].values


def _write_sequence_csvs(tmp_path, gt_rows, pred_rows):
    """Write a ground-truth and predictions CSV pair for a sequence task.

    Args:
        tmp_path: Directory to write the CSV files to.
        gt_rows: Rows of the ground-truth CSV.
        pred_rows: Rows of the predictions CSV.

    Returns:
        A tuple of the ground-truth and predictions CSV paths.
    """
    _write_csv(tmp_path / "gt.csv", gt_rows)
    _write_csv(tmp_path / "pred.csv", pred_rows)

    return tmp_path / "gt.csv", tmp_path / "pred.csv"


def test_perfect_sequence_predictions(tmp_path):
    """Test that correct sequence predictions give a balanced accuracy
    of 1.0, even where individual frames are wrong.
    """
    gt_path, pred_path = _write_sequence_csvs(
        tmp_path,
        [
            {"image_id": "a_0", "label": 0, "sequence_id": "a"},
            {"image_id": "a_1", "label": 0, "sequence_id": "a"},
            {"image_id": "b_0", "label": 1, "sequence_id": "b"},
            {"image_id": "b_1", "label": 1, "sequence_id": "b"},
        ],
        [
            {"image_id": "a_0", "label": 0, "prob_0": 0.9, "prob_1": 0.1},
            {"image_id": "a_1", "label": 1, "prob_0": 0.4, "prob_1": 0.6},
            {"image_id": "b_0", "label": 1, "prob_0": 0.2, "prob_1": 0.8},
            {"image_id": "b_1", "label": 1, "prob_0": 0.3, "prob_1": 0.7},
        ],
    )

    ev = ClassificationEvaluator(
        gt_path, pred_path, tmp_path, aggregate_by="sequence_id"
    )
    assert ev.evaluate()["balanced_accuracy"] == pytest.approx(1.0)


def test_mean_probability_overrides_frame_majority(tmp_path):
    """Check that the mean probability, not a majority vote over frames,
    decides the sequence prediction.

    Two frames weakly favour class 0 and one frame strongly favours
    class 1, so a majority vote would predict class 0.
    """
    gt_path, pred_path = _write_sequence_csvs(
        tmp_path,
        [{"image_id": f"a_{i}", "label": 1, "sequence_id": "a"} for i in range(3)],
        [
            {"image_id": "a_0", "label": 0, "prob_0": 0.55, "prob_1": 0.45},
            {"image_id": "a_1", "label": 0, "prob_0": 0.55, "prob_1": 0.45},
            {"image_id": "a_2", "label": 1, "prob_0": 0.0, "prob_1": 1.0},
        ],
    )

    ev = ClassificationEvaluator(
        gt_path, pred_path, tmp_path, aggregate_by="sequence_id"
    )
    ev.evaluate()

    per_sequence = pd.read_csv(tmp_path / "per_sequence_predictions.csv")
    assert per_sequence["label_pred"].tolist() == [1]


def test_per_sequence_csv_written(tmp_path):
    """Check that the pooled per-sequence predictions are written out."""
    gt_path, pred_path = _write_sequence_csvs(
        tmp_path,
        [
            {"image_id": "a_0", "label": 0, "sequence_id": "a"},
            {"image_id": "b_0", "label": 1, "sequence_id": "b"},
        ],
        [
            {"image_id": "a_0", "label": 0, "prob_0": 0.7, "prob_1": 0.3},
            {"image_id": "b_0", "label": 1, "prob_0": 0.1, "prob_1": 0.9},
        ],
    )

    out_dir = tmp_path / "out"
    ev = ClassificationEvaluator(
        gt_path, pred_path, out_dir, aggregate_by="sequence_id"
    )
    ev.evaluate()

    per_sequence = pd.read_csv(out_dir / "per_sequence_predictions.csv")
    assert set(per_sequence.columns) == {
        "sequence_id",
        "label_gt",
        "label_pred",
        "prob_0",
        "prob_1",
    }
    assert per_sequence["sequence_id"].tolist() == ["a", "b"]


def test_missing_probability_columns_raise(tmp_path):
    """Check that sequence-level evaluation requires probabilities."""
    gt_path, pred_path = _write_sequence_csvs(
        tmp_path,
        [{"image_id": "a_0", "label": 0, "sequence_id": "a"}],
        [{"image_id": "a_0", "label": 0}],
    )

    ev = ClassificationEvaluator(
        gt_path, pred_path, tmp_path, aggregate_by="sequence_id"
    )
    with pytest.raises(ValueError, match="no 'prob_\\*' columns"):
        ev.evaluate()


def test_missing_sequence_column_raises(tmp_path):
    """Check that a ground truth without the grouping column is
    rejected.
    """
    gt_path, pred_path = _write_sequence_csvs(
        tmp_path,
        [{"image_id": "a_0", "label": 0}],
        [{"image_id": "a_0", "label": 0, "prob_0": 0.9, "prob_1": 0.1}],
    )

    ev = ClassificationEvaluator(
        gt_path, pred_path, tmp_path, aggregate_by="sequence_id"
    )
    with pytest.raises(ValueError, match="missing columns"):
        ev.evaluate()


def test_conflicting_sequence_labels_raise(tmp_path):
    """Check that a sequence with two ground-truth labels is rejected."""
    gt_path, pred_path = _write_sequence_csvs(
        tmp_path,
        [
            {"image_id": "a_0", "label": 0, "sequence_id": "a"},
            {"image_id": "a_1", "label": 1, "sequence_id": "a"},
        ],
        [
            {"image_id": "a_0", "label": 0, "prob_0": 0.9, "prob_1": 0.1},
            {"image_id": "a_1", "label": 1, "prob_0": 0.1, "prob_1": 0.9},
        ],
    )

    ev = ClassificationEvaluator(
        gt_path, pred_path, tmp_path, aggregate_by="sequence_id"
    )
    with pytest.raises(ValueError, match="more than one ground-truth label"):
        ev.evaluate()


def test_probability_columns_ignored_without_aggregation(tmp_path):
    """Check that predictions with probability columns still evaluate
    per image when no aggregation is requested.
    """
    gt_path, pred_path = _write_sequence_csvs(
        tmp_path,
        [
            {"image_id": "a_0", "label": 0, "sequence_id": "a"},
            {"image_id": "a_1", "label": 0, "sequence_id": "a"},
        ],
        [
            {"image_id": "a_0", "label": 0, "prob_0": 0.9, "prob_1": 0.1},
            {"image_id": "a_1", "label": 1, "prob_0": 0.4, "prob_1": 0.6},
        ],
    )

    ev = ClassificationEvaluator(gt_path, pred_path, tmp_path)
    assert ev.evaluate()["balanced_accuracy"] == pytest.approx(0.5)
    assert not (tmp_path / "per_sequence_predictions.csv").exists()
