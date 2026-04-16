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
