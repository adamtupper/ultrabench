"""Unit tests for generating ground-truth CSVs."""

import dataclasses

import datasets
from hydra import compose, initialize_config_module
import numpy as np
import pandas as pd
import pytest

from ultrabench.evaluate.ground_truth.builder import (
    _write_classification_ground_truth,
    _write_segmentation_ground_truth,
    ground_truth_path,
)
from ultrabench.evaluate.tasks import get_task
from ultrabench.core.rle import rle_to_mask

SIZE = 8


def _classification_split(with_sequences: bool) -> datasets.Dataset:
    """Build a small classification test split.

    Args:
        with_sequences: Whether to include the sequence key columns.

    Returns:
        The test split.
    """
    records = [
        {"filename": "p1_a.png", "image_quality": 0, "patient": 1, "view": "2CH"},
        {"filename": "p1_b.png", "image_quality": 0, "patient": 1, "view": "2CH"},
        {"filename": "p2_a.png", "image_quality": 1, "patient": 2, "view": "4CH"},
    ]
    if not with_sequences:
        records = [
            {"filename": r["filename"], "image_quality": r["image_quality"]}
            for r in records
        ]

    return datasets.Dataset.from_list(records)


def _segmentation_split(size: int = SIZE) -> datasets.Dataset:
    """Build a small segmentation test split with two foreground classes.

    Args:
        size: The height and width of the masks.

    Returns:
        The test split.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[: size // 2, : size // 2] = 1
    mask[size // 2 :, size // 2 :] = 2

    return datasets.Dataset.from_list(
        [
            {"filename": "a.png", "psfh_mask": mask.tolist()},
            {"filename": "b.png", "psfh_mask": mask.tolist()},
        ]
    )


def test_classification_ground_truth_columns(tmp_path):
    """Check that a per-image task gets image IDs and labels."""
    spec = dataclasses.replace(get_task("camus_classification"), sequence_by=None)
    path = tmp_path / "gt.csv"

    rows = _write_classification_ground_truth(
        spec, _classification_split(with_sequences=False), path
    )

    frame = pd.read_csv(path)
    assert rows == 3
    assert list(frame.columns) == ["image_id", "label"]


def test_sequence_ground_truth_carries_the_sequence_id(tmp_path):
    """Check that a sequence task gets a `sequence_id` column.

    Without it the evaluator cannot pool frames, and the task would be
    scored per frame, weighting patients by their frame count.
    """
    spec = get_task("camus_classification")
    path = tmp_path / "gt.csv"

    _write_classification_ground_truth(
        spec, _classification_split(with_sequences=True), path
    )

    frame = pd.read_csv(path)
    assert list(frame.columns) == ["image_id", "label", "sequence_id"]
    assert frame["sequence_id"].tolist() == ["1_2CH", "1_2CH", "2_4CH"]


def test_segmentation_ground_truth_has_a_row_per_foreground_class(tmp_path):
    """Check that the background is excluded and each class gets a row."""
    spec = dataclasses.replace(get_task("psfhs_segmentation"), image_size=SIZE)
    path = tmp_path / "gt.csv"

    rows = _write_segmentation_ground_truth(spec, _segmentation_split(), path)

    frame = pd.read_csv(path)
    assert rows == 4
    assert list(frame.columns) == ["image_id", "class_id", "encoded_pixels"]
    assert sorted(frame["class_id"].unique()) == [1, 2]


def test_segmentation_ground_truth_masks_decode_at_the_task_size(tmp_path):
    """Check that the encoded masks match the size of the predictions.

    The published segmentation ground truth encodes full-resolution
    masks despite its `test.256` name, which silently produces a
    near-zero score. Regenerated ground truth must not repeat that.
    """
    spec = dataclasses.replace(get_task("psfhs_segmentation"), image_size=SIZE)
    path = tmp_path / "gt.csv"

    _write_segmentation_ground_truth(spec, _segmentation_split(SIZE), path)

    frame = pd.read_csv(path, converters={"encoded_pixels": str})
    for encoded in frame["encoded_pixels"]:
        assert rle_to_mask(encoded, SIZE, SIZE).sum() == (SIZE // 2) ** 2


def test_segmentation_ground_truth_rejects_a_mismatched_mask(tmp_path):
    """Check that a mask of the wrong size is rejected up front."""
    spec = get_task("psfhs_segmentation")

    with pytest.raises(ValueError, match="expected \\(256, 256\\)"):
        _write_segmentation_ground_truth(
            spec, _segmentation_split(SIZE), tmp_path / "gt.csv"
        )


def test_ground_truth_path_is_named_after_the_task(tmp_path):
    """Check that each task writes to its own CSV."""
    spec = get_task("tn3k_segmentation")

    assert ground_truth_path(spec, tmp_path).name == "tn3k_segmentation.csv"


def test_cli_config_supplies_the_builder_arguments():
    """Check that the CLI config composes into the builder's arguments.

    The ground-truth CLI is the only entry point into this module, so a
    config key renamed out from under it would otherwise go unnoticed.
    """
    with initialize_config_module(
        config_module="ultrabench.evaluate.ground_truth.conf", version_base=None
    ):
        cfg = compose(
            "config",
            overrides=[
                "output_dir=/tmp/ground_truth",
                "tasks=[tn3k_segmentation]",
                "overwrite=true",
            ],
        )

    assert cfg.output_dir == "/tmp/ground_truth"
    assert list(cfg.tasks) == ["tn3k_segmentation"]
    assert cfg.overwrite is True
