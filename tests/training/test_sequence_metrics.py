"""Tests for the sequence-level test metrics of UltraBenchModule."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from ultrabench.evaluate.classification import ClassificationEvaluator
from ultrabench.training.lightning import UltraBenchModule


def make_module(
    num_classes: int,
    backbone: MagicMock | None = None,
    head: MagicMock | None = None,
) -> UltraBenchModule:
    """Build a module with stubbed components for metric testing.

    Args:
        num_classes: The number of output classes.
        backbone: Optional stub backbone. Only needed by tests that run a
            forward pass.
        head: Optional stub head. Only needed by tests that run a forward
            pass.

    Returns:
        A classification module with sequence aggregation enabled.
    """
    return UltraBenchModule(
        backbone=backbone or MagicMock(),
        head=head or MagicMock(),
        criterion=MagicMock(),
        optimizer_cfg=None,
        scheduler_cfg=None,
        task_type="classification",
        output_dir="unused",
        num_classes=num_classes,
        aggregate_sequences=True,
    )


def test_sequence_metrics_pool_probabilities():
    """Sequence metrics score one pooled prediction per sequence.

    Sequence "a" has two frames whose mean probability favours class 1
    even though each frame's argmax is class 0, so a correct
    implementation scores it right and a per-frame one scores it wrong.
    """
    module = make_module(2)
    module.on_test_epoch_start()
    module._test_sequence_ids = ["a", "a", "b", "b"]
    module._test_probabilities = [
        np.array([[0.6, 0.4], [0.0, 1.0], [0.9, 0.1], [0.8, 0.2]])
    ]
    module._test_targets = [1, 1, 0, 0]

    metrics = module._compute_sequence_metrics()

    assert metrics["seq_accuracy"] == pytest.approx(1.0)
    assert metrics["seq_balanced_accuracy"] == pytest.approx(1.0)


def test_sequence_metrics_empty_without_predictions():
    """No accumulated predictions yields no sequence metrics."""
    module = make_module(2)
    module.on_test_epoch_start()

    assert module._compute_sequence_metrics() == {}


def test_disabled_for_segmentation():
    """Sequence aggregation is ignored for segmentation tasks."""
    module = UltraBenchModule(
        backbone=MagicMock(),
        head=MagicMock(),
        criterion=MagicMock(),
        optimizer_cfg=None,
        scheduler_cfg=None,
        task_type="segmentation",
        output_dir="unused",
        num_classes=2,
        aggregate_sequences=True,
    )

    assert module.aggregate_sequences is False


def test_training_and_evaluation_layers_agree(tmp_path):
    """Both layers report the same sequence-level balanced accuracy.

    The training layer computes it inline with torchmetrics and the
    evaluation layer recomputes it from the CSVs with scikit-learn. They
    must not disagree.
    """
    rng = np.random.default_rng(0)
    sequence_ids = [f"s{i // 5}" for i in range(40)]
    sequence_labels = {sid: int(sid[1:]) % 3 for sid in sequence_ids}
    labels = [sequence_labels[sid] for sid in sequence_ids]
    probabilities = rng.dirichlet(np.ones(3), size=len(sequence_ids))

    module = make_module(3)
    module.on_test_epoch_start()
    module._test_sequence_ids = sequence_ids
    module._test_probabilities = [probabilities]
    module._test_targets = labels
    training_metrics = module._compute_sequence_metrics()

    ground_truth = tmp_path / "gt.csv"
    predictions = tmp_path / "pred.csv"
    pd.DataFrame(
        {
            "image_id": [f"img{i}" for i in range(len(sequence_ids))],
            "label": labels,
            "sequence_id": sequence_ids,
        }
    ).to_csv(ground_truth, index=False)
    pd.DataFrame(
        {
            "image_id": [f"img{i}" for i in range(len(sequence_ids))],
            "label": probabilities.argmax(axis=1),
            "prob_0": probabilities[:, 0],
            "prob_1": probabilities[:, 1],
            "prob_2": probabilities[:, 2],
        }
    ).to_csv(predictions, index=False)

    evaluator = ClassificationEvaluator(
        ground_truth, predictions, tmp_path, aggregate_by="sequence_id"
    )
    evaluation_metrics = evaluator.evaluate()

    assert training_metrics["seq_balanced_accuracy"] == pytest.approx(
        evaluation_metrics["balanced_accuracy"], abs=1e-3
    )


def test_test_step_records_probabilities():
    """The classification test step writes one probability per class."""
    module = make_module(
        3,
        backbone=MagicMock(return_value=SimpleNamespace(features=torch.zeros(2, 4))),
        head=MagicMock(return_value=torch.tensor([[2.0, 1.0, 0.0], [0.0, 0.0, 5.0]])),
    )
    module.on_test_epoch_start()

    module.test_step(
        {
            "image": torch.zeros(2, 3, 4, 4),
            "target": torch.tensor([0, 2]),
            "filename": ["a_0", "b_0"],
            "sequence_id": ["a", "b"],
        },
        batch_idx=0,
    )

    assert [row["image_id"] for row in module._test_rows] == ["a_0", "b_0"]
    assert [row["label"] for row in module._test_rows] == [0, 2]
    for row in module._test_rows:
        probs = [row[f"prob_{i}"] for i in range(3)]
        assert sum(probs) == pytest.approx(1.0)
    assert module._test_sequence_ids == ["a", "b"]
    assert module._test_targets == [0, 2]
