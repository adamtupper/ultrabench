"""An evaluator for computing classification metrics given CSV files
containing a model's predictions and the ground-truth labels.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from ultrabench.core.aggregation import aggregate_sequence_probabilities

logger = logging.getLogger(__name__)

PROBABILITY_PREFIX = "prob_"
"""Prefix of the per-class probability columns in a predictions CSV."""


class ClassificationEvaluator:
    """Evaluate a classification model.

    Compute classification metrics given CSV files containing model
    predictions and ground-truth labels.

    Tasks whose labels apply to a whole sequence rather than a single
    image (CAMUS image quality, COVID-BLUES, MASLD) are scored per
    sequence. Set `aggregate_by` to pool the per-image predictions into
    one prediction per sequence by averaging the predicted probabilities
    over the images of the sequence.

    Args:
        ground_truth_path: Path to the ground truth CSV with columns
            `image_id`, `label`, and, for sequence-level tasks,
            `sequence_id`.
        predictions_path: Path to the predictions CSV with columns
            `image_id`, `label`, and, for sequence-level tasks, one
            `prob_{class}` column per class.
        output_dir: Directory where the summary CSV with the computed
            metrics will be written.
        aggregate_by: Ground-truth column to pool predictions over before
            computing metrics, normally `sequence_id`. If None, metrics
            are computed per image.
    """

    def __init__(
        self,
        ground_truth_path: str | Path,
        predictions_path: str | Path,
        output_dir: str | Path,
        aggregate_by: str | None = None,
    ) -> None:
        self.ground_truth_path = Path(ground_truth_path)
        self.predictions_path = Path(predictions_path)
        self.output_dir = Path(output_dir)
        self.aggregate_by = aggregate_by

    def evaluate(self) -> dict[str, float]:
        """Evaluate the predictions against the ground truth.

        Compare the ground truth and predictions CSV files, compute the
        classification metrics, and write a summary CSV with the results.

        Returns:
            A dictionary containing the computed classification metrics.

        Raises:
            ValueError: If either CSV is missing a required column, or if
                a sequence holds more than one ground-truth label.
        """
        gt = pd.read_csv(self.ground_truth_path)
        pred = pd.read_csv(self.predictions_path)

        for df, name in [(gt, "ground_truth"), (pred, "predictions")]:
            missing = {"image_id", "label"} - set(df.columns)
            if missing:
                raise ValueError(f"{name} CSV missing columns: {missing}")

        gt_images = set(gt["image_id"].unique())
        pred_images = set(pred["image_id"].unique())
        for image in gt_images - pred_images:
            logger.warning(f"Image in GT not in predictions: {image}")
        for image in pred_images - gt_images:
            logger.warning(f"Image in predictions not in GT: {image}")

        merged = gt.merge(pred, on="image_id", suffixes=("_gt", "_pred"))

        if self.aggregate_by is None:
            labels_gt = merged["label_gt"]
            labels_pred = merged["label_pred"]
        else:
            labels_gt, labels_pred = self._aggregate(merged)

        metrics = {"balanced_accuracy": balanced_accuracy_score(labels_gt, labels_pred)}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = [{"metric": k, "value": v} for k, v in metrics.items()]
        pd.DataFrame(summary_rows).to_csv(self.output_dir / "summary.csv", index=False)

        return metrics

    def _aggregate(self, merged: pd.DataFrame) -> tuple[list[int], list[int]]:
        """Pool per-image predictions into one prediction per group.

        Average the predicted probabilities over the images of each group
        and take the argmax. Writes the pooled predictions to
        `per_sequence_predictions.csv` in the output directory.

        Args:
            merged: The ground truth and predictions joined on `image_id`.

        Returns:
            A tuple of the ground-truth and predicted labels, one of each
            per group.

        Raises:
            ValueError: If the grouping column or the probability columns
                are absent, or if a group holds more than one
                ground-truth label.
        """
        if self.aggregate_by not in merged.columns:
            raise ValueError(
                f"ground_truth CSV missing columns: {{'{self.aggregate_by}'}}. It is "
                f"required to evaluate this task at the sequence level."
            )

        probability_columns = sorted(
            (c for c in merged.columns if c.startswith(PROBABILITY_PREFIX)),
            key=lambda c: int(c.removeprefix(PROBABILITY_PREFIX)),
        )
        if not probability_columns:
            raise ValueError(
                f"predictions CSV has no '{PROBABILITY_PREFIX}*' columns. "
                f"Sequence-level evaluation averages the predicted probabilities "
                f"over the images of a sequence, so one probability column per "
                f"class is required."
            )

        group_ids, probabilities, labels_gt = aggregate_sequence_probabilities(
            merged[self.aggregate_by].astype(str).tolist(),
            merged[probability_columns].to_numpy(),
            merged["label_gt"].tolist(),
        )
        labels_pred = probabilities.argmax(axis=1).tolist()
        logger.info(
            "Aggregated %d images into %d sequences by '%s'.",
            len(merged),
            len(group_ids),
            self.aggregate_by,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        results = pd.DataFrame(
            {
                self.aggregate_by: group_ids,
                "label_gt": labels_gt,
                "label_pred": labels_pred,
            }
        )
        for index, column in enumerate(probability_columns):
            results[column] = probabilities[:, index]
        results.to_csv(self.output_dir / "per_sequence_predictions.csv", index=False)

        return labels_gt, labels_pred  # ty:ignore[invalid-return-type]
