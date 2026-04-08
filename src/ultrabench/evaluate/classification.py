"""An evaluator for computing classification metrics given CSV files
containing a model's predictions and the ground-truth labels.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score

logger = logging.getLogger(__name__)


class ClassificationEvaluator:
    """Evaluate a classification model.

    Compute classification metrics given CSV files containing model
    predictions and ground-truth labels.

    Args:
        ground_truth_path: Path to the ground truth CSV with columns
            `image_id`, `label`.
        predictions_path: Path to the predictions CSV with the same
            columns.
        output_dir: Directory where the summary CSV with the computed
            metrics will be written.
    """

    def __init__(
        self,
        ground_truth_path: str | Path,
        predictions_path: str | Path,
        output_dir: str | Path,
    ) -> None:
        self.ground_truth_path = Path(ground_truth_path)
        self.predictions_path = Path(predictions_path)
        self.output_dir = Path(output_dir)

    def evaluate(self) -> dict[str, float]:
        """Evaluate the predictions against the ground truth.

        Compare the ground truth and predictions CSV files, compute the
        classification metrics, and write a summary CSV with the results.

        Returns:
            A dictionary containing the computed classification metrics.
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
        score = balanced_accuracy_score(merged["label_gt"], merged["label_pred"])
        metrics = {"balanced_accuracy": score}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = [{"metric": k, "value": v} for k, v in metrics.items()]
        pd.DataFrame(summary_rows).to_csv(self.output_dir / "summary.csv", index=False)

        return metrics
