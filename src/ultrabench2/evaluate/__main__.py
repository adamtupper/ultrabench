"""Entry point for the evaluation layer.

Usage::

    python -m ultrabench2.evaluate task=segmentation \\
        ground_truth=... predictions=...

Tasks with sequence-level labels (CAMUS image quality, COVID-BLUES,
MASLD) are scored on one prediction per sequence. Pass
``aggregate_by=sequence_id`` to average the predicted probabilities over
the images of each sequence before computing the metrics::

    python -m ultrabench2.evaluate task=classification \\
        aggregate_by=sequence_id ground_truth=... predictions=...
"""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build the configured evaluator and run evaluation.

    Args:
        cfg: Hydra configuration.
    """
    evaluator = hydra.utils.instantiate(cfg.task)
    metrics = evaluator.evaluate()
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
